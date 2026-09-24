"""
The neural network that imitates the simulator: a 1D Fourier Neural Operator (FNO).

    input:  the wave now,            shape [batch, n_points]
    output: the wave one step later, shape [batch, n_points]

Why an FNO and not an ordinary convolutional network
----------------------------------------------------
The simulator works on the wave's sine-wave ingredients (see solver.py), and so does
this network. Each layer runs two paths and adds them together:

  * The Fourier path mixes the broad shapes of the wave. Every point can affect every
    other point in a single layer, which suits physics, where information travels.
  * The pointwise path mixes channels at each grid point on its own. It carries the
    fine detail that the Fourier path throws away, such as sharp shock fronts.

Scaling is built in
-------------------
forward() takes a real wave and returns a real wave, and does the scaling internally.
Rollout code therefore cannot forget to undo it, which is an easy bug that looks
exactly like the drift we are trying to study.

Day 4 reads activations from `model.layers[hook_layer - 1]` with a PyTorch forward hook,
without changing anything in this file.
"""

import torch
from torch import nn


class SpectralConv1d(nn.Module):
    """
    The Fourier path of one layer.

    Converts the input to sine-wave ingredients, keeps the `modes` broadest ones, mixes
    the channels with learned weights (a separate width x width matrix for each kept
    mode), drops the rest, and converts back.
    """

    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        # One complex [width, width] mixing matrix per kept mode. Dividing by sqrt(width)
        # keeps the numbers a sensible size as they pass through the layers.
        self.weights = nn.Parameter(torch.randn(width, width, modes, dtype=torch.cfloat) / width**0.5)

    def forward(self, x):  # x: [batch, width, n_points]
        n_points = x.shape[-1]
        x_hat = torch.fft.rfft(x)  # [batch, width, n_points // 2 + 1]

        # Anything we don't fill in stays zero, which is how the finer modes get dropped.
        out_hat = torch.zeros_like(x_hat)
        # "bim,oim->bom": for each kept mode m, mix input channels i into output channels o.
        out_hat[..., : self.modes] = torch.einsum(
            "bim,oim->bom", x_hat[..., : self.modes], self.weights
        )

        return torch.fft.irfft(out_hat, n=n_points)


class FourierLayer(nn.Module):
    """One FNO layer: the Fourier path plus the pointwise path, then an activation function."""

    def __init__(self, width, modes):
        super().__init__()
        self.spectral = SpectralConv1d(width, modes)
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)  # a learned mix at each point

    def forward(self, x):
        return nn.functional.gelu(self.spectral(x) + self.pointwise(x))


class FNO1d(nn.Module):
    """
    The full model: widen the input, run several Fourier layers, narrow back to one wave.

    Args:
        modes:    how many of the broadest sine-wave modes each layer keeps
        width:    channels used inside the network
        n_layers: how many Fourier layers
        predict:  "direct" predicts the next wave; "residual" predicts the change and adds it on
    """

    def __init__(self, modes, width, n_layers, predict="direct"):
        super().__init__()
        self.predict = predict
        self.lift = nn.Conv1d(2, width, kernel_size=1)  # 2 numbers per point: the wave, and the position
        self.layers = nn.ModuleList(FourierLayer(width, modes) for _ in range(n_layers))
        self.project = nn.Sequential(
            nn.Conv1d(width, 128, kernel_size=1), nn.GELU(), nn.Conv1d(128, 1, kernel_size=1)
        )

        # The training set's average and spread, used to scale inputs and outputs.
        # Stored as buffers so they are saved and loaded along with the weights.
        self.register_buffer("mean", torch.zeros(()))
        self.register_buffer("std", torch.ones(()))

    def set_normalization(self, mean, std):
        """Record the training set's average and spread. Call this once before training."""
        self.mean.fill_(float(mean))
        self.std.fill_(float(std))

    def forward(self, u):  # u: [batch, n_points], a real wave
        n_points = u.shape[-1]
        scaled = (u - self.mean) / self.std

        # Tell the network where each point sits on the ring, as a second input channel.
        position = torch.arange(n_points, device=u.device, dtype=u.dtype) / n_points
        x = torch.stack([scaled, position.expand_as(scaled)], dim=1)  # [batch, 2, n_points]

        x = self.lift(x)
        for layer in self.layers:
            x = layer(x)
        out = self.project(x).squeeze(1)

        if self.predict == "residual":
            out = scaled + out
        return out * self.std + self.mean

    def n_parameters(self):
        """How many numbers the network learns."""
        return sum(p.numel() * (2 if p.is_complex() else 1) for p in self.parameters())
