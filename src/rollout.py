"""
Run each trained model on its own predictions, measuring errors and warning signals
(Days 3 and 4).

For every saved model:

  1. Reference runs. Feed the model the TRUE wave at every step, on training
     trajectories, to learn what each signal normally looks like at each step.
  2. Rollouts. Start from the true wave at step 0 and feed the model its own
     predictions from then on, for the calibration, dev and test splits. At every step,
     record the errors and every signal, then score each signal against the reference.

Errors recorded at every step:
  * relative error - size of the mistake divided by size of the true wave
  * raw error      - size of the mistake on its own
  * true size      - size of the true wave on its own

Raw error and true size are kept separately, because the true wave shrinks over time and
a mistake of constant size would still become a bigger percentage (PROJECT_PLAN.md,
Section 12).

Activations are read with a PyTorch forward hook (ActivationCatcher), so the model's code
is never changed. To use a different model, hand ActivationCatcher any of its layers.

Everything is saved to results/rollouts/ for Days 5 and 6.

    python src/rollout.py --config configs/full.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from model import FNO1d
from signals import SIGNALS, add_train_dist, measure, reference_statistics, scores

# Calibration sets thresholds, dev feeds the setup checks, test is scored once, in the frozen
# run. Training trajectories are only used for the reference runs.
SPLITS = ["calibration", "dev", "test"]


class ActivationCatcher:
    """
    Keeps the most recent output of one layer, using a PyTorch forward hook.

    A forward hook is a function PyTorch calls every time the layer runs. Ours just stores
    the layer's output. Works on any nn.Module, not only this project's FNO.
    """

    def __init__(self, layer):
        self.activations = None
        self._handle = layer.register_forward_hook(self._store)

    def _store(self, layer, inputs, output):
        self.activations = output.detach()

    def remove(self):
        self._handle.remove()


def load_model(path, device):
    """Load a model saved by train.py, set up exactly as it was when trained."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = FNO1d(**checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def run_model(model, catcher, u_true, feed_back):
    """
    Call the model once per step, measuring errors and signals along the way.

    Args:
        u_true:    the true waves, [n_trajectories, n_steps, n_points], on the model's device
        feed_back: True for a rollout (each prediction becomes the next input).
                   False for a reference run (the true wave is the input at every step).

    Returns:
        relative_error, raw_error, true_size: [n_trajectories, n_steps] each
        signals: dict of raw signals, [n_trajectories, n_steps] each (summary has an extra axis)

    Column s holds what was measured during the model call that produced the prediction
    for step s. The activations, the prediction and its error all come out of that one call
    at the same moment, so they share an index. Column 0 is the true starting wave, before
    any model call: its errors are 0 and its signals are NaN.
    """
    n_trajectories, n_steps, _ = u_true.shape
    true_size = torch.linalg.vector_norm(u_true, dim=-1).cpu()
    relative_error = torch.zeros(n_trajectories, n_steps)
    raw_error = torch.zeros(n_trajectories, n_steps)
    measured = []  # one dict of signals per step

    wave = u_true[:, 0]
    for step in range(1, n_steps):
        prediction = model(wave)  # the hook catches the activations during this call
        measured.append({name: values.cpu() for name, values in measure(catcher.activations, wave, prediction).items()})

        mistake = torch.linalg.vector_norm(prediction - u_true[:, step], dim=-1).cpu()
        raw_error[:, step] = mistake
        relative_error[:, step] = mistake / true_size[:, step]

        wave = prediction if feed_back else u_true[:, step]

    # Stack the steps together, with a NaN column in front for step 0.
    signals = {}
    for name in measured[0]:
        stacked = torch.stack([m[name] for m in measured], dim=1)
        signals[name] = torch.cat([torch.full_like(stacked[:, :1], float("nan")), stacked], dim=1)

    return relative_error, raw_error, true_size, signals


def main():
    parser = argparse.ArgumentParser(description="Run rollouts and measure warning signals.")
    parser.add_argument("--config", default="configs/full.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    data_dir = Path(config["data_dir"])
    output_dir = Path(config["results_dir"]) / "rollouts"
    output_dir.mkdir(parents=True, exist_ok=True)

    train = np.load(data_dir / "train.npz")
    n_reference = config["signals"]["n_reference"]
    reference_ids = train["ids"][:n_reference]
    reference_waves = torch.from_numpy(train["u"][:n_reference]).to(device)

    for seed in config["training"]["seeds"]:
        model = load_model(Path(config["results_dir"]) / "models" / f"model_seed{seed}.pt", device)
        layer = model.layers[config["signals"]["hook_layer"] - 1]  # config counts layers from 1
        catcher = ActivationCatcher(layer)

        # 1. What "normal" looks like at each step.
        _, _, _, reference = run_model(model, catcher, reference_waves, feed_back=False)
        normal = reference_statistics(reference)
        np.savez(
            output_dir / f"reference_seed{seed}.npz",
            ids=reference_ids,
            summary=normal["summary"].numpy(),
            **{f"{name}_mean": normal[name][0].numpy() for name in SIGNALS},
            **{f"{name}_spread": normal[name][1].numpy() for name in SIGNALS},
        )

        # 2. Rollouts, scored against normal.
        for split in SPLITS:
            data = np.load(data_dir / f"{split}.npz")
            assert not set(data["ids"]) & set(reference_ids), f"{split} overlaps the reference trajectories"

            u_true = torch.from_numpy(data["u"]).to(device)
            relative_error, raw_error, true_size, signals = run_model(model, catcher, u_true, feed_back=True)
            signals = add_train_dist(signals, normal["summary"])
            signal_scores = scores(signals, normal)

            np.savez(
                output_dir / f"{split}_seed{seed}.npz",
                relative_error=relative_error.numpy(),
                raw_error=raw_error.numpy(),
                true_size=true_size.numpy(),
                ids=data["ids"],
                seed=seed,
                **{name: signals[name].numpy() for name in SIGNALS},
                **{f"score_{name}": signal_scores[name].numpy() for name in SIGNALS},
            )
            print(f"  seed {seed} {split:>11}: {len(u_true)} rollouts, "
                  f"median error at the last step {100 * relative_error[:, -1].median():.1f}%")

        catcher.remove()


if __name__ == "__main__":
    main()
