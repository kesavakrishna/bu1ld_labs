"""
The five warning signals, and how each one becomes a score (Day 4).

Every signal is measured during a single model call, from three things: the wave that
goes in, the wave that comes out, and the activations captured inside the network.

Internal signals (need to see inside the network):
    act_norm        Are the activations getting bigger?
    eff_rank        How many independent patterns are the activations using?
    train_dist      How far are the activations from what is normal at this step?

Output-side signals (need only the network's prediction):
    step_change     How much did this prediction change the wave?
    spectral_drift  What share of the prediction's energy sits in the finest wiggles?

Turning a signal into a score
-----------------------------
Correct physics changes a lot over a trajectory, so a raw signal can't tell "the physics
is dramatic right now" from "the model is going wrong". So each signal is compared with
reference runs, where the model is given the true wave at every step:

    score(t) = |signal(t) - normal mean at step t| / normal spread at step t

A score around 1 or below is ordinary. Bigger means more unusual for that step.
"""

import torch

INTERNAL = ["act_norm", "eff_rank", "train_dist"]
OUTPUT = ["step_change", "spectral_drift"]
SIGNALS = INTERNAL + OUTPUT


def effective_rank(activations):
    """
    How many independent patterns the activations use.

    Take the singular values of each [channels, n_points] matrix, turn them into shares
    that add up to 1, and compute exp(entropy) of those shares. If all 64 shares were
    equal the answer would be 64; if one pattern dominated, it would be close to 1.

    Returns NaN where the activations aren't finite, because the SVD can't run on those.
    """
    rank = torch.full(activations.shape[:1], float("nan"), device=activations.device)
    finite = torch.isfinite(activations).all(dim=(1, 2))
    if finite.any():
        a = activations[finite]
        # Shares don't depend on overall size, so scale each matrix to at most 1 first.
        # This keeps the maths stable when a diverging rollout makes the numbers huge.
        a = a / a.abs().amax(dim=(1, 2), keepdim=True)
        # Singular values of A are the square roots of the eigenvalues of the small
        # [channels, channels] matrix A·Aᵀ. Same answer as torch.linalg.svdvals(a) to about
        # 0.003%, but around 200x faster on the GPU, which matters at one call per step.
        singular_values = torch.linalg.eigvalsh(a @ a.transpose(1, 2)).clamp_min(0).sqrt()
        shares = singular_values / singular_values.sum(dim=-1, keepdim=True)
        rank[finite] = torch.exp(torch.special.entr(shares).sum(dim=-1))  # entr(p) = -p·log(p)
    return rank


def fine_detail_share(wave):
    """
    Share of each wave's energy in the top third of its sine-wave modes.

    True waves have no energy there at all, because the simulator's 2/3 rule keeps that
    band empty. So anything in it was put there by the model.
    """
    energy_per_mode = torch.fft.rfft(wave).abs() ** 2
    first_fine_mode = 2 * energy_per_mode.shape[-1] // 3
    return energy_per_mode[..., first_fine_mode:].sum(dim=-1) / energy_per_mode.sum(dim=-1)


def measure(activations, wave_in, wave_out):
    """
    Raw signals from one model call, for a batch of trajectories.

    Args:
        activations: the captured layer output, [batch, channels, n_points]
        wave_in:     the wave given to the model, [batch, n_points]
        wave_out:    the model's prediction, [batch, n_points]

    Returns a dict of [batch] tensors, plus "summary" ([batch, 2 * channels]). train_dist
    is built from the summary later, once the normal summary for each step is known.
    """
    norm = torch.linalg.vector_norm
    return {
        "act_norm": norm(activations, dim=-1).mean(dim=-1),
        "eff_rank": effective_rank(activations),
        # Each channel's average and spread across the grid: a short description of the state.
        "summary": torch.cat([activations.mean(dim=-1), activations.std(dim=-1)], dim=-1),
        "step_change": norm(wave_out - wave_in, dim=-1) / norm(wave_in, dim=-1),
        "spectral_drift": fine_detail_share(wave_out),
    }


def add_train_dist(measurements, normal_summary):
    """
    Add train_dist: the cosine distance between each step's activation summary and the
    normal summary for that step. 0 means pointing the same way as normal; up to 2 means
    pointing the opposite way.
    """
    similarity = torch.nn.functional.cosine_similarity(measurements["summary"], normal_summary, dim=-1)
    return {**measurements, "train_dist": 1 - similarity}


def reference_statistics(reference):
    """
    What each signal normally looks like at each step, from reference runs.

    Args:
        reference: measurements from runs fed the true wave at every step, each shaped
                   [n_reference, n_steps], plus "summary" [n_reference, n_steps, 2 * channels]

    Returns a dict holding the normal summary at each step ([n_steps, 2 * channels]) and,
    for every signal, its normal mean and spread at each step ([n_steps] each).
    """
    normal = {"summary": reference["summary"].mean(dim=0)}
    reference = add_train_dist(reference, normal["summary"])
    for name in SIGNALS:
        normal[name] = (reference[name].mean(dim=0), reference[name].std(dim=0))
    return normal


def scores(measurements, normal):
    """
    Turn every signal into a score: how many normal spreads it sits from the normal mean
    at the same step. Call add_train_dist on the measurements first.

    Returns a dict of [n_trajectories, n_steps] tensors.
    """
    result = {}
    for name in SIGNALS:
        mean, spread = normal[name]
        score = (measurements[name] - mean).abs() / spread
        # A reading that overflowed is as unusual as it gets, so it counts as an alarm.
        score[~torch.isfinite(score)] = float("inf")
        # Step 0 is the true starting wave. No model call has happened, so nothing to score.
        score[:, 0] = 0
        result[name] = score
    return result
