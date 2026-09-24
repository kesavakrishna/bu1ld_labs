"""
Day 4 check: what do the warning signals do as a rollout approaches failure?
(PROJECT_PLAN.md, Section 9)

Every failing test rollout fails at a different step. To compare them, we line them up
so that step 0 on the x-axis is each rollout's own failure step: -20 means "20 steps
before this rollout failed". Then, for each signal, we show the middle (median) score
and the middle half of scores (25th to 75th percentile) at each of those offsets.

A useful signal climbs clearly above normal BEFORE 0. A flat one stays near normal.
This is a look, not the verdict: Days 5 and 6 tune thresholds fairly and measure lead time.
It uses the dev split, never test.

    python src/check_signals.py --config configs/full.yaml
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from leadtime import failure_steps
from signals import INTERNAL, OUTPUT, SIGNALS

STEPS_BEFORE, STEPS_AFTER = 60, 20  # how much of each rollout to show around its failure
DISPLAY_CAP = 1e6  # overflowed (infinite) scores are drawn at this height; they can't be plotted


def line_up_at_failure(values, failed_at):
    """
    Cut a window around each rollout's failure step, so row i, column j holds the value
    at (failure step + offset j). Parts of the window outside the rollout are NaN.

    Returns the windows [n_failing, STEPS_BEFORE + STEPS_AFTER + 1] and the offsets.
    """
    offsets = np.arange(-STEPS_BEFORE, STEPS_AFTER + 1)
    steps = failed_at[:, None] + offsets[None, :]
    inside = (steps >= 1) & (steps < values.shape[1])  # step 0 has no model call, so no score
    windows = np.full(steps.shape, np.nan)
    rows = np.arange(len(failed_at))[:, None].repeat(len(offsets), axis=1)
    windows[inside] = values[rows[inside], steps[inside]]
    return windows, offsets


def save_figure(aligned, examples, offsets, threshold, path):
    """One panel for the error and one per signal, all lined up at failure."""
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
        "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelcolor": "#52514e", "xtick.color": "#898781", "ytick.color": "#898781",
    })
    colour = {"relative_error": "#52514e", **{n: "#2a78d6" for n in INTERNAL}, **{n: "#eb6834" for n in OUTPUT}}
    kind = {**{n: "internal" for n in INTERNAL}, **{n: "output-side" for n in OUTPUT}}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, constrained_layout=True)
    for ax, name in zip(axes.flat, ["relative_error"] + SIGNALS):
        windows = np.minimum(aligned[name], DISPLAY_CAP)
        scale = 100 if name == "relative_error" else 1

        ax.plot(offsets, scale * np.minimum(examples[name], DISPLAY_CAP).T, color=colour[name], linewidth=0.8, alpha=0.25)
        low, middle, high = np.nanpercentile(windows, [25, 50, 75], axis=0)
        ax.fill_between(offsets, scale * low, scale * high, color=colour[name], alpha=0.2, linewidth=0)
        ax.plot(offsets, scale * middle, color=colour[name], linewidth=2.2)

        ax.axvline(0, color="#52514e", linewidth=1, linestyle="--")
        if name == "relative_error":
            ax.axhline(100 * threshold, color="#52514e", linewidth=1, linestyle=":")
            ax.set(title="Relative error (%)", ylabel="relative error (%)")
        else:
            ax.axhline(1, color="#52514e", linewidth=1, linestyle=":")
            ax.set(title=f"{name} ({kind[name]})", ylabel="score (1 = one normal spread)")
        ax.set_yscale("log")

    for ax in axes[1]:
        ax.set_xlabel("steps relative to failure (0 = the failure step)")

    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Day 4: signals lined up at failure.")
    parser.add_argument("--config", default="configs/full.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    threshold = config["rollout"]["failure_threshold"]
    rollout_dir = Path(config["results_dir"]) / "rollouts"

    # Pool the failing dev rollouts from every seed.
    aligned = {name: [] for name in ["relative_error"] + SIGNALS}
    for seed in config["training"]["seeds"]:
        data = np.load(rollout_dir / f"dev_seed{seed}.npz")
        failed_at = failure_steps(data["relative_error"], threshold)
        failing = failed_at >= 0
        for name in aligned:
            key = name if name == "relative_error" else f"score_{name}"
            windows, offsets = line_up_at_failure(data[key][failing], failed_at[failing])
            aligned[name].append(windows)
    aligned = {name: np.concatenate(parts) for name, parts in aligned.items()}
    examples = {name: windows[:10] for name, windows in aligned.items()}  # thin lines: 10 rollouts

    print(f"\nDay 4: median score around failure ({len(aligned['relative_error'])} failing dev rollouts, all seeds)")
    print("A score of 1 is one normal spread; for pure noise the median would be about 0.67.\n")
    show = [-40, -20, -10, -5, 0]
    print(f"  {'signal':>15} {'type':>12}" + "".join(f"{f'{k} steps':>11}" if k else f"{'at failure':>11}" for k in show))
    for name in SIGNALS:
        medians = [np.nanmedian(aligned[name][:, offsets == k]) for k in show]
        kind = "internal" if name in INTERNAL else "output-side"
        print(f"  {name:>15} {kind:>12}" + "".join(f"{m:>11.2f}" for m in medians))
    counts = [int(np.isfinite(aligned["relative_error"][:, offsets == k]).sum()) for k in show]
    print(f"  {'rollouts':>15} {'':>12}" + "".join(f"{c:>11}" for c in counts))
    print("  (early failures have no data 40 steps before, so fewer rollouts count there)")

    figure_path = Path(config["results_dir"]) / "day4_signals.png"
    save_figure(aligned, examples, offsets, threshold, figure_path)
    print(f"\n  Figure saved to {figure_path}")


if __name__ == "__main__":
    main()
