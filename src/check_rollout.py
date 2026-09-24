"""
Day 3 checks: does the setup gate pass? (PROJECT_PLAN.md, Section 8)

Reads the dev rollouts made by rollout.py and asks:

  1. Do enough rollouts actually fail?                  pass / fail
  2. Do they fail late enough to leave warning room?    pass / fail
  3. Is "failure" really just the true wave shrinking?  reported, and shown in the figure

Check 3 matters because relative error divides by the size of the true wave, and that
wave loses energy over time. A mistake that never grows still becomes a bigger
percentage. So we report how much the wave shrank and how much the raw error grew, and
let the two be compared.

Uses the dev split, never test. If the gate fails, the script stops with an error so
run_all.sh goes no further.

    python src/check_rollout.py --config configs/full.yaml
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from leadtime import failure_steps

# Shocks have formed by roughly this step (see the Day 1 figure). The shrinking
# diagnostic measures growth and shrinkage from here, not from step 0.
REFERENCE_STEP = 20


def save_figure(rollouts, failures, threshold, path):
    """Three panels: error curves, the shrinking check, and when rollouts fail."""
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
        "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelcolor": "#52514e", "xtick.color": "#898781", "ytick.color": "#898781",
    })
    blue, orange, green = "#2a78d6", "#eb6834", "#1baf7a"
    seeds = sorted(rollouts)
    first = rollouts[seeds[0]]
    steps = np.arange(first["relative_error"].shape[1])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)

    # Panel 1: every test rollout's error, for the first seed.
    ax = axes[0]
    ax.plot(steps[1:], 100 * first["relative_error"][:, 1:].T, color=blue, linewidth=1, alpha=0.35)
    ax.axhline(100 * threshold, color="#52514e", linewidth=1, linestyle="--")
    ax.text(steps[-1], 100 * threshold, "failure line", color="#52514e", ha="right", va="bottom")
    ax.set(title=f"Rollout error, seed {seeds[0]}", xlabel="step", ylabel="relative error (%)", yscale="log")

    # Panel 2: is failure the wave shrinking? Both lines are wave sizes, same units.
    ax = axes[1]
    show = slice(0, 10)
    ax.plot(steps, first["true_size"][show].T, color=blue, linewidth=1.2, alpha=0.7)
    ax.plot(steps[1:], first["raw_error"][show, 1:].T, color=orange, linewidth=1.2, alpha=0.7)
    ax.plot([], [], color=blue, linewidth=2, label="size of the true wave")
    ax.plot([], [], color=orange, linewidth=2, label="size of the mistake")
    ax.set(title="Is failure just the wave shrinking?", xlabel="step", ylabel="size", yscale="log")
    # Rollouts that blow up reach enormous values and would squash everything else into a
    # line, so cut the view just above the true wave. Those lines leave the top of the panel.
    ax.set_ylim(top=5 * first["true_size"].max())
    ax.legend(frameon=False, loc="lower left")

    # Panel 3: when rollouts fail.
    ax = axes[2]
    edges = np.linspace(0, steps[-1], 21)
    for seed, shade in zip(seeds, [blue, orange, green]):
        failed = failures[seed][failures[seed] >= 0]
        ax.hist(failed, bins=edges, histtype="step", linewidth=2, color=shade, label=f"seed {seed}")
    ax.set(title="When rollouts fail", xlabel="failure step", ylabel="rollouts")
    ax.legend(frameon=False)

    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Day 3 setup gate checks.")
    parser.add_argument("--config", default="configs/full.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    settings = config["rollout"]
    threshold = settings["failure_threshold"]

    rollout_dir = Path(config["results_dir"]) / "rollouts"
    rollouts = {seed: np.load(rollout_dir / f"dev_seed{seed}.npz") for seed in config["training"]["seeds"]}
    failures = {seed: failure_steps(data["relative_error"], threshold) for seed, data in rollouts.items()}

    print(f"\nDay 3 setup gate (dev rollouts, failure = relative error above {100 * threshold:.0f}%)\n")
    print(f"  {'seed':>4} {'rollouts that fail':>19} {'median failure step':>20} "
          f"{'wave size at failure':>21} {'mistake growth':>15}   gate")

    all_pass = True
    for seed, data in rollouts.items():
        failed_at = failures[seed]
        failed = failed_at >= 0
        fraction = failed.mean()
        median_step = np.median(failed_at[failed]) if failed.any() else float("nan")

        # For rollouts that failed: how much did the true wave shrink, and how much did the
        # raw mistake grow, between REFERENCE_STEP and the failure step? If the wave shrank
        # a lot while the mistake barely grew, "failure" is mostly shrinking.
        rows = np.flatnonzero(failed)
        at_failure = failed_at[failed]
        wave_ratio = data["true_size"][rows, at_failure] / data["true_size"][rows, REFERENCE_STEP]
        mistake_ratio = data["raw_error"][rows, at_failure] / data["raw_error"][rows, REFERENCE_STEP]

        passed = fraction >= settings["min_failing_fraction"] and median_step >= settings["min_median_failure_step"]
        all_pass = all_pass and passed

        print(f"  {seed:>4} {100 * fraction:>17.0f}% {median_step:>20.0f} "
              f"{100 * np.median(wave_ratio):>20.0f}% {np.median(mistake_ratio):>14.1f}x   "
              f"{'PASS' if passed else 'FAIL'}")

    print(f"\n  To pass, a seed needs at least {100 * settings['min_failing_fraction']:.0f}% of rollouts failing")
    print(f"  and a median failure step of {settings['min_median_failure_step']} or later.")
    print(f"  The last two columns are measured from step {REFERENCE_STEP}, once shocks have formed:")
    print("  the true wave's size at failure, and how much the raw mistake grew.")
    print(f"\n  Gate: {'PASS' if all_pass else 'FAIL'}")

    figure_path = Path(config["results_dir"]) / "day3_rollouts.png"
    save_figure(rollouts, failures, threshold, figure_path)
    print(f"  Figure saved to {figure_path}")

    if not all_pass:
        sys.exit("Day 3 gate failed: the setup isn't usable for measuring warnings. Stopping here.")


if __name__ == "__main__":
    main()
