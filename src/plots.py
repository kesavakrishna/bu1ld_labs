"""
Draw the headline figure from leadtime.py's results (Day 6).

  Left:  how much warning each alarm gives, across every failing test rollout and seed.
  Right: internal against output-side alarms, rollout by rollout.

Boxes cover the middle half of values (25th to 75th percentile) and whiskers run from the
5th to the 95th. Left of the dashed line means too late.

    python src/plots.py --config configs/full.yaml
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from leadtime import ALARMS
from signals import INTERNAL, OUTPUT

COLOUR = {**{n: "#2a78d6" for n in INTERNAL}, **{n: "#eb6834" for n in OUTPUT}, "clock": "#898781"}


def box_rows(ax, rows, colours):
    """Horizontal boxes, first row at the top. `rows` is a list of arrays."""
    positions = np.arange(len(rows), 0, -1)
    boxes = ax.boxplot(rows, positions=positions, vert=False, widths=0.55, whis=(5, 95),
                       showfliers=False, patch_artist=True, medianprops={"color": "#0b0b0b", "linewidth": 2})
    for patch, colour in zip(boxes["boxes"], colours):
        patch.set(facecolor=colour, alpha=0.35, edgecolor=colour)
    ax.axvline(0, color="#52514e", linewidth=1, linestyle="--")
    return positions


def main():
    parser = argparse.ArgumentParser(description="Draw the lead-time figure.")
    parser.add_argument("--config", default="configs/full.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    results_dir = Path(config["results_dir"])
    with open(results_dir / "summary.json") as f:
        summary = json.load(f)
    data = np.load(results_dir / "leadtimes.npz")

    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
        "axes.grid": True, "axes.grid.axis": "x", "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelcolor": "#52514e", "xtick.color": "#898781", "ytick.color": "#52514e",
    })
    fig, (left, right) = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)

    # Left: lead time per alarm, with each seed's median as a dot.
    alarms = summary["alarms"]
    positions = box_rows(left, [data[f"lead_{n}"] for n in ALARMS], [COLOUR[n] for n in ALARMS])
    for name, y in zip(ALARMS, positions):
        seed_medians = list(alarms[name]["lead_time_median_by_seed"].values())
        left.scatter(seed_medians, [y] * len(seed_medians), s=28, color="#0b0b0b", zorder=3)
    left.set_yticks(positions, [f"{n}\ndetects {100 * alarms[n]['detection_rate']:.0f}%" for n in ALARMS])
    left.set(title="How much warning each alarm gives (dots: each seed's median)",
             xlabel="lead time in steps (failure step minus alarm step)\nbelow 0 = the alarm came too late")

    # Right: internal minus output-side lead time, per failing rollout.
    pairs = [f"{i} vs {o}" for i in INTERNAL for o in OUTPUT]
    positions = box_rows(right, [data[f"difference_{p.replace(' ', '_')}"] for p in pairs], ["#2a78d6"] * len(pairs))
    right.set_yticks(positions, [f"{p}\ninternal wins {100 * summary['head_to_head'][p]['wins']:.0f}%" for p in pairs])
    right.set(title="Internal vs output-side, rollout by rollout",
              xlabel="internal lead time minus output-side lead time (steps)\nabove 0 = the internal alarm warned earlier")

    verdict = "SUPPORTED" if summary["verdict"]["supported"] else "NOT SUPPORTED"
    fig.suptitle(f"Failure = relative error above {100 * summary['failure_threshold']:.0f}%. "
                 f"Every alarm tuned to {100 * summary['false_alarm_rate_target']:.0f}% false alarms on calibration. "
                 f"Internal beats output-only baselines: {verdict}", fontsize=11, color="#0b0b0b")

    path = results_dir / "leadtime.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved to {path}")


if __name__ == "__main__":
    main()
