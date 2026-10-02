"""
Draw the result figures from leadtime.py's output (Day 6 and the shift sprint).

    1_lead_time.png          how much warning each alarm gives, in each setting
    2_false_alarms.png       how often each alarm cries wolf, in each setting
    3_tradeoff.png           detection against false alarms as each alarm is made stricter or looser
    4_head_to_head.png       internal against output-side alarms, rollout by rollout
    5_signals_at_failure.png what each signal does around the failure step
    6_rollout_error.png      how rollouts fail in each setting

Box plots: the box covers the middle half of values (25th to 75th percentile), whiskers run
from the 5th to the 95th, and the black line is the median.

    python src/plots.py --config configs/full.yaml
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from check_signals import line_up_at_failure
from leadtime import ALARMS, HEAD_TO_HEAD, KIND, failure_steps
from signals import SIGNALS

TYPE_COLOUR = {"internal": "#2a78d6", "output": "#eb6834", "baseline": "#898781"}
SETTING_COLOUR = ["#52514e", "#1baf7a", "#4a3aa7", "#e87ba4"]  # control first, then each shift
DISPLAY_CAP = 1e6  # infinite scores can't be drawn, so they are drawn at this height


def style():
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
        "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelcolor": "#52514e", "xtick.color": "#898781", "ytick.color": "#52514e",
    })


def box_rows(ax, rows, colours):
    """Horizontal box plots, first row at the top. Returns the row positions."""
    positions = np.arange(len(rows), 0, -1)
    boxes = ax.boxplot(rows, positions=positions, vert=False, widths=0.55, whis=(5, 95), showfliers=False,
                       patch_artist=True, medianprops={"color": "#0b0b0b", "linewidth": 2})
    for patch, colour in zip(boxes["boxes"], colours):
        patch.set(facecolor=colour, alpha=0.35, edgecolor=colour)
    ax.axvline(0, color="#52514e", linewidth=1, linestyle="--")
    ax.grid(axis="y", visible=False)
    return positions


def lead_times(rows, setting, alarm):
    return np.array([int(r["lead_time"]) for r in rows
                     if r["setting"] == setting and r["alarm"] == alarm and r["lead_time"] != ""])


def label(setting, summary):
    verdict = "supported" if summary["settings"][setting]["supported"] else "not supported"
    return f"{setting.replace('_', ' ')} ({verdict})"


def figure_lead_time(summary, rows, settings, path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, constrained_layout=True)
    for ax, setting in zip(axes.flat, settings):
        alarms = summary["settings"][setting]["alarms"]
        positions = box_rows(ax, [lead_times(rows, setting, a) for a in ALARMS], [TYPE_COLOUR[KIND[a]] for a in ALARMS])
        for alarm, y in zip(ALARMS, positions):
            medians = list(alarms[alarm]["lead_time_median_by_seed"].values())
            ax.scatter(medians, [y] * len(medians), s=24, color="#0b0b0b", zorder=3)
        ax.set_yticks(positions, [f"{a}\ndetects {100 * alarms[a]['detection_rate']:.0f}%" for a in ALARMS])
        ax.set_title(label(setting, summary))
    for ax in axes[1]:
        ax.set_xlabel("lead time in steps (failure step minus alarm step); below 0 = too late")
    fig.suptitle("How much warning each alarm gives. Blue: internal, orange: output-side, grey: clock. "
                 "Dots: each seed's median.", fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_false_alarms(summary, settings, path):
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    positions = np.arange(len(ALARMS), 0, -1)
    offsets = np.linspace(0.25, -0.25, len(settings))  # same top-to-bottom order as the legend
    for setting, colour, offset in zip(settings, SETTING_COLOUR, offsets):
        rates = [100 * summary["settings"][setting]["alarms"][a]["false_alarm_rate"] for a in ALARMS]
        ax.scatter(rates, positions + offset, s=45, color=colour, label=setting.replace("_", " "), zorder=3)
    ax.axvline(100 * summary["false_alarm_rate_target"], color="#52514e", linewidth=1, linestyle="--")
    ax.text(100 * summary["false_alarm_rate_target"], positions[-1] - 0.55, " target, set on the original physics",
            color="#52514e", va="top", fontsize=9)
    ax.set_yticks(positions, ALARMS)
    ax.set_ylim(positions[-1] - 1, positions[0] + 0.6)
    ax.grid(axis="y", visible=False)
    ax.set(title="How often each alarm goes off before failure is anywhere near (false alarms)",
           xlabel="false-alarm rate on test rollouts (%)")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_tradeoff(summary, settings, path):
    shades = {"act_norm": "#86b6ef", "eff_rank": "#2a78d6", "train_dist": "#104281",
              "step_change": "#f2a07d", "spectral_drift": "#eb6834", "pde_residual": "#a33a12", "clock": "#898781"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, sharey=True, constrained_layout=True)
    for ax, setting in zip(axes.flat, settings):
        for alarm in ALARMS:
            points = summary["tradeoff"][setting][alarm]
            x = [100 * p["false_alarm_rate"] for p in points]
            y = [100 * p["detection_rate"] for p in points]
            dashes = "-" if KIND[alarm] != "baseline" else "--"
            ax.plot(x, y, dashes, marker="o", markersize=4, linewidth=2, color=shades[alarm], label=alarm)
        ax.set_title(setting.replace("_", " "))
    axes[0, 0].legend(frameon=False, fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("false-alarm rate on test rollouts (%)")
    for ax in axes[:, 0]:
        ax.set_ylabel("detection rate: warned before failure (%)")
    fig.suptitle("Detection against false alarms, as each alarm is tuned from strict (1% false alarms on "
                 "calibration) to loose (50%). Up and to the left is better.", fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_head_to_head(summary, rows, settings, path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True, constrained_layout=True)
    for ax, setting in zip(axes.flat, settings):
        differences = []
        for a, b in HEAD_TO_HEAD:
            by_rollout = {}
            for r in rows:
                if r["setting"] == setting and r["lead_time"] != "" and r["alarm"] in (a, b):
                    by_rollout.setdefault((r["seed"], r["trajectory_id"]), {})[r["alarm"]] = int(r["lead_time"])
            differences.append(np.array([d[a] - d[b] for d in by_rollout.values()]))
        positions = box_rows(ax, differences, [TYPE_COLOUR["internal"]] * len(HEAD_TO_HEAD))
        pairs = summary["settings"][setting]["head_to_head"]
        ax.set_yticks(positions, [f"{a} vs {b}\ninternal wins {100 * pairs[f'{a} vs {b}']['wins']:.0f}%"
                                  for a, b in HEAD_TO_HEAD], fontsize=8)
        ax.set_title(label(setting, summary))
    for ax in axes[1]:
        ax.set_xlabel("internal minus output-side lead time (steps); above 0 = internal warned earlier")
    fig.suptitle("Internal against output-side alarms, on the same rollouts", fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_signals_at_failure(rollouts, threshold, settings, path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, constrained_layout=True)
    for ax, name in zip(axes.flat, SIGNALS):
        for setting, colour in zip(settings, SETTING_COLOUR):
            windows = []
            for data in rollouts[setting]:
                failed_at = failure_steps(data["relative_error"], threshold)
                failing = failed_at >= 0
                window, offsets = line_up_at_failure(data[f"score_{name}"][failing], failed_at[failing])
                windows.append(window)
            middle = np.nanmedian(np.minimum(np.concatenate(windows), DISPLAY_CAP), axis=0)
            ax.plot(offsets, middle, color=colour, linewidth=2, label=setting.replace("_", " "))
        ax.axvline(0, color="#52514e", linewidth=1, linestyle="--")
        ax.axhline(1, color="#52514e", linewidth=1, linestyle=":")
        ax.set(title=f"{name} ({KIND[name]})", yscale="log")
    axes[0, 0].legend(frameon=False, fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("steps relative to failure (0 = the failure step)")
    for ax in axes[:, 0]:
        ax.set_ylabel("median score (1 = one normal spread)")
    fig.suptitle("What each signal does around the failure step (median over failing rollouts, all seeds)", fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_rollout_error(rollouts, threshold, settings, path):
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for setting, colour in zip(settings, SETTING_COLOUR):
        error = np.concatenate([d["relative_error"] for d in rollouts[setting]])
        error = np.where(np.isfinite(error), error, np.inf)
        steps = np.arange(error.shape[1])
        left.plot(steps[1:], 100 * np.median(error[:, 1:], axis=0), color=colour, linewidth=2,
                  label=setting.replace("_", " "))
        failed_at = np.concatenate([failure_steps(d["relative_error"], threshold) for d in rollouts[setting]])
        share_failed = [(failed_at[failed_at >= 0] <= s).sum() / len(failed_at) for s in steps]
        right.plot(steps, 100 * np.array(share_failed), color=colour, linewidth=2, label=setting.replace("_", " "))
    left.axhline(100 * threshold, color="#52514e", linewidth=1, linestyle=":")
    left.set(title="Median rollout error", xlabel="step", ylabel="relative error (%)", yscale="log")
    right.set(title="Share of rollouts that have failed by each step", xlabel="step", ylabel="rollouts failed (%)")
    left.legend(frameon=False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Draw the result figures.")
    parser.add_argument("--config", default="configs/full.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    results_dir = Path(config["results_dir"])
    with open(results_dir / "summary.json") as f:
        summary = json.load(f)
    with open(results_dir / "leadtimes.csv") as f:
        rows = list(csv.DictReader(f))

    settings = list(summary["settings"])
    files = {"control": "test", **{name: f"shift_{name}" for name in config["shifts"]["settings"]}}
    rollouts = {setting: [dict(np.load(results_dir / "rollouts" / f"{files[setting]}_seed{seed}.npz"))
                          for seed in config["training"]["seeds"]] for setting in settings}
    threshold = summary["failure_threshold"]

    style()
    out = results_dir / "figures"
    out.mkdir(exist_ok=True)
    figure_lead_time(summary, rows, settings, out / "1_lead_time.png")
    figure_false_alarms(summary, settings, out / "2_false_alarms.png")
    figure_tradeoff(summary, settings, out / "3_tradeoff.png")
    figure_head_to_head(summary, rows, settings, out / "4_head_to_head.png")
    figure_signals_at_failure(rollouts, threshold, settings, out / "5_signals_at_failure.png")
    figure_rollout_error(rollouts, threshold, settings, out / "6_rollout_error.png")
    print(f"  Six figures saved to {out}")


if __name__ == "__main__":
    main()
