"""
Tune every alarm fairly, then measure how much warning each one gives, in every setting
(Days 5 and 6, and the shift sprint).

THE RULES IN THIS FILE ARE FROZEN (see FREEZE.md). They were fixed before any test or shift
data was scored, and must not be changed after seeing any result.

For each seed:
  1. Tune, on calibration rollouts only. These use the physics the model was trained on,
     just as a real user would only have data from the setting they built the model for.
     Each alarm gets the threshold that gives `false_alarm_rate` of calibration rollouts a
     false alarm (an alarm inside the safe window). The clock is tuned by the same rule.
  2. Score every setting with those thresholds frozen: the control (the test split, same
     physics as training) and each shift setting. For each failing rollout, find the first
     step each alarm goes off. Lead time = failure step - alarm step.

Reported per setting: lead time, detection rate, false-alarm rate, head-to-head
comparisons and the verdict; then an overall verdict across settings. Also repeated at
other failure thresholds, and at other false-alarm targets (the trade-off curve).

Outputs:
    results/summary.json            every reported number
    results/leadtimes.csv           one row per setting, seed, rollout and alarm
    results/rollout_steps.csv.gz    one row per setting, seed, rollout and step: the error and
                                    every score, so anything here can be recomputed from scratch

    python src/leadtime.py --config configs/full.yaml
"""

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import yaml

from signals import INTERNAL, OUTPUT, SIGNALS

ALARMS = SIGNALS + ["clock"]
KIND = {**{n: "internal" for n in INTERNAL}, **{n: "output" for n in OUTPUT}, "clock": "baseline"}
HEAD_TO_HEAD = [(i, o) for i in INTERNAL for o in OUTPUT]
AGAINST_CLOCK = [(name, "clock") for name in SIGNALS]


def failure_steps(relative_error, threshold):
    """
    First step where each rollout's relative error goes above `threshold`.
    Rollouts that never get there are marked -1.

    Badly diverged rollouts overflow to infinity, so anything not finite counts as failed.
    Those rollouts passed the threshold long before overflowing anyway.
    """
    failed = ~np.isfinite(relative_error) | (relative_error > threshold)
    first = failed.argmax(axis=1)  # argmax finds the first True
    return np.where(failed.any(axis=1), first, -1)


def safe_window_ends(failed_at, horizon, safe_fraction):
    """
    Where each rollout's safe window stops. Steps t with 1 <= t < end are "safe": an alarm
    there is a false alarm. end = safe_fraction x the failure step, or safe_fraction x the
    horizon for rollouts that never fail.
    """
    return np.where(failed_at >= 0, safe_fraction * failed_at, safe_fraction * horizon)


def first_alarm(score, threshold):
    """
    First step (1 or later) where the score goes above the threshold. An alarm that never
    goes off counts as going off at the last step, so silence is never rewarded.
    """
    above = score[:, 1:] > threshold
    horizon = score.shape[1] - 1
    return np.where(above.any(axis=1), above.argmax(axis=1) + 1, horizon)


def tune_threshold(score, window_end, false_alarm_rate):
    """
    The threshold that gives `false_alarm_rate` of rollouts a false alarm: take each
    rollout's highest score inside its safe window, then the (1 - false_alarm_rate)
    quantile of those highest scores.
    """
    steps = np.arange(score.shape[1])
    in_window = (steps >= 1) & (steps < window_end[:, None])
    # Scores are never negative, so -1 stands for "no safe steps at all": a rollout that
    # fails very early can't false-alarm. Infinite scores become the largest ordinary
    # number. Both keep the quantile a real number (numpy turns inf - inf into NaN, and
    # a NaN threshold would silently never fire).
    highest = np.where(in_window, score, -1.0).max(axis=1)
    highest = np.minimum(highest, np.finfo(np.float64).max)
    return float(np.quantile(highest, 1 - false_alarm_rate))


def tune_clock(window_end, false_alarm_rate, horizon):
    """
    The clock always says "failure at step k". It false-alarms on a rollout when k falls in
    that rollout's safe window. Pick the smallest k where at most `false_alarm_rate` of
    rollouts get a false alarm.
    """
    for k in range(1, horizon + 1):
        if np.mean(k < window_end) <= false_alarm_rate:
            return k
    return horizon


def tune(calibration, failure_threshold, false_alarm_rate, safe_fraction):
    """Thresholds for every alarm, from calibration rollouts only."""
    horizon = calibration["relative_error"].shape[1] - 1
    failed_at = failure_steps(calibration["relative_error"], failure_threshold)
    window_end = safe_window_ends(failed_at, horizon, safe_fraction)
    thresholds = {name: tune_threshold(calibration[f"score_{name}"], window_end, false_alarm_rate) for name in SIGNALS}
    thresholds["clock"] = tune_clock(window_end, false_alarm_rate, horizon)
    return thresholds


def score_rollouts(rollouts, thresholds, failure_threshold, safe_fraction):
    """
    Apply frozen thresholds to one set of rollouts. Returns each rollout's failure step,
    the end of its safe window, and the first alarm step of every alarm.
    """
    horizon = rollouts["relative_error"].shape[1] - 1
    failed_at = failure_steps(rollouts["relative_error"], failure_threshold)
    window_end = safe_window_ends(failed_at, horizon, safe_fraction)
    alarm_step = {name: first_alarm(rollouts[f"score_{name}"], thresholds[name]) for name in SIGNALS}
    alarm_step["clock"] = np.full(len(failed_at), thresholds["clock"])
    return failed_at, window_end, alarm_step


def median(values):
    return float(np.median(values)) if len(values) else float("nan")


def share(flags):
    return float(np.mean(flags)) if len(flags) else float("nan")


def describe(values):
    """Median and interquartile range (25th to 75th percentile), as plain numbers."""
    if not len(values):
        return {"median": float("nan"), "q25": float("nan"), "q75": float("nan")}
    q25, middle, q75 = np.percentile(values, [25, 50, 75])
    return {"median": float(middle), "q25": float(q25), "q75": float(q75)}


def analyse_setting(by_seed, failure_threshold, settings, false_alarm_rate):
    """
    The full analysis of one setting at one failure threshold, for every seed.

    Args:
        by_seed: {seed: (calibration rollouts, rollouts to score)}, as saved by rollout.py

    Returns:
        summary: every reported number for this setting
        rows:    one dict per seed, rollout and alarm (the raw record)
    """
    summary = {"seeds": {}, "alarms": {}, "head_to_head": {}}
    leads, false_alarms, rows = {}, {}, []

    for seed, (calibration, rollouts) in by_seed.items():
        thresholds = tune(calibration, failure_threshold, false_alarm_rate, settings["safe_fraction"])
        failed_at, window_end, alarm_step = score_rollouts(rollouts, thresholds, failure_threshold, settings["safe_fraction"])
        failing = failed_at >= 0
        leads[seed] = {name: failed_at[failing] - step[failing] for name, step in alarm_step.items()}
        false_alarms[seed] = {name: step < window_end for name, step in alarm_step.items()}
        summary["seeds"][seed] = {
            "rollouts": int(len(failed_at)),
            "failing": int(failing.sum()),
            "median_failure_step": median(failed_at[failing]),
            "thresholds": {name: float(value) for name, value in thresholds.items()},
        }
        for i, trajectory in enumerate(rollouts["ids"]):
            for name in ALARMS:
                fails = bool(failing[i])
                rows.append({
                    "seed": seed, "trajectory_id": int(trajectory), "alarm": name, "alarm_type": KIND[name],
                    "threshold": thresholds[name], "failure_step": int(failed_at[i]), "alarm_step": int(alarm_step[name][i]),
                    "lead_time": int(failed_at[i] - alarm_step[name][i]) if fails else "",
                    "detected": int(alarm_step[name][i] < failed_at[i]) if fails else "",
                    "false_alarm": int(false_alarms[seed][name][i]),
                })

    seeds = list(by_seed)
    for name in ALARMS:
        pooled = np.concatenate([leads[s][name] for s in seeds])
        summary["alarms"][name] = {
            "lead_time": describe(pooled),
            "lead_time_median_by_seed": {s: median(leads[s][name]) for s in seeds},
            "detection_rate": share(pooled > 0),
            "detection_rate_by_seed": {s: share(leads[s][name] > 0) for s in seeds},
            "false_alarm_rate": share(np.concatenate([false_alarms[s][name] for s in seeds])),
            "false_alarm_rate_by_seed": {s: share(false_alarms[s][name]) for s in seeds},
        }

    for a, b in HEAD_TO_HEAD + AGAINST_CLOCK:
        by_seed_difference = {s: leads[s][a] - leads[s][b] for s in seeds}
        pooled = np.concatenate(list(by_seed_difference.values()))
        summary["head_to_head"][f"{a} vs {b}"] = {
            "difference": describe(pooled),
            "difference_median_by_seed": {s: median(d) for s, d in by_seed_difference.items()},
            "wins": share(pooled > 0),
            "ties": share(pooled == 0),
            "losses": share(pooled < 0),
        }

    # The verdict: some internal alarm must beat EVERY output-side alarm in EVERY seed.
    summary["supported"] = any(
        all(summary["head_to_head"][f"{i} vs {o}"]["difference_median_by_seed"][s] > 0 for o in OUTPUT for s in seeds)
        for i in INTERNAL
    )
    return summary, rows


def analyse(rollouts, failure_threshold, settings, false_alarm_rate):
    """Every setting at one failure threshold, plus the overall verdict."""
    result = {"failure_threshold": failure_threshold, "settings": {}}
    rows = []
    for setting, by_seed in rollouts.items():
        result["settings"][setting], setting_rows = analyse_setting(by_seed, failure_threshold, settings, false_alarm_rate)
        rows += [{"setting": setting, **row} for row in setting_rows]
    supported = [name for name, s in result["settings"].items() if s["supported"]]
    result["settings_supported"] = supported
    result["overall_supported"] = len(supported) >= settings["min_settings_supported"]
    return result, rows


def print_setting(name, summary):
    print(f"\n  {name}: {sum(s['failing'] for s in summary['seeds'].values())} of "
          f"{sum(s['rollouts'] for s in summary['seeds'].values())} rollouts fail")
    print(f"  {'alarm':>15} {'type':>9} {'median lead':>12} {'IQR':>12} {'detected':>9} {'false alarms':>13}   median by seed")
    for alarm in ALARMS:
        a = summary["alarms"][alarm]
        lead = a["lead_time"]
        spread = f"[{lead['q25']:.0f}, {lead['q75']:.0f}]"
        by_seed = " / ".join(f"{m:.0f}" for m in a["lead_time_median_by_seed"].values())
        print(f"  {alarm:>15} {KIND[alarm]:>9} {lead['median']:>12.0f} {spread:>12} "
              f"{100 * a['detection_rate']:>8.0f}% {100 * a['false_alarm_rate']:>12.0f}%   {by_seed}")
    print("  head-to-head, internal minus output-side lead time (median by seed):")
    for a, b in HEAD_TO_HEAD:
        p = summary["head_to_head"][f"{a} vs {b}"]
        by_seed = " / ".join(f"{m:.0f}" for m in p["difference_median_by_seed"].values())
        print(f"  {f'{a} vs {b}':>33}   {by_seed:>14}   wins {100 * p['wins']:>3.0f}%  losses {100 * p['losses']:>3.0f}%")
    print(f"  -> {'SUPPORTED' if summary['supported'] else 'NOT SUPPORTED'} in this setting")


def main():
    parser = argparse.ArgumentParser(description="Tune alarms on calibration, measure lead times in every setting.")
    parser.add_argument("--config", default="configs/full.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    settings = config["leadtime"]
    results_dir = Path(config["results_dir"])
    rollout_dir = results_dir / "rollouts"
    seeds = config["training"]["seeds"]

    # The control is the test split; every other setting is a shift.
    files = {"control": "test", **{name: f"shift_{name}" for name in config["shifts"]["settings"]}}
    calibration = {seed: dict(np.load(rollout_dir / f"calibration_seed{seed}.npz")) for seed in seeds}
    rollouts = {
        setting: {seed: (calibration[seed], dict(np.load(rollout_dir / f"{file}_seed{seed}.npz"))) for seed in seeds}
        for setting, file in files.items()
    }

    # Primary result.
    primary = config["rollout"]["failure_threshold"]
    rate = settings["false_alarm_rate"]
    summary, rows = analyse(rollouts, primary, settings, rate)
    summary["false_alarm_rate_target"] = rate
    summary["verdict_rule"] = (
        "In each setting: supported if at least one internal alarm has a median lead-time difference above zero "
        f"against EVERY output-side alarm, in EVERY seed. Overall: supported in at least "
        f"{settings['min_settings_supported']} of {len(files)} settings."
    )
    print(f"\nPrimary result: failure = relative error above {100 * primary:.0f}%, "
          f"every alarm tuned to {100 * rate:.0f}% false alarms on calibration (original physics)")
    for setting, setting_summary in summary["settings"].items():
        print_setting(setting, setting_summary)
    print(f"\n  Verdict rule: {summary['verdict_rule']}")
    print(f"  Supported in: {summary['settings_supported'] or 'no setting'}")
    print(f"  -> OVERALL: {'SUPPORTED' if summary['overall_supported'] else 'NOT SUPPORTED'}")

    # Descriptive: detection against false alarms as the alarms are made stricter or looser.
    summary["tradeoff"] = {setting: {alarm: [] for alarm in ALARMS} for setting in files}
    for target in settings["tradeoff_rates"]:
        result, _ = analyse(rollouts, primary, settings, target)
        for setting, s in result["settings"].items():
            for alarm in ALARMS:
                summary["tradeoff"][setting][alarm].append({
                    "target": target,
                    "false_alarm_rate": s["alarms"][alarm]["false_alarm_rate"],
                    "detection_rate": s["alarms"][alarm]["detection_rate"],
                })

    # Secondary: the same analysis at other failure thresholds.
    summary["sweep"] = {}
    print("\nSecondary: other failure thresholds (settings where the internal signals are supported)")
    for threshold in settings["failure_threshold_sweep"]:
        result, _ = analyse(rollouts, threshold, settings, rate)
        summary["sweep"][str(threshold)] = {
            setting: {
                "supported": s["supported"],
                "failing_share": sum(x["failing"] for x in s["seeds"].values()) / sum(x["rollouts"] for x in s["seeds"].values()),
                "lead_time_median": {n: s["alarms"][n]["lead_time"]["median"] for n in ALARMS},
                "detection_rate": {n: s["alarms"][n]["detection_rate"] for n in ALARMS},
                "false_alarm_rate": {n: s["alarms"][n]["false_alarm_rate"] for n in ALARMS},
            }
            for setting, s in result["settings"].items()
        }
        summary["sweep"][str(threshold)]["overall_supported"] = result["overall_supported"]
        print(f"  failure at {100 * threshold:>4.0f}%: supported in {result['settings_supported'] or 'no setting'}"
              f" -> overall {'supported' if result['overall_supported'] else 'not supported'}")

    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(results_dir / "leadtimes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # The per-step record behind everything above.
    with gzip.open(results_dir / "rollout_steps.csv.gz", "wt", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["setting", "seed", "trajectory_id", "step", "relative_error", "raw_error", "true_size"]
                        + [f"score_{name}" for name in SIGNALS])
        for setting, by_seed in rollouts.items():
            for seed, (_, r) in by_seed.items():
                for i, trajectory in enumerate(r["ids"]):
                    columns = [r["relative_error"][i], r["raw_error"][i], r["true_size"][i]]
                    columns += [r[f"score_{name}"][i] for name in SIGNALS]
                    for step in range(len(columns[0])):
                        writer.writerow([setting, seed, int(trajectory), step] + [f"{c[step]:.6g}" for c in columns])

    print(f"\n  Saved summary.json, leadtimes.csv ({len(rows)} rows) and rollout_steps.csv.gz to {results_dir}")


if __name__ == "__main__":
    main()
