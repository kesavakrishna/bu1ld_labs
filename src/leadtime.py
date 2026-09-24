"""
Tune every alarm fairly on calibration rollouts, then measure how much warning each one
gives on test rollouts (Days 5 and 6).

THE RULES IN THIS FILE ARE FROZEN (see FREEZE.md). They were fixed before the test split
was scored, and must not be changed after seeing any result.

For each seed:
  1. Tune, on calibration rollouts only. Each alarm gets the threshold that gives
     `false_alarm_rate` of calibration rollouts a false alarm, meaning an alarm inside the
     safe window. The clock is tuned by the same rule, so every alarm is equally trigger-happy.
  2. Score, on test rollouts only, with the thresholds frozen. For each failing rollout,
     find the first step each alarm goes off. Lead time = failure step - alarm step.

Then, across seeds: each alarm's lead time, detection rate and false-alarm rate; head-to-head
differences, rollout by rollout; and the verdict. The whole analysis is repeated at other
failure thresholds as a secondary check.

Every number goes to results/summary.json, and the per-rollout numbers behind the figure to
results/leadtimes.npz.

    python src/leadtime.py --config configs/full.yaml
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from signals import INTERNAL, OUTPUT, SIGNALS

ALARMS = SIGNALS + ["clock"]
PAIRS = [(i, o) for i in INTERNAL for o in OUTPUT] + [(name, "clock") for name in SIGNALS]


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


def evaluate_seed(calibration, test, failure_threshold, settings):
    """
    Tune every alarm on calibration rollouts, then find when each first goes off on test
    rollouts. Returns the thresholds, the test failure steps, the test safe-window ends,
    and each alarm's first alarm step on every test rollout.
    """
    rate, fraction = settings["false_alarm_rate"], settings["safe_fraction"]
    horizon = test["relative_error"].shape[1] - 1

    calibration_end = safe_window_ends(failure_steps(calibration["relative_error"], failure_threshold), horizon, fraction)
    test_fail = failure_steps(test["relative_error"], failure_threshold)
    test_end = safe_window_ends(test_fail, horizon, fraction)

    thresholds, alarm_step = {}, {}
    for name in SIGNALS:
        thresholds[name] = tune_threshold(calibration[f"score_{name}"], calibration_end, rate)
        alarm_step[name] = first_alarm(test[f"score_{name}"], thresholds[name])
    thresholds["clock"] = tune_clock(calibration_end, rate, horizon)
    alarm_step["clock"] = np.full(len(test_fail), thresholds["clock"])

    return thresholds, test_fail, test_end, alarm_step


def median(values):
    return float(np.median(values)) if len(values) else float("nan")


def describe(values):
    """Median and interquartile range (25th to 75th percentile), as plain numbers."""
    if not len(values):
        return {"median": float("nan"), "q25": float("nan"), "q75": float("nan")}
    q25, middle, q75 = np.percentile(values, [25, 50, 75])
    return {"median": float(middle), "q25": float(q25), "q75": float(q75)}


def analyse(rollouts, failure_threshold, settings):
    """
    The full analysis at one failure threshold, for every seed.

    Args:
        rollouts: {seed: (calibration, test)}, each a dict of arrays saved by rollout.py

    Returns:
        summary: every reported number, ready to save as JSON
        leads:   {seed: {alarm: lead times on that seed's failing test rollouts}}
    """
    summary = {"failure_threshold": failure_threshold, "seeds": {}, "alarms": {}, "head_to_head": {}}
    leads, false_alarms = {}, {}

    for seed, (calibration, test) in rollouts.items():
        thresholds, test_fail, test_end, alarm_step = evaluate_seed(calibration, test, failure_threshold, settings)
        failing = test_fail >= 0
        leads[seed] = {name: test_fail[failing] - step[failing] for name, step in alarm_step.items()}
        false_alarms[seed] = {name: step < test_end for name, step in alarm_step.items()}
        summary["seeds"][seed] = {
            "test_rollouts": int(len(test_fail)),
            "failing": int(failing.sum()),
            "thresholds": {name: float(value) for name, value in thresholds.items()},
        }

    seeds = list(rollouts)
    for name in ALARMS:
        pooled = np.concatenate([leads[s][name] for s in seeds])
        summary["alarms"][name] = {
            "lead_time": describe(pooled),
            "lead_time_median_by_seed": {s: median(leads[s][name]) for s in seeds},
            "detection_rate": float(np.mean(pooled > 0)) if len(pooled) else float("nan"),
            "detection_rate_by_seed": {s: float(np.mean(leads[s][name] > 0)) for s in seeds},
            "false_alarm_rate": float(np.mean(np.concatenate([false_alarms[s][name] for s in seeds]))),
        }

    for a, b in PAIRS:
        by_seed = {s: leads[s][a] - leads[s][b] for s in seeds}
        pooled = np.concatenate(list(by_seed.values()))
        summary["head_to_head"][f"{a} vs {b}"] = {
            "difference": describe(pooled),
            "difference_median_by_seed": {s: median(d) for s, d in by_seed.items()},
            "wins": float(np.mean(pooled > 0)),
            "ties": float(np.mean(pooled == 0)),
            "losses": float(np.mean(pooled < 0)),
        }

    summary["verdict"] = {
        "rule": ("Internal signals beat the output-only baselines if at least one internal alarm has a median "
                 "lead-time difference above zero against BOTH output-side alarms, in EVERY seed."),
        "supported": any(
            all(summary["head_to_head"][f"{i} vs {o}"]["difference_median_by_seed"][s] > 0 for o in OUTPUT for s in seeds)
            for i in INTERNAL
        ),
    }
    return summary, leads


def print_report(summary):
    alarms, pairs = summary["alarms"], summary["head_to_head"]
    kind = {**{n: "internal" for n in INTERNAL}, **{n: "output" for n in OUTPUT}, "clock": "baseline"}

    print(f"\n  {'alarm':>15} {'type':>9} {'median lead':>12} {'IQR':>12} {'detected':>9} {'false alarms':>13}   median by seed")
    for name in ALARMS:
        a = alarms[name]
        lead = a["lead_time"]
        spread = f"[{lead['q25']:.0f}, {lead['q75']:.0f}]"
        by_seed = " / ".join(f"{m:.0f}" for m in a["lead_time_median_by_seed"].values())
        print(f"  {name:>15} {kind[name]:>9} {lead['median']:>12.0f} {spread:>12} "
              f"{100 * a['detection_rate']:>8.0f}% {100 * a['false_alarm_rate']:>12.0f}%   {by_seed}")

    print("\n  Head-to-head: internal minus other lead time, per failing test rollout")
    for pair, p in pairs.items():
        by_seed = " / ".join(f"{m:.0f}" for m in p["difference_median_by_seed"].values())
        print(f"  {pair:>33}   median {p['difference']['median']:>5.0f}   wins {100 * p['wins']:>3.0f}%  "
              f"ties {100 * p['ties']:>3.0f}%  losses {100 * p['losses']:>3.0f}%   by seed {by_seed}")

    verdict = "SUPPORTED" if summary["verdict"]["supported"] else "NOT SUPPORTED"
    print(f"\n  Verdict: {summary['verdict']['rule']}\n  -> {verdict}")


def main():
    parser = argparse.ArgumentParser(description="Tune alarms on calibration, measure lead times on test.")
    parser.add_argument("--config", default="configs/full.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    settings = config["leadtime"]
    results_dir = Path(config["results_dir"])
    rollout_dir = results_dir / "rollouts"

    rollouts = {
        seed: (dict(np.load(rollout_dir / f"calibration_seed{seed}.npz")), dict(np.load(rollout_dir / f"test_seed{seed}.npz")))
        for seed in config["training"]["seeds"]
    }

    # Primary result.
    primary = config["rollout"]["failure_threshold"]
    summary, leads = analyse(rollouts, primary, settings)
    summary["false_alarm_rate_target"] = settings["false_alarm_rate"]
    print(f"\nPrimary result: failure = relative error above {100 * primary:.0f}%, "
          f"every alarm tuned to {100 * settings['false_alarm_rate']:.0f}% false alarms on calibration")
    print_report(summary)

    # Secondary: the same analysis at other failure thresholds.
    summary["sweep"] = {}
    print("\nSecondary: the same analysis at other failure thresholds (median lead time / detection rate)")
    print(f"  {'failure at':>10} {'failing':>8}  " + "  ".join(f"{name:>15}" for name in ALARMS) + "   verdict")
    for threshold in settings["failure_threshold_sweep"]:
        swept, _ = analyse(rollouts, threshold, settings)
        failing = sum(s["failing"] for s in swept["seeds"].values()) / sum(s["test_rollouts"] for s in swept["seeds"].values())
        summary["sweep"][str(threshold)] = {
            "failing_share": failing,
            "supported": swept["verdict"]["supported"],
            "lead_time_median": {n: swept["alarms"][n]["lead_time"]["median"] for n in ALARMS},
            "detection_rate": {n: swept["alarms"][n]["detection_rate"] for n in ALARMS},
        }
        cells = []
        for n in ALARMS:
            a = swept["alarms"][n]
            cells.append(f"{a['lead_time']['median']:.0f} / {100 * a['detection_rate']:.0f}%")
        verdict = "supported" if swept["verdict"]["supported"] else "not supported"
        print(f"  {100 * threshold:>9.0f}% {100 * failing:>7.0f}%  " + "  ".join(f"{c:>15}" for c in cells) + f"   {verdict}")

    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(
        results_dir / "leadtimes.npz",
        **{f"lead_{name}": np.concatenate([leads[s][name] for s in leads]) for name in ALARMS},
        **{f"difference_{a}_vs_{b}": np.concatenate([leads[s][a] - leads[s][b] for s in leads]) for a, b in PAIRS},
    )
    print(f"\n  Saved {results_dir / 'summary.json'} and {results_dir / 'leadtimes.npz'}")


if __name__ == "__main__":
    main()
