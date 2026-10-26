"""
Own-normal scores for the successor protocol (SUCCESSOR_PROTOCOL.md, Amendment 1).

DRAFT: not frozen and not run. So far this is checked only on made-up arrays
(tests/test_own_scores.py). Nothing here reads rollout data.

Freeze-v2 compared each signal with the normal range of all reference runs at the same step.
Here each rollout is compared with its own first `warmup` steps instead:

    z(t)         = (signal(t) - normal mean at step t) / normal spread at step t    (signed)
    baseline     = mean of z over steps 1..warmup
    own_score(t) = |z(t) - baseline|  for t > warmup, with anything not finite set to +inf
                 = 0                  for t <= warmup, whatever z is (the warm-up wins)

Thresholds come from the frozen freeze-v2 tune_threshold, unchanged. Alarms are searched for
only after the warm-up, because a zero warm-up score alone doesn't keep an alarm silent when
a threshold comes out negative.
"""

import numpy as np

from leadtime import failure_steps, safe_window_ends, tune_threshold


def own_scores(signal, mean, spread, warmup):
    """
    Args:
        signal:       [n_rollouts, n_steps], column 0 being the true starting wave
        mean, spread: the reference runs' normal mean and spread at each step, [n_steps]
        warmup:       steps 1..warmup are only watched, never scored

    Returns own_score, [n_rollouts, n_steps].
    """
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        z = (signal - mean) / spread
        window = z[:, 1 : warmup + 1]
        # One reading that isn't finite during the warm-up breaks the baseline.
        baseline = np.where(np.isfinite(window).all(axis=1), window.mean(axis=1), np.nan)
        score = np.abs(z - baseline[:, None])
    score[~np.isfinite(score)] = np.inf
    score[:, : warmup + 1] = 0  # applied last, so the warm-up beats the line above
    return score


def check_thresholds(thresholds):
    """Refuse a threshold that isn't finite: a NaN one would silently never fire."""
    for name, value in thresholds.items():
        if not np.isfinite(value):
            raise ValueError(f"{name}: threshold {value} is not finite, so nothing was scored")


def tune(calibration_scores, window_end, false_alarm_rate):
    """A threshold for every signal, from calibration own-scores, by the freeze-v2 rule."""
    thresholds = {
        name: tune_threshold(score, window_end, false_alarm_rate) for name, score in calibration_scores.items()
    }
    check_thresholds(thresholds)
    return thresholds


def first_alarm(score, threshold, warmup):
    """
    First step after the warm-up where the score goes above the threshold. As in freeze-v2,
    an alarm that never goes off counts as going off at the last step.
    """
    if not np.isfinite(threshold):
        raise ValueError(f"threshold {threshold} is not finite")
    horizon = score.shape[1] - 1
    above = score[:, warmup + 1 :] > threshold
    if above.shape[1] == 0:  # the rollout ends inside the warm-up
        return np.full(len(score), horizon)
    return np.where(above.any(axis=1), above.argmax(axis=1) + warmup + 1, horizon)


def alarm_steps(scores, thresholds, warmup):
    """First alarm step of every signal."""
    check_thresholds(thresholds)
    return {name: first_alarm(score, thresholds[name], warmup) for name, score in scores.items()}


def score_rollouts(relative_error, scores, thresholds, failure_threshold, safe_fraction, warmup):
    """
    Freeze-v2's score_rollouts with own-scores and the warm-up alarm. Returns each rollout's
    failure step, the end of its safe window, and the first alarm step of every signal.
    """
    horizon = relative_error.shape[1] - 1
    failed_at = failure_steps(relative_error, failure_threshold)
    window_end = safe_window_ends(failed_at, horizon, safe_fraction)
    return failed_at, window_end, alarm_steps(scores, thresholds, warmup)
