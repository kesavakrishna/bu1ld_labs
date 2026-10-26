"""
Made-up checks for the successor protocol's own-scores and warm-up alarm
(SUCCESSOR_PROTOCOL.md, Amendment 1). Every input is built here; no rollout data is read.

    python tests/test_own_scores.py
"""

import platform
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import leadtime  # noqa: E402  (freeze-v2, unchanged)
from own_scores import alarm_steps, first_alarm, own_scores, score_rollouts, tune  # noqa: E402

W = yaml.safe_load(open(ROOT / "configs" / "successor.yaml"))["warmup_steps"]
H = 200


def scores_of(z):
    """Own-scores for made-up z values. Normal mean 0 and spread 1, so the signal is z."""
    z = np.atleast_2d(np.asarray(z, dtype=float))
    return own_scores(z, np.zeros(z.shape[1]), np.ones(z.shape[1]), W)


def steady(value, steps=H):
    """One made-up rollout whose z sits at `value` at every step."""
    return np.full(steps + 1, float(value))


def calibrate(fail_step, z):
    """Thresholds from 20 made-up calibration rollouts that all fail at `fail_step`."""
    error = np.zeros((20, H + 1))
    error[:, fail_step:] = 0.5
    window_end = leadtime.safe_window_ends(leadtime.failure_steps(error, 0.10), H, 0.5)
    return tune({"act_norm": scores_of(z)}, window_end, 0.05)["act_norm"]


def test_config_matches_protocol():
    """0.  The config's warm-up is the protocol's 10 steps"""
    assert W == 10


def test_1():
    """1.  Jump from 1 to 4 at step 15: own score 3, alarm at 15"""
    z = steady(1.0)
    z[15:] = 4.0
    score = scores_of(z)
    assert score[0, 15] == 3.0 and first_alarm(score, 2.0, W)[0] == 15


def test_2():
    """2.  Big finite warm-up value (50 at step 3): warm-up scores 0, baseline pulled to 5, alarm at 11"""
    z = steady(0.0)
    z[3] = 50.0
    score = scores_of(z)
    assert (score[0, : W + 1] == 0).all() and score[0, W + 1] == 5.0
    assert first_alarm(score, 2.0, W)[0] == W + 1


def test_3():
    """3.  NaN, +inf or -inf at step 7: warm-up scores 0, broken baseline, +inf from 11, alarm at 11"""
    for bad in (np.nan, np.inf, -np.inf):
        z = steady(1.0)
        z[7] = bad
        score = scores_of(z)
        assert (score[0, : W + 1] == 0).all() and np.isposinf(score[0, W + 1 :]).all()
        assert first_alarm(score, 2.0, W)[0] == W + 1


def test_4a():
    """4a. All scores 0, threshold -0.5: alarm waits for 11 (the freeze-v2 search fires at 1)"""
    score = scores_of(steady(0.0))
    assert leadtime.first_alarm(score, -0.5)[0] == 1
    assert first_alarm(score, -0.5, W)[0] == W + 1


def test_4b():
    """4b. All scores 0, threshold 0: never fires, counts as the last step (200)"""
    assert first_alarm(scores_of(steady(0.0)), 0.0, W)[0] == H


def test_4c():
    """4c. All scores 0, threshold 2: never fires, counts as the last step (200)"""
    assert first_alarm(scores_of(steady(0.0)), 2.0, W)[0] == H


def test_5():
    """5.  Rollouts failing at 5 and 40: the early one stays in as a miss (lead -6), detection 50%"""
    error = np.zeros((2, H + 1))
    error[0, 5:] = error[1, 40:] = 0.5
    z = np.zeros((2, H + 1))
    z[0, W + 1 :] = z[1, 30:] = 9.0
    scores, thresholds = {"act_norm": scores_of(z)}, {"act_norm": 2.0}
    failed_at, _, alarm = score_rollouts(error, scores, thresholds, 0.10, 0.5, W)
    lead = failed_at - alarm["act_norm"]
    assert list(failed_at) == [5, 40] and list(lead) == [-6, 10]
    assert np.mean(lead[failed_at >= 0] > 0) == 0.5


def test_6a():
    """6a. Broken baseline, rollout 11 steps long: alarm at 11"""
    z = steady(1.0, steps=W + 1)
    z[7] = np.nan
    assert first_alarm(scores_of(z), 2.0, W)[0] == W + 1


def test_6b():
    """6b. Broken baseline, rollout 10 steps long: no step 11, counts as the last step (10)"""
    z = steady(1.0, steps=W)
    z[7] = np.nan
    assert first_alarm(scores_of(z), 2.0, W)[0] == W


def test_7():
    """7.  NaN, +inf or -inf at step 20 after a calm start: alarm at 20"""
    for bad in (np.nan, np.inf, -np.inf):
        z = steady(1.0)
        z[20] = bad
        assert first_alarm(scores_of(z), 2.0, W)[0] == 20


def test_8():
    """8.  Score never crosses the threshold: counts as the last step (200)"""
    assert first_alarm(scores_of(steady(1.0)), 2.0, W)[0] == H


def test_9():
    """9.  NaN, +inf or -inf threshold, given or tuned: error naming the signal, nothing scored"""
    score = {"act_norm": scores_of(steady(1.0))}
    attempts = [lambda bad=bad: alarm_steps(score, {"act_norm": bad}, W) for bad in (np.nan, np.inf, -np.inf)]
    attempts.append(lambda: tune({"act_norm": np.full((20, H + 1), np.nan)}, np.full(20, 100.0), 0.05))
    for attempt in attempts:
        try:
            attempt()
        except ValueError as error:
            assert "act_norm" in str(error)
        else:
            raise AssertionError("a threshold that isn't finite was accepted")


def test_10a():
    """10a. Calibration safe steps all inside the warm-up (big z there): each gives 0, threshold 0"""
    z = np.zeros((20, H + 1))
    z[:, 1 : W + 1] = 9.0
    assert calibrate(15, z) == 0.0


def test_10b():
    """10b. Calibration rollouts failing at 2 have no safe steps: threshold -1, alarm still at 11"""
    threshold = calibrate(2, np.zeros((20, H + 1)))
    assert threshold == -1.0 and first_alarm(scores_of(steady(0.0)), threshold, W)[0] == W + 1


if __name__ == "__main__":
    print(f"Python {platform.python_version()}, numpy {np.__version__}, warm-up {W} steps\n")
    tests = [test for name, test in list(globals().items()) if name.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__doc__}")
        except Exception as error:
            failed += 1
            print(f"FAIL  {test.__doc__}\n      {type(error).__name__}: {error}")
    print(f"\n{len(tests) - failed} of {len(tests)} checks passed")
    sys.exit(1 if failed else 0)
