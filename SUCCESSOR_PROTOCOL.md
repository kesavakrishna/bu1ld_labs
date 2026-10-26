# Successor protocol (draft): warning signals measured against each rollout's own start

**Status: DRAFT. Not frozen, not run, and no code written.** Nothing in this file has been computed on any data. It becomes a protocol only once it is frozen under its own tag (§8), after the lab has reviewed it.

This is a **new hypothesis**, not an amendment. [FREEZE.md](FREEZE.md) (freeze-v2, `de54e3c`) and the October 4 scored run (`09ab581`) stay the terminal evidence for the original question, and [MEMO.md](MEMO.md) stands as written whatever this protocol finds.

## 1. The question

The original run scored each signal against **"normal for everyone"**: the spread of 200 healthy reference runs at the same step. That spread is wide, so a rollout can drift a long way and still look normal.

This protocol scores each signal against **"normal for this rollout"**: its own first 10 steps, while it is still healthy.

> **Hypothesis.** When every signal is scored against its own rollout's first 10 steps, at least one internal signal warns of failure earlier than every output-only signal, at a matched 5% false-alarm rate.

**Where the idea came from.** In the October 4 run, internal scores rose before failure (figure 5) but stayed below thresholds set from the population spread. That observation is why the hypothesis exists, so it can't be tested on the data that produced it. Fresh data is required (§4).

## 2. What stays exactly as in freeze-v2

Everything except the score definition in §3:

- Solver, physics settings, grid, `Δt`, horizon `H = 200`
- The three trained models (seeds 0, 1, 2) from freeze-v2, used as they are. **No training.**
- Train, calibration and dev splits, and the reference runs (train IDs 0–199) with their per-step `μ(t)`, `σ(t)`
- The six signals and the clock, as defined in FREEZE.md §5–6
- Failure definition: error > 0.10, or non-finite
- Calibration, safe window, 5% false-alarm matching, lead time, detection rate, head-to-head and both verdict rules, exactly as in FREEZE.md §7

## 3. The one change: the score

For each signal `s` and each rollout:

1. **Remove the normal trend over time.** Signals change naturally as the wave evolves, so first take the *signed* version of the old score, using the unchanged freeze-v2 reference:
   `z(t) = (s(t) − μ(t)) / σ(t)`
2. **Find this rollout's own normal level.** `b = mean of z(t)` over steps `t = 1..10`.
3. **Score how far the rollout has moved from its own normal.**
   `own_score(t) = |z(t) − b|` for `t > 10`, and `own_score(t) = 0` for `t ≤ 10`.
4. As before, non-finite values become `+∞`. If `b` itself is non-finite, `own_score(t) = +∞` for every `t > 10`.

This is applied to **all six signals**, internal and output alike, so a win can't come from scoring one side differently. The clock doesn't use scores, so it is unchanged.

**Why only the offset, not dividing by the rollout's own spread as well:** a spread estimated from 10 steps is too noisy. Calibration (5% false alarms) already puts every alarm on the same scale.

**Known cost:** no own-scored signal can fire during steps 1–10, so a rollout that fails by step 10 counts as a miss. In the October 4 results, about 4% of rollouts in each setting failed by step 10 (12–28% by step 20). The window length was chosen knowing this (§7). Other known risks are listed in §6.

## 4. Data

Fresh test data for all four settings, generated **only after the freeze**, with seeds fixed here:

| Setting | Physics | Trajectories | IDs | Seed |
| --- | --- | --- | --- | --- |
| control | as freeze-v2 control | 100 | 1700–1799 | 20261017 |
| viscosity_up | as freeze-v2 | 100 | 1800–1899 | 2001 |
| viscosity_down | as freeze-v2 | 100 | 1900–1999 | 2002 |
| rougher_waves | as freeze-v2 | 100 | 2000–2099 | 2003 |

The old test and shift data (IDs 1300–1699) are **not** used for scoring, because their results have already been seen. Thresholds are tuned on the existing calibration split (IDs 1000–1199), which was used only for threshold tuning and never graded.

## 5. Decision rules (fixed before any result exists)

- **Primary verdict, per setting:** SUPPORTED if at least one internal signal has a per-seed median `d > 0` against every output signal, in each of seeds 0, 1 and 2, with all signals own-scored. This is the freeze-v2 rule, unchanged.
- **Overall:** SUPPORTED only if it holds in at least 3 of the 4 settings. Otherwise NOT SUPPORTED.
- **What a yes would mean:** own-normal scoring reveals warning information in the internal signals that population scoring hid. It would still not be a safety guarantee.
- **What a no would mean:** own-normal scoring doesn't rescue the internal signals either, and the negative result extends to this scoring.

Either way, every setting is reported, including those that fail the rule.

**Reported alongside, not part of the verdict:**

- **Does own-normal scoring help at all?** For each signal, the paired lead-time difference between its own-scored and population-scored versions, on the same fresh rollouts.
- **Each signal against the clock**, as in freeze-v2.
- **False-alarm rates under shift**, and the failure-threshold sweep and trade-off curve from FREEZE.md §7.

## 6. Known risks

Written down before any result, so none of them can be raised afterwards as an excuse.

1. **Healthy rollouts that change on their own.** §3 step 1 removes changes that all rollouts share, but not changes that only some healthy rollouts make. Calibration would then see healthy rollouts moving away from their own start and set a higher threshold, making the alarm less sensitive. If that happens often, own-normal scoring loses its advantage.
2. **The first 10 steps may already be unhealthy.** Errors build up from step 1, so some rollouts may already be drifting while their own normal is being measured. Part of the problem is then counted as normal, and the warning looks smaller.
3. **A shift gets counted as normal.** Under a shift, the model is slightly off from step 1, and own-normal scoring treats that offset as normal. This should cut false alarms (`pde_residual` false-alarmed on 76% of viscosity_up rollouts in the October 4 run), but if the shift is what drives the failure, part of the warning is cancelled too.
4. **It may help the output signals as much as the internal ones.** The verdict compares internal with output signals, so if own-normal scoring improves every signal equally, the answer is no even though the scoring helped. The own-vs-population comparison in §5 shows this case.
5. **This is a second attempt on the same question.** Each extra hypothesis tested on the same setup raises the chance that one passes by luck. This is the only successor protocol, with its rules fixed in advance. A yes would justify repeating the test, not settle the question.
6. **The bar is strict.** Passing needs a win in every seed, against every output signal, in 3 of 4 settings, with 100 rollouts per setting. A small real advantage could still fail, so a no means "no advantage large enough to pass this bar", not "no information at all". The bar is kept so the result is comparable with freeze-v2.

## 7. Disclosure

- The hypothesis came from inspecting the October 4 results (§1).
- The 10-step window was chosen after looking at the October 4 failure-step distribution, to keep the share of rollouts that are impossible to warn small (§3).
- Nothing in this protocol has been computed on any data, old or new.

## 8. Before anything is run

1. The lab reviews this draft.
2. The code is written and tested on the fast config only. Dev and the old test data are not used to test the new score.
3. The code and this file are committed and tagged `freeze-s1`, and the commit hash is sent to the lab.
4. Only then is the fresh data generated, and the run done once.

**Compute:** no training. About 400 new trajectories, plus rollouts with the three existing models: well under an hour on the RTX 4060.

---

## Amendment 1 (2026-10-21), after review

Everything above this line is the draft reviewed at `c89f8b5`, left unchanged. Where this amendment differs from it, the amendment applies. Still not frozen and not run.

### A1. False-alarm rate (replaces the hypothesis wording in §1)

> **Hypothesis.** When every signal is scored against its own rollout's first 10 steps, with thresholds calibrated to a 5% false-alarm rate on the in-distribution calibration split (IDs 1000–1199), at least one internal signal warns of failure earlier than every output-only signal.

- Thresholds are recomputed from the own-scores on the calibration split only, with the freeze-v2 `tune_threshold`, unchanged.
- The 5% target holds on calibration only. The false-alarm rate achieved in each setting is a result, reported separately, as in freeze-v2.
- No threshold, window or rule is chosen from fresh test outcomes.

### A2. Scope of the conclusion (replaces "What a yes would mean" in §5)

A yes means only this: under own-normal scoring and the frozen rule, at least one internal signal gave earlier warning than every output-only signal. It is not evidence that population scoring hid information. Any statement about what the scoring change did comes only from the paired own-vs-population analysis, reported as secondary.

### A3. Warm-up, non-finite values and alarms (makes §3 exact)

For each signal and rollout, with warm-up length `W = 10` and horizon `H`:

1. `z(t) = (s(t) − μ(t)) / σ(t)`, signed, as in §3.
2. Baseline `b = mean of z(t)` over `t = 1..10`. If any of those values is non-finite, `b` is non-finite.
3. For `t > 10`: `own_score(t) = |z(t) − b|`. A non-finite result, including one caused by a non-finite `b`, becomes `+∞`.
4. For `t ≤ 10`: `own_score(t) = 0`, whatever `z(t)` is. **The warm-up takes precedence over the non-finite rule.**
5. **Calibration** uses these scores as they are, so warm-up steps inside a safe window contribute 0. A rollout with an empty safe window still contributes −1, as in freeze-v2.
6. **Alarms** are searched for only at `t > 10`: `t_signal` = the first `t` in `11..H` with `own_score(t) > τ`, or `H` if there is none. This holds for any finite threshold, including a negative or zero one, because setting warm-up scores to 0 alone does not guarantee silence when `τ < 0`. If `H ≤ 10`, no alarm can fire and `t_signal = H`. The freeze-v2 `first_alarm` is left unchanged and is not used for own-scores.
7. **Thresholds** must be finite. A NaN or infinite threshold is reported as an error naming the signal, and nothing is scored with it.

What follows from these rules:

- **Early failures stay counted.** A rollout with `t_fail ≤ 10` stays among the failing rollouts. Its `t_signal ≥ 11`, so its lead time is negative: a miss for every own-scored signal. The clock is unaffected.
- **A broken baseline** gives `own_score(t) = +∞` for every `t > 10`, so the alarm fires at step 11, provided `τ` is finite and `H ≥ 11`.
- **A blow-up after step 10** gives `+∞` at that step, so the alarm fires there unless it already fired earlier.

### A4. This milestone

- **Implementation:** at most three new files, about 200 lines in total: `src/own_scores.py` (own-scores and the warm-up alarm), `configs/successor.yaml` (warm-up length and the fresh seeds and IDs from §4) and `tests/test_own_scores.py` (synthetic checks). No freeze-v2 file is edited.
- **Checks:** synthetic arrays only. No rollout data is read, no data is generated, and no model is loaded or trained. This replaces §8 step 2: the fast config is not run in this milestone.
- **Synthetic cases:** finite and non-finite warm-up values; negative, zero and positive finite thresholds; early failures kept as misses; a broken baseline giving the step-11 alarm (finite `τ`, `H ≥ 11`); a blow-up after step 10; the horizon fallback when no alarm fires; and invalid (non-finite) thresholds reported as errors.
- **Handoff:** the commit, its diff, the test command and the unedited test output go to the lab. The freeze (`freeze-s1`) and any run need separate agreement.
- **Cap:** about one week. If the work outgrows the three files or about 200 lines, it stops and is reported back.
