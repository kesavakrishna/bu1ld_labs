# Decision memo: early warning for neural PDE surrogate rollouts

Kesava · BU1LD Fall 2026 Research Labs, thread 05 · 2026-10-05
Repo: https://github.com/kesavakrishna/bu1ld_labs · evaluation reference `freeze-v2` (`de54e3c`, unchanged) · results committed on top at `09ab581`

> **Correction, 2026-10-06 (documentation only).** Added after review. The memo below is unchanged from `2e7b9a1`, and nothing in the frozen protocol, code or scored outputs has changed.
>
> **1. Two sentences in "What this means" overstate the result.**
> - *"The internal signals carried no information the output didn't already carry, so white-box access bought nothing here."* The supported statement is narrower: **the tested internal signals did not satisfy the frozen superiority rule against the tested output-only baselines, in these four settings.** Failing to beat those baselines does not establish that the activations contain no additional information.
> - *"How long the rollout has run was the best available predictor of failure."* Should read: among the alarms tested, the fixed-step clock was competitive with, or better than, every signal.
>
> **2. The suggested horizon limit plus step-change tripwire is not a certified safety guarantee.** It is a practical heuristic suggested by these results, for this surrogate and these settings.
>
> **The scoped result to carry forward:**
> - **Gradual drift** (10% error) was not usefully anticipated by the tested internal statistics.
> - **Under viscosity up, the PDE residual** mostly detected the model/physics mismatch, and paid for it with a **76% false-alarm rate**.
> - **Outright blow-ups** were much easier to detect, and simple output-side changes were strongest.

## Decision

**Don't invest further in activation-statistic early-warning detectors of this kind.** Under the frozen decision rule, the internal signals were not supported in any of the four settings. None of them warned of rollout failure earlier than simple output-only checks, and none beat a fixed rollout-length limit. For practical safety, use a **rollout horizon limit** plus an **output-side step-change tripwire**. Both are cheap and need no access to the model's internals.

## The question

An autoregressive neural surrogate drifts away from the true solution as its own errors feed back in, and in deployment there is no ground truth to show it happening. We asked whether the surrogate's internal activations change *before* the output visibly fails, and whether that beats signals readable from the output alone.

## Setup

A 1D FNO (4 Fourier layers, width 64, 16 modes; seeds 0, 1, 2) is trained on single-step prediction for viscous Burgers' equation (ν = 1e-3, 256 points), then rolled out for 200 steps. It is tested on the training physics (**control**) and on three shifts it never saw: **viscosity up** (2e-3), **viscosity down** (7e-4), and **rougher initial waves**. That's 100 rollouts per setting per seed.

A rollout **fails** at the first step where relative L2 error exceeds 10%. We compare:

- **Three internal signals**, read with a hook on Fourier layer 3: activation norm, effective rank, and distance from typical activations.
- **Three output-only baselines**: step-to-step change, high-wavenumber energy, and PDE residual.
- **A fixed-step clock** as a floor.

Every signal is scored against its normal range at the same step. Every alarm is tuned to a 5% per-rollout false-alarm rate on calibration rollouts from the training physics, then applied unchanged to all settings. Lead time is the failure step minus the first alarm step; an alarm that never fires counts as firing at step 200.

## Primary result: the frozen decision rule

The rule, exactly as frozen in `FREEZE.md` §7:

> **Per setting:** supported if at least one internal signal has a per-seed median paired lead-time difference above zero against **every** output signal (`step_change`, `spectral_drift`, `pde_residual`), in **each** of seeds 0, 1 and 2.
> **Overall:** supported only if the per-setting verdict holds in **at least 3 of the 4** settings.

| Setting | Rollouts failing | Closest internal signal, and its weakest of 9 per-seed medians | Verdict |
| --- | --- | --- | --- |
| control | 256 / 300 | `act_norm`: 0 steps against `pde_residual` (a tie; the rule needs > 0) | **Not supported** |
| viscosity up | 295 / 300 | `train_dist`: −192 steps against `pde_residual` | **Not supported** |
| viscosity down | 273 / 300 | `act_norm`: 0 steps against `pde_residual` (a tie; the rule needs > 0) | **Not supported** |
| rougher waves | 279 / 300 | `act_norm`: 0 steps against `pde_residual` (a tie; the rule needs > 0) | **Not supported** |

**Overall: not supported (0 of 4 settings).** The medians of 0 are ties, not near-wins: in most paired comparisons, neither alarm fired before failure.

How each alarm did on the control:

| Alarm | Detected before failure | Median lead (steps) | False alarms |
| --- | --- | --- | --- |
| Activation norm (internal) | 10% | −112 | 6% |
| Effective rank (internal) | 5% | −103 | 5% |
| Distance from typical (internal) | 3% | −98 | 2% |
| Step change (output) | 6% | −100 | 3% |
| High-wavenumber energy (output) | 12% | −36 | 4% |
| PDE residual (output) | 14% | −82 | 7% |
| Fixed-step clock | 17% | −48 | 0% |

Every number in this memo traces to `results/summary.json`.

## Secondary and descriptive findings

These were pre-registered as secondary analyses or figures. They are not part of the verdict.

**1. Gradual drift gives no usable warning** (Figures 3 and 5). At the 10% failure step on the control, five of the six signals have median scores of 0.52–0.79 standard deviations from normal, about the level of noise. The PDE residual is slightly higher at 1.16. Across the false-alarm rates the clock covers, no signal catches more than about 1 percentage point more failures than the clock, in any setting. The only exception is the PDE residual under viscosity up, which runs at false-alarm rates of 48% and above, beyond anything the clock reaches.

**2. The internal "unfamiliarity" signal didn't notice the shifts** (Figures 2 and 6). Distance from typical activations stayed at a 2–7% false-alarm rate across all four settings, the same as on the training physics. Meanwhile, every shift made the model fail more often: 91–98% of rollouts against 85% for the control. Even the smoother physics failed more.

**3. The PDE residual detects a parameter mismatch, not impending failure.** Under viscosity up it flagged 83% of failures with a median of 18 steps' warning, but with a 76% false-alarm rate. It fires from the first step because the model's dynamics don't match the new viscosity.

**4. Blow-ups are predictable, and output checks predict them best** (failure-threshold sweep). About a quarter of rollouts diverge completely. With failure defined as 100% error, the ranges across the four settings were:

| Signal | Detected | Median lead | False alarms |
| --- | --- | --- | --- |
| Step change (output) | 97–100% | 12–14 steps | 3–7% |
| High-wavenumber energy (output) | 96–100% | 9–12 steps | 4–11% |
| Effective rank (internal) | 90–99% | 7–10 steps | 3–9% |
| Distance from typical (internal) | 19–27% | −1 step | 2–10% |

The decision rule was still not supported at this threshold, nor at any other in the sweep (2% to 100%). Of the alarms that kept false alarms near the 5% target, the output-side step change gave the most warning in every setting.

## What this means

For this surrogate, the failure that matters (quiet drift to a plausible but wrong state) isn't visible in any signal we measured. The one failure that *is* visible (blow-up) is caught as well or better by cheap output checks. The internal signals carried no information the output didn't already carry, so white-box access bought nothing here. Without ground truth, how long the rollout has run was the best available predictor of failure.

## What this setup can't tell us yet

**Whether per-trajectory baselines would rescue internal signals.** Each signal was compared with the normal range across *all* trajectories at the same step. That range is wide, so a single rollout can drift a long way from its own truth and still look normal. Comparing each rollout against its own early behaviour needs no ground truth either. It is the strongest remaining reason the internal signals could have been flat, and would need its own freeze.

Smaller limits: one PDE, one architecture, one hook layer; the blow-up results come from about a quarter of rollouts; and decaying Burgers' damps errors, so a forced equation may behave differently.

## What was run where (as in `FREEZE.md` §9)

- **Simulator-only feasibility checks.** The shift settings were chosen by checking which candidate physics the solver resolves accurately. No model was involved.
- **Development data and configuration.** Setup choices were made on the dev split, and signals were viewed on it, before the freeze: the time step, learning rate and batch size, and one gate threshold (40 → 30). New code was tested only on a separate small configuration (`configs/fast.yaml`), with its own data, seeds and model.
- **Scored evaluation.** The test split and the three shift datasets were generated and scored once, in a single run of `bash run_all.sh` at `freeze-v2`, on 2026-10-04 from 17:06 to 17:16 MDT, after the freeze was sent. Every gate passed. Results are committed unedited, with the run log (`results/frozen_run.log`).
