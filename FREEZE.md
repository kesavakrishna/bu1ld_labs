# Freeze receipt (v2): early-warning signals for FNO rollouts on 1D Burgers'

Pre-registration for BU1LD Fall 2026 Research Labs, thread 05, and the PDE failure-prediction sprint due 2026-10-06. Everything below is fixed at git tag **`freeze-v2`** and does not change once any test or shift data has been scored. A null or negative result will be reported as such.

## What v2 changes from v1

`freeze-v1` covered one setting. The sprint brief asks for at least three parameter or initial-condition shift settings, simple output-residual baselines, a false-positive rate, raw CSV and 5+ plots. v2 adds exactly these. Everything in v1 that isn't listed here is unchanged. Nothing has been scored under either freeze.

| Added in v2 | Where |
| --- | --- |
| Three shift settings, scored with thresholds tuned only on the original physics | §2, §7 |
| A third output-only baseline, `pde_residual` (how badly each predicted step breaks Burgers' equation) | §6 |
| Verdict generalised from "both output baselines" to "every output baseline", plus an overall rule across settings | §7 |
| False-alarm rate and the trade-off curve (detection against false alarms) reported for every setting | §7 |
| Raw CSVs and six figures | §8 |

## Rerun command

```bash
git clone https://github.com/kesavakrishna/bu1ld_labs.git && cd bu1ld_labs
git checkout freeze-v2
pip install -r requirements.txt
bash run_all.sh
```

This regenerates all data, runs the validity gates, trains the three models, performs the rollouts, tunes thresholds on calibration, and scores the control and shift settings once. It writes the outputs listed in §8. It takes about 12 minutes on an RTX 4060 Laptop GPU. Each gate is a hard stop: if one fails, the script exits and no lead-time result is produced.

---

## 1. Repository

| Item | Value |
| --- | --- |
| Repo | https://github.com/kesavakrishna/bu1ld_labs |
| Tag | `freeze-v2` (the commit SHA is given in the accompanying email; it can't appear inside the commit it names) |

## 2. Solver, data, splits and shift settings

**Equation:** `u_t + u·u_x = ν·u_xx` on `x ∈ [0, 1)` with periodic boundaries.

| Setting | Value |
| --- | --- |
| Spatial method | Pseudo-spectral (real FFT), 2/3-rule dealiasing on the nonlinear term, float64 |
| Time stepping | Classical RK4 |
| Grid | `N = 256` |
| Viscosity (training physics) | `ν = 1e-3` |
| Solver step | `dt_solver = 1e-3` |
| Saved step (surrogate stride) | every 5 solver steps, `Δt = 0.005` |
| Horizon | 200 saved steps, `H = 200` (`t ∈ [0, 1]`) |
| Initial conditions (training physics) | `u₀(x) = Σ_{m=1..5} a_m sin(2πmx + φ_m)`, `a_m ~ N(0,1)/m`, `φ_m ~ U(0, 2π)`, rescaled so `max|u₀| = 0.5` |

| Split | Trajectories | IDs | Drawn from | Role |
| --- | --- | --- | --- | --- |
| train | 1000 | 0–999 | seed 0 | Model fitting; the first 200 (IDs 0–199) also give the per-step reference statistics |
| calibration | 200 | 1000–1199 | seed 0 | Alarm thresholds only |
| dev | 100 | 1200–1299 | seed 0 | Validity gates and exploratory checks (called "test" before freeze-v1, see §9) |
| test (**control** setting) | 100 | 1300–1399 | seed 20260924 | Scored once, in the frozen run |

**Shift settings** are test-only physics or starting waves the model never trained on. Each has 100 trajectories (IDs 1400–1699, in the order listed). Their true solutions are simulated on a 1024-point grid (solver step ÷4) and read at the model's 256 points, because the sharper shocks of `viscosity_down` are too thin for the 256-point solver. At `ν = 7e-4`, the 1024-point truth matches a 2048-point run to within 0.001% at the model's points.

| Setting | Viscosity | Peak amplitude | Modes | Amplitude decay `a_m ∝ 1/m^d` | Seed | What changes |
| --- | --- | --- | --- | --- | --- | --- |
| control | 1e-3 | 0.5 | 5 | 1.0 | 20260924 | Nothing (the test split) |
| `viscosity_up` | 2e-3 | 0.5 | 5 | 1.0 | 1001 | Smoother physics; shocks peak around step 136 instead of 90 |
| `viscosity_down` | 7e-4 | 0.5 | 5 | 1.0 | 1002 | Sharper shocks |
| `rougher_waves` | 1e-3 | 0.5 | 5 | 0.5 | 1003 | More small-scale starting detail; shocks form twice as early (around step 41) |

**Validity gates** (hard stops):

- Energy `mean(u²)` never increases between saved steps (relative tolerance 1e-5). This is checked on every split and on every shift setting's fine-grid solution.
- The 256-point solution agrees with a 1024-point run (solver step ÷4) within 1% relative L2 at every saved step, compared at the shared grid points. Checked on 20 fresh trajectories drawn with seed 12345.

## 3. Surrogate and seeds

| Setting | Value |
| --- | --- |
| Architecture | 1D FNO: lift `Conv1d(2→64)` → 4 × [spectral conv keeping the 16 lowest modes, with a complex 64×64 weight per mode, plus pointwise `Conv1d(64→64)`, then GELU] → `Conv1d(64→128)` → GELU → `Conv1d(128→1)` |
| Inputs | `u(t)` and the grid coordinate `x` |
| Output | `u(t+Δt)`, predicted directly (not as a residual) |
| Normalisation | Scalar mean and std of the training inputs, stored inside the model and applied to both input and output |
| Parameters | 549,569 real |
| Training data | Single-step pairs only, 20 random `(t, t+1)` pairs per training trajectory (20,000 pairs, sampled with seed 0). Training physics only |
| Loss, optimiser | Mean relative L2; Adam, learning rate 0.002, cosine annealing over 100 epochs, batch size 256 |
| Seeds | **0, 1, 2**. Each sets weight initialisation and batch order; the data is identical across seeds |
| Determinism | cuDNN deterministic mode; two runs of the same seed gave bit-identical weights on the reference machine |

**Gate** (hard stop): mean single-step relative L2 on dev below 1% for every seed.

## 4. Failure definition

- Rollouts are autoregressive from the true `u(0)` for `H = 200` steps.
- Relative error: `e(t) = ‖û(t) − u(t)‖₂ / ‖u(t)‖₂` over the 256 grid points.
- **Failure step** `t_fail`: the first `t ≥ 1` with `e(t) > 0.10`, or with a non-finite value (overflow). A rollout with no such `t` within `H` never fails.

**Gate** (hard stop), on dev, per seed: at least 70% of rollouts fail, and the median `t_fail` is at least 30.

## 5. Activation statistics (internal signals)

Activations are captured with a forward hook on the output of Fourier layer 3 of 4 (after GELU): `A ∈ ℝ^{64×256}` per trajectory per step.

| Signal | Definition |
| --- | --- |
| `act_norm` | Mean over the 64 channels of `‖A_c‖₂` |
| `eff_rank` | `exp(H(p))`, where `p = σ / Σσ` and `σ` are the singular values of `A`. Computed as `√eig(A·Aᵀ)` after scaling `A` by `max|A|`; this matches `svdvals` to within 3×10⁻⁵ relative |
| `train_dist` | `1 − cos(s, s̄(t))`, where `s` is each channel's mean and standard deviation over `x` (128 values) and `s̄(t)` is the mean of `s` over the reference runs at step `t` |

## 6. Output-only baselines

| Signal | Definition |
| --- | --- |
| `step_change` | `‖û(t) − û(t−1)‖₂ / ‖û(t−1)‖₂` |
| `spectral_drift` | Energy-spectrum drift: the share of the prediction's energy in the top third of wavenumbers, `Σ_{m≥86} |û_m|² / Σ_m |û_m|²` over real-FFT modes `m = 0..128`. The solver's 2/3 rule keeps this band empty in true solutions |
| `pde_residual` (v2) | Physics residual of the predicted step. With the midpoint `w = (û(t−1) + û(t))/2`: `‖r‖₂ / ‖(û(t) − û(t−1))/Δt‖₂`, where `r = (û(t) − û(t−1))/Δt + ∂ₓ(w²/2) − ν ∂ₓₓw`. Fourier derivatives are used, the 2/3 rule is applied to `w²/2`, and `ν` is the setting's true viscosity (always known to a user). On true dev steps it has a median of 0.001 and a 99th percentile of 0.004 |
| `clock` | A trivial floor: always predicts failure at a fixed step `k` |

### Scoring, the same for all six signals

- **Step labels.** Everything measured during the model call that produces `û(t)` is labelled `t`, the same index as `e(t)`.
- **Reference runs.** The model is applied once to the true `u(t−1)` at every step on the reference trajectories (train IDs 0–199, training physics), giving a per-step mean `μ(t)` and standard deviation `σ(t)` for each signal. Every setting, including the shifts, is scored against this one reference.
- **Score.** `score(t) = |signal(t) − μ(t)| / σ(t)`. Non-finite scores are set to `+∞`, and `score(0) = 0`.

## 7. Primary metric

All steps are applied separately for each seed and each alarm.

**Calibration** (calibration split only, which uses the training physics). Thresholds are set once and applied unchanged to every setting, as they would be in real deployment, where the shift isn't known in advance.

1. Safe window: `W = {t : 1 ≤ t < 0.5·t_fail}`, or `0.5·H` for rollouts that never fail.
2. Threshold `τ` = the 0.95 quantile (numpy `quantile`, linear interpolation) of each rollout's maximum score over `W`. A rollout with an empty `W` contributes −1; an infinite score is capped at the largest float.
3. Clock: `k` = the smallest `k ≥ 1` such that at most 5% of calibration rollouts have `k ∈ W`.

**Scoring**, in each setting (control and each shift):

1. `t_signal` = the first `t ≥ 1` with `score(t) > τ`, or `H` if there is none. For the clock, `t_signal = k`.
2. Lead time = `t_fail − t_signal`, for each failing rollout.

**Reported for every alarm, in every setting:**

- Median lead time and IQR, pooled over failing rollouts of all three seeds, plus each seed's median
- Detection rate: the share of failing rollouts with lead time > 0
- **False-alarm (false-positive) rate:** the share of rollouts with `t_signal ∈ W`. The 5% target holds on calibration; under shift, the observed rate is itself a result
- Counts of failing and non-failing rollouts

**Head-to-head:** for each internal × output pair (9 pairs) and each signal against the clock (6 pairs), compute `d = lead_a − lead_b` for every failing rollout. Report the median of `d`, pooled and per seed, and the shares of wins, ties and losses.

**Primary verdict, per setting.** "Internal signals beat the output-only baselines" is **SUPPORTED** in a setting if at least one internal signal has a per-seed median `d > 0` against **every** output signal (`step_change`, `spectral_drift` and `pde_residual`), in **each** of seeds 0, 1 and 2.

**Overall verdict:** **SUPPORTED** only if the per-setting verdict holds in **at least 3 of the 4 settings**. Otherwise it is **NOT SUPPORTED**.

**Secondary analyses**, reported whatever the outcome and not part of either verdict:

- The full procedure repeated with the failure threshold at 0.02, 0.05, 0.10, 0.20, 0.50 and 1.00, in every setting.
- A trade-off curve: the alarms re-tuned to calibration false-alarm targets of 1%, 2%, 5%, 10%, 20%, 30% and 50%, with test detection rate and false-alarm rate reported for each, in every setting.

**Stopping rule:** if any gate fails in the frozen run, no lead-time result is claimed, and the gate failure is reported.

## 8. Outputs

| File | Contents |
| --- | --- |
| `results/summary.json` | Every reported number |
| `results/leadtimes.csv` | One row per setting, seed, rollout and alarm: threshold, failure step, alarm step, lead time, detected, false alarm |
| `results/rollout_steps.csv.gz` | One row per setting, seed, rollout and step: relative error, raw error, true size and every score. Everything above can be recomputed from this |
| `results/figures/1_lead_time.png` to `6_rollout_error.png` | Lead time, false alarms, trade-off, head-to-head, signals around failure, and rollout error, per setting |
| `results/day1_*.png` to `day4_*.png` | Setup checks (simulator, training, dev rollouts, dev signals) |

## 9. Disclosure: what was run and inspected before the freeze

All choices below were made before any lead time was computed on full-configuration data. Any that were informed by looking at outcomes are marked as such.

**Settled in writing on 2026-09-16, before any rollout existed:**

- Per-rollout (rather than per-step) false-alarm matching
- Detection rate reported alongside lead time
- An alarm that never fires counts as `t_signal = H`
- Explicit rule for rollouts that never fail
- Per-step normalisation of signals against reference runs
- Paired head-to-head comparison and the verdict rule
- Failure threshold of 0.10 (from the original spec)
- The five original signals (from the original spec)

**Setup choices informed by results (before freeze-v1):**

| When | Change | What it was based on |
| --- | --- | --- |
| Day 1 | Initial amplitude 1.0 → 0.5 | The 256 vs 1024 resolution check failed, at up to 4% (exploratory batches, seeds 12345 and 777) |
| Day 2 | Batch size 64 → 256, learning rate 0.001 → 0.002 | Speed, and single-step error in 2–20-epoch runs on the split now called dev |
| Day 3 | `Δt` 0.02 → 0.005 | At `Δt = 0.02`, dev rollouts failed in a burst during shock formation (median failure step 13–16 across seeds). Starting rollouts later (steps 10–75) was also tried on dev and rejected |
| Day 3 | Horizon kept at 200 | A 400-step trial overflowed by step 300 |
| Day 3 | Gate minimum median failure step 40 → 30 | Seed 1's dev median was 38 (seeds 0 and 2: 46, 47). The gate decides whether the setup is usable; it does not enter the internal-vs-output comparison |

**Exploratory look at signals (Day 4, before freeze-v1):** all five original scores were computed on dev rollouts and viewed aligned to each rollout's failure step. At the failure step, median scores were 0.6–0.8 for every signal, about the level of noise. Internal scores were flat; output-side scores rose slightly before failure and spiked after it. Lead times were not computed. This look informed two decisions: adding 0.50 and 1.00 to the secondary sweep, and freezing with a fresh test split.

**Removed from the earlier plan** (moved to `NEXT_STEPS.md`, not run): a second failure definition (`‖û − u‖ / ‖u(0)‖`, removed because dev showed failure isn't caused by the wave shrinking), a windowed `eff_rank` retry, and a threshold-stability bootstrap. Removing these can only take away chances for internal signals to look better.

**Choices made for v2:**

- **Shift settings** were chosen with a solver-only feasibility check: 10 candidates, 10 trajectories each, seeds 4242–4251, checking resolution, energy and shock timing. No model was involved. Candidates that sharpen shocks (viscosity 7e-4 and 8e-4, amplitude 0.6 and 0.65) failed the 256-point resolution check, which led to simulating all shift truths on the 1024-point grid. The three settings cover easier physics, harder physics, and a starting-wave shift.
- **`pde_residual`** was added because the sprint brief asks for output-residual baselines. Its formula was checked only on true dev waves (no model) and on dev waves with 1% added noise, where the median residual jumps from 0.001 to 1.03. Adding a third output baseline makes the per-setting verdict harder for internal signals to pass.
- **The overall rule** (3 of 4 settings) was fixed before any shift data existed.
- **The v2 code was tested only on the small configuration** (`configs/fast.yaml`: its own data, test seed 99, shift seeds 91–93, and its own model).

**Not yet generated or seen:** the shift datasets for the full configuration are generated for the first time by `run_all.sh` at `freeze-v2`. The test split (seed 20260924) has had only its shape, IDs and seed checked. No rollout, error or score has been computed on either.

## 10. Environment

Python 3.13.1, PyTorch 2.13.0+cu126, numpy 2.3.2 (all pinned in `requirements.txt`), running on an RTX 4060 Laptop GPU with driver 616.92. Other GPUs or driver versions may produce small numerical differences.
