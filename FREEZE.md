# Freeze receipt: early-warning signals for FNO rollouts on 1D Burgers'

Pre-registration for BU1LD Fall 2026 Research Labs, thread 05. Everything below is fixed at git tag **`freeze-v1`** and does not change once the test split has been scored. A null or negative result will be reported as such.

## Rerun command

```bash
git checkout freeze-v1
pip install -r requirements.txt
bash run_all.sh
```

This regenerates the data, runs the validity gates, trains all three models, performs the rollouts, tunes thresholds on calibration, scores the test split once, and writes `results/summary.json` (every number) and `results/leadtime.png`. It takes about 8 minutes on an RTX 4060 Laptop GPU. Each gate is a hard stop: if one fails, the script exits and no lead-time result is produced.

---

## 1. Repository

| Item | Value |
| --- | --- |
| Tag | `freeze-v1` |
| Commit and repo URL | Given in the accompanying email; the SHA can't appear inside the commit it names |

## 2. Solver, data and splits

**Equation:** `u_t + u·u_x = ν·u_xx` on `x ∈ [0, 1)` with periodic boundaries.

| Setting | Value |
| --- | --- |
| Spatial method | Pseudo-spectral (real FFT), 2/3-rule dealiasing on the nonlinear term, float64 |
| Time stepping | Classical RK4 |
| Grid | `N = 256` |
| Viscosity | `ν = 1e-3` |
| Solver step | `dt_solver = 1e-3` |
| Saved step (surrogate stride) | every 5 solver steps, `Δt = 0.005` |
| Horizon | 200 saved steps, `H = 200` (`t ∈ [0, 1]`) |
| Initial conditions | `u₀(x) = Σ_{m=1..5} a_m sin(2πmx + φ_m)`, `a_m ~ N(0,1)/m`, `φ_m ~ U(0, 2π)`, rescaled so `max|u₀| = 0.5` |

| Split | Trajectories | IDs | Drawn from | Role |
| --- | --- | --- | --- | --- |
| train | 1000 | 0–999 | seed 0 | Model fitting; the first 200 (IDs 0–199) also give the per-step reference statistics |
| calibration | 200 | 1000–1199 | seed 0 | Alarm thresholds only |
| dev | 100 | 1200–1299 | seed 0 | Validity gates and exploratory checks (called "test" before the freeze, see §8) |
| test | 100 | 1300–1399 | seed 20260924 | Scored once, in the frozen run |

**Validity gates** (hard stops):

- Energy `mean(u²)` never increases between saved steps (relative tolerance 1e-5).
- The 256-point solution agrees with a 1024-point run (solver step ÷4) within 1% relative L2 at every saved step, compared at the shared grid points. Checked on 20 fresh trajectories drawn with seed 12345.

## 3. Surrogate and seeds

| Setting | Value |
| --- | --- |
| Architecture | 1D FNO: lift `Conv1d(2→64)` → 4 × [spectral conv keeping the 16 lowest modes, with a complex 64×64 weight per mode, plus pointwise `Conv1d(64→64)`, then GELU] → `Conv1d(64→128)` → GELU → `Conv1d(128→1)` |
| Inputs | `u(t)` and the grid coordinate `x` |
| Output | `u(t+Δt)`, predicted directly (not as a residual) |
| Normalisation | Scalar mean and std of the training inputs, stored inside the model and applied to both input and output |
| Parameters | 549,569 real |
| Training data | Single-step pairs only, 20 random `(t, t+1)` pairs per training trajectory (20,000 pairs, sampled with seed 0) |
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
| `clock` | A trivial floor: always predicts failure at a fixed step `k` |

### Scoring, the same for all five signals

- **Step labels.** Everything measured during the model call that produces `û(t)` is labelled `t`, the same index as `e(t)`.
- **Reference runs.** The model is applied once to the true `u(t−1)` at every step on the reference trajectories (train IDs 0–199), giving a per-step mean `μ(t)` and standard deviation `σ(t)` for each signal.
- **Score.** `score(t) = |signal(t) − μ(t)| / σ(t)`. Non-finite scores are set to `+∞`, and `score(0) = 0`.

## 7. Primary metric

All steps below are applied separately for each seed and each alarm.

**Calibration** (calibration split only):

1. Safe window: `W = {t : 1 ≤ t < 0.5·t_fail}`, or `0.5·H` for rollouts that never fail.
2. Threshold `τ` = the 0.95 quantile (numpy `quantile`, linear interpolation) of each rollout's maximum score over `W`. A rollout with an empty `W` contributes −1; an infinite score is capped at the largest float.
3. Clock: `k` = the smallest `k ≥ 1` such that at most 5% of calibration rollouts have `k ∈ W`.

This matches the false-alarm rate **per rollout**, at 5%, for every alarm.

**Test:**

1. `t_signal` = the first `t ≥ 1` with `score(t) > τ`, or `H` if there is none. For the clock, `t_signal = k`.
2. Lead time = `t_fail − t_signal`, for each failing test rollout.

**Reported for every alarm:**

- Median lead time and IQR, pooled over the failing test rollouts of all three seeds
- Each seed's median
- Detection rate: the share of failing rollouts with lead time > 0
- Test false-alarm rate: the share of rollouts with `t_signal ∈ W`
- Counts of failing and non-failing rollouts

**Head-to-head:** for each internal × output pair (6 pairs), and each signal against the clock (5 pairs), compute `d = lead_a − lead_b` for every failing test rollout. Report the median of `d`, pooled and per seed, and the shares of wins, ties and losses.

**Primary verdict.** "Internal signals beat the output-only baselines" is **SUPPORTED** if at least one internal signal has a per-seed median `d > 0` against **both** output signals in **each** of seeds 0, 1 and 2. Otherwise it is **NOT SUPPORTED**.

**Secondary analysis**, reported whatever the outcome and not part of the verdict: the whole procedure repeated with the failure threshold at 0.02, 0.05, 0.10, 0.20, 0.50 and 1.00.

**Stopping rule:** if any gate fails in the frozen run, no lead-time result is claimed, and the gate failure is reported.

## 8. Disclosure: what was run and inspected before the freeze

All choices below were made before the freeze, and before any lead time was computed on full-configuration data. Any that were informed by looking at outcomes are marked as such.

**Settled in writing on 2026-09-16, before any rollout existed:**

- Per-rollout (rather than per-step) false-alarm matching
- Detection rate reported alongside lead time
- An alarm that never fires counts as `t_signal = H`
- Explicit rule for rollouts that never fail
- Per-step normalisation of signals against reference runs
- Paired head-to-head comparison and the verdict rule
- Failure threshold of 0.10 (from the original spec)
- The five signals (from the original spec)

**Setup choices informed by results:**

| When | Change | What it was based on |
| --- | --- | --- |
| Day 1 | Initial amplitude 1.0 → 0.5 | The 256 vs 1024 resolution check failed, at up to 4% (exploratory batches, seeds 12345 and 777) |
| Day 2 | Batch size 64 → 256, learning rate 0.001 → 0.002 | Speed, and single-step error in 2–20-epoch runs on the split now called dev |
| Day 3 | `Δt` 0.02 → 0.005 | At `Δt = 0.02`, dev rollouts failed in a burst during shock formation (median failure step 13–16 across seeds). Starting rollouts later (steps 10–75) was also tried on dev and rejected |
| Day 3 | Horizon kept at 200 | A 400-step trial overflowed by step 300 |
| Day 3 | Gate minimum median failure step 40 → 30 | Seed 1's dev median was 38 (seeds 0 and 2: 46, 47). The gate decides whether the setup is usable; it does not enter the internal-vs-output comparison |

**Exploratory look at signals (Day 4):** all five scores were computed on dev rollouts and viewed aligned to each rollout's failure step. At the failure step, median scores were 0.6–0.8 for every signal, about the level of noise. Internal scores were flat; output-side scores rose slightly before failure and spiked after it. Lead times were not computed. This look informed two decisions:

- Adding 0.50 and 1.00 to the secondary sweep. The earlier plan listed 0.02–0.20.
- Freezing with a fresh test split, drawn with seed 20260924.

**Removed from the earlier plan** (moved to `NEXT_STEPS.md`, not run):

- A second failure definition, `‖û − u‖ / ‖u(0)‖`. Removed under the plan's own rule: dev rollouts showed failure is not caused by the true wave shrinking, since at failure the true wave is 96–98% of its step-20 size while the raw error has grown 1.5–1.7×.
- A windowed `eff_rank` retry.
- A threshold-stability bootstrap.

Removing these can only take away chances for internal signals to look better.

**How the scoring code was tested:** only on a separate small configuration (`configs/fast.yaml`, with its own data, test seed 99 and its own models). Nothing in `leadtime.py` was run on full-configuration rollouts before the freeze.

**The test split** (seed 20260924, IDs 1300–1399) was generated while preparing the freeze. Only its shape, IDs and seed were checked; no rollout, error or score has been computed on it. Regenerating the data reproduced train, calibration and dev bit for bit.

## 9. Environment

Python 3.13.1, PyTorch 2.13.0+cu126, numpy 2.3.2 (all pinned in `requirements.txt`), running on an RTX 4060 Laptop GPU with driver 616.92. Other GPUs or driver versions may produce small numerical differences.
