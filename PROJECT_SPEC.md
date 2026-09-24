# Early-Warning Harness for Neural PDE Surrogates

**Working title:** Do representations shift before rollouts fail?
**Context:** Week-1 deliverable for The BU1LD Fall 2026 Research Labs, thread 05 — *Dynamical Representation Phase Transitions for PDE Surrogates*.
**Time budget:** ~7 days, part-time alongside coursework. Roughly 12–18 hours total.
**Compute budget:** Must run start-to-finish in under 20 minutes on free Colab (T4) or a laptop CPU.

---

## 1. The question in plain terms

A neural surrogate is a model trained to imitate a physics solver because it runs much faster. You give it the state of a system at time `t`, it predicts time `t+1`, and you feed that prediction back in to keep going. This is called an autoregressive rollout.

The known failure mode: small errors compound. The rollout looks fine for a while, then drifts, then blows up. In real deployment you don't have the true solver to compare against, so you can't see the error growing — you only see the model's output, which may look plausible right up until it doesn't.

**Hypothesis:** the model's *internal activations* start behaving differently before the output error becomes obvious. If true, you get a warning signal that needs no ground truth.

**The thing that makes this research rather than a demo:** you must check whether internal signals actually beat signals you could read off the *output* alone. If a dumb output-side statistic warns just as early, the internal machinery buys nothing. That negative result is a legitimate and publishable finding — report it, don't bury it.

---

## 2. Scope discipline

This is deliberately small. Resist scope creep. **Out of scope for week 1:**

- More than one PDE (Burgers' only)
- More than one architecture (FNO only)
- 2D problems
- Sparse autoencoders, circuit-level interpretability, causal interventions
- Training anything large, or anything that needs a paid GPU
- Fixing the drift. We are *detecting* it, not preventing it.

If something above seems tempting, write it in `NEXT_STEPS.md` instead of building it.

---

## 3. Target repo structure

```
pde-early-warning/
├── README.md               # what this is, how to run it, headline result
├── requirements.txt        # pinned versions (==, not >=)
├── run_all.sh              # ONE command reproduces everything
├── configs/
│   └── default.yaml        # all hyperparameters, no magic numbers in code
├── src/
│   ├── solver.py           # ground-truth Burgers' solver + dataset generation
│   ├── model.py            # 1D FNO
│   ├── train.py            # single-step training loop
│   ├── rollout.py          # autoregressive rollout + activation capture
│   ├── signals.py          # internal signals + output-side baselines
│   ├── leadtime.py         # threshold calibration + lead-time computation
│   └── plots.py            # figure generation
├── results/                # generated, gitignored except the final figure
└── NEXT_STEPS.md           # what I'd do with more time
```

Use `argparse` + a YAML config. Do **not** pull in Hydra or a heavy experiment framework — it's overkill here and adds install friction for whoever reproduces this.

---

## 4. Implementation detail

### 4.1 Ground-truth solver (`solver.py`)

1D viscous Burgers' equation on a periodic domain:

```
∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²,   x ∈ [0, 1),   periodic
```

- Pseudo-spectral in space (FFT), RK4 in time, with 2/3-rule dealiasing on the nonlinear term.
- Grid: `N_x = 256`. Viscosity: `ν = 1e-3` (start here; see §6 for tuning).
- Initial conditions: superposition of random Fourier modes, amplitudes decaying with wavenumber, random phases. Seeded.
- Solver timestep `dt_solver` should be small for stability; the surrogate learns a coarser stride `Δt = k · dt_solver` (start with the coarse step ≈ 100× the solver step, tune so single-step change is visible but not chaotic).
- Generate: 1000 training trajectories, 100 calibration, 100 test. Save as `.npz` with the seed recorded in the file.

**Sanity check before moving on:** total energy `∫u² dx` should decay monotonically. Shocks should form and then smooth out. If energy grows, the solver is unstable — reduce `dt_solver`.

### 4.2 Surrogate (`model.py`, `train.py`)

Standard 1D Fourier Neural Operator:
- `modes = 16`, `width = 64`, 4 Fourier layers, GELU activations.
- Input: `u(t)` plus the spatial grid coordinate as an extra channel. Output: `u(t+Δt)`.
- Train on **single-step** prediction only (no rollout during training — that's part of why it drifts, and we want it to drift).
- Loss: relative L2. Normalize inputs using training-set mean/std, saved to disk.
- Adam, ~100 epochs, cosine decay. Should take a couple of minutes.

**Sanity check:** single-step relative L2 on test should be well under 1%. If it isn't, the surrogate is too weak and everything downstream is measuring the wrong thing.

### 4.3 Rollout + activation capture (`rollout.py`)

- Roll out 200+ steps autoregressively from each test initial condition.
- At each step, record ground truth from the solver alongside the prediction.
- **Error metric:** relative L2 error `||û - u|| / ||u||` at each step.
- **Failure step** `t_fail`: first step where relative L2 exceeds `0.10`.
- Capture hidden activations with a PyTorch forward hook on the **output of Fourier layer 3** (second to last). Don't fork the model class — use hooks, so the harness can be pointed at other models later. This matters for the "reusable infrastructure" claim.

### 4.4 Signals (`signals.py`)

**Internal signals** (computed from captured activations `A` of shape `[width, N_x]` at each step):

| Signal | Definition |
|---|---|
| `act_norm` | Mean L2 norm across channels of `A` |
| `eff_rank` | SVD of `A` → normalize singular values to sum 1 → `exp(entropy)` of that distribution |
| `train_dist` | Cosine distance between spatially-pooled `A` and the mean pooled activation over the training set |

**Output-side baselines** (need only the prediction, no internals, no ground truth):

| Signal | Definition |
|---|---|
| `step_change` | `\|\|û(t+1) - û(t)\|\| / \|\|û(t)\|\|` |
| `spectral_drift` | Fraction of energy in the top third of wavenumbers, vs. the training-set distribution of that fraction |

**Trivial floor:** a fixed step count (predict failure at step `k` always, `k` chosen on calibration data). Any signal that can't beat this is worthless.

### 4.5 Lead time (`leadtime.py`) — the part to get right

Each signal fires when it crosses a threshold. Thresholds must be **calibrated to a matched false-alarm rate**, otherwise the comparison is rigged — a trigger-happy signal always "warns earlier."

Procedure:
1. On the **calibration** trajectories, find each signal's threshold such that it fires on 5% of steps that occur well before `t_fail` (define "well before" as before `0.5 · t_fail`). This equalizes false-alarm rate across signals.
2. On the **test** trajectories, find `t_signal` = first step the signal crosses its calibrated threshold.
3. **Lead time** = `t_fail - t_signal`, in steps. Negative means the signal fired too late to be useful.
4. Report median and interquartile range across all test trajectories × 3 seeds.

A signal with a great median but an IQR straddling zero is not a usable detector. Say so.

### 4.6 The one figure (`plots.py`)

Horizontal box plot or dot plot: lead time in steps on the x-axis, one row per signal (3 internal, 2 output-side, 1 trivial floor), with a vertical line at zero. That single figure should let someone answer the research question in five seconds.

---

## 5. Definition of done

- [ ] `bash run_all.sh` on a clean clone regenerates data, trains, rolls out, computes lead times, and writes the figure — in under 20 minutes, no manual steps, no downloads.
- [ ] Every number in the write-up traces to a file in `results/`.
- [ ] 3 seeds throughout, spread reported everywhere a median is reported.
- [ ] `README.md` states the headline result in the first three lines, including if the result is null.
- [ ] Activation capture works via hooks and is documented as pointable at other models.
- [ ] `NEXT_STEPS.md` lists what a second month would test.

---

## 6. Known failure modes and how to handle them

**The rollout doesn't fail within 200 steps.** Most likely snag. Burgers' with moderate viscosity is well-behaved and a good FNO can track it for a long time. Fixes, in order of preference: lower `ν` toward `1e-4` to sharpen the dynamics; roll out longer; shrink the model (`width = 32`, `modes = 8`); train on fewer trajectories. Do *not* add noise to the rollout to force failure — that changes what you're measuring.

**The rollout fails immediately.** Surrogate undertrained or the coarse step `Δt` is too large. Check single-step error first.

**Effective rank is flat and uninformative.** Try it at a different layer, and also try it on the batch-of-timesteps matrix rather than the single-step channel matrix. If it stays flat, that's a finding — report it.

**All signals fire at essentially the same step.** Likely means the failure is abrupt rather than gradual, so there's nothing to warn about. Lower the failure threshold from 0.10 to 0.02 to catch the onset earlier, and say in the write-up that lead time depends on how failure is defined.

**Everything works and internal signals win big.** Be suspicious. Check for leakage: are your thresholds calibrated on data that includes the test trajectories? Is `train_dist` implicitly seeing test statistics? Re-verify the false-alarm matching before believing it.

---

## 7. Suggested order of work

| Day | Goal | Done when |
|---|---|---|
| 1 | Solver + dataset generation | Energy decays, shocks form, data saved with seeds |
| 2 | FNO + training loop | Single-step test error < 1% |
| 3 | Rollout + error curve | You can plot error vs. step and see clear failure |
| 4 | Activation hooks + internal signals | Three signal curves plotted alongside the error curve |
| 5 | Output-side baselines + calibration | All thresholds calibrated at matched false-alarm rate |
| 6 | Lead-time computation + the figure | One plot answers the question |
| 7 | README, write-up, cleanup, `run_all.sh` | Clean clone reproduces in one command |

If you fall behind, cut day 4's third signal before cutting day 5. The baseline comparison is the part that makes this research.

---

## 8. Writing up

Keep the write-up to roughly one page. Structure:

1. The question, in two sentences.
2. Setup, in one paragraph.
3. The figure.
4. What it shows — including, plainly, if internal signals did not beat output-side baselines.
5. The single most important thing this setup can't tell us yet.

Do not hedge a null result into sounding positive. Stating a clean negative clearly is a stronger signal about how you work than a marginal positive would be.
