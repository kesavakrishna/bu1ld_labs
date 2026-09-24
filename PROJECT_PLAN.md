# PDE Early-Warning Project Plan

This is the single plan for the project. It combines the lab's original brief ([PROJECT_SPEC.md](PROJECT_SPEC.md)) with the fixes from our review. Where they disagree, this file wins.

> **Frozen on 2026-09-24.** Before the headline run, every setting and the exact scoring rules were frozen in [FREEZE.md](FREEZE.md), at git tag `freeze-v1`, as the lab requested. **Where this plan and FREEZE.md differ, FREEZE.md wins.** Also since the freeze: the split this plan's Day 2–4 results call "test" was renamed **dev**, and a fresh **test** split that nobody has looked at was drawn for the headline run.

- **Project:** week-1 deliverable for The BU1LD Fall 2026 Research Labs, thread 05 (*Dynamical Representation Phase Transitions for PDE Surrogates*)
- **Time:** about 7 days part-time, 12–18 hours total
- **Hardware:** RTX 4060 laptop GPU; the full run must finish in under 20 minutes

## Contents

1. [The project in plain English](#1-the-project-in-plain-english)
2. [The physics you need, and nothing more](#2-the-physics-you-need-and-nothing-more)
3. [Words you'll see](#3-words-youll-see)
4. [Ground rules](#4-ground-rules)
5. [The pipeline at a glance](#5-the-pipeline-at-a-glance)
6. [Day 1: Simulator and data](#6-day-1-simulator-and-data)
7. [Day 2: Model and training](#7-day-2-model-and-training)
8. [Day 3: Rollouts and the setup gate](#8-day-3-rollouts-and-the-setup-gate)
9. [Day 4: The six alarms](#9-day-4-the-six-alarms)
10. [Day 5: Tuning the alarms fairly](#10-day-5-tuning-the-alarms-fairly)
11. [Day 6: Scoring and the figure](#11-day-6-scoring-and-the-figure)
12. [Checks so we don't fool ourselves](#12-checks-so-we-dont-fool-ourselves)
13. [Running on this laptop](#13-running-on-this-laptop)
14. [Schedule](#14-schedule)
15. [Definition of done](#15-definition-of-done)
16. [Writing it up](#16-writing-it-up)
17. [Open decisions](#17-open-decisions)
18. [Appendix: What changed from the original spec](#appendix-what-changed-from-the-original-spec)

---

## 1. The project in plain English

**The situation.** Physics simulators are accurate but slow. A common shortcut is to train a neural network to imitate one. The network, called a *surrogate*, runs much faster.

**How it's used.** You give the network the current state, and it predicts the state a moment later. You feed that prediction back in to get the next one, and repeat for hundreds of steps. This is called a *rollout*.

**The problem.** Every prediction is slightly wrong, and the small errors pile up. A rollout looks fine for a while, then drifts, then falls apart. In real use the slow simulator isn't running alongside, so you can't see the error growing. You only see the network's output, which can look believable right up until it isn't.

**Our question.** Can we get an early warning that a rollout is about to fail, without knowing the right answer?

There are two kinds of warning sign to test:

- **Internal:** numbers inside the network. The idea is that the network "notices" it's in unfamiliar territory before its output looks wrong.
- **Output-side:** simple checks on the prediction itself, such as "is it suddenly changing a lot?"

**The real test.** Internal signs are only worth the extra effort if they warn *earlier* than the simple output checks. If the simple checks do just as well, that's the answer and we report it plainly. A clear "no" is a perfectly good result.

### The smoke alarm picture

It helps to think of the project as testing smoke alarms.

| In the project | In the analogy |
| --- | --- |
| A rollout | A house over time |
| The rollout failing | The house catching fire |
| A warning signal | A brand of smoke alarm |
| Threshold | How sensitive the alarm is set |
| False alarm | Beeping when there's no fire |
| Lead time | How long before the fire it beeped |

We want to know which alarm gives the most warning. But a super-sensitive alarm beeps constantly: it technically "warns earliest" while being useless. So before comparing, every alarm is tuned to the same false-alarm rate. Most of the careful work in this project is about keeping that comparison fair.

---

## 2. The physics you need, and nothing more

### What we simulate

Picture a ring with 256 evenly spaced points. Each point holds a number `u`. Plotted around the ring, the numbers form a wave, and the wave changes shape over time.

The rule for how it changes is **Burgers' equation**:

```
du/dt  +  u · du/dx  =  viscosity · d²u/dx²
          ─────────     ─────────────────────
          steepening          smoothing
```

You never solve this by hand. You only need to know what the two effects do.

**Steepening (`u · du/dx`).** Each point of the wave travels at a speed equal to its own height. High points move fast; low points move slowly. So fast parts catch up with the slow parts ahead of them, and the front of the wave gets steeper and steeper.

Traffic is a good picture. Fast cars catch up with slow cars, and smooth traffic bunches into a sudden jam. That sudden jump is called a **shock**.

**Smoothing (`viscosity · d²u/dx²`).** Viscosity acts like friction. It rounds off sharp corners, stops a shock from becoming a true vertical cliff, and slowly flattens the whole wave.

**Viscosity is the difficulty dial.** High viscosity smooths everything quickly, so nothing interesting happens. Low viscosity makes razor-thin shocks that are hard to simulate.

### A free correctness check: energy

"Energy" here means the average of `u²` around the ring. For this equation, energy can **only go down**: friction removes it and nothing adds it back. That's a mathematical fact. So if a simulation ever shows energy going *up*, the simulation code is broken.

### Why the number of grid points matters

The simulator draws the wave with 256 points. A shock is a very steep front. If it gets thinner than the gap between points, the simulator can't draw it, and it produces fake ripples. Then our "true answer" is wrong.

Shocks get thinner when viscosity is lower or the wave is taller. So on Day 1 we run the same waves on 256 points and on 1024 points, and check that they agree.

### How the simulator works, in one paragraph

Any wave shape can be built by adding together sine waves of different wiggliness. Broad sine waves are *low modes*; fine, wiggly ones are *high modes*. The FFT (fast Fourier transform) converts a wave into its sine-wave ingredients and back. The simulator works with the ingredients, because measuring "how steep is the wave here" is exact and easy that way. It moves forward in time with **RK4**, a standard, accurate stepping method. One safeguard, the **2/3 rule**, throws away the finest third of the ingredients to stop a known source of fake ripples.

---

## 3. Words you'll see

| Word | Meaning here |
| --- | --- |
| Trajectory | One simulation: a starting wave and how it changes over time |
| Simulator / solver | The slow, accurate program that computes the true wave |
| Surrogate | The fast neural network trained to imitate the simulator |
| FNO | Fourier Neural Operator, the network type we use. Like the simulator, it works with sine-wave ingredients |
| Step | One jump forward in time by the model (`Δt`). The simulator takes many tiny steps (`dt_solver`) per model step |
| Rollout | Feeding the model its own predictions, step after step |
| `u`, `û` | The true wave, and the model's prediction of it |
| Size of a wave, `‖u‖` | Square every point, add them up, take the square root (the "L2 norm") |
| Relative error | Size of the mistake divided by size of the true wave: `‖û − u‖ / ‖u‖`. 0.10 means 10% off |
| Failure step, `t_fail` | First step where relative error goes above 10% |
| Activations | The numbers inside the network while it runs |
| Hook | A PyTorch feature for reading activations without changing the model's code |
| Signal / alarm | A number computed at every step that might warn of failure |
| Threshold | The level a signal must cross to count as an alarm |
| Lead time | Failure step minus alarm step. Positive means it warned in time |
| Detection rate | Share of failures where the alarm went off beforehand |
| Calibration set | Trajectories used only to set thresholds |
| Test set | Trajectories used only for final scoring |
| Seed | A number that fixes randomness so a run can be repeated exactly |
| Median, IQR | The middle value, and the range covering the middle half of values |

---

## 4. Ground rules

### Scope

Keep it small. These are **out of scope** this week; if one is tempting, write it in `NEXT_STEPS.md` instead:

- More than one equation (Burgers' only)
- More than one model type (FNO only)
- 2D problems
- Sparse autoencoders, circuit-level interpretability, causal interventions
- Training anything large, or anything needing a paid GPU
- Fixing the drift. We detect it; we don't prevent it.

### Working rules

- **Settings:** every number the code uses lives in a YAML config file. Scripts take the config path with `argparse`. No Hydra or other experiment frameworks.
- **Seeds:** everything random is seeded, and the seed is saved with the data.
- **Honesty:** every number in the write-up traces to a file in `results/`. If internal signals don't beat the output checks, the README says so in its first lines.

### Project layout

```
bui1d_labs/
├── PROJECT_PLAN.md        this plan
├── PROJECT_SPEC.md        the lab's original brief, kept for reference
├── README.md              what this is, how to run it, headline result
├── requirements.txt       exact package versions
├── run_all.sh             one command reproduces everything        (Day 7)
├── configs/
│   ├── full.yaml          the real run, on the GPU
│   └── fast.yaml          small version for a quick check, on CPU
├── src/
│   ├── solver.py          simulator + dataset generation            (Day 1)
│   ├── check_solver.py    Day 1 checks and plots                    (Day 1)
│   ├── model.py           the FNO                                   (Day 2)
│   ├── train.py           training                                  (Day 2)
│   ├── rollout.py         rollouts + activation hook                (Days 3–4)
│   ├── signals.py         the alarms                                (Day 4)
│   ├── leadtime.py        threshold tuning + scoring                (Days 5–6)
│   └── plots.py           the final figure                          (Day 6)
├── results/               figures and numbers (only the final figure and summary.json are committed)
└── NEXT_STEPS.md          what we'd do with more time
```

Generated data goes in `data/`, which git ignores. It is about 290 MB, and the project folder is OneDrive-synced, so OneDrive will upload it again after each regeneration. Pause syncing during heavy runs if that gets annoying.

---

## 5. The pipeline at a glance

```
Day 1   Simulate waves           solver.py, check_solver.py
  ↓
Day 2   Train the model          model.py, train.py
  ↓
Day 3   Roll out + setup gate    rollout.py
  ↓
Day 4   Compute the alarms       rollout.py, signals.py
  ↓
Day 5   Tune alarms fairly       leadtime.py
  ↓
Day 6   Score + figure           leadtime.py, plots.py
  ↓
Day 7   Clean up + write up      run_all.sh, README.md
```

### Which data is used where

Keeping these separate is what makes the comparison fair.

| Data | Count | Used for | Never used for |
| --- | --- | --- | --- |
| Training | 1000 | Training the model; building the "normal" reference (Day 4) | Thresholds, scoring |
| Calibration | 200 | Setting alarm thresholds | Training, scoring |
| Dev | 100 | Setup checks on Days 2–4 (called "test" before the freeze) | Thresholds, scoring |
| Test | 100 | Final scoring only, once, in the frozen run | Anything else |

The test split comes from its own random seed, drawn at the freeze, so no earlier choice could have been influenced by it.

---

## 6. Day 1: Simulator and data

**Goal:** a simulator we can trust, and a saved dataset.

### What to build

`src/solver.py`:

- **Starting waves.** Each is a sum of a few sine waves with random sizes and positions. Finer sine waves are made smaller, so every wave starts smooth. Each wave is then rescaled so its peak height equals `amplitude` from the config.
- **Simulate all trajectories at once** as one array of shape `[n_trajectories, 256]`. Doing one at a time in a loop would take hours; this is the most important speed detail in the project.
- **Use PyTorch (`torch.fft`)**, so the same code runs on the GPU or the CPU.
- **Run every trajectory for the full rollout length**, training ones included. Otherwise the model never sees what late-stage waves look like, and gets confused late in a rollout for the wrong reason.
- **Save** train, calibration and test as separate `.npz` files, with the seed and all simulator settings stored inside.

`src/check_solver.py` runs the Day 1 checks and saves a figure.

### Day 1 checks

| Check | Passes when | Why |
| --- | --- | --- |
| Energy | Never goes up, on any trajectory | It can only go down for this equation; going up means a bug |
| Resolution | 256-point and 1024-point runs agree within 1% at every step | Otherwise shocks are too thin for the grid, and the "true answer" is wrong |
| Shocks form | Steepness rises sharply, then falls | Confirms we have the interesting behaviour |
| Change per step | Reported, not pass/fail. Aim for roughly 5–20% on early steps | A starting guide for the model's step size `Δt` |

**How the resolution check compares:** run identical starting waves on both grids, then compare values at the 256 points the grids share (every 4th point of the fine grid). That asks exactly what we care about: are the values the model will train on correct?

**If resolution fails:** raise viscosity or lower `amplitude` until it passes. Never lower viscosity past the point where it passes.

The check script also reports **how much energy is left at the end**. If the wave has nearly died out, a 10% relative error gets easy to hit late in a rollout just because the wave is small (see [Section 12](#12-checks-so-we-dont-fool-ourselves)).

### Settings chosen on Day 1

`configs/full.yaml` is the source of truth; this table records what Day 1 settled on and why.

| Setting | Value | Meaning | Why this value |
| --- | --- | --- | --- |
| `n_points` | 256 | Grid points around the ring | From the spec |
| `viscosity` | 0.001 | Smoothing strength | From the spec |
| `amplitude` | 0.5 | Peak height of each starting wave | At 1.0, shocks were too thin: 256 points were 4% off the 1024-point run |
| `dt_solver` | 0.001 | The simulator's tiny time step | Stable, and matches the 4×-finer reference run within 1% |
| `steps_per_save` | 5 | Tiny steps per saved step, so `Δt` = 0.005 | Started at 20 (`Δt` = 0.02), then lowered on Day 3: the bigger step made every rollout fail in a burst with no warning room |
| `n_saves` | 200 | Saved steps per trajectory (the rollout length) | From the spec |
| `n_modes`, `decay` | 5, 1.0 | Sine waves in each starting wave, and how fast finer ones shrink | Smooth starts that still form several shocks |
| Split | 1000 / 200 / 100 | Train / calibration / test | Calibration raised from 100 (see Appendix) |

The spec suggested about 100 tiny steps per saved step; 20 passes the same accuracy check and runs 5× faster.

### Day 1 results

From `python src/check_solver.py` (20 fresh trajectories) and `python src/solver.py`:

Numbers below are with the final `Δt` of 0.005.

| Result | Value |
| --- | --- |
| Energy never goes up | Pass |
| 256 vs 1024 grid points | Pass: worst difference 0.62%, while shocks form around step 75 |
| Change per step | About 1.3% on early steps, 1.1% overall |
| Shocks | Steepness peaks around step 66, about 10× the starting slope |
| Energy left at step 200 | Median 33.5% (range 15–59%) |
| Dataset | 1300 trajectories in 3.3 s on the GPU, 268 MB |

**Good for Day 3:** a third of the energy is still there at step 200, so the true wave is only about 1.7× smaller at the end. The "is failure just the wave shrinking?" worry in [Section 12](#12-checks-so-we-dont-fool-ourselves) is much smaller than it was at `Δt` = 0.02, where only 4.5% of the energy survived.

**Done when:** all checks pass and the three data files are saved. ✅ Done 2026-09-16.

---

## 7. Day 2: Model and training

**Goal:** a model that predicts one step ahead accurately.

**The model:** a 1D FNO with 4 Fourier layers, 64 channels wide, using the lowest 16 sine-wave modes, with GELU activations.

- **Input:** the wave at step `t`, plus each point's position as a second input channel.
- **Output:** the wave at step `t+1`.

**Training:**

- **Single steps only.** We never train on rollouts, because we *want* the model to drift when rolled out.
- **About 20 random (wave, next wave) pairs** from each training trajectory, spread across the whole trajectory.
- **Loss:** relative error. **Optimizer:** Adam with cosine learning-rate decay, 100 epochs to start.
- **Normalize inputs** with the training set's mean and standard deviation, and save those to disk.
- **Predict the next wave directly**, not the change. This is a config flag (`predict: direct | residual`) because it strongly affects when rollouts fail.

**Scaling is built into the model.** `FNO1d.forward` takes a real wave and returns a real wave, scaling inside itself. A rollout can therefore never forget to undo it, which is an easy bug that looks exactly like drift.

**Seeds:** 3 seeds. Each changes the model's starting weights and the order it sees training data. The dataset stays the same.

**Done when:** single-step relative error on test data is under 1%. If it isn't, the model is too weak and everything afterwards measures the wrong thing.

### Settings chosen on Day 2

| Setting | Value | Why |
| --- | --- | --- |
| `modes`, `width`, `n_layers` | 16, 64, 4 | From the spec |
| `predict` | `direct` | From the spec; a config flag because it changes when rollouts fail |
| `pairs_per_trajectory` | 20 | 20,000 training pairs from 1000 trajectories |
| `epochs` | 100 | From the spec |
| `batch_size` | 256 | Batches of 64 left the GPU idle: 7 s per epoch against 2 s at 256 |
| `learning_rate` | 0.002 | Raised slightly for the bigger batch. Between 0.001 and 0.004 it made little difference |

Model size: 549,569 learned numbers.

### Day 2 results

From `python src/train.py --config configs/full.yaml`:

| Result | Value |
| --- | --- |
| Single-step test error | 0.251%, 0.270%, 0.235% for seeds 0, 1, 2 — all pass the 1% gate |
| Training time | 110 s per seed, 5.5 minutes for all three |
| Seed agreement | Training curves almost on top of each other |
| Error by step | Worst at step 0 (about 1.6%), falling steadily to about 0.03% by step 175 |

**For Day 3:** the model is weakest early, while shocks are forming, and very accurate later once the wave is smooth. So most error enters a rollout in its first 25 steps. Whether that is enough to push a rollout past 10% error is exactly what the Day 3 gate tests.

**Done when:** single-step test error is under 1%. ✅ Done 2026-09-20.

---

## 8. Day 3: Rollouts and the setup gate

**Goal:** confirm that rollouts fail in a way worth studying.

**Rollouts:** from each test starting wave, run the model on its own predictions for 200 steps. At every step record the relative error against the true simulation, and the true wave's size `‖u‖`.

**Make this plot** for about 10 rollouts: true wave size, raw error size, and relative error, over time on the same axes. It shows whether "failure" is real (see [Section 12](#12-checks-so-we-dont-fool-ourselves)).

### The setup gate

Everything here must pass before any alarm work. Record the final settings in the config and README.

| Check | Passes when | Why |
| --- | --- | --- |
| Energy | Never increases | The simulator is correct (Day 1) |
| Resolution | Grids agree within 1% | The true answer is correct (Day 1) |
| Single-step accuracy | Test error under 1% | The model is good (Day 2) |
| Rollouts fail | At least 70% of test rollouts pass 10% error within 200 steps | Enough failures to study |
| Not too fast | Median failure step is 30 or later | Room to warn early |

**The last limit was changed from 40 to 30 on Day 3, after seeing results.** That is the kind of move this plan says to distrust, so here is the reasoning in full. The three seeds gave nearly identical failure-step distributions (10th percentile 15–17, median 38–47, 90th percentile 95–118); seed 1's median of 38 sat two steps under a limit that [Section 17](#17-open-decisions) had already flagged as a guess rather than a derived number. The criterion exists to confirm there is room to warn before failure, and a median of 38 provides that as surely as 46 does. It matters that this gate only decides whether the **setup** is usable: it does not touch the comparison between internal and output-side alarms, so relaxing it cannot tilt the result either way. Lead times are reported per seed regardless, so a reader can check this judgement.

**If too few rollouts fail, try these in order:**

1. Roll out for more steps.
2. Use a bigger model step `Δt`, as long as single-step error stays under 1%.
3. Lower viscosity, but only as far as the resolution check still passes.
4. Make the model weaker (width 32, modes 8, or fewer training trajectories). Last resort: it can break the 1% check, and the write-up must say it was done.

**Never** add random noise to rollouts to force failure. That changes what we're measuring.

**If rollouts fail too early, use a smaller `Δt`.** This is the case we actually hit, and it wasn't in the spec's list. Failing early is not the same problem as failing rarely: it means the model's mistakes all arrive in a burst rather than building up. A smaller `Δt` spreads the same physics over more steps, so the error grows gradually and there is room to warn. Check single-step accuracy too, but ours was already fine.

### Settings chosen on Day 3

| Setting | Value | Why |
| --- | --- | --- |
| `steps_per_save` | 5, so `Δt` = 0.005 | At `Δt` = 0.02 every rollout's error arrived in a 15-step burst during shock formation, then flatlined. Median failure step was 14, leaving no warning room |
| `n_saves` | 200 | At 400 steps the rollouts overflowed to infinity by step 300, which would break the alarm calculations |
| `failure_threshold` | 0.10 | Gives 95% of rollouts failing, with failure steps spread between about 22 and 70 |

**This replaces Day 1's "5–20% change per step" guide**, which now sits at about 1.5%. That guide was a starting heuristic; the gate is the real test, and it prefers the smaller step.

### Day 3 results

| Seed | Rollouts that fail | Median failure step | Failure steps, 25th–75th | Wave size at failure | Mistake growth |
| --- | --- | --- | --- | --- | --- |
| 0 | 95% | 46 | 22–70 | 97% | 1.6× |
| 1 | 91% | 38 | 20–68 | 98% | 1.5× |
| 2 | 85% | 47 | 23–81 | 96% | 1.7× |

Gate: **pass** on all three seeds.

What the rollouts do: error climbs steadily from about 0.4% to 10% over the first 30–50 steps, and the median test rollout ends around 30% error. Roughly a third of rollouts then blow up completely, reaching absurd values; 9 out of 100 overflow to infinity before step 200. Those had crossed the failure line long before, so `failure_steps` counts anything non-finite as failed.

**The shrinking worry is settled.** At the moment of failure the true wave still holds 96–98% of its step-20 size, while the raw mistake has grown 1.5–1.7×. Failure is the model getting worse, not the wave fading. The second failure definition in [Section 12](#12-checks-so-we-dont-fool-ourselves) is therefore a robustness check rather than a necessity.

**For Day 4:** about 9% of rollouts contain non-finite values near the end. Alarm scores must treat non-finite as "alarm fires" rather than crashing or silently dropping those steps.

**Done when:** the gate passes. ✅ Done 2026-09-20.

---

## 9. Day 4: The six alarms

**Goal:** compute every alarm at every rollout step.

### Reading the model's insides

Use a PyTorch **forward hook** on the output of Fourier layer 3 (second to last). A hook reads activations without editing the model's code, so the same harness can later be pointed at other models. The layer is a config setting (`hook_layer`).

Compute alarms as the rollout runs, then throw the activations away; keeping them all would take over 1 GB per seed.

At each step the captured activations `A` have shape `[64 channels, 256 points]`.

### The alarms

Every alarm is computed from the same thing: the wave fed into the model at step `t`, and what the model does with it.

| Alarm | Type | Plain question | Exact definition |
| --- | --- | --- | --- |
| `act_norm` | Internal | Are internal numbers getting bigger? | Average over the 64 channels of each channel's size (L2 norm) |
| `eff_rank` | Internal | How many independent patterns are active? | SVD of `A`; divide the singular values by their sum; take `exp(entropy)` of the result |
| `train_dist` | Internal | How unfamiliar is the internal state? | Each channel's mean and standard deviation across the 256 points (128 numbers); cosine distance to the average of those from normal runs at the same step |
| `step_change` | Output | Is the prediction jumping around? | `‖û(t+1) − û(t)‖ / ‖û(t)‖` |
| `spectral_drift` | Output | Is energy piling up in fine wiggles? | Share of the prediction's energy in the top third of sine-wave modes |
| clock | Baseline | None; it ignores the data | Always says "failure at step `k`", with `k` tuned on calibration data |

The **clock** is the floor. An alarm that can't beat a fixed guess is worthless.

### Comparing against "normal at the same moment"

Correct physics changes a lot over time: early on shocks form (fast change, lots of fine detail), later the wave calms down. A raw signal can't tell "the physics is dramatic right now" from "the model is failing." So each signal becomes a score:

1. **Build a "normal" reference.** Take 200 training trajectories. At every step, feed the *true* wave into the model for one step and compute all five signals. Record each signal's average `μ(t)` and spread `σ(t)` at each step.
2. **Score each rollout step:** `score(t) = |signal(t) − μ(t)| / σ(t)`

In words: "how many spreads away from normal-for-this-step is this?" The absolute value means we don't have to guess in advance whether a signal rises or falls when things break.

This needs no true answers at run time, because you always know which step you're on.

**Done when:** all six score curves are plotted next to the error curve for about 10 rollouts, and any signal that stays flat is noted.

### How it was built

- **Which step a signal belongs to.** One model call produces the prediction for step `s`, its activations and both output signals, all at the same moment. So every signal from that call is labelled step `s`, the same index as that prediction's error. Labelling by the input step instead would hand every alarm one free step of lead time.
- **`eff_rank` uses a shortcut.** The singular values of `A` are the square roots of the eigenvalues of the 64×64 matrix `A·Aᵀ`. That gives the same answer as a full SVD to about 0.003%, but runs 200× faster on the GPU (4 ms against 785 ms per step).
- **Non-finite readings score as infinite**, so a rollout that overflows always sets off the alarm.
- **Speed:** reference runs plus 900 scored rollouts, across three seeds, take 29 seconds.
- **Figure:** `check_signals.py` lines every failing rollout up at its own failure step and plots the median score with the middle half shaded, plus 10 individual rollouts. That's more informative than 10 rollouts alone.

### Day 4 results

Median score around failure, pooled over the 271 failing test rollouts from all three seeds. A score of 1 is one normal spread; pure noise gives a median of about 0.67.

| Signal | Type | 40 steps before | 20 before | 10 before | At failure |
| --- | --- | --- | --- | --- | --- |
| `act_norm` | Internal | 0.80 | 0.80 | 0.77 | 0.79 |
| `eff_rank` | Internal | 0.45 | 0.51 | 0.62 | 0.63 |
| `train_dist` | Internal | 0.64 | 0.68 | 0.69 | 0.68 |
| `step_change` | Output | 0.42 | 0.51 | 0.67 | 0.79 |
| `spectral_drift` | Output | 0.42 | 0.42 | 0.51 | 0.61 |

**All five signals are flat at the 10% failure line.** When a typical rollout crosses 10% error, it looks normal to every signal, internal and output-side alike. Its wave is wrong but plausible, which is exactly the situation the project set out to probe. The three internal signals barely move at all. The two output signals creep upward slightly and only spike after failure, as rollouts start to blow up.

**What this means for Days 5–6:** with thresholds tuned fairly, most alarms will probably fire after the 10% line or not at all. That would make lead times mostly negative, and the clock may well win. That would be a legitimate null result, and it gets reported as one. Two pre-registered checks bear on it directly: the failure-threshold sweep ([Section 12](#12-checks-so-we-dont-fool-ourselves)), since signals may do better against "blew up" than against "10% off"; and one retry of `eff_rank` computed over the last 10 steps.

**Done when:** score curves are plotted and flat signals noted. ✅ Done 2026-09-24.

---

## 10. Day 5: Tuning the alarms fairly

> **Frozen.** Sections 10 and 11 are implemented exactly in `src/leadtime.py` and specified exactly in [FREEZE.md](FREEZE.md) §7, which wins over the plain-English version here.

**Goal:** make every alarm equally prone to false alarms, using calibration data only.

### Definitions

- **Failure step `t_fail`:** first step where relative error goes above 0.10.
- **Rollout length `H`:** 200 steps, unless the gate changed it.
- **Safe window:** the steps before `0.5 × t_fail`. An alarm here is a false alarm.
- **Rollouts that never fail:** their safe window is the steps before `0.5 × H`. They stay in; dropping them would tune thresholds only on the fastest failures.

### The tuning rule

For each alarm:

1. On each calibration rollout, find the alarm's **highest score** inside the safe window.
2. Set the threshold at the **95th percentile** of those highest scores.

Result: 95% of calibration rollouts get no false alarm and 5% do, for every alarm.

**Why not "fire on 5% of safe steps"**, the original spec's rule? Alarms are scored by when they *first* go off. An alarm beeping at random on 5% of steps, over a 50-step safe window, has a 92% chance of beeping at least once. Nearly every alarm would go off early by accident, and they'd all look the same.

### Tuning the clock

The clock false-alarms on a rollout if `k` falls inside that rollout's safe window. Pick the **smallest `k` where at most 5% of calibration rollouts get a false alarm.**

**Done when:** thresholds are set, and on test data every alarm's false-alarm rate comes out near 5%. If one is far off, the tuning didn't carry over and that alarm's results can't be trusted.

---

## 11. Day 6: Scoring and the figure

**Goal:** answer the question, using test data only with thresholds frozen.

### Per rollout, per alarm

- **Alarm step `t_signal`:** first step the score passes the threshold.
- **If the alarm never fires:** count it as firing at the last step `H`. A silent alarm then counts as at least as bad as a late one, instead of disappearing from the results.
- **Lead time** = `t_fail − t_signal`. Positive means a useful warning.
- **Rollouts that never fail** have no lead time. Leave them out of lead time and detection rate, but report how many there were.

### Reported for each alarm

| Number | Meaning |
| --- | --- |
| Detection rate | Share of failing rollouts where the alarm fired before failure |
| Lead time | Median and IQR, all seeds together |
| Per-seed medians | Each seed's median separately, to see whether seeds agree |
| Test false-alarm rate | Should be near 5% |

An alarm with a good median but an IQR that crosses zero is not a usable detector. Say so.

### Head-to-head: the comparison that answers the question

For each internal alarm against each output alarm (6 pairs), on every failing test rollout:

```
difference = lead time (internal) − lead time (output)
```

Report the median difference, and the share of rollouts where internal wins, ties and loses. Compare every alarm against the clock the same way.

**The verdict:** internal signals win only if some internal alarm has a median difference above zero against **both** output alarms, in **every** seed. Anything less is a null result, stated plainly.

### The figure

One figure, two panels:

- **Left, lead time per alarm.** One row per alarm, grouped as internal / output / clock. A box of lead times, a dot for each seed's median, a vertical line at zero, and detection rate printed at the right.
- **Right, head-to-head.** One row per internal-vs-output pair. A box of lead-time differences, a line at zero, and the internal win share printed at the right.

Save `results/leadtime.png`, and every number in it to `results/summary.json`. Commit both.

Also re-score everything with the second failure definition from [Section 12](#12-checks-so-we-dont-fool-ourselves).

**Done when:** the figure exists and `summary.json` holds every number in it.

---

## 12. Checks so we don't fool ourselves

### Is "failure" just the wave shrinking?

Energy only goes down, so the true wave gets smaller over time. Relative error divides by the true wave's size. If the wave shrinks a lot, a mistake that stays the same size becomes a bigger *percentage*, and we'd call it "failure" when the model didn't actually get worse.

- **Look** at the Day 3 plot. If relative error crosses 10% mainly because the wave shrank while raw error stayed flat, this is happening.
- **Guard:** also score everything with a second definition: error divided by the *starting* wave size.

**Outcome:** the Day 3 check showed failure is *not* driven by shrinking (at failure the true wave was still 96–98% of its step-20 size). Under this plan's own cut rule, the second definition was dropped from the frozen protocol and moved to [NEXT_STEPS.md](NEXT_STEPS.md).

### Flat signals

If a score barely moves before failure, report it as flat. That's a finding, not a bug. All five signals were flat at the failure line on Day 4, and that is reported. The windowed `eff_rank` retry was moved to [NEXT_STEPS.md](NEXT_STEPS.md) rather than run: it could only give internal signals another chance, so dropping it can't tilt the result in their favour.

### Leakage

Each data split has one job ([Section 5](#5-the-pipeline-at-a-glance)). Add code checks that test trajectory IDs never appear in the reference or calibration sets. If internal signals win by a lot, re-check this before believing it.

### If there's time

- **Failure threshold sweep:** re-score with failure at 2%, 5%, 10% and 20%. Nearly free, since error curves are saved. **Frozen as a secondary analysis**, with 50% and 100% added after the Day 4 look (disclosed in FREEZE.md).
- **Threshold stability:** resample the calibration set about 200 times, recompute thresholds, and check the conclusion doesn't flip. Moved to [NEXT_STEPS.md](NEXT_STEPS.md).
- **Extra hook layers:** also record layers 1 and 2, to show the hook works elsewhere. Moved to [NEXT_STEPS.md](NEXT_STEPS.md).

---

## 13. Running on this laptop

Checked 2026-09-16: RTX 4060 Laptop GPU (8 GB), PyTorch 2.13.0 with CUDA 12.6, Python 3.13.1.

| | `full` config | `fast` config |
| --- | --- | --- |
| Hardware | RTX 4060 GPU | GPU if available, otherwise CPU |
| Trajectories (train / calibration / dev / test) | 1000 / 200 / 100 / 100 | 200 / 50 / 50 / 50 |
| Seeds | 3 | 1 |
| Purpose | The real results | Checking the pipeline runs; never used for results |
| Time | About 8 min | About 2 min |

**Timing for `full`:** data about 4 s, training three models about 6 min, rollouts and signals about 40 s, everything else a few seconds.

**The `fast` config stops at the Day 3 gate, by design.** A model trained on 200 trajectories is too weak: its rollouts fail around step 10, so the gate correctly refuses to go on. The scoring code was tested by running `leadtime.py` and `plots.py` directly on its rollouts.

**Practical notes:**

- **Data lives in `data/`**, ignored by git. It is about 290 MB, and OneDrive re-uploads it after every regeneration; pause syncing during heavy runs if that gets annoying.
- **CUDA builds of PyTorch** come from PyTorch's own package index, so `requirements.txt` includes that index URL.
- **For timed runs,** plug the laptop in and use a performance power mode.
- `run_all.sh` runs through Git Bash on Windows. Colab also works for anyone without a GPU.

---

## 14. Schedule

| Day | Build | Done when |
| --- | --- | --- |
| 1 | Simulator, data, Day 1 checks | Energy never rises; 256 and 1024 grids agree within 1%; shocks form; data saved with seed |
| 2 | FNO and training | Single-step test error under 1% |
| 3 | Rollouts and setup gate | Gate passes, or fallbacks tried and recorded; wave-size plot made |
| 4 | Hook, alarms, "normal" reference | Score curves plotted next to error for about 10 rollouts |
| 5 | Threshold tuning | Test false-alarm rate near 5% for every alarm |
| 6 | 3 seeds, scoring, figure | Figure exists; `summary.json` has every number |
| 7 | `run_all.sh`, README, write-up, `NEXT_STEPS.md` | A clean copy reproduces the figure with one command |

### If you fall behind, cut in this order

1. The "if there's time" checks
2. The `fast` config
3. `eff_rank`, the least well-motivated alarm
4. The second failure definition, but only if the Day 3 plot shows failure isn't caused by the wave shrinking

**Never cut** the output alarms, the clock, the fair tuning rule, or the head-to-head comparison. They're what make this research rather than a demo.

---

## 15. Definition of done

- [ ] `bash run_all.sh` on a clean copy regenerates data, trains, rolls out, scores and writes the figure in under 20 minutes, with no manual steps.
- [ ] Every number in the write-up traces to a file in `results/`.
- [ ] 3 seeds throughout, with spread reported wherever a median is.
- [ ] Detection rate and test false-alarm rate are reported for every alarm.
- [ ] `README.md` states the headline result in its first three lines, even if it's a null result.
- [ ] Activation capture uses hooks, and is documented as usable on other models.
- [ ] `NEXT_STEPS.md` lists what a second month would test.

---

## 16. Writing it up

Keep it to about one page:

1. The question, in two sentences.
2. The setup, in one paragraph.
3. The figure.
4. What it shows, including plainly if internal signals did not beat the output checks.
5. The single most important thing this setup can't tell us yet.

Don't dress up a null result as a positive one. A clearly stated "no" says more about how you work than a weak "maybe yes."

---

## 17. Open decisions

These choices were made to complete the plan. Each is easy to change.

1. **The lab accepts the new tuning rule.** If not, report the original spec's rule as well, beside ours; the gap between them is informative in itself.
2. **Seeds change the model, not the dataset.** Regenerating data per seed would triple data generation time.
3. **Gate numbers are starting values, not derived:** 1% resolution agreement, 70% of rollouts failing, median failure step of 40, and 5–20% change per step. Adjust if needed, and record why.
4. **"Normal at the same moment" scoring is the main result.** If the lab wants the spec's raw signals as the main result, swap the roles.
5. **200 calibration trajectories** instead of the spec's 100.

---

## Appendix: What changed from the original spec

| # | Change | Why |
| --- | --- | --- |
| 1 | Tune alarms so **5% of safe rollouts** get any false alarm (spec: 5% of safe *steps*) | Under the spec's rule almost every alarm goes off by accident early, so they all look the same |
| 2 | Tune the clock by the same rule | Under the spec's rule the clock can't be tuned fairly |
| 3 | Report **detection rate** beside lead time; an alarm that never fires counts as firing at the last step | Otherwise an alarm that usually stays silent looks great |
| 4 | Explicit rule for rollouts that never fail | The spec doesn't say, and it may be a large share of the data |
| 5 | Score every signal against **normal runs at the same step** | Real shocks make output change fast and look jagged; we want alarms for failure, not for normal physics |
| 6 | Add a **head-to-head** comparison on each rollout | That's what actually answers "does internal beat output?" |
| 7 | Check whether "failure" is really the wave shrinking; score under two failure definitions | The true wave loses energy, which inflates percentage error by itself |
| 8 | Training trajectories cover the **full rollout length** | Otherwise late in a rollout the model is confused just because it never saw late-stage waves |
| 9 | Check the simulator against a 1024-point grid; set wave height explicitly | Shocks can get thinner than a grid gap, making the "true answer" wrong. The spec's first fix (lower viscosity) makes this worse |
| 10 | A **setup gate** with pass/fail criteria before alarm work | The spec says both "make the model accurate" and "make it weaker"; the gate settles which wins |
| 11 | Two configs: `full` (GPU) and `fast` (CPU) | Three training runs don't fit in 20 minutes on a CPU |
| 12 | Commit `results/summary.json` | The spec ignores `results/` in git but also says every number must trace to it |
| 13 | Run on the laptop's RTX 4060; keep data outside OneDrive | Hardware is available locally; OneDrive re-uploads large files |
