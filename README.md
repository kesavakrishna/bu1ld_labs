# PDE Early-Warning Harness

Do a neural surrogate's internal activations warn that a rollout is about to fail, earlier than simple checks on its output, including when the physics shifts away from what it was trained on?
**Headline result:** not yet available. The evaluation is frozen (see [FREEZE.md](FREEZE.md)) and nothing has been scored.
The full plan, in plain English, is in [PROJECT_PLAN.md](PROJECT_PLAN.md).

## Reproduce everything

```bash
pip install -r requirements.txt
bash run_all.sh
```

That single command generates the data, checks the simulator, trains three models, runs the rollouts, tunes the alarms, scores the control and three shift settings, writes the raw CSVs, and draws six figures. It takes about 12 minutes on an RTX 4060 laptop GPU. It installs the CUDA 12.6 build of PyTorch; the project was built with Python 3.13.

Every check stops the run if it fails, so a result only ever comes from a setup that passed its gates. `bash run_all.sh configs/fast.yaml` runs a small version of the pipeline. Its small model is too weak to pass the Day 3 gate, so that run stops there by design.

## What each step does

| Day | Script | What it does | Output |
| --- | --- | --- | --- |
| 1 | `src/check_solver.py` | Checks the simulator can be trusted: energy never rises, and 256 points match 1024 | `results/day1_solver_checks.png` |
| 1 | `src/solver.py` | Generates the train, calibration, dev and test splits, and the three shift settings | `data/*.npz` |
| 2 | `src/train.py` | Trains one model per seed; checks single-step error on dev | `results/models/`, `results/day2_training.png` |
| 3–4 | `src/rollout.py` | Learns what "normal" looks like, then runs every model on its own predictions in every split and setting, measuring errors and all six warning signals | `results/rollouts/` |
| 3 | `src/check_rollout.py` | Setup gate on dev: do rollouts fail, late enough, for the right reason? | `results/day3_rollouts.png` |
| 4 | `src/check_signals.py` | What each signal does as dev rollouts approach failure | `results/day4_signals.png` |
| 5–6 | `src/leadtime.py` | Tunes every alarm on calibration, scores every setting, gives the verdicts | `results/summary.json`, `results/leadtimes.csv`, `results/rollout_steps.csv.gz` |
| 6 | `src/plots.py` | The six result figures | `results/figures/` |

Each script takes `--config configs/full.yaml` (the default).

## The settings

The model is trained on one kind of physics, then tested on that same physics (the control) and on three shifts it never saw. Alarm thresholds are set on the original physics only, just as they would be in real use.

| Setting | What's different | Data file |
| --- | --- | --- |
| control | Nothing: same physics as training | `data/test.npz` |
| viscosity up | Twice the viscosity: smoother waves, shocks form later | `data/shift_viscosity_up.npz` |
| viscosity down | 70% of the viscosity: sharper shocks | `data/shift_viscosity_down.npz` |
| rougher waves | Starting waves with more fine detail: shocks form twice as early | `data/shift_rougher_waves.npz` |

## The data

| Split | Trajectories | Used for |
| --- | --- | --- |
| train | 1000 | Training the models; the first 200 also define "normal" signal levels |
| calibration | 200 | Setting alarm thresholds |
| dev | 100 | Setup checks on Days 2–4 |
| test, and each shift setting | 100 each | Scored once, in the frozen run |

Everything lands in `data/`, which git ignores because it's about 350 MB. Each `.npz` file contains:

| Key | Shape | Meaning |
| --- | --- | --- |
| `u` | `[n_trajectories, 201, 256]` | The wave at each saved step. Step 0 is the starting wave; steps are 0.005 time units apart |
| `ids` | `[n_trajectories]` | Trajectory IDs, unique across all files |
| `seed` | number | Random seed used to make the starting waves |
| `viscosity` | number | The physics the file was simulated with |
| `dt` | number | Time between saved steps |
| `config` | text | Every setting used to generate the file |

```python
import numpy as np

data = np.load("data/dev.npz")
u = data["u"]        # u[trajectory, step, point]
print(u.shape)       # (100, 201, 256)
```

## The results files

| File | One row per | Columns |
| --- | --- | --- |
| `results/leadtimes.csv` | setting, seed, rollout, alarm | threshold, failure step, alarm step, lead time, detected, false alarm |
| `results/rollout_steps.csv.gz` | setting, seed, rollout, step | relative error, raw error, true wave size, and the score of every signal |
