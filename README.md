# PDE Early-Warning Harness

Do a neural surrogate's internal activations warn that a rollout is about to fail, earlier than simple checks on its output?
**Headline result:** not yet available. The setup is frozen (see [FREEZE.md](FREEZE.md)) and the test split has not been scored.
The full plan, in plain English, is in [PROJECT_PLAN.md](PROJECT_PLAN.md).

## Reproduce everything

```bash
pip install -r requirements.txt
bash run_all.sh
```

That single command generates the data, checks the simulator, trains three models, runs the rollouts, tunes the alarms, scores the test split and draws the figure. It takes about 8 minutes on an RTX 4060 laptop GPU. It installs the CUDA 12.6 build of PyTorch; the project was built with Python 3.13.

Every check stops the run if it fails, so a result only ever comes from a setup that passed its gates. `bash run_all.sh configs/fast.yaml` runs a small version of the pipeline. Its small model is too weak to pass the Day 3 gate, so that run stops there by design.

## What each step does

| Day | Script | What it does | Output |
| --- | --- | --- | --- |
| 1 | `src/check_solver.py` | Checks the simulator can be trusted: energy never rises, and 256 points match 1024 | `results/day1_solver_checks.png` |
| 1 | `src/solver.py` | Generates the train, calibration, dev and test splits | `data/*.npz` |
| 2 | `src/train.py` | Trains one model per seed; checks single-step error on dev | `results/models/`, `results/day2_training.png` |
| 3–4 | `src/rollout.py` | Learns what "normal" looks like, then runs every model on its own predictions and measures the errors and all five warning signals | `results/rollouts/` |
| 3 | `src/check_rollout.py` | Setup gate on dev: do rollouts fail, late enough, for the right reason? | `results/day3_rollouts.png` |
| 4 | `src/check_signals.py` | What each signal does as dev rollouts approach failure | `results/day4_signals.png` |
| 5–6 | `src/leadtime.py` | Tunes every alarm on calibration, scores the test split, gives the verdict | `results/summary.json` |
| 6 | `src/plots.py` | The headline figure | `results/leadtime.png` |

Each script takes `--config configs/full.yaml` (the default).

## The data

| Split | Trajectories | Used for |
| --- | --- | --- |
| train | 1000 | Training the models; the first 200 also define "normal" signal levels |
| calibration | 200 | Setting alarm thresholds |
| dev | 100 | Setup checks on Days 2–4 |
| test | 100 | Scored once, in the frozen run |

Everything lands in `data/`, which git ignores because it's about 290 MB. Each `.npz` file contains:

| Key | Shape | Meaning |
| --- | --- | --- |
| `u` | `[n_trajectories, 201, 256]` | The wave at each saved step. Step 0 is the starting wave; steps are 0.005 time units apart |
| `ids` | `[n_trajectories]` | Trajectory IDs, unique across all splits |
| `seed` | number | Random seed used to make the starting waves |
| `dt` | number | Time between saved steps |
| `config` | text | Every setting used to generate the file |

```python
import numpy as np

data = np.load("data/dev.npz")
u = data["u"]        # u[trajectory, step, point]
print(u.shape)       # (100, 201, 256)
```
