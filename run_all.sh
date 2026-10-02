#!/usr/bin/env bash
# Reproduce everything from scratch: data, checks, models, rollouts, lead times, figures.
#
#   bash run_all.sh                     # the full run on the GPU, about 12 minutes
#   bash run_all.sh configs/fast.yaml   # a small version, to check the pipeline works
#
# Every check stops the run if it fails, so a result only ever comes from a valid setup.
set -euo pipefail

CONFIG="${1:-configs/full.yaml}"

python src/check_solver.py  --config "$CONFIG"   # Day 1: can the simulator be trusted?
python src/solver.py        --config "$CONFIG"   # Day 1: all splits, plus the three shift settings
python src/train.py         --config "$CONFIG"   # Day 2: one model per seed, checked on dev
python src/rollout.py       --config "$CONFIG"   # Days 3-4: rollouts and warning signals, every split and setting
python src/check_rollout.py --config "$CONFIG"   # Day 3: setup gate, on dev
python src/check_signals.py --config "$CONFIG"   # Day 4: signals around failure, on dev
python src/leadtime.py      --config "$CONFIG"   # Days 5-6: tune on calibration, score every setting, raw CSVs
python src/plots.py         --config "$CONFIG"   # Day 6: the six result figures
