#!/usr/bin/env bash
# Real-mesh physics-vs-DCO scaling sweep. Small configs calibrate the control step;
# the paper-scale config reuses step 1.2 and gradient checkpointing (memory).
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.
nix develop --command python3 scripts/physics_scaling_sweep.py --configs 0,1,2 --steps 0.6,1.2,2.4 \
    --out physics_scaling_sweep_real_012.csv > results/logs/real012.progress 2> results/logs/real012.log
nix develop --command python3 scripts/physics_scaling_sweep.py --configs 4 --steps 1.2 --checkpoint \
    --out physics_scaling_sweep_real_4.csv > results/logs/real4.progress 2> results/logs/real4.log
