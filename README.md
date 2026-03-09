# Chamfer Distance Optimization for Physics-Based Shape Morphing

Coupled Chamfer distance optimization with differentiable MPM physics simulation for shape morphing.

## Overview

This repository implements a **coupled reverse-smooth Chamfer distance** approach that integrates geometric loss guidance into a differentiable Material Point Method (MPM) physics simulation. The method optimizes deformation gradient fields to morph a source shape into a target shape while maintaining physical plausibility.

### Key Results (s→t Chamfer Distance at ep39)

| Shape  | Physics-only | Ours   | Improvement |
|--------|-------------|--------|-------------|
| Bunny  | 0.181       | 0.157  | -13.3%      |
| Duck   | 0.184       | 0.173  | -6.0%       |
| Cow    | 0.178       | 0.167  | -6.2%       |
| Teapot | 0.180       | 0.148  | -17.8%      |

## Project Structure

```
├── DiffMPMLib3D/          # C++ differentiable MPM simulation core
│   ├── CompGraph.cpp/h    # Computation graph with Chamfer integration
│   ├── E2ESession.h       # End-to-end session management
│   └── ForwardSimulation.cpp
├── bind/                  # Python bindings (pybind11)
├── utils/
│   └── training_loop.py   # Main training loop with coupled optimization
├── run.py                 # Entry point
├── loss.py                # Loss functions
├── configs/               # Experiment configurations
│   ├── pairwise_coupled_v3/    # Coupled chamfer configs
│   ├── pairwise_physics_only/  # Physics-only baselines
│   ├── pairwise_rev_smooth/    # Reverse-smooth configs
│   ├── dcd_ppc4/               # DCO/DCD baseline configs
│   ├── ablation_*.yaml         # Ablation study configs
│   └── teaser_*.yaml           # Teaser figure configs
├── tools/                 # Experiment & visualization scripts
│   ├── plot_cd_loss.py          # CD loss convergence plots
│   ├── render_4method_pbr.py    # 4-method comparison (PBR)
│   ├── render_teaser_4row_pbr.py # Teaser figure rendering
│   ├── compute_hausdorff_f1.py  # Hausdorff & F1 metrics
│   ├── measure_ablation.py      # Ablation study measurement
│   └── run_*.py                 # Experiment runners
├── assets/                # 3D mesh files (.obj)
├── figs/                  # Generated figures
├── paper2.md              # Main paper draft
└── supplementary.md       # Supplementary material
```

## Method

### Coupled Reverse-Smooth Chamfer Optimization

1. **Physics simulation** (DiffMPM): Neo-Hookean elasticity with differentiable MPM
2. **Chamfer guidance**: Two-sided Chamfer distance between simulated particles and target mesh
3. **Coupled schedule**: Chamfer loss weight ramps up as physics converges
   - Episodes 0–14: Physics-only (weight = 0)
   - Episodes 15–39: Linearly increasing Chamfer weight (0 → 1.0)
4. **Physics decay**: Gradually reduce physics constraints to allow geometric refinement

### Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Grid resolution | 64³ |
| Material model | Neo-Hookean |
| Young's modulus | 1000 Pa |
| Poisson's ratio | 0.3 |
| Particles per cell | 4 (PPC=4) |
| Timesteps | 25 per episode |
| dt | 5×10⁻⁴ |

## Usage

### Environment Setup

```bash
conda activate diffmpm_v2.3.0
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
```

### Running Experiments

```bash
# Single shape morphing
python run.py -c configs/sphere_to_bunny_rev_smooth.yaml

# Pairwise evaluation (all shapes)
python tools/run_pairwise_all.py

# DCO/DCD baselines
python tools/run_dco_dcd.py

# Ablation studies
python tools/measure_ablation.py
```

### Visualization

```bash
# CD loss convergence plot
python tools/plot_cd_loss.py

# 4-method comparison (PBR rendering)
python tools/render_4method_pbr.py

# Teaser trajectory figure
python tools/render_teaser_4row_pbr.py

# Compute Hausdorff distance & F1 score
python tools/compute_hausdorff_f1.py
```

## Baselines

- **Physics-only**: Pure MPM simulation without geometric guidance
- **DCO** (Differentiable Chamfer Optimization): Direct position optimization via Chamfer
- **DCD** (DCO + Density Control): DCO with density-aware corrections

## Citation

If you use this code, please cite our paper (ECCV 2026 submission).
