#!/usr/bin/env python3
"""Generate configs + runner for pairwise coupled-rev experiments."""
import os, yaml
from pathlib import Path
from itertools import permutations

SHAPES = ['bunny', 'bob', 'cow', 'spot', 'dragon', 'armadilo']
MESH_MAP = {s: f'assets/{s}.obj' for s in SHAPES}

OUTPUT_ROOT = Path('output/pairwise_coupled')
CONFIG_DIR = Path('configs/pairwise_coupled')
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATE = {
    'simulation': {
        'grid_dx': 1,
        'points_per_cell_cuberoot': 4,
        'grid_min_point': [-16.0, -16.0, -16.0],
        'grid_max_point': [16.0, 16.0, 16.0],
        'lam': 38888.89,
        'mu': 58333.3,
        'density': 75.0,
        'dt': 0.00833333333,
        'drag': 0.5,
        'external_force': [0.0, 0.0, 0.0],
        'smoothing_factor': 0.955,
    },
    'optimization': {
        'num_animations': 40,
        'num_timesteps': 10,
        'control_stride': 1,
        'num_passes': 3,
        'max_gd_iters': 1,
        'max_ls_iters': 5,
        'initial_alpha': 0.01,
        'adaptive_alpha_enabled': True,
        'adaptive_alpha_target_norm': 2500.0,
        'adaptive_alpha_min_scale': 0.1,
        'gd_tol': 0.0,
        'use_session_mode': True,
        'use_pcgrad': False,
        'w_render_base': 0.0,
        'j_barrier_weight': 0.0,
        'j_barrier_target': 0.8,
        'render_adam_enabled': False,
        # Native C++ Chamfer with coupled rev schedule
        'w_chamfer': 10.0,
        'chamfer_start_ep': 5,
        'chamfer_ramp_ep': 10,
        'chamfer_rev_weight': 0.3,
        'chamfer_rev_mode': 'coupled',    # rev = rev_base × physics_weight
        # Physics weight schedule
        'physics_weight_start_ep': 15,
        'physics_weight_ramp_ep': 5,
        'physics_weight_final': 0.3,
        'smoothing_start_ep': 15,
        'smoothing_value': 0.7,
        'chamfer_in_callback': False,
        'loss': {
            'enabled': True,
            'w_alpha': 0.0, 'w_depth': 0.0, 'w_photo': 0.0, 'w_edge': 0.0,
            'w_cov_align': 0.0, 'w_cov_reg': 0.0, 'w_det_barrier': 0.0,
            'schedule': 'constant',
        },
    },
    'upsample': {
        'use_simple_pipeline': True,
        'render_loss_weight': 1.0,
        'debug': {'verbose': False},
        'covariance': {
            'use_curvature_for_target': True,
            'sigma_isotropic': 0.038, 'sigma0': 0.25, 'k_F': 32,
            'use_multiscale_F': True, 'k_F_coarse': 64, 'k_F_fine': 16,
            'multiscale_blend_mode': 'adaptive',
            'enable_subdivision': True, 'subdivision_target': 100000,
            'subdivision_jitter': 0.15, 'sv_min': 0.60, 'sv_max': 1.30,
        },
    },
    'camera': {
        'width': 640, 'height': 360,
        'fx': 237.5, 'fy': 237.5, 'cx': 320.0, 'cy': 180.0,
        'znear': 0.01, 'zfar': 100.0,
        'lookat': {'eye': [20.0, -25.0, 12.5], 'target': [0.0, 0.0, 0.0], 'up': [0.0, 0.0, 1.0]},
    },
    'cameras': [{'eye': [-25.0, 20.0, 12.5], 'target': [0.0, 0.0, 0.0], 'up': [0.0, 0.0, 1.0]}],
    'render': {
        'num_frames': 1, 'schedule': 'uniform', 'bg': [1.0, 1.0, 1.0],
        'training_resolution_scale': 0.5, 'particle_color': [0.27, 0.51, 0.71],
        'surface_mask_ratio': 0.20, 'surface_mask_mode': 'last',
    },
}


def main():
    configs = []
    for src, tgt in permutations(SHAPES, 2):
        name = f'{src}_to_{tgt}'
        cfg = dict(TEMPLATE)
        cfg['input_mesh_path'] = MESH_MAP[src]
        cfg['target_mesh_path'] = MESH_MAP[tgt]
        cfg['output_dir'] = str(OUTPUT_ROOT / name) + '/'

        config_path = CONFIG_DIR / f'{name}.yaml'
        with open(config_path, 'w') as f:
            # Add header comment
            f.write(f'# Pairwise coupled-rev: {src} -> {tgt}\n')
            f.write(f'# rev_mode=coupled, rev_base=0.3, physics_final=0.3\n\n')
            f.write(f'input_mesh_path: "{MESH_MAP[src]}"\n')
            f.write(f'target_mesh_path: "{MESH_MAP[tgt]}"\n')
            f.write(f'output_dir: "{OUTPUT_ROOT / name}/"\n\n')
            yaml.dump({k: v for k, v in TEMPLATE.items()}, f, default_flow_style=False, sort_keys=False)

        configs.append((name, str(config_path)))
        print(f'  {config_path}')

    # Generate runner script
    runner_path = Path('tools/run_pairwise.sh')
    with open(runner_path, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('# Run all pairwise coupled-rev experiments\n')
        f.write('# Usage: bash tools/run_pairwise.sh\n\n')
        f.write('set -e\n')
        f.write('export LD_LIBRARY_PATH=/home/chayo/anaconda3/envs/diffmpm_v2.3.0/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH\n\n')
        for name, cfg_path in configs:
            f.write(f'echo "\\n===== {name} ====="\n')
            f.write(f'conda run -n diffmpm_v2.3.0 python run.py -c {cfg_path} || echo "FAILED: {name}"\n\n')

    os.chmod(runner_path, 0o755)
    print(f'\nGenerated {len(configs)} configs in {CONFIG_DIR}/')
    print(f'Runner script: {runner_path}')
    print(f'\nTo run: bash {runner_path}')


if __name__ == '__main__':
    main()
