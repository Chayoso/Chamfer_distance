#!/usr/bin/env python3
"""Compute Hausdorff distance and F1-score for Physics-only vs Ours (sphere source).
Target particles are generated via C++ voxelization (same as training).
"""
import sys
sys.path.insert(0, '/home/chayo/Desktop/Shape-morphing-binder')

import torch
import numpy as np
from pathlib import Path
from utils.physics_utils import build_opt_input, initialize_point_clouds, extract_target_point_cloud

WORKSPACE = Path("/home/chayo/Desktop/Shape-morphing-binder")
SHAPES = ['bunny', 'bob', 'spot', 'teapot']
TAU = 0.2  # 1/5 of grid_dx=1 (one-fifth grid cell spacing)

PHYS_DIR = "pairwise_physics_only/sphere_to_{shape}"
OURS_DIR = "sphere_to_{shape}_rev_smooth"

# Matching Section 5 setup
SIM_CFG = {
    'grid_dx': 1,
    'points_per_cell_cuberoot': 3,
    'grid_min_point': [-16.0, -16.0, -16.0],
    'grid_max_point': [16.0, 16.0, 16.0],
    'lam': 38888.89, 'mu': 58333.3, 'density': 75.0,
    'dt': 0.00833333333, 'drag': 0.5,
    'external_force': [0.0, 0.0, 0.0],
    'smoothing_factor': 0.955,
}
OPT_CFG = {
    'num_animations': 1, 'num_timesteps': 1, 'control_stride': 1,
    'num_passes': 1, 'max_gd_iters': 0, 'max_ls_iters': 0,
    'initial_alpha': 0.01, 'gd_tol': 0.0,
    'adaptive_alpha_enabled': False,
    'adaptive_alpha_target_norm': 2500.0,
    'adaptive_alpha_min_scale': 0.1,
    'use_session_mode': False,
}


def get_target_particles(shape):
    """Generate target particles via C++ voxelization."""
    cfg = {
        'input_mesh_path': str(WORKSPACE / 'assets/isosphere.obj'),
        'target_mesh_path': str(WORKSPACE / f'assets/{shape}.obj'),
        'simulation': SIM_CFG,
        'optimization': OPT_CFG,
    }
    opt = build_opt_input(cfg)
    _, target_pc = initialize_point_clouds(opt)
    x_tgt, _ = extract_target_point_cloud(target_pc)
    if isinstance(x_tgt, torch.Tensor):
        return x_tgt.float().numpy()
    return np.asarray(x_tgt, dtype=np.float32)


def load_particles(path):
    data = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(data, torch.Tensor):
        return data.float().numpy()
    raise ValueError(f"Unexpected format: {type(data)}")


def nn_dists(src, tgt):
    chunk = 4096
    dists = []
    tgt_t = torch.from_numpy(tgt).float()
    for i in range(0, len(src), chunk):
        s = torch.from_numpy(src[i:i+chunk]).float()
        d = torch.cdist(s, tgt_t)
        dists.append(d.min(dim=1).values.numpy())
    return np.concatenate(dists)


def hausdorff(src, tgt):
    d_fwd = nn_dists(src, tgt)
    d_rev = nn_dists(tgt, src)
    return float(max(d_fwd.max(), d_rev.max()))


def f1_score(src, tgt, tau):
    d_fwd = nn_dists(src, tgt)
    d_rev = nn_dists(tgt, src)
    precision = float((d_fwd < tau).mean())
    recall = float((d_rev < tau).mean())
    if precision + recall < 1e-10:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main():
    print(f"{'Shape':<8} {'Method':<10} {'Hausdorff':>10} {'F1':>10}")
    print("-" * 42)

    results = {}
    for shape in SHAPES:
        tgt = get_target_particles(shape)
        print(f"[{shape}] target: {len(tgt)} pts")

        results[shape] = {}
        for method, tmpl in [('Physics', PHYS_DIR), ('Ours', OURS_DIR)]:
            d = tmpl.format(shape=shape)
            path = WORKSPACE / "output" / d / "ep039" / "ep039_particles.pt"
            pts = load_particles(path)

            h = hausdorff(pts, tgt)
            f1 = f1_score(pts, tgt, TAU)
            results[shape][method] = {'h': h, 'f1': f1}
            print(f"  {method:<10} {h:>10.4f} {f1:>10.4f}  ({len(pts)} pts)")

    print("\n=== LaTeX rows ===")
    for shape in SHAPES:
        r = results[shape]
        ph, ou = r['Physics'], r['Ours']
        h_p = f"\\textbf{{{ph['h']:.3f}}}" if ph['h'] < ou['h'] else f"{ph['h']:.3f}"
        h_o = f"\\textbf{{{ou['h']:.3f}}}" if ou['h'] < ph['h'] else f"{ou['h']:.3f}"
        f_p = f"\\textbf{{{ph['f1']:.3f}}}" if ph['f1'] > ou['f1'] else f"{ph['f1']:.3f}"
        f_o = f"\\textbf{{{ou['f1']:.3f}}}" if ou['f1'] > ph['f1'] else f"{ou['f1']:.3f}"
        print(f"{shape.capitalize():<8} & {h_p} & {h_o} & {f_p} & {f_o} \\\\")


if __name__ == '__main__':
    main()
