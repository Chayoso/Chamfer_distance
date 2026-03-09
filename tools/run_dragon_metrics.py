#!/usr/bin/env python3
"""Compute Hausdorff/F1 for dragon table (tab:dragon in paper2.md)."""
import sys
sys.path.insert(0, '/home/chayo/Desktop/Shape-morphing-binder')

import torch
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from utils.physics_utils import build_opt_input, initialize_point_clouds, extract_target_point_cloud

WORKSPACE = Path('/home/chayo/Desktop/Shape-morphing-binder')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_target(ppc):
    cfg = {
        'input_mesh_path': str(WORKSPACE / 'assets/isosphere.obj'),
        'target_mesh_path': str(WORKSPACE / 'assets/dragon.obj'),
        'simulation': {
            'grid_dx': 1, 'points_per_cell_cuberoot': ppc,
            'grid_min_point': [-16.0, -16.0, -16.0],
            'grid_max_point': [16.0, 16.0, 16.0],
            'lam': 38888.89, 'mu': 58333.3, 'density': 75.0,
            'dt': 0.00833333333, 'drag': 0.5,
            'external_force': [0.0, 0.0, 0.0], 'smoothing_factor': 0.955,
        },
        'optimization': {
            'num_animations': 1, 'num_timesteps': 1, 'control_stride': 1,
            'num_passes': 1, 'max_gd_iters': 0, 'max_ls_iters': 0,
            'initial_alpha': 0.01, 'gd_tol': 0.0,
            'adaptive_alpha_enabled': False,
            'adaptive_alpha_target_norm': 2500.0,
            'adaptive_alpha_min_scale': 0.1,
            'use_session_mode': False,
        },
    }
    opt = build_opt_input(cfg)
    ipc, tpc = initialize_point_clouds(opt)
    x_tgt, _ = extract_target_point_cloud(tpc)
    if isinstance(x_tgt, torch.Tensor):
        x_tgt = x_tgt.float().numpy()
    else:
        x_tgt = np.asarray(x_tgt, dtype=np.float32)
    x_src = np.array(ipc.get_positions(), dtype=np.float32)
    return x_src, x_tgt


def metrics(pts, tgt, tau=0.2):
    tree_tgt = cKDTree(tgt)
    tree_src = cKDTree(pts)
    d_fwd, _ = tree_tgt.query(pts)
    d_rev, _ = tree_src.query(tgt)
    s2t = float(np.sqrt(np.mean(d_fwd**2)))
    t2s = float(np.sqrt(np.mean(d_rev**2)))
    cd2 = float(np.sqrt(s2t**2 + t2s**2))
    h = float(max(d_fwd.max(), d_rev.max()))
    prec = float((d_fwd < tau).mean())
    rec = float((d_rev < tau).mean())
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return s2t, t2s, cd2, h, f1


def load(path):
    p = Path(path)
    if p.suffix == '.pt':
        return torch.load(str(p), map_location='cpu', weights_only=False).float().numpy()
    return np.load(str(p)).astype(np.float32)


def compute_density_weights(x_np, k=8):
    tree = cKDTree(x_np)
    dists, _ = tree.query(x_np, k=k + 1)
    avg_knn_dist = dists[:, 1:].mean(axis=1)
    weights = 1.0 / (avg_knn_dist + 1e-8)
    weights = weights * (len(x_np) / weights.sum())
    return weights.astype(np.float32)


def run_dco(src_np, tgt_np, mode='standard', num_eps=40, adam_steps=30, lr=0.01):
    x = torch.tensor(src_np, dtype=torch.float32, device=DEVICE, requires_grad=True)
    x_tgt_t = torch.tensor(tgt_np, dtype=torch.float32, device=DEVICE)
    tree_tgt = cKDTree(tgt_np)
    optimizer = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.999), eps=1e-4, amsgrad=True)

    weights_src = None
    weights_tgt = None
    if mode == 'dcd':
        w_tgt_np = compute_density_weights(tgt_np, k=8)
        weights_tgt = torch.tensor(w_tgt_np, dtype=torch.float32, device=DEVICE)

    for ep in range(num_eps):
        if mode == 'dcd' and ep % 5 == 0:
            x_np_tmp = x.detach().cpu().numpy().astype(np.float32)
            w_src_np = compute_density_weights(x_np_tmp, k=8)
            weights_src = torch.tensor(w_src_np, dtype=torch.float32, device=DEVICE)

        for step in range(adam_steps):
            optimizer.zero_grad()
            x_np = x.detach().cpu().numpy().astype(np.float32)

            _, nn_s2t = tree_tgt.query(x_np, k=1, workers=4)
            x_nn_s2t = x_tgt_t[torch.from_numpy(nn_s2t).long().to(DEVICE)]
            sq_s2t = (x - x_nn_s2t).pow(2).sum(dim=1)

            tree_src = cKDTree(x_np)
            _, nn_t2s = tree_src.query(tgt_np, k=1, workers=4)
            x_nn_t2s = x[torch.from_numpy(nn_t2s).long().to(DEVICE)]
            sq_t2s = (x_tgt_t - x_nn_t2s).pow(2).sum(dim=1)

            if mode == 'standard':
                loss = 0.5 * (sq_s2t.mean() + sq_t2s.mean())
            else:
                loss = 0.5 * ((weights_src * sq_s2t).mean() +
                              (weights_tgt * sq_t2s).mean())

            loss.backward()
            optimizer.step()

        if ep % 10 == 0:
            print(f'  ep{ep}: loss={loss.item():.4f}', flush=True)

    return x.detach().cpu().numpy()


def main():
    print('Generating targets...', flush=True)
    src3, tgt3 = make_target(3)
    _, tgt4 = make_target(4)
    print(f'PPC=3: src={len(src3)}, tgt={len(tgt3)} | PPC=4: tgt={len(tgt4)}',
          flush=True)

    print('\n--- Running DCO (PPC=3, 40ep x 30steps) ---', flush=True)
    dco_pts = run_dco(src3, tgt3, mode='standard')
    print('DCO done.', flush=True)

    print('\n--- Running DCD (PPC=3, 40ep x 30steps) ---', flush=True)
    dcd_pts = run_dco(src3, tgt3, mode='dcd')
    print('DCD done.', flush=True)

    rows = [
        ('DCO', dco_pts, tgt3),
        ('DCD', dcd_pts, tgt3),
        ('Physics-only', load(
            WORKSPACE / 'output/pairwise_physics_only/sphere_to_dragon/ep039/ep039_particles.pt'
        ), tgt3),
        ('Ours PPC=3', load(
            WORKSPACE / 'output/sphere_to_dragon_rev_smooth/ep039/ep039_particles.pt'
        ), tgt3),
        ('Ours PPC=4', load(
            WORKSPACE / 'output/sphere_to_dragon_native_chamfer/ep039/ep039_particles.pt'
        ), tgt4),
    ]

    hdr = f"{'Method':<16} {'#pts':>6} {'s->t':>8} {'t->s':>8} {'2-sided':>8} {'Hausdorff':>10} {'F1@0.2':>8}"
    print(f'\n{hdr}')
    print('=' * 68)
    for name, pts, tgt in rows:
        s2t, t2s, cd2, h, f1 = metrics(pts, tgt)
        print(f'{name:<16} {len(pts):>6} {s2t:>8.3f} {t2s:>8.3f} {cd2:>8.3f} '
              f'{h:>10.3f} {f1:>8.3f}')


if __name__ == '__main__':
    main()
