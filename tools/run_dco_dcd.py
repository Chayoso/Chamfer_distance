#!/usr/bin/env python3
"""
DCD (Density-aware Chamfer Distance) experiment.

Compares standard two-sided DCO vs DCD as optimization objective.
DCD re-weights each point's contribution by inverse local density,
which should in principle reduce many-to-one collapse.

Key question: Does DCD fix the collapse without physics?

Usage:
    export LD_LIBRARY_PATH=.../torch/lib:$LD_LIBRARY_PATH
    conda run -n diffmpm_v2.3.0 python tools/run_dco_dcd.py -c configs/dcd_ppc4/sphere_to_bunny.yaml
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import json
import argparse
import time
from scipy.spatial import cKDTree
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_source_target(cfg_path):
    """Load source and target point clouds using the physics pipeline."""
    import yaml
    import diffmpm_bindings
    from utils.physics_utils import (build_opt_input, initialize_point_clouds,
                                      initialize_grids, extract_target_point_cloud)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    opt = build_opt_input({
        'input_mesh_path': cfg['input_mesh_path'],
        'target_mesh_path': cfg['target_mesh_path'],
        'simulation': cfg['simulation'],
        'optimization': cfg['optimization'],
    })
    input_pc, target_pc = initialize_point_clouds(opt)
    input_grid, target_grid = initialize_grids(opt)
    diffmpm_bindings.calculate_point_cloud_volumes(input_pc, input_grid)
    diffmpm_bindings.calculate_point_cloud_volumes(target_pc, target_grid)

    tgt_raw, _ = extract_target_point_cloud(target_pc)
    tgt_np = np.array(tgt_raw, dtype=np.float32)
    src_raw = input_pc.get_positions()
    src_np = np.array(src_raw, dtype=np.float32)

    ppc = cfg['simulation'].get('points_per_cell_cuberoot', 3)
    shape_name = Path(cfg['target_mesh_path']).stem
    return src_np, tgt_np, ppc, shape_name


def measure_chamfer(x_np, tgt_np):
    """Standard two-sided CD metric (same as paper)."""
    tree_tgt = cKDTree(tgt_np)
    d_s2t, _ = tree_tgt.query(x_np, k=1, workers=4)
    cd_s2t = float(np.sqrt((d_s2t**2).mean()))

    tree_src = cKDTree(x_np)
    d_t2s, _ = tree_src.query(tgt_np, k=1, workers=4)
    cd_t2s = float(np.sqrt((d_t2s**2).mean()))

    return cd_s2t, cd_t2s, 0.5 * (cd_s2t + cd_t2s)


def compute_density_weights(x_np, k=8):
    """Compute DCD-style density weights: w(p) = 1 / avg_knn_dist(p).
    Points in dense regions get lower weight, sparse regions get higher weight.
    Returns normalized weights that sum to N."""
    tree = cKDTree(x_np)
    dists, _ = tree.query(x_np, k=k+1)
    avg_knn_dist = dists[:, 1:].mean(axis=1)  # exclude self
    # Inverse density weighting
    weights = 1.0 / (avg_knn_dist + 1e-8)
    # Normalize so weights sum to N (same scale as uniform)
    weights = weights * (len(x_np) / weights.sum())
    return weights.astype(np.float32)


def run_dco(src_np, tgt_np, mode='standard', num_eps=40, adam_steps=30, lr=0.01,
            dcd_k=8, dcd_recompute_every=5):
    """Run DCO with standard CD or DCD loss.

    mode: 'standard' = uniform weighting, 'dcd' = density-aware weighting
    """
    x = torch.tensor(src_np, dtype=torch.float32, device=DEVICE, requires_grad=True)
    x_tgt = torch.tensor(tgt_np, dtype=torch.float32, device=DEVICE)
    tree_tgt = cKDTree(tgt_np)
    optimizer = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.999), eps=1e-4, amsgrad=True)

    results = []
    weights_src = None
    weights_tgt = None

    # Precompute target density weights (fixed)
    if mode == 'dcd':
        w_tgt_np = compute_density_weights(tgt_np, k=dcd_k)
        weights_tgt = torch.tensor(w_tgt_np, dtype=torch.float32, device=DEVICE)

    for ep in range(num_eps):
        # Recompute source density weights periodically
        if mode == 'dcd' and (ep % dcd_recompute_every == 0):
            x_np_tmp = x.detach().cpu().numpy().astype(np.float32)
            w_src_np = compute_density_weights(x_np_tmp, k=dcd_k)
            weights_src = torch.tensor(w_src_np, dtype=torch.float32, device=DEVICE)

        for step in range(adam_steps):
            optimizer.zero_grad()
            x_np = x.detach().cpu().numpy().astype(np.float32)

            # s->t: each source point to nearest target
            _, nn_s2t = tree_tgt.query(x_np, k=1, workers=4)
            x_nn_s2t = x_tgt[torch.from_numpy(nn_s2t).long().to(DEVICE)]
            sq_s2t = (x - x_nn_s2t).pow(2).sum(dim=1)  # [N_src]

            # t->s: each target point to nearest source
            tree_src = cKDTree(x_np)
            _, nn_t2s = tree_src.query(tgt_np, k=1, workers=4)
            x_nn_t2s = x[torch.from_numpy(nn_t2s).long().to(DEVICE)]
            sq_t2s = (x_tgt - x_nn_t2s).pow(2).sum(dim=1)  # [N_tgt]

            if mode == 'standard':
                loss = 0.5 * (sq_s2t.mean() + sq_t2s.mean())
            elif mode == 'dcd':
                # Density-weighted: high-density points contribute less
                loss_s2t = (weights_src * sq_s2t).mean()
                loss_t2s = (weights_tgt * sq_t2s).mean()
                loss = 0.5 * (loss_s2t + loss_t2s)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            loss.backward()
            optimizer.step()

        # Measure with STANDARD metric (fair comparison)
        x_np = x.detach().cpu().numpy().astype(np.float32)
        cd_s2t, cd_t2s, cd_2s = measure_chamfer(x_np, tgt_np)
        results.append({'ep': ep, 's2t': cd_s2t, 't2s': cd_t2s, '2sided': cd_2s})

        if ep % 5 == 0 or ep == num_eps - 1:
            print(f"    ep{ep:02d}: s->t={cd_s2t:.5f}  t->s={cd_t2s:.5f}  2sided={cd_2s:.5f}")

    final_np = x.detach().cpu().numpy().astype(np.float32)
    return results, final_np


def render_point_clouds(outdir, shape_name, src_np, tgt_np, final_std, final_dcd):
    """Render 3D scatter plots of source, target, DCO result, DCD result."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa

        fig = plt.figure(figsize=(20, 5))
        point_sets = [
            ('Source (sphere)', src_np, '#888888'),
            ('Target', tgt_np, '#55A868'),
            ('DCO result', final_std, '#C44E52'),
            ('DCD result', final_dcd, '#4C72B0'),
        ]
        # Subsample for rendering
        max_pts = 5000
        for i, (title, pts, color) in enumerate(point_sets):
            ax = fig.add_subplot(1, 4, i+1, projection='3d')
            idx = np.random.choice(len(pts), min(max_pts, len(pts)), replace=False)
            p = pts[idx]
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=color, s=0.3, alpha=0.6)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlim(-12, 12); ax.set_ylim(-12, 12); ax.set_zlim(-12, 12)
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            ax.view_init(elev=25, azim=135)

        fig.suptitle(f'sphere -> {shape_name} (PPC=4)', fontsize=14, fontweight='bold')
        fig.tight_layout()
        render_path = outdir / f"{shape_name}_render.png"
        fig.savefig(str(render_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved render: {render_path}")
    except Exception as e:
        print(f"[WARN] Render failed: {e}")


def plot_convergence(outdir, shape_name, results_std, results_dcd, phys_cd, dcd_k):
    """Plot CD convergence curves and s->t vs t->s bar chart."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Left: two-sided convergence
        ax = axes[0]
        eps_std = [r['ep'] for r in results_std]
        eps_dcd = [r['ep'] for r in results_dcd]
        ax.plot(eps_std, [r['2sided'] for r in results_std],
                label='DCO (standard)', color='#C44E52', linewidth=2)
        ax.plot(eps_dcd, [r['2sided'] for r in results_dcd],
                label=f'DCD (k={dcd_k})', color='#4C72B0', linewidth=2)
        if phys_cd:
            ax.axhline(phys_cd['2sided'], color='#55A868', linestyle='--',
                       linewidth=2, alpha=0.7, label=f"Physics ({phys_cd['2sided']:.3f})")
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Two-sided CD", fontsize=12)
        ax.set_title("Two-sided CD Convergence", fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2)

        # Middle: s->t convergence
        ax = axes[1]
        ax.plot(eps_std, [r['s2t'] for r in results_std],
                label='DCO s->t', color='#C44E52', linewidth=2)
        ax.plot(eps_dcd, [r['s2t'] for r in results_dcd],
                label='DCD s->t', color='#4C72B0', linewidth=2)
        ax.plot(eps_std, [r['t2s'] for r in results_std],
                label='DCO t->s', color='#C44E52', linewidth=2, linestyle='--')
        ax.plot(eps_dcd, [r['t2s'] for r in results_dcd],
                label='DCD t->s', color='#4C72B0', linewidth=2, linestyle='--')
        if phys_cd:
            ax.axhline(phys_cd['s2t'], color='#55A868', linestyle=':',
                       linewidth=1.5, alpha=0.5, label=f"Phys s->t ({phys_cd['s2t']:.3f})")
            ax.axhline(phys_cd['t2s'], color='#55A868', linestyle='--',
                       linewidth=1.5, alpha=0.5, label=f"Phys t->s ({phys_cd['t2s']:.3f})")
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("CD", fontsize=12)
        ax.set_title("s->t vs t->s Components", fontsize=13, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

        # Right: bar chart
        ax = axes[2]
        methods = ['DCO', 'DCD']
        s2t_vals = [results_std[-1]['s2t'], results_dcd[-1]['s2t']]
        t2s_vals = [results_std[-1]['t2s'], results_dcd[-1]['t2s']]
        x_pos = np.arange(len(methods))
        w = 0.3
        ax.bar(x_pos - w/2, s2t_vals, w, label='s->t', color='#4C72B0', alpha=0.8)
        ax.bar(x_pos + w/2, t2s_vals, w, label='t->s', color='#C44E52', alpha=0.8)
        if phys_cd:
            ax.axhline(phys_cd['2sided'], color='#55A868', linestyle='--',
                       linewidth=1.5, alpha=0.5, label='Physics baseline')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, fontsize=12)
        ax.set_ylabel("Chamfer Distance", fontsize=12)
        ax.set_title("Final s->t vs t->s", fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2, axis='y')

        fig.suptitle(f'sphere -> {shape_name} (PPC=4)', fontsize=14, fontweight='bold', y=1.02)
        fig.tight_layout()
        plot_path = outdir / f"{shape_name}_convergence.png"
        fig.savefig(str(plot_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot: {plot_path}")
    except Exception as e:
        print(f"[WARN] Plot failed: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config",
                    default="configs/dcd_ppc4/sphere_to_bunny.yaml")
    ap.add_argument("--num-eps", type=int, default=40)
    ap.add_argument("--adam-steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--dcd-k", type=int, default=8,
                    help="k for DCD density estimation")
    args = ap.parse_args()

    print(f"[DCD Experiment] Config: {args.config}")
    src_np, tgt_np, ppc, shape_name = load_source_target(args.config)
    print(f"Source: {len(src_np)} pts, Target: {len(tgt_np)} pts, PPC={ppc}, Shape={shape_name}")

    # Per-shape output directory
    outdir = Path(f"output/dcd_ppc4/{shape_name}")
    outdir.mkdir(parents=True, exist_ok=True)

    # Initial CD
    s0, t0, ts0 = measure_chamfer(src_np, tgt_np)
    print(f"Initial: s->t={s0:.5f}  t->s={t0:.5f}  2sided={ts0:.5f}")

    # Also measure physics-only endpoint if available
    phys_cd = None
    for phys_dir in [
        Path(f"output/pairwise_physics_only/sphere_to_{shape_name}"),
        Path(f"output/render_ctrl/expM_phys_only"),
    ]:
        phys_ep39 = phys_dir / "ep039" / "ep039_particles.pt"
        if phys_ep39.exists():
            phys_pos = torch.load(str(phys_ep39), weights_only=False, map_location='cpu')
            if isinstance(phys_pos, torch.Tensor):
                phys_np = phys_pos.numpy().astype(np.float32)
            else:
                phys_np = np.array(phys_pos, dtype=np.float32)
            ps, pt_, p2 = measure_chamfer(phys_np, tgt_np)
            phys_cd = {'s2t': ps, 't2s': pt_, '2sided': p2}
            print(f"Physics ep39 ({phys_dir}): s->t={ps:.5f}  t->s={pt_:.5f}  2sided={p2:.5f}")
            break

    all_results = {}

    # -- Run standard DCO -------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  DCO (standard two-sided)")
    print(f"{'='*60}")
    t0_time = time.time()
    results_std, final_std = run_dco(
        src_np, tgt_np, mode='standard',
        num_eps=args.num_eps, adam_steps=args.adam_steps, lr=args.lr)
    dt_std = time.time() - t0_time

    all_results['dco_standard'] = {
        'label': 'DCO (standard)',
        'results': results_std,
        'final': results_std[-1],
        'time_s': dt_std,
    }
    np.save(str(outdir / "dco_standard_final.npy"), final_std)

    # -- Run DCD ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  DCD (density-aware, k={args.dcd_k})")
    print(f"{'='*60}")
    t0_time = time.time()
    results_dcd, final_dcd = run_dco(
        src_np, tgt_np, mode='dcd',
        num_eps=args.num_eps, adam_steps=args.adam_steps, lr=args.lr,
        dcd_k=args.dcd_k)
    dt_dcd = time.time() - t0_time

    all_results['dco_dcd'] = {
        'label': f'DCD (k={args.dcd_k})',
        'results': results_dcd,
        'final': results_dcd[-1],
        'time_s': dt_dcd,
    }
    np.save(str(outdir / "dcd_final.npy"), final_dcd)

    # -- Summary ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"SUMMARY [{shape_name}]")
    print(f"{'='*60}")
    print(f"{'Method':<25} {'s->t':>8} {'t->s':>8} {'2sided':>8}")
    print("-" * 55)
    for name, data in all_results.items():
        r = data['final']
        print(f"{data['label']:<25} {r['s2t']:>8.5f} {r['t2s']:>8.5f} {r['2sided']:>8.5f}")
    if phys_cd:
        print(f"{'Physics-only':<25} {phys_cd['s2t']:>8.5f} {phys_cd['t2s']:>8.5f} {phys_cd['2sided']:>8.5f}")
    print("-" * 55)

    # DCD vs DCO comparison
    std_final = results_std[-1]['2sided']
    dcd_final = results_dcd[-1]['2sided']
    diff_pct = (dcd_final - std_final) / std_final * 100
    print(f"\nDCD vs DCO: {diff_pct:+.1f}% ({'better' if diff_pct < 0 else 'worse'})")
    if phys_cd:
        print(f"DCO vs Physics: {std_final/phys_cd['2sided']:.2f}x")
        print(f"DCD vs Physics: {dcd_final/phys_cd['2sided']:.2f}x")

    # Save JSON
    save_data = {
        'config': args.config,
        'shape': shape_name,
        'ppc': ppc,
        'num_source': len(src_np),
        'num_target': len(tgt_np),
        'physics_cd': phys_cd,
        'dco_standard': all_results['dco_standard'],
        'dco_dcd': all_results['dco_dcd'],
    }
    out_json = outdir / f"{shape_name}_dcd_results.json"
    with open(out_json, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {out_json}")

    # -- Plots ------------------------------------------------------------------
    plot_convergence(outdir, shape_name, results_std, results_dcd, phys_cd, args.dcd_k)

    # -- 3D Render --------------------------------------------------------------
    render_point_clouds(outdir, shape_name, src_np, tgt_np, final_std, final_dcd)

    print(f"\nDONE [{shape_name}].")


if __name__ == "__main__":
    main()
