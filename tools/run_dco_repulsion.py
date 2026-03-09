#!/usr/bin/env python3
"""
DCO + Repulsion Baseline: Direct Chamfer Optimization with local regularizers.

ECCV Tier 2 experiment: Tests whether simple regularization (repulsion,
Laplacian smoothness, volume penalty) can fix DCO collapse without physics.

Conditions:
  - DCO-only: pure two-sided Chamfer (same as run_cd_only.py)
  - DCO + repulsion: Chamfer + λ_rep * repulsion loss
  - DCO + repulsion + smoothness: + λ_smooth * Laplacian smoothness
  - DCO + repulsion + volume: + λ_vol * volume preservation

Key claim: local regularizers resist collapse locally but cannot enforce
global mass redistribution → physics constraints remain necessary.

Usage:
    export LD_LIBRARY_PATH=.../torch/lib:$LD_LIBRARY_PATH
    conda run -n diffmpm_v2.3.0 python tools/run_dco_repulsion.py [--target bunny]
    conda run -n diffmpm_v2.3.0 python tools/run_dco_repulsion.py --config configs/dcd_ppc4/sphere_to_bunny.yaml
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

WORKSPACE = Path("/home/chayo/Desktop/Shape-morphing-binder")
OUTDIR = WORKSPACE / "output" / "dco_repulsion"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Shape registry ───────────────────────────────────────────────────────────
SHAPES = {
    'bunny': {
        'source': 'output/render_ctrl/expM_phys_only/ep000/ep000_particles.pt',
        'target_npy': 'output/render_ctrl/expM_chamfer_postprocess/target_positions.npy',
        'mesh': 'assets/bunny.obj',
    },
    'teapot': {
        'source': 'output/sphere_to_teapot/ep000/ep000_particles.pt',
        'target_npy': 'output/sphere_to_teapot_pp/target_positions.npy',
        'mesh': 'assets/teapot.obj',
    },
    'armadilo': {
        'source': 'output/sphere_to_armadilo/ep000/ep000_particles.pt',
        'target_npy': 'output/sphere_to_armadilo_pp/target_positions.npy',
        'mesh': 'assets/armadilo.obj',
    },
}


def load_pts(path):
    path = Path(path)
    if path.suffix == '.pt':
        t = torch.load(str(path), weights_only=False, map_location='cpu')
        if isinstance(t, torch.Tensor):
            return t.numpy().astype(np.float32)
        return np.array(t, dtype=np.float32)
    return np.load(str(path)).astype(np.float32)


def load_target(info):
    npy_path = WORKSPACE / info['target_npy']
    if npy_path.exists():
        return np.load(str(npy_path)).astype(np.float32)
    import trimesh
    mesh = trimesh.load_mesh(str(WORKSPACE / info['mesh']))
    pts, _ = trimesh.sample.sample_surface(mesh, 80000)
    return pts.astype(np.float32)


def load_source_target_from_config(cfg_path):
    """Load source and target from a YAML config via the physics pipeline.
    Same method as run_dco_dcd.py for consistency."""
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

    shape_name = Path(cfg['target_mesh_path']).stem
    return src_np, tgt_np, shape_name


def measure_chamfer(x_np, tgt_np, tree_tgt=None):
    if tree_tgt is None:
        tree_tgt = cKDTree(tgt_np)
    d_s2t, _ = tree_tgt.query(x_np, k=1, workers=4)
    cd_s2t = float(np.sqrt((d_s2t**2).mean()))
    tree_src = cKDTree(x_np)
    d_t2s, _ = tree_src.query(tgt_np, k=1, workers=4)
    cd_t2s = float(np.sqrt((d_t2s**2).mean()))
    return cd_s2t, cd_t2s, 0.5 * (cd_s2t + cd_t2s)


def precompute_knn(x, k=12):
    """Precompute k-NN indices (called once per episode, not every step)."""
    x_np = x.detach().cpu().numpy().astype(np.float32)
    tree = cKDTree(x_np)
    dists, idx = tree.query(x_np, k=k+1)
    return dists[:, 1:], idx[:, 1:]  # exclude self


def repulsion_loss_knn(x, knn_idx, knn_dists, k=8):
    """Repulsion using precomputed k-NN indices (fast)."""
    idx = knn_idx[:, :k]
    dists = knn_dists[:, :k]
    mean_dist = float(dists.mean())
    threshold = mean_dist * 0.5

    neighbors = x[torch.from_numpy(idx.astype(np.int64)).to(x.device)]
    diff = x.unsqueeze(1) - neighbors
    sq_dist = (diff ** 2).sum(dim=2)

    safe_sq = sq_dist.clamp(min=1e-8)
    penalty = (1.0 / safe_sq) * (sq_dist < threshold**2).float()
    return penalty.mean()


def smoothness_loss_knn(x, knn_idx, k=12):
    """Laplacian smoothness using precomputed k-NN indices (fast)."""
    idx = knn_idx[:, :k]
    neighbors = x[torch.from_numpy(idx.astype(np.int64)).to(x.device)]
    centroid = neighbors.mean(dim=1)
    return ((x - centroid) ** 2).sum(dim=1).mean()


def volume_loss(x, source_vol=None):
    """Volume preservation: penalize deviation from source bounding box volume."""
    bbox_min = x.min(dim=0).values
    bbox_max = x.max(dim=0).values
    vol = (bbox_max - bbox_min).prod()
    if source_vol is None:
        return torch.tensor(0.0)
    return ((vol - source_vol) / source_vol) ** 2


def run_dco_condition(src_np, tgt_np, condition, num_eps=40, adam_steps=30, lr=0.01):
    """Run DCO with specified regularization condition."""
    x = torch.tensor(src_np, dtype=torch.float32, device=DEVICE, requires_grad=True)
    x_tgt = torch.tensor(tgt_np, dtype=torch.float32, device=DEVICE)
    tree_tgt = cKDTree(tgt_np)

    # Source volume for volume penalty
    src_bbox = src_np.max(axis=0) - src_np.min(axis=0)
    src_vol = float(src_bbox.prod())

    optimizer = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.999), eps=1e-4, amsgrad=True)

    # Parse regularization weights from condition
    lambda_rep = condition.get('lambda_rep', 0.0)
    lambda_smooth = condition.get('lambda_smooth', 0.0)
    lambda_vol = condition.get('lambda_vol', 0.0)
    k_rep = condition.get('k_rep', 8)

    results = []
    k_max = max(k_rep, 12)  # precompute enough neighbors for both losses

    for ep in range(num_eps):
        # Precompute k-NN once per episode (not every step!)
        knn_dists, knn_idx = None, None
        if lambda_rep > 0 or lambda_smooth > 0:
            knn_dists, knn_idx = precompute_knn(x, k=k_max)

        for step in range(adam_steps):
            optimizer.zero_grad()
            x_np = x.detach().cpu().numpy().astype(np.float32)

            # s→t Chamfer
            _, nn_s2t = tree_tgt.query(x_np, k=1, workers=4)
            x_nn_s2t = x_tgt[torch.from_numpy(nn_s2t).long().to(DEVICE)]
            loss_s2t = (x - x_nn_s2t).pow(2).sum(dim=1).mean()

            # t→s Chamfer
            tree_src = cKDTree(x_np)
            _, nn_t2s = tree_src.query(tgt_np, k=1, workers=4)
            x_nn_t2s = x[torch.from_numpy(nn_t2s).long().to(DEVICE)]
            loss_t2s = (x_tgt - x_nn_t2s).pow(2).sum(dim=1).mean()

            loss = 0.5 * (loss_s2t + loss_t2s)

            # Regularization terms (using precomputed k-NN)
            if lambda_rep > 0 and knn_idx is not None:
                loss = loss + lambda_rep * repulsion_loss_knn(x, knn_idx, knn_dists, k=k_rep)
            if lambda_smooth > 0 and knn_idx is not None:
                loss = loss + lambda_smooth * smoothness_loss_knn(x, knn_idx)
            if lambda_vol > 0:
                loss = loss + lambda_vol * volume_loss(
                    x, torch.tensor(src_vol, device=DEVICE))

            loss.backward()
            optimizer.step()

        # Measure final CD this episode
        x_np = x.detach().cpu().numpy().astype(np.float32)
        cd_s2t, cd_t2s, cd_2s = measure_chamfer(x_np, tgt_np, tree_tgt)
        results.append({
            'ep': ep, 'cd_s2t': cd_s2t, 'cd_t2s': cd_t2s,
            'cd_2sided': cd_2s,
        })

        if ep % 5 == 0 or ep == num_eps - 1:
            print(f"    ep{ep:02d}: s→t={cd_s2t:.5f}  t→s={cd_t2s:.5f}  2sided={cd_2s:.5f}")

    # Save final positions
    final_np = x.detach().cpu().numpy().astype(np.float32)
    return results, final_np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="bunny", choices=list(SHAPES.keys()))
    ap.add_argument("--config", "-c", default=None, help="YAML config for source/target (overrides --target)")
    ap.add_argument("--num-eps", type=int, default=40)
    ap.add_argument("--adam-steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.01)
    args = ap.parse_args()

    if args.config:
        src_np, tgt_np, shape_name = load_source_target_from_config(args.config)
        outdir = WORKSPACE / "output" / "dco_repulsion_ppc4" / shape_name
        outdir.mkdir(parents=True, exist_ok=True)
        args.target = shape_name
    else:
        outdir = OUTDIR
        outdir.mkdir(parents=True, exist_ok=True)
        info = SHAPES[args.target]
        src_np = load_pts(WORKSPACE / info['source'])
        tgt_np = load_target(info)
        shape_name = args.target

    print(f"Source: {src_np.shape[0]} pts, Target: {tgt_np.shape[0]} pts")
    cd_init = measure_chamfer(src_np, tgt_np)
    print(f"Initial: s→t={cd_init[0]:.5f}  t→s={cd_init[1]:.5f}  2sided={cd_init[2]:.5f}")

    # ── Conditions ────────────────────────────────────────────────────────
    CONDITIONS = {
        'dco_only': {
            'label': 'DCO (no regularization)',
            'lambda_rep': 0.0, 'lambda_smooth': 0.0, 'lambda_vol': 0.0,
        },
        'dco_rep_weak': {
            'label': 'DCO + Repulsion (λ=0.01)',
            'lambda_rep': 0.01, 'lambda_smooth': 0.0, 'lambda_vol': 0.0,
        },
        'dco_rep_strong': {
            'label': 'DCO + Repulsion (λ=0.1)',
            'lambda_rep': 0.1, 'lambda_smooth': 0.0, 'lambda_vol': 0.0,
        },
        'dco_rep_smooth': {
            'label': 'DCO + Repulsion + Smoothness',
            'lambda_rep': 0.05, 'lambda_smooth': 0.01, 'lambda_vol': 0.0,
        },
        'dco_full_reg': {
            'label': 'DCO + Repulsion + Smooth + Volume',
            'lambda_rep': 0.05, 'lambda_smooth': 0.01, 'lambda_vol': 0.1,
        },
    }

    all_results = {}

    for cond_name, cond in CONDITIONS.items():
        print(f"\n{'='*60}")
        print(f"  {cond['label']}")
        print(f"{'='*60}")

        t0 = time.time()
        results, final_np = run_dco_condition(
            src_np, tgt_np, cond,
            num_eps=args.num_eps, adam_steps=args.adam_steps, lr=args.lr
        )
        dt = time.time() - t0

        r_final = results[-1]
        r_best = min(results, key=lambda r: r['cd_2sided'])

        print(f"\n  Final: 2sided={r_final['cd_2sided']:.5f}  ({dt:.1f}s)")
        print(f"  Best ep{r_best['ep']}: 2sided={r_best['cd_2sided']:.5f}")

        # Save final positions
        np.save(str(outdir / f"{args.target}_{cond_name}_final.npy"), final_np)

        all_results[cond_name] = {
            'label': cond['label'],
            'params': {k: v for k, v in cond.items() if k != 'label'},
            'cd_final': {
                's2t': r_final['cd_s2t'], 't2s': r_final['cd_t2s'],
                '2sided': r_final['cd_2sided'],
            },
            'cd_best': {
                'ep': r_best['ep'], '2sided': r_best['cd_2sided'],
            },
            'time_s': dt,
            'history': results,
        }

    # ── Measure physics-only baseline ────────────────────────────────────
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
            print(f"\nPhysics ep39 ({phys_dir}): s→t={ps:.5f}  t→s={pt_:.5f}  2sided={p2:.5f}")
            break
    all_results['_physics'] = phys_cd

    # ── Save results ──────────────────────────────────────────────────────
    out_json = outdir / f"{args.target}_dco_repulsion_results.json"
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_json}")

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"SUMMARY: {args.target} — DCO + Regularization Ablation")
    print(f"{'='*70}")
    print(f"{'Condition':<35} {'s→t':>8} {'t→s':>8} {'2sided':>8}")
    print("-" * 65)
    for cond_name, data in all_results.items():
        if cond_name.startswith('_'):
            continue
        r = data['cd_final']
        print(f"{data['label']:<35} {r['s2t']:>8.5f} {r['t2s']:>8.5f} {r['2sided']:>8.5f}")
    print("-" * 65)
    print(f"{'Physics-only [Xu et al.]':<35} {'':>8} {'':>8} {'0.131':>8}")
    print(f"{'PhysMorph (ours)':<35} {'':>8} {'':>8} {'0.056':>8}")

    # ── Generate comparison plot ──────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        colors_cond = {
            'dco_only': '#C44E52',
            'dco_rep_weak': '#DD8452',
            'dco_rep_strong': '#CCB974',
            'dco_rep_smooth': '#64B5CD',
            'dco_full_reg': '#8172B3',
        }

        # Left: convergence curves (two-sided CD over episodes)
        ax = axes[0]
        for cond_name, data in all_results.items():
            if cond_name.startswith('_'):
                continue
            h = data['history']
            eps = [r['ep'] for r in h]
            cds = [r['cd_2sided'] for r in h]
            ax.plot(eps, cds, label=data['label'],
                    color=colors_cond.get(cond_name, 'gray'), linewidth=2)

        ax.axhline(0.131, color='#4C72B0', linestyle='--', linewidth=2,
                    alpha=0.7, label='Physics-only (0.131)')
        ax.axhline(0.056, color='#55A868', linestyle='--', linewidth=2,
                    alpha=0.7, label='PhysMorph (0.056)')

        ax.set_xlabel("Optimization Episode", fontsize=13)
        ax.set_ylabel("Two-sided Chamfer Distance", fontsize=13)
        ax.set_title("DCO + Regularization: Convergence", fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

        # Right: s→t vs t→s final values (shows collapse asymmetry)
        ax = axes[1]
        names = []
        s2t_vals = []
        t2s_vals = []
        for cond_name, data in all_results.items():
            if cond_name.startswith('_'):
                continue
            names.append(data['label'].replace('DCO + ', 'DCO+\n').replace(' + ', '+\n'))
            s2t_vals.append(data['cd_final']['s2t'])
            t2s_vals.append(data['cd_final']['t2s'])

        x_pos = np.arange(len(names))
        w = 0.35
        ax.bar(x_pos - w/2, s2t_vals, w, label='s→t', color='#4C72B0', alpha=0.8)
        ax.bar(x_pos + w/2, t2s_vals, w, label='t→s', color='#C44E52', alpha=0.8)

        # Physics reference
        ax.axhline(0.131, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.text(len(names)-0.5, 0.135, 'Physics baseline', fontsize=8, alpha=0.6)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel("Chamfer Distance", fontsize=13)
        ax.set_title("s→t vs t→s Asymmetry (Collapse Diagnostic)", fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.2, axis='y')

        fig.tight_layout()
        plot_path = outdir / f"{args.target}_dco_repulsion_plot.png"
        fig.savefig(str(plot_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot: {plot_path}")
    except Exception as e:
        print(f"[WARN] Could not generate plot: {e}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
