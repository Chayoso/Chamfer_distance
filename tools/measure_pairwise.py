#!/usr/bin/env python3
"""Measure CD + generate ablation graphs for all pairwise coupled-rev experiments.

Usage:
    conda run -n diffmpm_v2.3.0 python tools/measure_pairwise.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from itertools import permutations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SHAPES = ['bunny', 'bob', 'cow', 'spot', 'dragon', 'armadilo']
MESH_MAP = {s: f'assets/{s}.obj' for s in SHAPES}
OUTPUT_ROOT = Path('output/pairwise_coupled')
NUM_EPS = 40


def load_pts(pt_path):
    pt_path = str(pt_path)
    if pt_path.endswith('.pt') or pt_path.endswith('.pth'):
        import torch
        data = torch.load(pt_path, map_location='cpu', weights_only=False)
        if isinstance(data, torch.Tensor):
            return data.numpy().astype(np.float64)
        elif isinstance(data, dict):
            for k in ['positions', 'pos', 'points']:
                if k in data:
                    v = data[k]
                    return v.numpy().astype(np.float64) if isinstance(v, torch.Tensor) else np.array(v, dtype=np.float64)
        raise KeyError(f'No positions in {pt_path}')
    else:
        d = dict(np.load(pt_path, allow_pickle=True))
        for k in ['positions', 'pos']:
            if k in d:
                return np.array(d[k], dtype=np.float64)
        raise KeyError(f'No positions in {pt_path}')


def load_target(mesh_path, n_samples=80000):
    import trimesh
    m = trimesh.load(mesh_path, force='mesh')
    pts, _ = trimesh.sample.sample_surface(m, n_samples)
    return np.array(pts, dtype=np.float64)


def compute_cd(pts, tgt_pts, tgt_tree=None):
    # Skip NaN/inf
    mask = np.isfinite(pts).all(axis=1)
    if mask.sum() < 10:
        return None
    pts = pts[mask]
    if tgt_tree is None:
        tgt_tree = cKDTree(tgt_pts)
    src_tree = cKDTree(pts)
    d_fwd, _ = tgt_tree.query(pts)
    d_rev, _ = src_tree.query(tgt_pts)
    s2t = float(np.sqrt(np.mean(d_fwd**2)))
    t2s = float(np.sqrt(np.mean(d_rev**2)))
    two_sided = float(np.sqrt(np.mean(d_fwd**2) + np.mean(d_rev**2)))
    return {'s2t': s2t, 't2s': t2s, 'two_sided': two_sided}


def measure_trajectory(exp_dir, tgt_pts, tgt_tree):
    results = {}
    for ep in range(NUM_EPS):
        pt_file = exp_dir / f'ep{ep:03d}/ep{ep:03d}_particles.pt'
        if not pt_file.exists():
            continue
        try:
            pts = load_pts(pt_file)
            results[ep] = compute_cd(pts, tgt_pts, tgt_tree)
        except Exception as e:
            print(f'  Warning: ep{ep:03d} failed: {e}')
    return results


def measure_bbox_trajectory(exp_dir):
    results = {}
    for ep in range(NUM_EPS):
        pt_file = exp_dir / f'ep{ep:03d}/ep{ep:03d}_particles.pt'
        if not pt_file.exists():
            continue
        try:
            pts = load_pts(pt_file)
            mn, mx = pts.min(0), pts.max(0)
            ext = mx - mn
            results[ep] = float(ext[0] * ext[1] * ext[2])
        except:
            pass
    return results


def main():
    all_results = {}
    all_trajectories = {}
    all_bboxes = {}

    pairs = [(s, t) for s, t in permutations(SHAPES, 2)]

    print(f'Measuring {len(pairs)} pairwise experiments...\n')

    for src, tgt in pairs:
        name = f'{src}_to_{tgt}'
        exp_dir = OUTPUT_ROOT / name
        ep39 = exp_dir / 'ep039/ep039_particles.pt'

        if not ep39.exists():
            print(f'  {name}: NOT COMPLETE (no ep039)')
            continue

        print(f'  {name}...', end=' ', flush=True)

        tgt_pts = load_target(MESH_MAP[tgt])
        tgt_tree = cKDTree(tgt_pts)

        # Final CD
        pts = load_pts(ep39)
        cd = compute_cd(pts, tgt_pts, tgt_tree)
        if cd is None:
            print(f'DIVERGED (NaN particles)')
            continue
        all_results[name] = cd

        # Trajectory
        traj = measure_trajectory(exp_dir, tgt_pts, tgt_tree)
        all_trajectories[name] = {str(k): v for k, v in traj.items() if v is not None}

        # Bounding box volume trajectory
        bbox = measure_bbox_trajectory(exp_dir)
        all_bboxes[name] = {str(k): v for k, v in bbox.items()}

        print(f's2t={cd["s2t"]:.4f} t2s={cd["t2s"]:.4f} 2s={cd["two_sided"]:.4f}')

    # Save JSON
    out_json = OUTPUT_ROOT / 'all_results.json'
    with open(out_json, 'w') as f:
        json.dump({
            'final_cd': all_results,
            'trajectories': all_trajectories,
            'bbox_volumes': all_bboxes,
        }, f, indent=2)
    print(f'\nResults saved to {out_json}')

    if not all_results:
        print('No completed experiments found.')
        return

    # ── Summary table ──
    print('\n' + '='*80)
    print('PAIRWISE CD RESULTS (two-sided, ep039)')
    print('='*80)
    print(f'{"Pair":<25} {"s2t":>8} {"t2s":>8} {"2-sided":>8}')
    print('-'*55)
    for name in sorted(all_results.keys()):
        cd = all_results[name]
        print(f'  {name:<23} {cd["s2t"]:>8.4f} {cd["t2s"]:>8.4f} {cd["two_sided"]:>8.4f}')

    # ── Matrix heatmap ──
    n = len(SHAPES)
    mat = np.full((n, n), np.nan)
    for i, src in enumerate(SHAPES):
        for j, tgt in enumerate(SHAPES):
            if i == j:
                continue
            name = f'{src}_to_{tgt}'
            if name in all_results:
                mat[i, j] = all_results[name]['two_sided']

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap='RdYlGn_r', vmin=0, vmax=np.nanmax(mat)*1.1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(SHAPES, rotation=45, ha='right')
    ax.set_yticklabels(SHAPES)
    ax.set_xlabel('Target')
    ax.set_ylabel('Source')
    ax.set_title('Pairwise Two-Sided CD (coupled rev, ep039)')
    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f'{mat[i,j]:.3f}', ha='center', va='center', fontsize=8,
                        color='white' if mat[i,j] > np.nanmedian(mat) else 'black')
    plt.colorbar(im, label='Two-sided CD')
    plt.tight_layout()
    fig.savefig(str(OUTPUT_ROOT / 'cd_matrix_heatmap.png'), dpi=150)
    plt.close()
    print(f'Heatmap saved to {OUTPUT_ROOT / "cd_matrix_heatmap.png"}')

    # ── Trajectory plots (per source) ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()
    colors = plt.cm.Set2(np.linspace(0, 1, len(SHAPES)))
    for idx, src in enumerate(SHAPES):
        ax = axes[idx]
        ax.set_title(f'Source: {src}')
        for j, tgt in enumerate(SHAPES):
            if src == tgt:
                continue
            name = f'{src}_to_{tgt}'
            if name not in all_trajectories:
                continue
            traj = all_trajectories[name]
            eps = sorted([int(k) for k in traj.keys()])
            vals = [traj[str(e)]['two_sided'] for e in eps]
            ax.plot(eps, vals, label=f'→{tgt}', color=colors[j], linewidth=1.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Two-sided CD')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(OUTPUT_ROOT / 'cd_trajectories_by_source.png'), dpi=150)
    plt.close()
    print(f'Trajectory plots saved to {OUTPUT_ROOT / "cd_trajectories_by_source.png"}')

    # ── BBox volume plots (shrinkage check) ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()
    for idx, src in enumerate(SHAPES):
        ax = axes[idx]
        ax.set_title(f'BBox Volume: {src} →')
        for j, tgt in enumerate(SHAPES):
            if src == tgt:
                continue
            name = f'{src}_to_{tgt}'
            if name not in all_bboxes:
                continue
            bbox = all_bboxes[name]
            eps = sorted([int(k) for k in bbox.keys()])
            vals = [bbox[str(e)] for e in eps]
            ax.plot(eps, vals, label=f'→{tgt}', color=colors[j], linewidth=1.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('BBox Volume')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=15, color='red', linestyle='--', alpha=0.3, label='phys ramp start')
    plt.tight_layout()
    fig.savefig(str(OUTPUT_ROOT / 'bbox_volume_trajectories.png'), dpi=150)
    plt.close()
    print(f'BBox volume plots saved to {OUTPUT_ROOT / "bbox_volume_trajectories.png"}')

    # ── Best results per pair ──
    print('\n' + '='*80)
    print('BEST RESULTS (lowest two-sided CD across trajectory)')
    print('='*80)
    print(f'{"Pair":<25} {"Best ep":>8} {"Best 2s":>10} {"ep039 2s":>10}')
    print('-'*60)
    best_results = {}
    for name in sorted(all_results.keys()):
        traj = all_trajectories.get(name, {})
        if not traj:
            continue
        best_ep = min(traj.keys(), key=lambda e: traj[e]['two_sided'])
        best_cd = traj[best_ep]['two_sided']
        ep39_cd = all_results[name]['two_sided']
        best_results[name] = {
            'best_ep': int(best_ep),
            'best_two_sided': best_cd,
            'ep039_two_sided': ep39_cd,
        }
        print(f'  {name:<23} {best_ep:>8} {best_cd:>10.4f} {ep39_cd:>10.4f}')

    with open(OUTPUT_ROOT / 'best_results.json', 'w') as f:
        json.dump(best_results, f, indent=2)
    print(f'\nBest results saved to {OUTPUT_ROOT / "best_results.json"}')


if __name__ == '__main__':
    main()
