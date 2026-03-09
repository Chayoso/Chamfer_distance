#!/usr/bin/env python3
"""Measure CD for both physics-only and coupled experiments, compare ep30+ performance.

Usage:
    conda run -n diffmpm_v2.3.0 python tools/measure_and_compare.py
    conda run -n diffmpm_v2.3.0 python tools/measure_and_compare.py --coupled_dir output/pairwise_coupled_v2
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SHAPES = ['sphere', 'bunny', 'spot', 'C', 'dragon', 'E', 'teapot', 'V', 'armadilo', 'bob']
MESH_MAP = {s: f'assets/{s}.obj' for s in SHAPES}
MESH_MAP['sphere'] = 'assets/isosphere.obj'
NUM_EPS = 40
LATE_EPS = list(range(30, 40))  # ep30-ep39


def load_pts(pt_path):
    import torch
    data = torch.load(str(pt_path), map_location='cpu', weights_only=False)
    if isinstance(data, torch.Tensor):
        return data.numpy().astype(np.float64)
    elif isinstance(data, dict):
        for k in ['positions', 'pos', 'points']:
            if k in data:
                v = data[k]
                return v.numpy().astype(np.float64) if isinstance(v, torch.Tensor) else np.array(v, dtype=np.float64)
    raise KeyError(f'No positions in {pt_path}')


def load_target(mesh_path, n_samples=80000):
    import trimesh
    m = trimesh.load(mesh_path, force='mesh')
    pts, _ = trimesh.sample.sample_surface(m, n_samples)
    return np.array(pts, dtype=np.float64)


def compute_cd(pts, tgt_pts, tgt_tree=None):
    mask = np.isfinite(pts).all(axis=1)
    if mask.sum() < 10:
        return None
    pts = pts[mask]
    if tgt_tree is None:
        tgt_tree = cKDTree(tgt_pts)
    src_tree = cKDTree(pts)
    d_fwd, _ = tgt_tree.query(pts)
    d_rev, _ = src_tree.query(tgt_pts)
    two_sided = float(np.sqrt(np.mean(d_fwd**2) + np.mean(d_rev**2)))
    return two_sided


def measure_experiment(output_root, name, tgt_pts, tgt_tree):
    """Measure CD trajectory for one experiment."""
    exp_dir = output_root / name
    results = {}
    for ep in range(NUM_EPS):
        pt_file = exp_dir / f'ep{ep:03d}/ep{ep:03d}_particles.pt'
        if not pt_file.exists():
            continue
        try:
            pts = load_pts(pt_file)
            cd = compute_cd(pts, tgt_pts, tgt_tree)
            if cd is not None:
                results[ep] = cd
        except Exception as e:
            pass
    return results


def measure_bbox(output_root, name):
    """Measure BBox volume trajectory for shrinkage check."""
    exp_dir = output_root / name
    results = {}
    for ep in range(NUM_EPS):
        pt_file = exp_dir / f'ep{ep:03d}/ep{ep:03d}_particles.pt'
        if not pt_file.exists():
            continue
        try:
            pts = load_pts(pt_file)
            mask = np.isfinite(pts).all(axis=1)
            if mask.sum() < 10:
                continue
            pts = pts[mask]
            mn, mx = pts.min(0), pts.max(0)
            ext = mx - mn
            results[ep] = float(ext[0] * ext[1] * ext[2])
        except:
            pass
    return results


def late_ep_mean(traj, eps=LATE_EPS):
    """Mean CD over late episodes (30+)."""
    vals = [traj[e] for e in eps if e in traj]
    if not vals:
        # fallback: use last available
        if traj:
            last_ep = max(traj.keys())
            return traj[last_ep]
        return None
    return np.mean(vals)


def check_shrinkage(bbox_traj):
    """Check if BBox volume has a significant dip (shrinkage) in the trajectory."""
    if len(bbox_traj) < 10:
        return False, 0, 0
    eps = sorted(bbox_traj.keys())
    vols = [bbox_traj[e] for e in eps]
    if len(vols) < 5:
        return False, 0, 0

    vol_start = np.mean(vols[:3])  # average of first 3
    vol_min = min(vols[3:])        # min after initial
    vol_end = np.mean(vols[-3:])   # average of last 3

    # Shrinkage = vol drops > 30% then recovers > 50% of the drop
    if vol_start > 0:
        drop_ratio = (vol_start - vol_min) / vol_start
        if drop_ratio > 0.3 and vol_end > vol_min * 1.3:
            return True, drop_ratio, vol_min / vol_start
    return False, 0, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--physics_dir', type=str, default='output/pairwise_physics_only')
    ap.add_argument('--coupled_dir', type=str, default='output/pairwise_coupled_v2')
    args = ap.parse_args()

    phys_root = Path(args.physics_dir)
    coup_root = Path(args.coupled_dir)

    # Discover all pairs from configs
    from itertools import permutations
    all_pairs = [f'{s}_to_{t}' for s, t in permutations(SHAPES, 2)]

    print(f'=== Pairwise Comparison: Physics-only vs Coupled ===')
    print(f'  Physics dir: {phys_root}')
    print(f'  Coupled dir: {coup_root}')
    print(f'  Total pairs: {len(all_pairs)}')
    print()

    phys_results = {}
    coup_results = {}
    phys_bbox = {}
    coup_bbox = {}

    # Cache target point clouds
    tgt_cache = {}

    for name in sorted(all_pairs):
        tgt_name = name.split('_to_')[1]

        # Load target once
        if tgt_name not in tgt_cache:
            try:
                tgt_pts = load_target(MESH_MAP[tgt_name])
                tgt_cache[tgt_name] = (tgt_pts, cKDTree(tgt_pts))
            except Exception as e:
                print(f'  SKIP {name}: target load failed ({e})')
                continue
        tgt_pts, tgt_tree = tgt_cache[tgt_name]

        # Check if experiments exist
        phys_exists = (phys_root / name / 'ep039/ep039_particles.pt').exists() or \
                      (phys_root / name / 'ep030/ep030_particles.pt').exists()
        coup_exists = (coup_root / name / 'ep039/ep039_particles.pt').exists() or \
                      (coup_root / name / 'ep030/ep030_particles.pt').exists()

        if phys_exists:
            phys_results[name] = measure_experiment(phys_root, name, tgt_pts, tgt_tree)
            phys_bbox[name] = measure_bbox(phys_root, name)

        if coup_exists:
            coup_results[name] = measure_experiment(coup_root, name, tgt_pts, tgt_tree)
            coup_bbox[name] = measure_bbox(coup_root, name)

    # ── Comparison Table ──
    print('\n' + '='*100)
    print(f'{"Pair":<25} {"Phys ep30+":<12} {"Coup ep30+":<12} {"Winner":<10} {"Improv":<10} {"Shrink?":<8}')
    print('-'*100)

    coupled_wins = 0
    physics_wins = 0
    both_exist = 0
    shrink_pairs = []

    for name in sorted(all_pairs):
        phys_traj = phys_results.get(name, {})
        coup_traj = coup_results.get(name, {})

        if not phys_traj and not coup_traj:
            continue

        phys_late = late_ep_mean(phys_traj)
        coup_late = late_ep_mean(coup_traj)

        # Shrinkage check on coupled
        has_shrink = False
        if name in coup_bbox:
            has_shrink, drop_r, _ = check_shrinkage(coup_bbox[name])
            if has_shrink:
                shrink_pairs.append(name)

        shrink_str = 'YES' if has_shrink else ''

        if phys_late is not None and coup_late is not None:
            both_exist += 1
            if coup_late < phys_late:
                winner = 'COUPLED'
                coupled_wins += 1
                improv = f'-{(1 - coup_late/phys_late)*100:.1f}%'
            else:
                winner = 'PHYSICS'
                physics_wins += 1
                improv = f'+{(coup_late/phys_late - 1)*100:.1f}%'
            print(f'  {name:<23} {phys_late:>10.4f}  {coup_late:>10.4f}  {winner:<8} {improv:<8} {shrink_str}')
        elif phys_late is not None:
            print(f'  {name:<23} {phys_late:>10.4f}  {"N/A":>10}  {"---":<8} {"---":<8} {shrink_str}')
        elif coup_late is not None:
            print(f'  {name:<23} {"N/A":>10}  {coup_late:>10.4f}  {"---":<8} {"---":<8} {shrink_str}')

    print(f'\n=== SCOREBOARD (ep30+ mean) ===')
    print(f'  Pairs compared: {both_exist}')
    print(f'  Coupled wins: {coupled_wins}')
    print(f'  Physics wins: {physics_wins}')
    print(f'  Shrinkage detected: {len(shrink_pairs)}')
    if shrink_pairs:
        print(f'  Shrink pairs: {shrink_pairs}')

    # ── Find pairs where coupled loses ──
    losers = []
    for name in sorted(all_pairs):
        phys_traj = phys_results.get(name, {})
        coup_traj = coup_results.get(name, {})
        phys_late = late_ep_mean(phys_traj)
        coup_late = late_ep_mean(coup_traj)
        if phys_late is not None and coup_late is not None and coup_late >= phys_late:
            losers.append((name, phys_late, coup_late))

    if losers:
        print(f'\n=== PAIRS WHERE COUPLED LOSES (need re-run) ===')
        for name, pl, cl in losers:
            print(f'  {name}: phys={pl:.4f} coup={cl:.4f} (coup {(cl/pl-1)*100:+.1f}% worse)')

    # Save results
    comparison = {
        'physics_only_dir': str(phys_root),
        'coupled_dir': str(coup_root),
        'physics_trajectories': {k: {str(ep): v for ep, v in t.items()} for k, t in phys_results.items()},
        'coupled_trajectories': {k: {str(ep): v for ep, v in t.items()} for k, t in coup_results.items()},
        'coupled_wins': coupled_wins,
        'physics_wins': physics_wins,
        'losers': [(n, pl, cl) for n, pl, cl in losers] if losers else [],
        'shrink_pairs': shrink_pairs,
    }

    out_json = coup_root / 'comparison_results.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f'\nResults saved to {out_json}')

    # ── Generate comparison plot ──
    if both_exist > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Scatter: physics vs coupled
        phys_vals = []
        coup_vals = []
        names_plot = []
        for name in sorted(all_pairs):
            phys_traj = phys_results.get(name, {})
            coup_traj = coup_results.get(name, {})
            pl = late_ep_mean(phys_traj)
            cl = late_ep_mean(coup_traj)
            if pl is not None and cl is not None:
                phys_vals.append(pl)
                coup_vals.append(cl)
                names_plot.append(name)

        ax1.scatter(phys_vals, coup_vals, alpha=0.7, s=40)
        lim = max(max(phys_vals), max(coup_vals)) * 1.1
        ax1.plot([0, lim], [0, lim], 'r--', alpha=0.5, label='y=x (equal)')
        ax1.set_xlabel('Physics-only CD (ep30+ mean)')
        ax1.set_ylabel('Coupled CD (ep30+ mean)')
        ax1.set_title('Physics-only vs Coupled (ep30+ mean CD)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bar: improvement per pair
        improvements = [(cl - pl) / pl * 100 for pl, cl in zip(phys_vals, coup_vals)]
        colors = ['green' if imp < 0 else 'red' for imp in improvements]
        ax2.barh(range(len(names_plot)), improvements, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(names_plot)))
        ax2.set_yticklabels(names_plot, fontsize=6)
        ax2.set_xlabel('Coupled vs Physics-only (%)')
        ax2.set_title('Improvement (negative = coupled better)')
        ax2.axvline(x=0, color='black', linewidth=0.5)
        ax2.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        fig.savefig(str(coup_root / 'comparison_plot.png'), dpi=150)
        plt.close()
        print(f'Comparison plot saved to {coup_root / "comparison_plot.png"}')


if __name__ == '__main__':
    main()
