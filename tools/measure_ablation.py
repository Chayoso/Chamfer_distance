#!/usr/bin/env python3
"""Measure CD metrics for ablation experiments A and D on V and E meshes."""
import sys, os, json
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

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
        raise KeyError(f"No positions in torch file {pt_path}: {list(data.keys()) if isinstance(data, dict) else type(data)}")
    else:
        d = dict(np.load(pt_path, allow_pickle=True))
        if 'positions' in d:
            return np.array(d['positions'], dtype=np.float64)
        elif 'pos' in d:
            return np.array(d['pos'], dtype=np.float64)
        else:
            raise KeyError(f"No positions in {pt_path}: {list(d.keys())}")

def load_target(mesh_path, n_samples=80000):
    import trimesh
    m = trimesh.load(mesh_path, force='mesh')
    pts, _ = trimesh.sample.sample_surface(m, n_samples)
    return np.array(pts, dtype=np.float64)

def compute_cd_metrics(pts, tgt_pts, tgt_tree=None):
    if tgt_tree is None:
        tgt_tree = cKDTree(tgt_pts)
    src_tree = cKDTree(pts)
    d_fwd, _ = tgt_tree.query(pts)
    d_rev, _ = src_tree.query(tgt_pts)
    s2t = float(np.sqrt(np.mean(d_fwd**2)))
    t2s = float(np.sqrt(np.mean(d_rev**2)))
    two_sided = float(np.sqrt(np.mean(d_fwd**2) + np.mean(d_rev**2)))
    return {'s2t': s2t, 't2s': t2s, 'two_sided': two_sided}

# ── Define experiments ──
EXPERIMENTS = {
    # Ablation experiments
    'V_expA': {'dir': 'output/ablation_V_expA', 'target': 'assets/V.obj', 'desc': 'V rev=0.0 phys=0.1'},
    'V_expD': {'dir': 'output/ablation_V_expD', 'target': 'assets/V.obj', 'desc': 'V rev=0.3 phys=0.3'},
    'E_expA': {'dir': 'output/ablation_E_expA', 'target': 'assets/E.obj', 'desc': 'E rev=0.0 phys=0.1'},
    'E_expD': {'dir': 'output/ablation_E_expD', 'target': 'assets/E.obj', 'desc': 'E rev=0.3 phys=0.3'},
    # Original bidirectional (rev=1.0, phys=0.1) for comparison
    'V_orig': {'dir': 'output/sphere_to_V_native_chamfer', 'target': 'assets/V.obj', 'desc': 'V rev=1.0 phys=0.1 (original)'},
    'E_orig': {'dir': 'output/sphere_to_E_native_chamfer', 'target': 'assets/E.obj', 'desc': 'E rev=1.0 phys=0.1 (original)'},
}

# Physics-only baselines
PHYSICS_BASELINES = {
    'V_phys': {'dir': 'output/sphere_to_V', 'target': 'assets/V.obj', 'desc': 'V physics-only'},
    'E_phys': {'dir': 'output/sphere_to_E', 'target': 'assets/E.obj', 'desc': 'E physics-only'},
}

def find_ep39_file(exp_dir):
    """Find ep039 particle file."""
    candidates = [
        Path(exp_dir) / 'ep039' / 'ep039_particles.pt',
        Path(exp_dir) / 'ep039' / 'ep039_particles.npz',
    ]
    for c in candidates:
        if c.exists():
            return c
    # Try listing
    ep39_dir = Path(exp_dir) / 'ep039'
    if ep39_dir.exists():
        files = list(ep39_dir.glob('*particles*'))
        if files:
            return files[0]
    return None

def measure_trajectory(exp_dir, target_path, label):
    """Measure CD at multiple epochs for trajectory analysis."""
    tgt_pts = load_target(target_path)
    tgt_tree = cKDTree(tgt_pts)

    results = {}
    for ep in range(0, 40):
        ep_dir = Path(exp_dir) / f'ep{ep:03d}'
        if not ep_dir.exists():
            continue
        pt_files = list(ep_dir.glob('*particles*'))
        if not pt_files:
            continue
        try:
            pts = load_pts(str(pt_files[0]))
            m = compute_cd_metrics(pts, tgt_pts, tgt_tree)
            results[ep] = m
        except Exception as e:
            print(f"  Warning: ep{ep:03d} failed: {e}")
    return results

def main():
    print("="*80)
    print("ABLATION EXPERIMENT RESULTS: rev_weight x physics_weight_final")
    print("="*80)

    all_results = {}

    # Measure physics baselines
    print("\n--- Physics-Only Baselines ---")
    for key, cfg in PHYSICS_BASELINES.items():
        f = find_ep39_file(cfg['dir'])
        if f:
            pts = load_pts(str(f))
            tgt = load_target(cfg['target'])
            m = compute_cd_metrics(pts, tgt)
            all_results[key] = m
            print(f"  {cfg['desc']}: s2t={m['s2t']:.4f}  t2s={m['t2s']:.4f}  two_sided={m['two_sided']:.4f}")
        else:
            print(f"  {cfg['desc']}: ep039 NOT FOUND in {cfg['dir']}")

    # Measure ablation experiments
    print("\n--- Ablation Experiments (ep039) ---")
    for key, cfg in EXPERIMENTS.items():
        f = find_ep39_file(cfg['dir'])
        if f:
            pts = load_pts(str(f))
            tgt = load_target(cfg['target'])
            m = compute_cd_metrics(pts, tgt)
            all_results[key] = m
            print(f"  {cfg['desc']}: s2t={m['s2t']:.4f}  t2s={m['t2s']:.4f}  two_sided={m['two_sided']:.4f}")
        else:
            print(f"  {cfg['desc']}: ep039 NOT FOUND in {cfg['dir']}")

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Experiment':<35} {'s2t':>8} {'t2s':>8} {'2-sided':>8} {'vs phys':>10}")
    print("-"*75)

    for shape in ['V', 'E']:
        phys_key = f'{shape}_phys'
        phys_2s = all_results.get(phys_key, {}).get('two_sided', 0)

        # Physics baseline
        if phys_key in all_results:
            m = all_results[phys_key]
            print(f"  {shape} physics-only               {m['s2t']:>8.4f} {m['t2s']:>8.4f} {m['two_sided']:>8.4f}   baseline")

        # Original (rev=1.0)
        orig_key = f'{shape}_orig'
        if orig_key in all_results:
            m = all_results[orig_key]
            delta = (1 - m['two_sided']/phys_2s)*100 if phys_2s > 0 else 0
            sign = '+' if delta > 0 else ''
            print(f"  {shape} rev=1.0 phys=0.1 (orig)   {m['s2t']:>8.4f} {m['t2s']:>8.4f} {m['two_sided']:>8.4f}   {sign}{delta:.1f}%")

        # Exp A (rev=0.0)
        a_key = f'{shape}_expA'
        if a_key in all_results:
            m = all_results[a_key]
            delta = (1 - m['two_sided']/phys_2s)*100 if phys_2s > 0 else 0
            sign = '+' if delta > 0 else ''
            print(f"  {shape} rev=0.0 phys=0.1 (expA)   {m['s2t']:>8.4f} {m['t2s']:>8.4f} {m['two_sided']:>8.4f}   {sign}{delta:.1f}%")

        # Exp D (rev=0.3)
        d_key = f'{shape}_expD'
        if d_key in all_results:
            m = all_results[d_key]
            delta = (1 - m['two_sided']/phys_2s)*100 if phys_2s > 0 else 0
            sign = '+' if delta > 0 else ''
            print(f"  {shape} rev=0.3 phys=0.3 (expD)   {m['s2t']:>8.4f} {m['t2s']:>8.4f} {m['two_sided']:>8.4f}   {sign}{delta:.1f}%")

        print()

    # Trajectory analysis (key epochs: 0, 10, 15, 20, 25, 30, 35, 39)
    print("\n" + "="*80)
    print("TRAJECTORY (two-sided CD at key epochs)")
    print("="*80)
    key_eps = [0, 5, 10, 15, 20, 25, 30, 35, 39]
    header = f"{'Experiment':<35}" + "".join(f"{'ep'+str(e):>8}" for e in key_eps)
    print(header)
    print("-"*len(header))

    for key, cfg in {**PHYSICS_BASELINES, **EXPERIMENTS}.items():
        traj = measure_trajectory(cfg['dir'], cfg['target'], key)
        if not traj:
            continue
        row = f"  {cfg['desc']:<33}"
        for e in key_eps:
            if e in traj:
                row += f"{traj[e]['two_sided']:>8.4f}"
            else:
                row += f"{'---':>8}"
        print(row)

    # Save JSON
    out_path = 'output/ablation_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == '__main__':
    main()
