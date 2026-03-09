#!/usr/bin/env python3
"""Auto-retune: find pairs where coupled loses to physics-only at ep30+,
adjust parameters, regenerate configs, and re-run.

Strategy progression:
  Round 1: physics_weight_final=0.5, chamfer_rev_weight=0.3
  Round 2: physics_weight_final=0.6, chamfer_rev_weight=0.2
  Round 3: physics_weight_final=0.7, chamfer_rev_weight=0.15
  Round 4: physics_weight_final=0.8, chamfer_rev_weight=0.1

Each round only re-runs pairs where coupled still loses.

Usage:
    conda run -n diffmpm_v2.3.0 python tools/auto_retune.py --workers 4
"""
import sys, os, json, argparse, subprocess, time, shutil, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from itertools import permutations
from concurrent.futures import ProcessPoolExecutor, as_completed

SHAPES = ['sphere', 'bunny', 'spot', 'C', 'dragon', 'E', 'teapot', 'V', 'armadilo', 'bob']
MESH_MAP = {s: f'assets/{s}.obj' for s in SHAPES}
MESH_MAP['sphere'] = 'assets/isosphere.obj'
NUM_EPS = 40
LATE_EPS = list(range(30, 40))

# Parameter progression for re-tuning
RETUNE_SCHEDULE = [
    {'physics_weight_final': 0.5, 'chamfer_rev_weight': 0.3, 'w_chamfer': 10.0},
    {'physics_weight_final': 0.6, 'chamfer_rev_weight': 0.2, 'w_chamfer': 8.0},
    {'physics_weight_final': 0.7, 'chamfer_rev_weight': 0.15, 'w_chamfer': 6.0},
    {'physics_weight_final': 0.8, 'chamfer_rev_weight': 0.1, 'w_chamfer': 5.0},
]


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
    return float(np.sqrt(np.mean(d_fwd**2) + np.mean(d_rev**2)))


def late_ep_mean(output_root, name, tgt_pts, tgt_tree):
    """Get mean CD over ep30-39."""
    vals = []
    for ep in LATE_EPS:
        pt_file = output_root / name / f'ep{ep:03d}/ep{ep:03d}_particles.pt'
        if not pt_file.exists():
            continue
        try:
            pts = load_pts(pt_file)
            cd = compute_cd(pts, tgt_pts, tgt_tree)
            if cd is not None:
                vals.append(cd)
        except:
            pass
    return np.mean(vals) if vals else None


def run_one(name, config_path, output_root):
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = (
        '/home/chayo/anaconda3/envs/diffmpm_v2.3.0/lib/python3.10/site-packages/torch/lib:'
        + env.get('LD_LIBRARY_PATH', '')
    )
    log_file = output_root / f'{name}.log'
    t0 = time.time()
    try:
        with open(log_file, 'w') as logf:
            result = subprocess.run(
                ['python', 'run.py', '-c', str(config_path)],
                env=env, stdout=logf, stderr=subprocess.STDOUT, timeout=3600,
            )
        return name, result.returncode == 0, time.time() - t0
    except subprocess.TimeoutExpired:
        return name, False, time.time() - t0
    except Exception as e:
        return name, False, time.time() - t0


def regenerate_config(name, params, round_num):
    """Regenerate a coupled config with new parameters."""
    src, tgt = name.split('_to_')
    src_mesh = MESH_MAP[src]
    tgt_mesh = MESH_MAP[tgt]

    config_dir = Path(f'configs/pairwise_coupled_v2')
    cfg_path = config_dir / f'{name}.yaml'

    # Load existing config and update parameters
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    cfg['optimization']['physics_weight_final'] = params['physics_weight_final']
    cfg['optimization']['chamfer_rev_weight'] = params['chamfer_rev_weight']
    cfg['optimization']['w_chamfer'] = params['w_chamfer']
    cfg['output_dir'] = f'output/pairwise_coupled_v2/{name}/'

    # Add round info as comment
    with open(cfg_path, 'w') as f:
        f.write(f'# Round {round_num}: pw_final={params["physics_weight_final"]}, '
                f'rev_w={params["chamfer_rev_weight"]}, w_ch={params["w_chamfer"]}\n')
        yaml.dump(cfg, f, default_flow_style=False)

    return cfg_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--physics_dir', type=str, default='output/pairwise_physics_only')
    ap.add_argument('--coupled_dir', type=str, default='output/pairwise_coupled_v2')
    ap.add_argument('--start_round', type=int, default=0,
                    help='Start from this round (0=initial, 1=first retune, etc)')
    args = ap.parse_args()

    phys_root = Path(args.physics_dir)
    coup_root = Path(args.coupled_dir)

    all_pairs = [f'{s}_to_{t}' for s, t in permutations(SHAPES, 2)]

    # Cache targets
    tgt_cache = {}
    for name in all_pairs:
        tgt_name = name.split('_to_')[1]
        if tgt_name not in tgt_cache:
            try:
                tgt_pts = load_target(MESH_MAP[tgt_name])
                tgt_cache[tgt_name] = (tgt_pts, cKDTree(tgt_pts))
            except:
                pass

    for round_num in range(args.start_round, len(RETUNE_SCHEDULE)):
        params = RETUNE_SCHEDULE[round_num]
        print(f'\n{"="*80}')
        print(f'ROUND {round_num}: pw_final={params["physics_weight_final"]}, '
              f'rev_w={params["chamfer_rev_weight"]}, w_ch={params["w_chamfer"]}')
        print(f'{"="*80}\n')

        # Find pairs where coupled loses
        losers = []
        for name in all_pairs:
            tgt_name = name.split('_to_')[1]
            if tgt_name not in tgt_cache:
                continue

            tgt_pts, tgt_tree = tgt_cache[tgt_name]
            phys_late = late_ep_mean(phys_root, name, tgt_pts, tgt_tree)
            coup_late = late_ep_mean(coup_root, name, tgt_pts, tgt_tree)

            if phys_late is None:
                continue  # no physics baseline yet

            if coup_late is None or coup_late >= phys_late:
                losers.append((name, phys_late, coup_late))

        if not losers:
            print(f'All pairs: coupled wins! No re-tuning needed.')
            break

        print(f'Found {len(losers)} pairs where coupled loses or missing:')
        for name, pl, cl in losers[:10]:
            cl_str = f'{cl:.4f}' if cl is not None else 'N/A'
            pl_str = f'{pl:.4f}' if pl is not None else 'N/A'
            print(f'  {name}: phys={pl_str} coup={cl_str}')
        if len(losers) > 10:
            print(f'  ... and {len(losers) - 10} more')

        # Regenerate configs for losers
        print(f'\nRegenerating configs for {len(losers)} pairs...')
        configs_to_run = []
        for name, _, _ in losers:
            # Clear old output
            old_dir = coup_root / name
            if old_dir.exists():
                shutil.rmtree(old_dir)
            cfg_path = regenerate_config(name, params, round_num)
            configs_to_run.append((name, cfg_path))

        # Run experiments
        print(f'Running {len(configs_to_run)} experiments with {args.workers} workers...')
        done = 0
        failed = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_one, name, cfg_path, coup_root): name
                for name, cfg_path in configs_to_run
            }
            for future in as_completed(futures):
                fname = futures[future]
                try:
                    name, ok, elapsed = future.result()
                    if ok:
                        done += 1
                        print(f'  OK [{done}/{len(configs_to_run)}] {name} ({elapsed:.0f}s)')
                    else:
                        failed.append(name)
                        print(f'  FAIL {name} ({elapsed:.0f}s)')
                except Exception as e:
                    failed.append(fname)
                    print(f'  ERROR: {fname}: {e}')

        print(f'\nRound {round_num} complete: {done}/{len(configs_to_run)} succeeded, {len(failed)} failed')

    # Final check
    print(f'\n{"="*80}')
    print(f'FINAL COMPARISON')
    print(f'{"="*80}')

    coupled_wins = 0
    physics_wins = 0
    total = 0
    for name in sorted(all_pairs):
        tgt_name = name.split('_to_')[1]
        if tgt_name not in tgt_cache:
            continue
        tgt_pts, tgt_tree = tgt_cache[tgt_name]
        phys_late = late_ep_mean(phys_root, name, tgt_pts, tgt_tree)
        coup_late = late_ep_mean(coup_root, name, tgt_pts, tgt_tree)
        if phys_late is not None and coup_late is not None:
            total += 1
            if coup_late < phys_late:
                coupled_wins += 1
            else:
                physics_wins += 1

    print(f'  Total compared: {total}')
    print(f'  Coupled wins: {coupled_wins}')
    print(f'  Physics wins: {physics_wins}')
    if physics_wins > 0:
        print(f'\n  ⚠ {physics_wins} pairs still lose. Consider further tuning.')
    else:
        print(f'\n  ✓ Coupled wins ALL pairs at ep30+!')


if __name__ == '__main__':
    main()
