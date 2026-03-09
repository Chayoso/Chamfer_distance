#!/usr/bin/env python3
"""Run all 30 pairwise rev_smooth experiments sequentially.
Skips experiments that already have ep039 output.

Usage:
    conda run -n diffmpm_v2.3.0 python tools/run_pairwise_rev_smooth.py
"""
import subprocess, sys, time, os
from pathlib import Path
from itertools import permutations

SHAPES = ['sphere', 'bunny', 'dragon', 'bob', 'spot', 'teapot']
CONFIG_DIR = Path('configs/pairwise_rev_smooth')
OUTPUT_ROOT = Path('output/pairwise_rev_smooth')

def is_done(name):
    """Check if experiment already has ep039 (check both pairwise and legacy dirs)."""
    # Check pairwise output dir
    ep39 = OUTPUT_ROOT / name / 'ep039' / 'ep039_particles.pt'
    if ep39.exists():
        return True
    # Check legacy sphere_to_X_rev_smooth output dir
    if name.startswith('sphere_to_'):
        tgt = name.replace('sphere_to_', '')
        legacy = Path(f'output/sphere_to_{tgt}_rev_smooth/ep039/ep039_particles.pt')
        if legacy.exists():
            return True
    return False

def run_one(name):
    cfg = CONFIG_DIR / f'{name}.yaml'
    if not cfg.exists():
        print(f'  CONFIG NOT FOUND: {cfg}')
        return False

    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = (
        '/home/chayo/anaconda3/envs/diffmpm_v2.3.0/lib/python3.10/site-packages/torch/lib:'
        + env.get('LD_LIBRARY_PATH', '')
    )

    t0 = time.time()
    result = subprocess.run(
        ['python', 'run.py', '-c', str(cfg)],
        env=env,
        capture_output=False,
        timeout=7200,  # 2hr max per experiment
    )
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f'  DONE: {name} ({elapsed:.0f}s)')
        return True
    else:
        print(f'  FAILED: {name} (rc={result.returncode}, {elapsed:.0f}s)')
        return False

def main():
    pairs = [(s, t) for s, t in permutations(SHAPES, 2)]
    total = len(pairs)

    print(f'=== Pairwise Rev-Smooth Experiments ({total} pairs) ===')
    print(f'Config dir: {CONFIG_DIR}')
    print(f'Output dir: {OUTPUT_ROOT}\n')

    done = 0
    skipped = 0
    failed = []

    for i, (src, tgt) in enumerate(pairs):
        name = f'{src}_to_{tgt}'
        if is_done(name):
            print(f'[{i+1}/{total}] SKIP (already done): {name}')
            skipped += 1
            done += 1
            continue

        print(f'\n[{i+1}/{total}] Running: {name}')
        try:
            ok = run_one(name)
            if ok:
                done += 1
            else:
                failed.append(name)
        except subprocess.TimeoutExpired:
            print(f'  TIMEOUT: {name}')
            failed.append(name)
        except KeyboardInterrupt:
            print(f'\n\nInterrupted at {name}. {done} done, {len(failed)} failed.')
            break

    print(f'\n{"="*50}')
    print(f'=== Summary ===')
    print(f'  Total: {total}')
    print(f'  Done: {done} (skipped: {skipped})')
    print(f'  Failed: {len(failed)}')
    if failed:
        print(f'  Failed list: {failed}')

if __name__ == '__main__':
    main()
