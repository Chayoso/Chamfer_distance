#!/usr/bin/env python3
"""Run all 30 pairwise coupled-rev experiments sequentially.
Skips experiments that already have ep039 output.

Usage:
    conda run -n diffmpm_v2.3.0 python tools/run_pairwise_all.py
"""
import subprocess, sys, time, os
from pathlib import Path
from itertools import permutations

SHAPES = ['bunny', 'bob', 'cow', 'spot', 'dragon', 'armadilo']
CONFIG_DIR = Path('configs/pairwise_coupled')
OUTPUT_ROOT = Path('output/pairwise_coupled')

def is_done(name):
    """Check if experiment already has ep039."""
    ep39 = OUTPUT_ROOT / name / 'ep039' / 'ep039_particles.pt'
    return ep39.exists()

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
        capture_output=False,  # let output stream to console
        timeout=3600,  # 1hr max per experiment
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

    print(f'=== Pairwise Coupled-Rev Experiments ({total} pairs) ===\n')

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

    print(f'\n=== Summary ===')
    print(f'  Total: {total}')
    print(f'  Done: {done} (skipped: {skipped})')
    print(f'  Failed: {len(failed)}')
    if failed:
        print(f'  Failed list: {failed}')

if __name__ == '__main__':
    main()
