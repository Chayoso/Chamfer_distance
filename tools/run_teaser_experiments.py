#!/usr/bin/env python3
"""Run 4 teaser experiments (coupled_v3, PPC=3):
  1. teapot -> V
  2. sphere -> C
  3. spot   -> C
  4. heart  -> E

Usage:
    conda run -n diffmpm_v2.3.0 python tools/run_teaser_experiments.py
"""
import subprocess, sys, time, os
from pathlib import Path

EXPERIMENTS = [
    'teapot_to_V',
    'sphere_to_C',
    'spot_to_C',
    'heart_to_E',
]

CONFIG_DIR = Path('configs/pairwise_coupled_v3')
OUTPUT_ROOT = Path('output/pairwise_coupled_v3')

def is_done(name):
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
        capture_output=False,
        timeout=7200,  # 2hr max per experiment
    )
    elapsed = time.time() - t0
    ok = result.returncode == 0
    status = 'OK' if ok else f'FAIL (rc={result.returncode})'
    print(f'  {name}: {status} in {elapsed/60:.1f} min')
    return ok

def main():
    print(f'=== Teaser Experiments (4 pairs, coupled_v3 PPC=3) ===')
    results = {}
    for name in EXPERIMENTS:
        if is_done(name):
            print(f'[SKIP] {name} — already has ep039')
            results[name] = 'skip'
            continue
        print(f'[RUN]  {name}')
        ok = run_one(name)
        results[name] = 'ok' if ok else 'FAIL'

    print('\n=== Summary ===')
    for name, status in results.items():
        print(f'  {name}: {status}')

if __name__ == '__main__':
    main()
