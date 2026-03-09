#!/usr/bin/env python3
"""Run pairwise coupled-rev experiments in parallel (N workers).

Usage:
    conda run -n diffmpm_v2.3.0 python tools/run_pairwise_parallel.py --workers 4
"""
import subprocess, sys, os, time, argparse
from pathlib import Path
from itertools import permutations
from concurrent.futures import ProcessPoolExecutor, as_completed

SHAPES = ['bunny', 'bob', 'cow', 'spot', 'dragon', 'armadilo']
CONFIG_DIR = Path('configs/pairwise_coupled')
OUTPUT_ROOT = Path('output/pairwise_coupled')


def is_done(name):
    return (OUTPUT_ROOT / name / 'ep039' / 'ep039_particles.pt').exists()


def run_one(name):
    cfg = CONFIG_DIR / f'{name}.yaml'
    if not cfg.exists():
        return name, False, 0, 'CONFIG NOT FOUND'

    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = (
        '/home/chayo/anaconda3/envs/diffmpm_v2.3.0/lib/python3.10/site-packages/torch/lib:'
        + env.get('LD_LIBRARY_PATH', '')
    )

    log_file = OUTPUT_ROOT / f'{name}.log'
    t0 = time.time()
    try:
        with open(log_file, 'w') as logf:
            result = subprocess.run(
                ['python', 'run.py', '-c', str(cfg)],
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=3600,
            )
        elapsed = time.time() - t0
        return name, result.returncode == 0, elapsed, ''
    except subprocess.TimeoutExpired:
        return name, False, time.time() - t0, 'TIMEOUT'
    except Exception as e:
        return name, False, time.time() - t0, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    pairs = [f'{s}_to_{t}' for s, t in permutations(SHAPES, 2)]

    todo = [p for p in pairs if not is_done(p)]
    skipped = len(pairs) - len(todo)

    print(f'=== Pairwise Parallel Runner ===')
    print(f'  Total pairs: {len(pairs)}')
    print(f'  Already done: {skipped}')
    print(f'  To run: {len(todo)}')
    print(f'  Workers: {args.workers}')
    print()

    if not todo:
        print('All experiments already complete!')
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    done = 0
    failed = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_one, name): name for name in todo}
        for future in as_completed(futures):
            name = futures[future]
            try:
                name, ok, elapsed, err = future.result()
                if ok:
                    done += 1
                    print(f'  OK [{done + skipped}/{len(pairs)}] {name} ({elapsed:.0f}s)')
                else:
                    failed.append(name)
                    print(f'  FAIL [{done + skipped}/{len(pairs)}] {name} ({elapsed:.0f}s) {err}')
            except Exception as e:
                failed.append(name)
                print(f'  ERROR: {name}: {e}')

    print(f'\n=== Summary ===')
    print(f'  Completed: {done + skipped}/{len(pairs)}')
    print(f'  Failed: {len(failed)}')
    if failed:
        print(f'  Failed: {failed}')


if __name__ == '__main__':
    main()
