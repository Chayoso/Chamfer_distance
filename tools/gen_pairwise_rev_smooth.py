#!/usr/bin/env python3
"""Generate pairwise rev_smooth configs for all non-sphere shape pairs.
Uses sphere_to_bunny_rev_smooth.yaml as template.

Usage:
    python tools/gen_pairwise_rev_smooth.py
"""
from pathlib import Path
from itertools import permutations
import yaml

SHAPES = ['sphere', 'bunny', 'dragon', 'bob', 'spot', 'teapot']
TEMPLATE = Path('configs/sphere_to_bunny_rev_smooth.yaml')
OUT_DIR = Path('configs/pairwise_rev_smooth')

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(TEMPLATE) as f:
        tpl = yaml.safe_load(f)

    count = 0
    for src, tgt in permutations(SHAPES, 2):
        name = f'{src}_to_{tgt}'
        cfg = dict(tpl)  # shallow copy top-level

        # Update paths (sphere uses isosphere.obj)
        src_mesh = 'isosphere' if src == 'sphere' else src
        tgt_mesh = 'isosphere' if tgt == 'sphere' else tgt
        cfg['input_mesh_path'] = f'assets/{src_mesh}.obj'
        cfg['target_mesh_path'] = f'assets/{tgt_mesh}.obj'
        cfg['output_dir'] = f'output/pairwise_rev_smooth/{name}/'

        out_path = OUT_DIR / f'{name}.yaml'
        with open(out_path, 'w') as f:
            f.write(f'# {src} -> {tgt} + Native Chamfer with rev_grad smoothing\n')
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        count += 1
        print(f'  Generated: {out_path}')

    print(f'\nTotal: {count} configs in {OUT_DIR}/')

if __name__ == '__main__':
    main()
