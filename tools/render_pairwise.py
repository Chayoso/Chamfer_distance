#!/usr/bin/env python3
"""Render pairwise experiment results: ep000, ep019, ep039, target for each pair.
Creates per-pair comparison strips + per-pair videos.

Usage:
    conda run -n diffmpm_v2.3.0 python tools/render_pairwise.py
    conda run -n diffmpm_v2.3.0 python tools/render_pairwise.py --pair bunny_to_bob
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, subprocess
import numpy as np
from PIL import Image
from pathlib import Path
from itertools import permutations

from tools.render_eccv_figures import (
    render_pointcloud, load_pts, load_mesh_pts, hstack_panels, C_TARGET
)
from tools.render_zaxis_view import render_from_axis

SHAPES = ['bunny', 'bob', 'cow', 'spot', 'dragon', 'armadilo']
MESH_MAP = {s: f'assets/{s}.obj' for s in SHAPES}
OUTPUT_ROOT = Path('output/pairwise_coupled')
SIZE = 500
# Color for morphed particles (0-1 float!)
C_MORPH = np.array([0.30, 0.52, 0.74])  # steel blue


def render_pair(name):
    """Render all 40 episodes for a pair."""
    exp_dir = OUTPUT_ROOT / name
    tgt_name = name.split('_to_')[1]
    tgt_mesh = MESH_MAP[tgt_name]

    renders_dir = exp_dir / 'renders'
    renders_dir.mkdir(exist_ok=True)

    # Render target once
    tgt_path = renders_dir / 'target.png'
    if not tgt_path.exists():
        tgt_pts = load_mesh_pts(tgt_mesh, 80000)
        tgt_img = render_from_axis(tgt_pts, C_TARGET, 'x', None, SIZE, rot_x=-90)
        Image.fromarray(tgt_img).save(tgt_path)

    # Render each episode
    rendered = 0
    for ep in range(40):
        out_path = renders_dir / f'ep{ep:03d}.png'
        if out_path.exists():
            rendered += 1
            continue
        pt_file = exp_dir / f'ep{ep:03d}/ep{ep:03d}_particles.pt'
        if not pt_file.exists():
            continue
        pts = load_pts(pt_file)
        img = render_from_axis(pts, C_MORPH, 'x', None, SIZE, rot_x=-90)
        Image.fromarray(img).save(out_path)
        rendered += 1

    # Make video
    vid_path = renders_dir / f'{name}.mp4'
    if not vid_path.exists() and rendered >= 30:
        cmd = [
            'ffmpeg', '-y', '-framerate', '12',
            '-i', str(renders_dir / 'ep%03d.png'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
            str(vid_path)
        ]
        subprocess.run(cmd, capture_output=True)

    return rendered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', type=str, default=None, help='Specific pair to render')
    args = ap.parse_args()

    if args.pair:
        pairs = [args.pair]
    else:
        pairs = [f'{s}_to_{t}' for s, t in permutations(SHAPES, 2)]

    for name in pairs:
        ep39 = OUTPUT_ROOT / name / 'ep039/ep039_particles.pt'
        if not ep39.exists():
            continue
        print(f'Rendering {name}...', end=' ', flush=True)
        n = render_pair(name)
        print(f'{n} frames')

    print('Done!')


if __name__ == '__main__':
    main()
