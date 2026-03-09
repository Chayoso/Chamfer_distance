#!/usr/bin/env python3
"""
Render ablation comparison: Physics-only | Original(rev=1.0) | ExpA(rev=0.0) | ExpD(rev=0.3) | Target
Creates per-frame PNGs and ffmpeg videos at 12fps.

Usage:
    conda run -n diffmpm_v2.3.0 python tools/render_ablation.py
    conda run -n diffmpm_v2.3.0 python tools/render_ablation.py --shapes V
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, subprocess, argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from scipy.spatial import cKDTree

from tools.render_eccv_figures import (
    render_pointcloud, load_pts, load_mesh_pts, hstack_panels,
    C_PHYSICS, C_PP, C_TARGET
)
from tools.render_zaxis_view import render_from_axis

WORKSPACE = Path("/home/chayo/Desktop/Shape-morphing-binder")

try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    FONT_SM = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
except:
    FONT = ImageFont.load_default()
    FONT_SM = FONT

SIZE = 500
NUM_EPS = 40

# Colors for each experiment (0-1 float, NOT 0-255!)
C_ORIG = np.array([70/255, 130/255, 180/255])   # steel blue (original chamfer)
C_EXPA = np.array([200/255, 80/255, 80/255])    # red (expA - unidirectional)
C_EXPD = np.array([60/255, 160/255, 80/255])    # green (expD - rev=0.3)

ABLATION_SHAPES = {
    'V': {
        'mesh': 'assets/V.obj',
        'phys_dir': 'output/sphere_to_V',
        'orig_dir': 'output/sphere_to_V_native_chamfer',
        'expA_dir': 'output/ablation_V_expA',
        'expD_dir': 'output/ablation_V_expD',
    },
    'E': {
        'mesh': 'assets/E.obj',
        'phys_dir': 'output/sphere_to_E',
        'orig_dir': 'output/sphere_to_E_native_chamfer',
        'expA_dir': 'output/ablation_E_expA',
        'expD_dir': 'output/ablation_E_expD',
    },
}


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


def measure_cd_trajectory(exp_dir, tgt_pts, tgt_tree):
    """Measure CD for all episodes in a directory."""
    cd_data = {}
    for ep in range(NUM_EPS):
        pt_file = exp_dir / f"ep{ep:03d}/ep{ep:03d}_particles.pt"
        if not pt_file.exists():
            continue
        pts = load_pts(pt_file).astype(np.float64)
        cd_data[ep] = compute_cd_metrics(pts, tgt_pts, tgt_tree)
    return cd_data


def add_label(img_np, title, cd_val, color_title=(40, 40, 40), color_cd=(200, 50, 50)):
    """Add title at top and CD value at bottom."""
    img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), title, font=FONT)
    draw.text(((img.width - (bbox[2] - bbox[0])) // 2, 6), title, fill=color_title, font=FONT)
    if cd_val is not None:
        cd_text = f"CD={cd_val:.4f}"
        bbox2 = draw.textbbox((0, 0), cd_text, font=FONT_SM)
        draw.text(((img.width - (bbox2[2] - bbox2[0])) // 2, img.height - 22),
                  cd_text, fill=color_cd, font=FONT_SM)
    return np.array(img)


def render_shape(shape_name, info):
    """Render all frames for a shape's ablation study."""
    import trimesh

    phys_dir = WORKSPACE / info['phys_dir']
    orig_dir = WORKSPACE / info['orig_dir']
    expA_dir = WORKSPACE / info['expA_dir']
    expD_dir = WORKSPACE / info['expD_dir']
    mesh_path = WORKSPACE / info['mesh']

    out_dir = WORKSPACE / 'output' / f'ablation_{shape_name}_renders'
    frames_dir = out_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Load target
    tgt_mesh = trimesh.load(str(mesh_path), force='mesh')
    tgt_pts, _ = trimesh.sample.sample_surface(tgt_mesh, 80000)
    tgt_pts = np.array(tgt_pts, dtype=np.float64)
    tgt_tree = cKDTree(tgt_pts)

    # Measure CDs
    print(f"  Measuring CD trajectories...")
    cd_phys = measure_cd_trajectory(phys_dir, tgt_pts, tgt_tree)
    cd_orig = measure_cd_trajectory(orig_dir, tgt_pts, tgt_tree)
    cd_expA = measure_cd_trajectory(expA_dir, tgt_pts, tgt_tree)
    cd_expD = measure_cd_trajectory(expD_dir, tgt_pts, tgt_tree)

    # Save CD data
    cd_all = {'phys': {str(k): v for k, v in cd_phys.items()},
              'orig': {str(k): v for k, v in cd_orig.items()},
              'expA': {str(k): v for k, v in cd_expA.items()},
              'expD': {str(k): v for k, v in cd_expD.items()}}
    with open(out_dir / 'cd_trajectories.json', 'w') as f:
        json.dump(cd_all, f, indent=2)
    print(f"  CD data saved to {out_dir / 'cd_trajectories.json'}")

    # Render target panel
    tgt_render_pts = load_mesh_pts(mesh_path, 80000)
    tgt_img = render_from_axis(tgt_render_pts, C_TARGET, 'x', None, SIZE, rot_x=-90)
    tgt_panel = add_label(tgt_img, "Target", None, color_cd=(100, 100, 100))

    # Render each episode
    rendered = 0
    for ep in range(NUM_EPS):
        frame_path = frames_dir / f"frame_{ep:03d}.png"
        if frame_path.exists():
            rendered += 1
            continue

        # Check all files exist
        phys_pt = phys_dir / f"ep{ep:03d}/ep{ep:03d}_particles.pt"
        orig_pt = orig_dir / f"ep{ep:03d}/ep{ep:03d}_particles.pt"
        expA_pt = expA_dir / f"ep{ep:03d}/ep{ep:03d}_particles.pt"
        expD_pt = expD_dir / f"ep{ep:03d}/ep{ep:03d}_particles.pt"

        if not all(p.exists() for p in [phys_pt, orig_pt, expA_pt, expD_pt]):
            continue

        print(f"  [ep{ep:02d}]", end=" ", flush=True)

        # Get CD values
        p_cd = cd_phys.get(ep, {}).get('two_sided', 0)
        o_cd = cd_orig.get(ep, {}).get('two_sided', 0)
        a_cd = cd_expA.get(ep, {}).get('two_sided', 0)
        d_cd = cd_expD.get(ep, {}).get('two_sided', 0)

        # Render each
        img_p = render_from_axis(load_pts(phys_pt), C_PHYSICS, 'x', None, SIZE, rot_x=-90)
        img_p = add_label(img_p, f"Physics-only ep{ep}", p_cd)

        img_o = render_from_axis(load_pts(orig_pt), C_ORIG, 'x', None, SIZE, rot_x=-90)
        img_o = add_label(img_o, f"rev=1.0 ep{ep}", o_cd)

        img_a = render_from_axis(load_pts(expA_pt), C_EXPA, 'x', None, SIZE, rot_x=-90)
        img_a = add_label(img_a, f"rev=0.0 ep{ep}", a_cd)

        img_d = render_from_axis(load_pts(expD_pt), C_EXPD, 'x', None, SIZE, rot_x=-90)
        img_d = add_label(img_d, f"rev=0.3 ep{ep}", d_cd)

        strip = hstack_panels([img_p, img_o, img_a, img_d, tgt_panel], gap=3)
        Image.fromarray(strip).save(frame_path)
        rendered += 1
        print("done", flush=True)

    print(f"  {rendered} frames ready")
    return frames_dir, rendered


def make_video(frames_dir, output_path, fps=12):
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  Video saved: {output_path}")
    else:
        print(f"  Video FAILED: {result.stderr[:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", default=list(ABLATION_SHAPES.keys()))
    args = ap.parse_args()

    for shape_name in args.shapes:
        if shape_name not in ABLATION_SHAPES:
            print(f"Unknown shape: {shape_name}")
            continue
        info = ABLATION_SHAPES[shape_name]
        print(f"\n{'='*60}")
        print(f"Rendering ablation: {shape_name}")
        print(f"{'='*60}")

        frames_dir, n_frames = render_shape(shape_name, info)

        if n_frames > 0:
            vid_path = WORKSPACE / 'output' / f'ablation_{shape_name}_renders' / f'{shape_name}_ablation.mp4'
            make_video(frames_dir, vid_path)

    print("\nDone!")


if __name__ == '__main__':
    main()
