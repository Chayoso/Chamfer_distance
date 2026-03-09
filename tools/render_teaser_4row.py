#!/usr/bin/env python3
"""
Render teaser_4row.png: 4-row trajectory figure.
Cols: Target | Source | ep5 | ep10 | ep15 | best_ep

Rows:
  Heart → E:    source=heart,     eps=[5,10,15,38]
  Spot → C:     source=spot,      eps=[5,10,15,40]
  Bob → C:      source=bob_y90,   eps=[5,10,15,40]  (ep40: origin filter r>=3.8)
  Sphere → V:   source=isosphere, eps=[5,10,15,59]

Usage:
    conda run -n diffmpm_v2.3.0 python tools/render_teaser_4row.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import torch
import trimesh

pv.start_xvfb()

WORKSPACE = Path("/home/chayo/Desktop/Shape-morphing-binder")
SIZE = 600

try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
    FONT_SM = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
except:
    FONT = ImageFont.load_default()
    FONT_SM = FONT

ROWS = [
    {
        'label': 'Heart → E',
        'source_mesh': 'assets/heart.obj',
        'target_mesh': 'assets/E.obj',
        'output_dir': 'output/teaser_heart_to_E',
        'episodes': [5, 10, 15, 38],
    },
    {
        'label': 'Spot → C',
        'source_mesh': 'assets/spot.obj',
        'target_mesh': 'assets/C.obj',
        'output_dir': 'output/teaser_spot_to_C',
        'episodes': [5, 10, 15, 40],
        'origin_filter': {40: 3.9},  # ep40: remove artifact particles within r=3.9 from origin
    },
    {
        'label': 'Bob → C',
        'source_mesh': 'assets/bob_y90.obj',
        'target_mesh': 'assets/C.obj',
        'output_dir': 'output/teaser_bob_y90_to_C',
        'episodes': [5, 10, 15, 40],
        'bbox_filter': {40: ('y', 7.5)},  # ep40: remove artifact particles with Y > 7.5
    },
    {
        'label': 'Sphere → V',
        'source_mesh': 'assets/isosphere.obj',
        'target_mesh': 'assets/V.obj',
        'output_dir': 'output/teaser_sphere_to_V',
        'episodes': [5, 10, 15, 59],
    },
]

COLOR_SRC = [0.24, 0.68, 0.44]
COLOR_TARGET = [0.85, 0.65, 0.13]


def load_pts(path):
    p = Path(path)
    if p.suffix == '.pt':
        return torch.load(str(p), weights_only=False, map_location='cpu').numpy().astype(np.float64)
    return np.load(str(p)).astype(np.float64)


def load_mesh_pts(obj_path, n_samples=80000):
    mesh = trimesh.load(str(obj_path), force='mesh')
    pts, _ = trimesh.sample.sample_surface(mesh, n_samples)
    return pts.astype(np.float64)


def render_pyvista(points, color, point_size=8, cam_pos=None, focal=None):
    cloud = pv.PolyData(points)
    pl = pv.Plotter(off_screen=True, window_size=[SIZE, SIZE])
    pl.set_background('white')
    pl.add_points(
        cloud, color=color, point_size=point_size,
        render_points_as_spheres=True,
        ambient=0.3, diffuse=0.7, specular=0.2, specular_power=20,
    )
    if cam_pos is not None and focal is not None:
        pl.camera.position = cam_pos
        pl.camera.focal_point = focal
        pl.camera.up = (0, 0, 1)
        pl.camera.view_angle = 30
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def compute_camera(all_points_list):
    all_pts = np.vstack(all_points_list)
    center = all_pts.mean(axis=0)
    extent = (all_pts.max(axis=0) - all_pts.min(axis=0)).max()
    eye = center + np.array([extent * 2.5, 0, 0])
    return eye, center


def hstack_panels(panels, gap=4, bg=255):
    h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        if p.shape[0] < h:
            pad = np.full((h - p.shape[0], p.shape[1], 3), bg, dtype=np.uint8)
            p = np.vstack([p, pad])
        padded.append(p)
        if gap > 0:
            padded.append(np.full((h, gap, 3), bg, dtype=np.uint8))
    if gap > 0 and padded:
        padded = padded[:-1]
    return np.hstack(padded)


def vstack_panels(panels, gap=4, bg=255):
    w = max(p.shape[1] for p in panels)
    padded = []
    for p in panels:
        if p.shape[1] < w:
            pad = np.full((p.shape[0], w - p.shape[1], 3), bg, dtype=np.uint8)
            p = np.hstack([p, pad])
        padded.append(p)
        if gap > 0:
            padded.append(np.full((gap, w, 3), bg, dtype=np.uint8))
    if gap > 0 and padded:
        padded = padded[:-1]
    return np.vstack(padded)


def add_label(img_array, text, font=None):
    if font is None:
        font = FONT_SM
    pil = Image.fromarray(img_array)
    draw = ImageDraw.Draw(pil)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    label_w = tw + 20
    label_h = img_array.shape[0]
    label_img = np.full((label_h, label_w, 3), 255, dtype=np.uint8)
    label_pil = Image.fromarray(label_img)
    label_draw = ImageDraw.Draw(label_pil)
    y = (label_h - th) // 2
    label_draw.text((10, y), text, fill=(80, 80, 80), font=font)
    return np.hstack([np.array(label_pil), img_array])


def render_row(row_def, point_size=8):
    label = row_def['label']
    out_dir = WORKSPACE / row_def['output_dir']
    source_mesh = WORKSPACE / row_def['source_mesh']
    target_mesh = WORKSPACE / row_def['target_mesh']
    episodes = row_def['episodes']
    origin_filter = row_def.get('origin_filter', {})
    bbox_filter = row_def.get('bbox_filter', {})

    print(f"\n=== {label} ===")

    # Collect all point clouds for shared camera
    all_pts_list = []

    # Target mesh surface points
    tgt_pts = load_mesh_pts(target_mesh, 80000)
    all_pts_list.append(tgt_pts)

    # Source: ep000 particles (point cloud, not mesh)
    src_path = out_dir / 'ep000' / 'ep000_particles.pt'
    src_pts = load_pts(src_path)
    all_pts_list.append(src_pts)

    # Episode particles
    ep_pts = {}
    for ep in episodes:
        ep_path = out_dir / f'ep{ep:03d}' / f'ep{ep:03d}_particles.pt'
        if ep_path.exists():
            pts = load_pts(ep_path)
            if ep in origin_filter:
                r = origin_filter[ep]
                before = len(pts)
                pts = pts[np.linalg.norm(pts, axis=1) >= r]
                print(f"  ep{ep:03d}: origin filter r>={r}, removed {before - len(pts)} particles")
            if ep in bbox_filter:
                axis_name, threshold = bbox_filter[ep]
                axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis_name]
                before = len(pts)
                pts = pts[pts[:, axis_idx] <= threshold]
                print(f"  ep{ep:03d}: bbox filter {axis_name}<={threshold}, removed {before - len(pts)} particles")
            ep_pts[ep] = pts
            all_pts_list.append(pts)
        else:
            print(f"  WARNING: {ep_path} not found")

    # Shared camera
    cam_pos, focal = compute_camera(all_pts_list)

    panels = []

    # 1) Target
    print(f"  Target...", end=" ", flush=True)
    img = render_pyvista(tgt_pts, COLOR_TARGET, point_size=point_size,
                         cam_pos=cam_pos, focal=focal)
    panels.append(img)
    print("done")

    # 2) Source (point cloud)
    print(f"  Source...", end=" ", flush=True)
    img = render_pyvista(src_pts, COLOR_SRC, point_size=point_size,
                         cam_pos=cam_pos, focal=focal)
    panels.append(img)
    print("done")

    # 3) Episodes
    for ep in episodes:
        if ep in ep_pts:
            print(f"  ep{ep:03d}...", end=" ", flush=True)
            img = render_pyvista(ep_pts[ep], COLOR_SRC, point_size=point_size,
                                 cam_pos=cam_pos, focal=focal)
            panels.append(img)
            print("done")

    return hstack_panels(panels, gap=4)


def main():
    print("Rendering teaser_4row.png...")

    row_images = []
    for row_def in ROWS:
        row_img = render_row(row_def, point_size=8)
        row_img = add_label(row_img, row_def['label'], font=FONT_SM)
        row_images.append(row_img)

    final = vstack_panels(row_images, gap=8)

    outpath = WORKSPACE / "figs" / "teaser_4row.png"
    outpath.parent.mkdir(exist_ok=True)
    Image.fromarray(final).save(str(outpath), dpi=(300, 300))
    print(f"\nSaved: {outpath}")
    print(f"Size: {final.shape[1]}x{final.shape[0]} px")


if __name__ == '__main__':
    main()
