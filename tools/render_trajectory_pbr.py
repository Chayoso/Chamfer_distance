#!/usr/bin/env python3
"""
Render trajectory frames using the same PBR sphere-mesh pipeline as render_4method_pbr.py.

Usage:
    conda run -n diffmpm_v2.3.0 python tools/render_trajectory_pbr.py --shape spot --frames 1 2 3 5 10
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import torch

WORKSPACE = Path("/home/chayo/Desktop/Shape-morphing-binder")
SIZE = 800
MAX_SPHERES = 30000

try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 72)
except Exception:
    FONT = ImageFont.load_default()

# Same green as "Ours" in 4method_pbr
COLOR_OURS = [0.20, 0.90, 0.40, 1.0]


def load_pts(path):
    p = Path(path)
    if p.suffix == '.pt':
        return torch.load(str(p), weights_only=False, map_location='cpu').numpy().astype(np.float64)
    return np.load(str(p)).astype(np.float64)


def build_sphere_mesh(points, radius, resolution=5):
    if len(points) > MAX_SPHERES:
        idx = np.random.choice(len(points), MAX_SPHERES, replace=False)
        points = points[idx]

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    sphere.compute_vertex_normals()
    n_v = len(sphere.vertices)

    base_verts = np.asarray(sphere.vertices)
    base_tris = np.asarray(sphere.triangles)

    all_verts = (base_verts[None, :, :] + points[:, None, :]).reshape(-1, 3)
    offsets = np.arange(len(points))[:, None, None] * n_v
    all_tris = (base_tris[None, :, :] + offsets).reshape(-1, 3)

    combined = o3d.geometry.TriangleMesh()
    combined.vertices = o3d.utility.Vector3dVector(all_verts)
    combined.triangles = o3d.utility.Vector3iVector(all_tris)
    combined.compute_vertex_normals()
    return combined


def make_renderer():
    r = rendering.OffscreenRenderer(SIZE, SIZE)
    r.scene.set_background([1.0, 1.0, 1.0, 1.0])
    r.scene.show_skybox(False)
    r.scene.scene.enable_indirect_light(False)
    r.scene.scene.enable_sun_light(False)
    r.scene.scene.add_directional_light('key', [1.0, 0.98, 0.95], [-1.0, 0.0, -0.2], 100000, True)
    r.scene.scene.add_directional_light('fill', [0.85, 0.88, 0.95], [-0.3, -0.8, -0.3], 70000, True)
    r.scene.scene.add_directional_light('rim', [0.9, 0.9, 1.0], [0.6, 0.3, -0.5], 40000, True)
    r.scene.scene.add_directional_light('bottom', [0.8, 0.82, 0.85], [0.0, 0.0, 1.0], 30000, True)
    return r


def render_spheres(points, color_rgba):
    extent = (points.max(axis=0) - points.min(axis=0)).max()
    n = min(len(points), MAX_SPHERES)
    radius = extent * 0.5 / (n ** (1/3))

    mesh = build_sphere_mesh(points, radius)

    renderer = make_renderer()
    mat = rendering.MaterialRecord()
    mat.shader = 'defaultLit'
    mat.base_color = list(color_rgba)
    mat.base_metallic = 0.0
    mat.base_roughness = 0.75

    renderer.scene.add_geometry('s', mesh, mat)

    center = points.mean(axis=0)
    eye = center + np.array([extent * 2.5, 0, 0])
    renderer.setup_camera(30.0, center.tolist(), eye.tolist(), [0, 0, 1])

    img = np.asarray(renderer.render_to_image()).copy()
    depth = np.asarray(renderer.render_to_depth_image()).copy()
    img[depth >= 0.9999] = 255

    renderer.scene.clear_geometry()
    del renderer
    return img


def hstack_panels(panels, gap=6, bg=255):
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


def make_header_row(labels, panel_width, gap=6):
    header_h = 100
    total_w = panel_width * len(labels) + gap * (len(labels) - 1)
    header = np.full((header_h, total_w, 3), 255, dtype=np.uint8)
    pil = Image.fromarray(header)
    draw = ImageDraw.Draw(pil)
    for i, label in enumerate(labels):
        cx = i * (panel_width + gap) + panel_width // 2
        bbox = draw.textbbox((0, 0), label, font=FONT)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((cx - tw // 2, (header_h - th) // 2), label, fill=(40, 40, 40), font=FONT)
    return np.array(pil)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="spot")
    ap.add_argument("--frames", nargs="+", type=int, default=[1, 2, 3, 5, 10])
    ap.add_argument("--method", default="rev_smooth", help="rev_smooth or physics_only")
    ap.add_argument("--color", nargs=4, type=float, default=None, help="RGBA color override")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    color = args.color if args.color else COLOR_OURS

    if args.method == "physics_only":
        base_dir = WORKSPACE / f"output/pairwise_physics_only/sphere_to_{args.shape}"
    else:
        base_dir = WORKSPACE / f"output/sphere_to_{args.shape}_rev_smooth"

    labels = [f"Ep {f}" for f in args.frames]
    panels = []

    for f in args.frames:
        pt_path = base_dir / f"ep{f:03d}/ep{f:03d}_particles.pt"
        if not pt_path.exists():
            print(f"  SKIP ep{f:03d}: not found")
            panels.append(np.full((SIZE, SIZE, 3), 255, dtype=np.uint8))
            continue

        pts = load_pts(pt_path)
        print(f"  [{args.shape}] ep{f:03d}: {len(pts)} pts ...", end=" ", flush=True)
        img = render_spheres(pts, color)
        panels.append(img)
        print("done")

    row = hstack_panels(panels, gap=6)
    header = make_header_row(labels, SIZE, gap=6)

    if header.shape[1] < row.shape[1]:
        header = np.hstack([header, np.full((header.shape[0], row.shape[1] - header.shape[1], 3), 255, dtype=np.uint8)])
    elif header.shape[1] > row.shape[1]:
        header = header[:, :row.shape[1]]

    final = vstack_panels([header, row], gap=4)

    outname = args.output or f"figs/trajectory_pbr_{args.shape}.png"
    outpath = WORKSPACE / outname
    outpath.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(final).save(str(outpath))
    print(f"\nSaved: {outpath}")
    print(f"Size: {final.shape[1]}x{final.shape[0]} px")


if __name__ == '__main__':
    main()
