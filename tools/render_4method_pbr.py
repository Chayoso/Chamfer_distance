#!/usr/bin/env python3
"""
Render 4-method comparison using Open3D sphere-mesh PBR (matte).
Target | DCO | DCD | Physics | Ours

Usage:
    conda run -n diffmpm_v2.3.0 python tools/render_4method_pbr.py
    conda run -n diffmpm_v2.3.0 python tools/render_4method_pbr.py --shapes bunny dragon
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
import trimesh

WORKSPACE = Path("/home/chayo/Desktop/Shape-morphing-binder")
SIZE = 800
MAX_SPHERES = 30000  # max points for sphere mesh (avoid OOM)

try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 72)
except Exception:
    FONT = ImageFont.load_default()

SHAPES = {
    'bunny':  {'mesh': 'assets/bunny.obj'},
    'dragon': {'mesh': 'assets/dragon.obj'},
    'bob':    {'mesh': 'assets/bob.obj'},
    'spot':   {'mesh': 'assets/spot.obj'},
    'teapot': {'mesh': 'assets/teapot.obj'},
}

METHOD_ORDER = ['DCO', 'DCD', 'Physics', 'Ours']

# Matte colors (RGBA for Open3D) — vivid
COLORS = {
    'Target':  [1.00, 0.82, 0.10, 1.0],
    'Physics': [0.30, 0.60, 1.00, 1.0],
    'Ours':    [0.20, 0.90, 0.40, 1.0],
    'DCO':     [1.00, 0.35, 0.30, 1.0],
    'DCD':     [0.70, 0.40, 1.00, 1.0],
}


def load_pts(path):
    p = Path(path)
    if p.suffix == '.pt':
        return torch.load(str(p), weights_only=False, map_location='cpu').numpy().astype(np.float64)
    return np.load(str(p)).astype(np.float64)


def load_mesh_pts(obj_path, n_samples=80000):
    mesh = trimesh.load(str(obj_path), force='mesh')
    pts, _ = trimesh.sample.sample_surface(mesh, n_samples)
    return pts.astype(np.float64)


def get_particle_paths(shape, ep=39):
    return {
        'Physics': WORKSPACE / f'output/pairwise_physics_only/sphere_to_{shape}/ep{ep:03d}/ep{ep:03d}_particles.pt',
        'Ours':    WORKSPACE / f'output/sphere_to_{shape}_rev_smooth/ep{ep:03d}/ep{ep:03d}_particles.pt',
        'DCO':     WORKSPACE / f'output/dcd_ppc4/{shape}/dco_standard_final.npy',
        'DCD':     WORKSPACE / f'output/dcd_ppc4/{shape}/dcd_final.npy',
    }


def build_sphere_mesh(points, radius, resolution=5):
    """Create merged sphere mesh at each point location (vectorized)."""
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
    """Create OffscreenRenderer with 4-point studio lighting."""
    r = rendering.OffscreenRenderer(SIZE, SIZE)
    r.scene.set_background([1.0, 1.0, 1.0, 1.0])
    r.scene.show_skybox(False)
    r.scene.scene.enable_indirect_light(False)
    r.scene.scene.enable_sun_light(False)
    # Key: from camera direction (front-lit)
    r.scene.scene.add_directional_light('key', [1.0, 0.98, 0.95], [-1.0, 0.0, -0.2], 100000, True)
    # Fill: from left-above
    r.scene.scene.add_directional_light('fill', [0.85, 0.88, 0.95], [-0.3, -0.8, -0.3], 70000, True)
    # Rim: from behind-above
    r.scene.scene.add_directional_light('rim', [0.9, 0.9, 1.0], [0.6, 0.3, -0.5], 40000, True)
    # Bottom fill: reduce under-chin shadows
    r.scene.scene.add_directional_light('bottom', [0.8, 0.82, 0.85], [0.0, 0.0, 1.0], 30000, True)
    return r


def render_spheres(points, color_rgba, width=800, height=800):
    """Render point cloud as sphere meshes with PBR matte material."""
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


def render_one_shape(shape, info, ep=39):
    mesh_path = WORKSPACE / info['mesh']
    paths = get_particle_paths(shape, ep=ep)

    missing = [m for m, p in paths.items() if not p.exists()]
    if missing:
        print(f"  WARNING: missing {missing} for {shape}")
        return None

    tgt_pts = load_mesh_pts(mesh_path, 80000)

    panels = []

    # Target
    print(f"  [{shape}] Target...", end=" ", flush=True)
    img = render_spheres(tgt_pts, COLORS['Target'])
    panels.append(img)
    print("done")

    # Methods
    for method in METHOD_ORDER:
        print(f"  [{shape}] {method}...", end=" ", flush=True)
        pts = load_pts(paths[method])
        img = render_spheres(pts, COLORS[method])
        panels.append(img)
        print("done")

    return hstack_panels(panels, gap=6)


def make_header_row(panel_width, num_panels, gap=6):
    labels = ['Target', 'DCO', 'DCD', 'Physics', 'Ours']
    header_h = 100
    total_w = panel_width * num_panels + gap * (num_panels - 1)
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
    ap.add_argument("--shapes", nargs="+", default=list(SHAPES.keys()))
    ap.add_argument("--ep", type=int, default=None)
    ap.add_argument("--output", default="output/4method_comparison_pbr.png")
    args = ap.parse_args()

    rows = []
    for shape in args.shapes:
        if shape not in SHAPES:
            continue
        print(f"\n{'='*50}")
        print(f"  Rendering {shape.upper()} (sphere mesh PBR)")
        print(f"{'='*50}")
        ep = args.ep if args.ep is not None else 39
        row = render_one_shape(shape, SHAPES[shape], ep=ep)
        if row is not None:
            rows.append(row)

    if rows:
        header = make_header_row(SIZE, 5, gap=6)
        row_w = rows[0].shape[1]
        if header.shape[1] < row_w:
            header = np.hstack([header,
                np.full((header.shape[0], row_w - header.shape[1], 3), 255, dtype=np.uint8)])
        elif header.shape[1] > row_w:
            header = header[:, :row_w]

        final = vstack_panels([header] + rows, gap=4)
        outpath = WORKSPACE / args.output
        outpath.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(final).save(str(outpath))
        print(f"\nSaved: {outpath}")
        print(f"Size: {final.shape[1]}x{final.shape[0]} pixels")


if __name__ == "__main__":
    main()
