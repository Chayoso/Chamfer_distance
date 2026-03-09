#!/usr/bin/env python3
"""
Cross-section rendering using the SAME PBR sphere-mesh pipeline as render_4method_pbr.py.
Slices particles at a given plane, then renders with Open3D PBR matte material.

Usage:
    conda run -n diffmpm_v2.3.0 python tools/render_cross_section_pbr.py --shape spot --center 0.9
    conda run -n diffmpm_v2.3.0 python tools/render_cross_section_pbr.py --shape spot --center 0.9 --thickness 2.0
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
MAX_SPHERES = 30000

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
METHODS = ['DCO', 'DCD', 'Physics', 'Ours']

# Same PBR colors as render_4method_pbr.py
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


def get_paths(shape, ep=39):
    return {
        'Target':  WORKSPACE / f'assets/{shape}.obj',
        'DCO':     WORKSPACE / f'output/dcd_ppc4/{shape}/dco_standard_final.npy',
        'DCD':     WORKSPACE / f'output/dcd_ppc4/{shape}/dcd_final.npy',
        'Physics': WORKSPACE / f'output/pairwise_physics_only/sphere_to_{shape}/ep{ep:03d}/ep{ep:03d}_particles.pt',
        'Ours':    WORKSPACE / f'output/sphere_to_{shape}_rev_smooth/ep{ep:03d}/ep{ep:03d}_particles.pt',
    }


def build_sphere_mesh(points, radius, resolution=5):
    """Create merged sphere mesh at each point location."""
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


def make_renderer_topdown():
    """Create OffscreenRenderer with lighting adapted for top-down view."""
    r = rendering.OffscreenRenderer(SIZE, SIZE)
    r.scene.set_background([1.0, 1.0, 1.0, 1.0])
    r.scene.show_skybox(False)
    r.scene.scene.enable_indirect_light(False)
    r.scene.scene.enable_sun_light(False)
    # Key: from above (camera direction)
    r.scene.scene.add_directional_light('key', [1.0, 0.98, 0.95], [0.0, 0.0, -1.0], 100000, True)
    # Fill: from front-left
    r.scene.scene.add_directional_light('fill', [0.85, 0.88, 0.95], [-0.5, -0.7, -0.3], 70000, True)
    # Rim: from back-right
    r.scene.scene.add_directional_light('rim', [0.9, 0.9, 1.0], [0.5, 0.5, -0.2], 40000, True)
    # Bottom fill: from below
    r.scene.scene.add_directional_light('bottom', [0.8, 0.82, 0.85], [0.0, 0.0, 1.0], 30000, True)
    return r


def render_spheres_topdown(points, color_rgba):
    """Render point cloud as sphere meshes, viewed from above (looking down z-axis)."""
    extent = (points.max(axis=0) - points.min(axis=0)).max()
    n = min(len(points), MAX_SPHERES)
    radius = extent * 0.35 / (n ** (1/3))

    mesh = build_sphere_mesh(points, radius)

    renderer = make_renderer_topdown()
    mat = rendering.MaterialRecord()
    mat.shader = 'defaultLit'
    mat.base_color = list(color_rgba)
    mat.base_metallic = 0.0
    mat.base_roughness = 0.75

    renderer.scene.add_geometry('s', mesh, mat)

    center = points.mean(axis=0)
    # Camera looking down z-axis from above
    eye = center + np.array([0, 0, extent * 2.5])
    # Up vector: -x so that y goes horizontal (body length), x goes vertical
    renderer.setup_camera(30.0, center.tolist(), eye.tolist(), [-1, 0, 0])

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


def make_header_row(panel_width, num_panels, gap=6):
    labels = METHODS
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
    ap.add_argument("--shape", default="spot")
    ap.add_argument("--shapes", nargs="+", default=None)
    ap.add_argument("--axis", default="z", choices=["x", "y", "z"])
    ap.add_argument("--center", type=float, default=0.9)
    ap.add_argument("--thickness", type=float, default=1.5)
    ap.add_argument("--ep", type=int, default=39)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    shapes = args.shapes if args.shapes else [args.shape]
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[args.axis]

    rows = []
    for shape in shapes:
        if shape not in SHAPES:
            print(f"Unknown shape: {shape}")
            continue

        print(f"\n{'='*50}")
        print(f"  {shape.upper()} cross-section: {args.axis}={args.center}, thickness={args.thickness}")
        print(f"{'='*50}")

        paths = get_paths(shape, ep=args.ep)

        # Load all method points
        method_pts = {}
        for method in METHODS:
            if method == 'Target':
                method_pts[method] = load_mesh_pts(paths['Target'], 80000)
            else:
                if not paths[method].exists():
                    print(f"  SKIP {method}: {paths[method]} not found")
                    continue
                method_pts[method] = load_pts(paths[method])

        available = [m for m in METHODS if m in method_pts]

        # Slice and render each method
        panels = []
        for method in available:
            pts = method_pts[method]
            mask = np.abs(pts[:, axis_idx] - args.center) < args.thickness / 2
            sliced = pts[mask]
            print(f"  [{shape}] {method}: {len(sliced)}/{len(pts)} pts in slice ...", end=" ", flush=True)

            if len(sliced) < 10:
                print("TOO FEW, blank panel")
                panels.append(np.full((SIZE, SIZE, 3), 255, dtype=np.uint8))
                continue

            img = render_spheres_topdown(sliced, COLORS[method])
            panels.append(img)
            print("done")

        row = hstack_panels(panels, gap=6)
        rows.append(row)

    if rows:
        header = make_header_row(SIZE, len(METHODS), gap=6)
        row_w = rows[0].shape[1]
        if header.shape[1] < row_w:
            header = np.hstack([header,
                np.full((header.shape[0], row_w - header.shape[1], 3), 255, dtype=np.uint8)])
        elif header.shape[1] > row_w:
            header = header[:, :row_w]

        final = vstack_panels([header] + rows, gap=4)
        outname = args.output or f"figs/cross_section_pbr_{args.axis}{args.center}.png"
        outpath = WORKSPACE / outname
        outpath.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(final).save(str(outpath))
        print(f"\nSaved: {outpath}")
        print(f"Size: {final.shape[1]}x{final.shape[0]} px")


if __name__ == '__main__':
    main()
