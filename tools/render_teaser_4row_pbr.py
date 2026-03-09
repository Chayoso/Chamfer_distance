#!/usr/bin/env python3
"""
Render teaser_4row_pbr.png: 4-row trajectory with Open3D sphere-mesh PBR.
Cols: Source (mesh→ptcloud, gold) | ep5 | ep10 | ep15 | best_ep

Usage:
    conda run -n diffmpm_v2.3.0 python tools/render_teaser_4row_pbr.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
RADIUS_SCALE = 0.65  # bigger than default 0.5 → larger particles

try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
    FONT_SM = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
except Exception:
    FONT = ImageFont.load_default()
    FONT_SM = FONT

COLOR_SOURCE = [1.00, 0.82, 0.10, 1.0]   # gold for source mesh

# Per-row episode colors (vivid, saturated)
ROW_COLORS = {
    'Heart → E':  [0.15, 0.45, 0.90, 1.0],   # cobalt blue
    'Spot → C':   [0.00, 0.75, 0.55, 1.0],    # teal
    'Bob → C':    [0.90, 0.25, 0.30, 1.0],    # crimson
    'Sphere → V': [0.75, 0.60, 1.00, 1.0],    # lavender
}

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
        'origin_filter': {40: 3.9},
    },
    {
        'label': 'Bob → C',
        'source_mesh': 'assets/bob_y90.obj',
        'target_mesh': 'assets/C.obj',
        'output_dir': 'output/teaser_bob_y90_to_C',
        'episodes': [5, 10, 15, 40],
        'bbox_filter': {40: ('y', 7.5)},
    },
    {
        'label': 'Sphere → V',
        'source_mesh': 'assets/isosphere.obj',
        'target_mesh': 'assets/V.obj',
        'output_dir': 'output/teaser_sphere_to_V',
        'episodes': [5, 10, 15, 59],
    },
]


def load_pts(path):
    p = Path(path)
    if p.suffix == '.pt':
        return torch.load(str(p), weights_only=False, map_location='cpu').numpy().astype(np.float64)
    return np.load(str(p)).astype(np.float64)


def load_mesh_pts(obj_path, n_samples=80000):
    mesh = trimesh.load(str(obj_path), force='mesh')
    pts, _ = trimesh.sample.sample_surface(mesh, n_samples)
    return pts.astype(np.float64)


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


def render_spheres(points, color_rgba, cam_center, cam_eye):
    extent = (points.max(axis=0) - points.min(axis=0)).max()
    n = min(len(points), MAX_SPHERES)
    radius = extent * RADIUS_SCALE / (n ** (1/3))

    mesh = build_sphere_mesh(points, radius)

    renderer = make_renderer()
    mat = rendering.MaterialRecord()
    mat.shader = 'defaultLit'
    mat.base_color = list(color_rgba)
    mat.base_metallic = 0.0
    mat.base_roughness = 0.75

    renderer.scene.add_geometry('s', mesh, mat)
    renderer.setup_camera(30.0, cam_center.tolist(), cam_eye.tolist(), [0, 0, 1])

    img = np.asarray(renderer.render_to_image()).copy()
    depth = np.asarray(renderer.render_to_depth_image()).copy()
    img[depth >= 0.9999] = 255

    renderer.scene.clear_geometry()
    del renderer
    return img


def compute_camera(all_points_list):
    all_pts = np.vstack(all_points_list)
    center = all_pts.mean(axis=0)
    extent = (all_pts.max(axis=0) - all_pts.min(axis=0)).max()
    eye = center + np.array([extent * 2.5, 0, 0])
    return center, eye


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


def render_row(row_def):
    label = row_def['label']
    out_dir = WORKSPACE / row_def['output_dir']
    source_mesh = WORKSPACE / row_def['source_mesh']
    target_mesh = WORKSPACE / row_def['target_mesh']
    episodes = row_def['episodes']
    origin_filter = row_def.get('origin_filter', {})
    bbox_filter = row_def.get('bbox_filter', {})
    ep_color = ROW_COLORS.get(label, [0.20, 0.90, 0.40, 1.0])

    print(f"\n=== {label} ===")

    # Collect all point clouds for shared camera
    all_pts_list = []

    # Source mesh → point cloud
    src_mesh_pts = load_mesh_pts(source_mesh, 80000)
    all_pts_list.append(src_mesh_pts)

    # ep000 particles
    ep0_path = out_dir / 'ep000' / 'ep000_particles.pt'
    ep0_pts = load_pts(ep0_path)
    all_pts_list.append(ep0_pts)

    # Target mesh → point cloud
    tgt_mesh_pts = load_mesh_pts(target_mesh, 80000)
    all_pts_list.append(tgt_mesh_pts)

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
    cam_center, cam_eye = compute_camera(all_pts_list)

    panels = []

    # 1) Source mesh as point cloud (gold)
    print(f"  Source mesh...", end=" ", flush=True)
    img = render_spheres(src_mesh_pts, COLOR_SOURCE, cam_center, cam_eye)
    panels.append(img)
    print("done")

    # 2) ep000 (per-row color)
    print(f"  ep000...", end=" ", flush=True)
    img = render_spheres(ep0_pts, ep_color, cam_center, cam_eye)
    panels.append(img)
    print("done")

    # 3) Episodes (per-row color)
    for ep in episodes:
        if ep in ep_pts:
            print(f"  ep{ep:03d}...", end=" ", flush=True)
            img = render_spheres(ep_pts[ep], ep_color, cam_center, cam_eye)
            panels.append(img)
            print("done")

    # 4) Target mesh as point cloud (gold)
    print(f"  Target mesh...", end=" ", flush=True)
    img = render_spheres(tgt_mesh_pts, COLOR_SOURCE, cam_center, cam_eye)
    panels.append(img)
    print("done")

    return hstack_panels(panels, gap=6)


def main():
    print("Rendering teaser_4row_pbr.png ...")

    row_images = []
    for row_def in ROWS:
        row_img = render_row(row_def)
        row_images.append(row_img)

    final = vstack_panels(row_images, gap=8)

    outpath = WORKSPACE / "figs" / "teaser_4row_pbr.png"
    outpath.parent.mkdir(exist_ok=True)
    Image.fromarray(final).save(str(outpath), dpi=(300, 300))
    print(f"\nSaved: {outpath}")
    print(f"Size: {final.shape[1]}x{final.shape[0]} px")


if __name__ == '__main__':
    main()
