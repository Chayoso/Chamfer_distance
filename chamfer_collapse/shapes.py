"""Source / target point clouds for the 3D experiments (Sec. 3, 5).

The paper evaluates on an isosphere source and Stanford-Bunny-family targets
(Bunny, Duck, Cow/Spot, Teapot, Dragon) whose exact mesh files are not
distributed with the paper. ``load_shape`` resolves each name as follows:

1. If ``assets/meshes/<name>.obj`` (or ``.ply``/``.stl``) exists, it is
   loaded with trimesh and surface-sampled -- drop the real assets there
   (e.g. from the Stanford 3D Scanning Repository, or your own copies of
   the meshes used in the paper) to reproduce the exact figures.
2. Otherwise a procedural stand-in with comparable qualitative geometry
   (concavity / genus / thin structures) is generated so every experiment
   in this repository still runs out of the box. These are clearly
   labelled approximations, not the original assets.

Run ``scripts/fetch_stanford_meshes.py`` to download the Bunny and Dragon
from the canonical Stanford Computer Graphics Laboratory 3D Scanning
Repository (the only two assets with one stable, well-known public URL).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "meshes"

_MESH_EXTENSIONS = (".obj", ".ply", ".stl")


#: Real-asset file names accepted for each experiment shape name, in priority order.
#: ``sphere`` is the paper's isosphere source; ``cow`` is the "Cow/Spot" target.
ASSET_ALIASES = {
    "sphere": ("sphere", "isosphere"),
    "cow": ("cow", "spot"),
}


def _find_mesh_file(name: str) -> Path | None:
    for candidate in ASSET_ALIASES.get(name, (name,)):
        for ext in _MESH_EXTENSIONS:
            p = ASSETS_DIR / f"{candidate}{ext}"
            if p.exists():
                return p
    return None


def _load_asset(path: Path) -> trimesh.Trimesh:
    # process=False + explicit merge: OBJ exporters often split vertices per face
    # (normals/UVs), which would otherwise leave an isosphere "non-watertight".
    mesh = trimesh.load(path, force="mesh", process=False)
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    return mesh


def shape_provenance(names=None) -> dict:
    """{name: 'real:<file>' | 'procedural stand-in'} for the experiment shapes."""
    out = {}
    for n in (names or ("sphere", "bunny", "duck", "cow", "teapot", "dragon")):
        f = _find_mesh_file(n)
        out[n] = f"real:{f.name}" if f else "procedural stand-in"
    return out


def sample_surface(mesh: trimesh.Trimesh, n_points: int, seed: int = 0) -> np.ndarray:
    """Area-weighted uniform surface sampling (matches the "surface-sampled"
    protocol described for the Bunny target in Sec. 3.1)."""
    rng = np.random.default_rng(seed)
    pts, _ = trimesh.sample.sample_surface(mesh, n_points, seed=rng)
    return np.asarray(pts, dtype=np.float64)


def _normalize(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.bounding_box.centroid)
    scale = 1.0 / max(mesh.bounding_box.extents.max(), 1e-8)
    mesh.apply_scale(scale)
    return mesh


# ----------------------------------------------------------------------
# Procedural stand-ins
# ----------------------------------------------------------------------

def _procedural_sphere() -> trimesh.Trimesh:
    return trimesh.creation.icosphere(subdivisions=4, radius=1.0)


def _procedural_bunny() -> trimesh.Trimesh:
    """Ellipsoid body + two long ear cones: a simple non-convex silhouette."""
    body = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    body.apply_scale([0.8, 0.95, 1.0])
    ear_l = trimesh.creation.cone(radius=0.14, height=0.9, sections=24)
    ear_l.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-20), [0, 0, 1]))
    ear_l.apply_translation([-0.32, 0.15, 0.85])
    ear_r = ear_l.copy()
    ear_r.apply_transform(trimesh.transformations.rotation_matrix(np.radians(40), [0, 0, 1], point=[0, 0, 0]))
    ear_r.apply_translation([0.64, 0, 0])
    return trimesh.util.concatenate([body, ear_l, ear_r])


def _procedural_duck() -> trimesh.Trimesh:
    """Squashed-sphere body with a conical beak: asymmetric, mildly concave."""
    body = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    body.apply_scale([0.85, 1.1, 0.85])
    head = trimesh.creation.icosphere(subdivisions=3, radius=0.55)
    head.apply_translation([0, 0.95, 0.35])
    beak = trimesh.creation.cone(radius=0.22, height=0.5, sections=16)
    beak.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0]))
    beak.apply_translation([0, 1.35, 0.35])
    return trimesh.util.concatenate([body, head, beak])


def _procedural_cow() -> trimesh.Trimesh:
    """Elongated body + four cylindrical legs + head + ears: genuinely
    non-convex (concavities between the legs), akin to Spot the cow."""
    body = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    body.apply_scale([1.3, 0.7, 0.7])
    parts = [body]
    leg_positions = [(-0.8, -0.55, 0.5), (0.8, -0.55, 0.5), (-0.8, -0.55, -0.5), (0.8, -0.55, -0.5)]
    for x, y, z in leg_positions:
        leg = trimesh.creation.cylinder(radius=0.13, height=0.7, sections=16)
        leg.apply_translation([x, y, z])
        parts.append(leg)
    head = trimesh.creation.icosphere(subdivisions=3, radius=0.45)
    head.apply_translation([1.5, 0.15, 0])
    parts.append(head)
    for s in (-1, 1):
        ear = trimesh.creation.cone(radius=0.12, height=0.3, sections=12)
        ear.apply_translation([1.55, 0.55, s * 0.25])
        parts.append(ear)
    return trimesh.util.concatenate(parts)


def _procedural_teapot() -> trimesh.Trimesh:
    """Rounded ellipsoid body + torus handle + conical spout: nearly convex
    overall except for the handle, matching the "nearly convex" description
    in Sec. 3.4 / Table 2."""
    body = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    body.apply_scale([1.0, 0.75, 1.0])
    handle = trimesh.creation.torus(major_radius=0.35, minor_radius=0.09, major_sections=32, minor_sections=16)
    handle.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90), [0, 0, 1]))
    handle.apply_translation([-1.0, 0.05, 0])
    spout = trimesh.creation.cone(radius=0.2, height=0.7, sections=16)
    spout.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-70), [0, 0, 1]))
    spout.apply_translation([1.05, 0.15, 0])
    lid = trimesh.creation.icosphere(subdivisions=2, radius=0.35)
    lid.apply_translation([0, 0.85, 0])
    return trimesh.util.concatenate([body, handle, spout, lid])


def _procedural_dragon() -> trimesh.Trimesh:
    """Serpentine, high-genus stand-in: a twisted torus body with limbs and
    thin wing-like appendages, meant to be the most topologically extreme
    target (deep concavities, thin structures) as described in Sec. 5.4."""
    body = trimesh.creation.torus(major_radius=0.9, minor_radius=0.28, major_sections=48, minor_sections=20)
    verts = body.vertices.copy()
    # twist the torus along its main axis to break symmetry and add
    # deep concavities / self-proximate thin structure.
    theta = np.arctan2(verts[:, 1], verts[:, 0])
    twist = 0.6 * np.sin(3 * theta)
    verts[:, 2] += twist
    body.vertices = verts
    parts = [body]
    for i, ang in enumerate(np.linspace(0, 2 * np.pi, 4, endpoint=False)):
        leg = trimesh.creation.cone(radius=0.08, height=0.5, sections=10)
        leg.apply_translation([0.9 * np.cos(ang), 0.9 * np.sin(ang), -0.35])
        parts.append(leg)
    head = trimesh.creation.icosphere(subdivisions=2, radius=0.32)
    head.apply_translation([1.15, 0, 0.15])
    parts.append(head)
    for s in (-1, 1):
        wing = trimesh.creation.box(extents=[0.05, 0.75, 0.45])
        wing.apply_translation([-0.1, s * 0.55, 0.35])
        parts.append(wing)
    tail = trimesh.creation.cone(radius=0.12, height=0.9, sections=10)
    tail.apply_transform(trimesh.transformations.rotation_matrix(np.radians(100), [0, 1, 0]))
    tail.apply_translation([-1.3, 0, -0.1])
    parts.append(tail)
    return trimesh.util.concatenate(parts)


_PROCEDURAL_BUILDERS = {
    "sphere": _procedural_sphere,
    "bunny": _procedural_bunny,
    "duck": _procedural_duck,
    "cow": _procedural_cow,
    "spot": _procedural_cow,
    "teapot": _procedural_teapot,
    "dragon": _procedural_dragon,
}


def load_shape_mesh(name: str) -> tuple[trimesh.Trimesh, bool]:
    """Return (mesh, is_real_asset). Prefers a real mesh file dropped into
    ``assets/meshes``; falls back to the procedural stand-in otherwise."""
    name = name.lower()
    mesh_path = _find_mesh_file(name)
    if mesh_path is not None:
        return _normalize(_load_asset(mesh_path)), True
    if name not in _PROCEDURAL_BUILDERS:
        raise ValueError(f"Unknown shape '{name}'. Known shapes: {sorted(_PROCEDURAL_BUILDERS)}")
    mesh = _PROCEDURAL_BUILDERS[name]()
    return _normalize(mesh), False


def load_shape_points(name: str, n_points: int, seed: int = 0) -> tuple[np.ndarray, bool]:
    """Surface-sample ``n_points`` points from the named shape.

    Returns (points [n_points, 3], is_real_asset).
    """
    mesh, is_real = load_shape_mesh(name)
    pts = sample_surface(mesh, n_points, seed=seed)
    return pts, is_real


SHAPE_NAMES = tuple(sorted(set(_PROCEDURAL_BUILDERS) - {"spot"}))
