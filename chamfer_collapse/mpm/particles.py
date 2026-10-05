"""Volumetric particle sampling for MPM.

The paper seeds MPM particles on a regular per-cell sub-grid ("N particles
per cell" = PPC^{1/3} along each axis, giving ~37K particles at PPC^{1/3}=3
on a 32^3 grid, ~89K at PPC^{1/3}=4, Supplementary Table 1). We instead use
rejection-sampled uniform volumetric sampling at an *equivalent average
density* (occupied-cell count x PPC), which is statistically equivalent for
elastic-solid MPM and far simpler to make robust across the procedural,
possibly self-intersecting union meshes used as shape stand-ins here (exact
per-cell lattice seeding assumes a clean watertight mesh, which the unions
in ``shapes.py`` are not guaranteed to be).
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy import ndimage


def cap_boundary_loops(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Close simple open boundary loops (e.g. the Stanford Bunny's base) with a
    centroid fan. Non-simple boundaries (non-manifold edges, e.g. the dragon) are
    left alone; the voxel fill below handles those."""
    import networkx as nx
    edges = mesh.edges_sorted
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = uniq[counts == 1]
    if len(boundary) == 0:
        return mesh
    g = nx.Graph()
    g.add_edges_from(map(tuple, boundary))
    new_v, new_f, nv = [], [], len(mesh.vertices)
    for comp in nx.connected_components(g):
        sub = g.subgraph(comp)
        if any(d != 2 for _, d in sub.degree()):
            continue
        cycle = [u for u, _ in nx.find_cycle(sub)]
        ci = nv + len(new_v)
        new_v.append(mesh.vertices[cycle].mean(0))
        new_f += [[a, b, ci] for a, b in zip(cycle, cycle[1:] + cycle[:1])]
    if not new_f:
        return mesh
    out = trimesh.Trimesh(np.vstack([mesh.vertices, np.array(new_v)]),
                          np.vstack([mesh.faces, np.array(new_f)]), process=False)
    out.fix_normals()
    return out


class VolumeSampler:
    """Inside/outside test and volume for a (possibly imperfect) mesh.

    1. cap simple open loops; 2. voxelize the surface at ``resolution`` voxels along the
    longest axis and flood-fill the interior. A voxel lookup replaces ``mesh.contains``:
    it works on open / non-manifold / multi-body meshes and is cheap in memory (the
    ray-casting path used >20 GB on 37K-particle runs). ``volume`` is the filled volume;
    ``exact_volume`` is the mesh's own volume when it is watertight (used to validate)."""

    def __init__(self, mesh: trimesh.Trimesh, resolution: int = 192):
        self.repaired = cap_boundary_loops(mesh)
        self.watertight = bool(self.repaired.is_watertight)
        self.exact_volume = abs(self.repaired.volume) if self.watertight else None
        self.pitch = float(self.repaired.bounding_box.extents.max()) / resolution
        vg = self.repaired.voxelized(self.pitch)
        occ = np.pad(vg.matrix, 2)
        filled = ndimage.binary_fill_holes(occ)
        if not self.watertight:                     # seal small gaps before filling
            filled = ndimage.binary_fill_holes(ndimage.binary_closing(occ, iterations=2))
        self.filled = filled
        self.origin = vg.transform[:3, 3] - 2 * self.pitch
        self.voxel_volume = float(filled.sum()) * self.pitch ** 3
        # voxel fill over-estimates by ~half a voxel of surface shell (+2-5% at 192^3), so use
        # the exact volume whenever the repaired mesh is watertight.
        self.volume = self.exact_volume if self.watertight else self.voxel_volume

    def contains(self, pts: np.ndarray) -> np.ndarray:
        ijk = np.floor((pts - self.origin) / self.pitch).astype(int)
        ok = np.all((ijk >= 0) & (ijk < np.array(self.filled.shape)), axis=1)
        out = np.zeros(len(pts), dtype=bool)
        out[ok] = self.filled[tuple(ijk[ok].T)]
        return out


_SAMPLER_CACHE: dict = {}


def get_sampler(mesh: trimesh.Trimesh, resolution: int = 192) -> VolumeSampler:
    key = (id(mesh), len(mesh.vertices), resolution)
    if key not in _SAMPLER_CACHE:
        _SAMPLER_CACHE[key] = VolumeSampler(mesh, resolution)
    return _SAMPLER_CACHE[key]


def mesh_volume(mesh: trimesh.Trimesh) -> float:
    """Interior volume used for particle mass (robust to open / non-manifold meshes)."""
    return get_sampler(mesh).volume


def estimate_particle_count(mesh: trimesh.Trimesh, dx: float, ppc: int) -> int:
    """Estimate the total particle count for a target density of ``ppc``
    particles per grid cell of size ``dx``, given the mesh's volume."""
    occupied_cells = max(mesh_volume(mesh) / (dx ** 3), 1.0)
    return max(int(round(occupied_cells * ppc)), 64)


def sample_volume_points(mesh: trimesh.Trimesh, n_points: int, seed: int = 0,
                          max_tries: int = 40) -> np.ndarray:
    """Rejection-sample ``n_points`` points inside the mesh's (repaired) interior."""
    rng = np.random.default_rng(seed)
    sampler = get_sampler(mesh)
    lo, hi = sampler.repaired.bounds
    collected, n_have, tries = [], 0, 0
    batch = max(n_points * 2, 512)
    while n_have < n_points and tries < max_tries:
        candidates = rng.uniform(lo, hi, size=(batch, 3))
        inside = candidates[sampler.contains(candidates)]
        collected.append(inside)
        n_have += inside.shape[0]
        tries += 1
        batch = int(batch * 1.5)
    pts = np.concatenate(collected, axis=0) if collected else np.zeros((0, 3))
    if pts.shape[0] < n_points:
        if pts.shape[0] == 0:
            raise RuntimeError("volumetric sampling failed to find any interior points")
        extra_idx = rng.integers(0, pts.shape[0], size=n_points - pts.shape[0])
        pts = np.concatenate([pts, pts[extra_idx]], axis=0)
    return pts[rng.permutation(pts.shape[0])[:n_points]]
