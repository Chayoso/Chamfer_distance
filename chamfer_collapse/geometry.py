"""Nearest-neighbor plumbing shared by every experiment.

Target point sets are always treated as constants, so the standard trick for
a differentiable Chamfer distance is used throughout the project: the
nearest-neighbor *index* is found with an exact k-d tree (non-differentiable,
recomputed every step since the assignment can change), and the *coordinates*
of the matched point are gathered into the autograd graph so PyTorch can
differentiate the resulting squared distance. This reproduces the gradient of
Eq. (1) exactly (see ``tests/test_propositions.py``).
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree


def nn_indices(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Index in ``reference`` of the nearest neighbor of every row of ``query``.

    Computed with an exact k-d tree; not part of the autograd graph (matches
    the standard nearest-neighbor-assignment treatment used by every Chamfer
    distance implementation, including the paper's own setup, Sec. 3.1).
    """
    if query.is_cuda:
        return _nn_indices_gpu(query.detach(), reference.detach())
    ref_np = reference.detach().cpu().numpy()
    query_np = query.detach().cpu().numpy()
    tree = cKDTree(ref_np)
    _, idx = tree.query(query_np, k=1)
    idx = np.atleast_1d(idx)
    return torch.as_tensor(idx, dtype=torch.long, device=query.device)


_GPU_CHUNK_BYTES = 256 * 2 ** 20


def _nn_indices_gpu(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Exact brute-force nearest neighbor on the GPU, chunked over queries so the
    [chunk, M] distance block stays under ~256 MB. Uses the direct (non-matmul)
    Euclidean formula so ties/near-ties resolve like the k-d tree up to float32."""
    chunk = max(1, _GPU_CHUNK_BYTES // (4 * max(reference.shape[0], 1)))
    out = torch.empty(query.shape[0], dtype=torch.long, device=query.device)
    for s in range(0, query.shape[0], chunk):
        d = torch.cdist(query[s:s + chunk], reference, compute_mode="donot_use_mm_for_euclid_dist")
        out[s:s + chunk] = d.argmin(dim=1)
    return out


def nn_indices_and_dists(query: np.ndarray, reference: np.ndarray):
    """NumPy convenience wrapper returning (distances, indices)."""
    tree = cKDTree(reference)
    dists, idx = tree.query(query, k=1)
    return dists, idx


def knn_indices(points: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the ``k`` nearest neighbors of each point *within the same
    set* (excluding the point itself). Used by local regularizers."""
    if points.is_cuda:
        chunk = max(1, _GPU_CHUNK_BYTES // (4 * points.shape[0]))
        rows = []
        for s in range(0, points.shape[0], chunk):
            d = torch.cdist(points[s:s + chunk].detach(), points.detach(), compute_mode="donot_use_mm_for_euclid_dist")
            rows.append(d.topk(k + 1, dim=1, largest=False).indices[:, 1:])
        return torch.cat(rows, 0)
    pts_np = points.detach().cpu().numpy()
    tree = cKDTree(pts_np)
    _, idx = tree.query(pts_np, k=k + 1)
    idx = np.atleast_2d(idx)[:, 1:]  # drop self-match (column 0)
    return torch.as_tensor(idx, dtype=torch.long, device=points.device)


def fibonacci_sphere(n: int, radius: float = 1.0, seed: int | None = None) -> np.ndarray:
    """Quasi-uniform point sampling on a sphere via the Fibonacci lattice."""
    i = np.arange(n)
    golden = (1 + 5 ** 0.5) / 2
    theta = 2 * np.pi * i / golden
    z = 1 - (2 * i + 1) / n
    r_xy = np.sqrt(np.clip(1 - z ** 2, 0.0, 1.0))
    x = r_xy * np.cos(theta)
    y = r_xy * np.sin(theta)
    pts = np.stack([x, y, z], axis=-1) * radius
    if seed is not None:
        rng = np.random.default_rng(seed)
        pts = pts[rng.permutation(n)]
    return pts
