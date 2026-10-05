"""Local regularizers added to Direct Chamfer Optimization in Table 1 /
Sec. 3.3 ("Insufficiency of Local Regularizers") and formalized in
Proposition 3 / the supplementary remark that follows it.

Every regularizer here is built only from *relative* positions of points
within a local k-NN neighborhood, so each is translationally invariant by
construction -- the structural property Proposition 3 requires:

    R({p_i + v}) = R({p_i})  for any v in R^3

The exact functional forms for repulsion, smoothness and volume
preservation are not pinned down numerically in the main paper beyond
"repulsion", "smoothness", and "volume preservation"; we follow the
standard instantiations named in the supplementary remark (Sec. 1.3):
a pairwise inverse-power repulsive force, graph-Laplacian smoothing, and a
local-volume penalty built from the covariance of each neighborhood (itself
translation invariant, since it depends only on p_j - p_i).
"""

from __future__ import annotations

import torch

from .geometry import knn_indices


def repulsion_loss(points: torch.Tensor, k: int = 6, alpha: float = 1.0) -> torch.Tensor:
    """Pairwise inverse-power repulsion energy whose gradient has the form
    f(p_i, p_j) = (p_i - p_j) / ||p_i - p_j||^(2+alpha) given in the
    supplementary remark, satisfying f(p_i, p_j) = -f(p_j, p_i).

    Energy: E = (1/(N*k)) * sum_i sum_{j in kNN(i)} 1 / ||p_i - p_j||^alpha
    """
    idx = knn_indices(points, k)  # [N, k], no grad
    neighbors = points[idx]  # [N, k, D]
    diff = points.unsqueeze(1) - neighbors  # [N, k, D]
    dist = diff.norm(dim=-1).clamp_min(1e-8)
    energy = (1.0 / dist.pow(alpha)).mean()
    return energy


def laplacian_smoothness_loss(points: torch.Tensor, k: int = 6) -> torch.Tensor:
    """Graph-Laplacian smoothness: mean squared deviation of each point from
    the centroid of its k-NN neighborhood. Translationally invariant since
    p_i - mean_j(p_j) only depends on relative offsets."""
    idx = knn_indices(points, k)
    neighbors = points[idx]  # [N, k, D]
    centroid = neighbors.mean(dim=1)
    return ((points - centroid) ** 2).sum(-1).mean()


def volume_preservation_loss(points: torch.Tensor, ref_local_scale: torch.Tensor, k: int = 6) -> torch.Tensor:
    """Penalize relative change of a local-volume proxy (the determinant of
    the covariance of each k-NN neighborhood's offsets from its centroid)
    against a reference computed once at initialization. Translationally
    invariant since it only depends on p_j - centroid.

    ``ref_local_scale`` is the per-point reference volume proxy, precomputed
    with :func:`local_volume_proxy` on the initial point cloud.
    """
    current = local_volume_proxy(points, k=k)
    rel = (current - ref_local_scale) / ref_local_scale.clamp_min(1e-8)
    return (rel ** 2).mean()


def local_volume_proxy(points: torch.Tensor, k: int = 6) -> torch.Tensor:
    """Determinant of the covariance matrix of each point's k-NN
    neighborhood offsets -- a smooth, translation-invariant proxy for the
    local volume spanned by the neighborhood."""
    idx = knn_indices(points, k)
    neighbors = points[idx]  # [N, k, D]
    centroid = neighbors.mean(dim=1, keepdim=True)
    offsets = neighbors - centroid  # [N, k, D]
    cov = torch.einsum("nkd,nke->nde", offsets, offsets) / k
    # add a small ridge for numerical stability of the determinant
    eye = torch.eye(cov.shape[-1], device=points.device, dtype=points.dtype)
    cov = cov + 1e-6 * eye
    return torch.linalg.det(cov)
