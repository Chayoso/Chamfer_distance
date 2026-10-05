"""Density-aware Chamfer Distance (DCD), following the simplified
formulation described in the remark to Proposition 3 (supplementary
material, Sec. 1.3): each point's contribution to the forward loss is
re-weighted by the inverse of a local density estimate,

    w_i = 1 / rho_hat(p_i),          L_DCD_fwd = (1/N) sum_i w_i * d_fwd_i^2

with weights normalized to average 1 so the loss stays on the same scale as
unweighted Chamfer distance. Density rho_hat(p_i) is estimated from the mean
distance to the k nearest neighbors *within the source point cloud*: dense
regions (small mean NN distance) get large density / small weight, sparse
regions get small density / large weight -- boosting the gradient on
under-covered regions relative to plain DCO.

This is the design used in Wu et al. 2021 ("Density-aware Chamfer Distance
as a Comprehensive Metric for Point Cloud Completion") in spirit: per-point
weighting by local density, applied here in the simplified single-weight
form the paper's own remark uses when explaining why DCD's t->s component
is unchanged from DCO's near collapse (the secondary density-gradient term
vanishes as ||p_i - q*|| -> 0, Sec. 1.3).
"""

from __future__ import annotations

import torch

from .geometry import knn_indices, nn_indices


def local_density(points: torch.Tensor, k: int = 8) -> torch.Tensor:
    """Local density estimate: inverse of the mean distance to the k nearest
    neighbors within ``points`` (detached -- density is treated as a fixed
    per-step weight, matching standard DCD practice)."""
    with torch.no_grad():
        idx = knn_indices(points, k)
        neighbors = points[idx]
        mean_nn_dist = (points.unsqueeze(1) - neighbors).norm(dim=-1).mean(dim=1)
        density = 1.0 / mean_nn_dist.clamp_min(1e-6)
    return density


def dcd_forward_loss(source: torch.Tensor, target: torch.Tensor, k: int = 8) -> torch.Tensor:
    fwd_idx = nn_indices(source, target)
    fwd_nn = target.index_select(0, fwd_idx)
    d_fwd_sq = ((source - fwd_nn) ** 2).sum(-1)
    density = local_density(source, k=k)
    weight = 1.0 / density
    weight = weight / weight.mean().clamp_min(1e-8)
    return (weight * d_fwd_sq).mean()


def dcd_reverse_loss(source: torch.Tensor, target: torch.Tensor, k: int = 8) -> torch.Tensor:
    """Reverse term re-weighted by target-side density, for symmetry."""
    rev_idx = nn_indices(target, source)
    rev_nn = source.index_select(0, rev_idx)
    d_rev_sq = ((target - rev_nn) ** 2).sum(-1)
    density = local_density(target, k=k)
    weight = 1.0 / density
    weight = weight / weight.mean().clamp_min(1e-8)
    return (weight * d_rev_sq).mean()


def dcd_two_sided(source: torch.Tensor, target: torch.Tensor, k: int = 8):
    fwd = dcd_forward_loss(source, target, k=k)
    rev = dcd_reverse_loss(source, target, k=k)
    return fwd, rev, (fwd + rev).sqrt()
