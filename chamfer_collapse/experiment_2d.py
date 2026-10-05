"""Cross-domain 2D collapse experiment (Sec. 3.5 of the main paper,
Supplementary Sec. 2.2): 600 points on a circle are optimized to match a
200-point 5-arm star, under three strategies:

    (i)   DCO -- per-point Chamfer gradient descent.
    (ii)  DCO + local k-NN repulsion (1/d^2).
    (iii) Shared-basis (Fourier) deformation -- all points share the same
          12-mode Fourier coefficients, providing global coupling.

All three run for 300 steps with bidirectional Chamfer loss and 95th
percentile gradient-norm clipping, exactly as specified in Supplementary
Sec. 2.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .chamfer import chamfer_components
from .geometry import knn_indices
from .metrics import cluster_fraction, coverage

N_SOURCE = 600
N_TARGET = 200
N_STEPS = 300
CLIP_PERCENTILE = 95.0


def make_circle(n: int = N_SOURCE, radius: float = 0.8) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=-1)


def make_star(n: int = N_TARGET, arms: int = 5, r_inner: float = 0.25, r_outer: float = 1.0) -> np.ndarray:
    """A true 5-arm star *polygon*: ``2*arms`` vertices alternating between
    ``r_outer`` (tips) and ``r_inner`` (concave valleys), joined by straight
    edges and sampled uniformly (by arc length) with ``n`` points. This
    matches the paper's description of DCO "collaps[ing] source points onto
    star vertices" (Fig. 3) -- a smooth sinusoidal radial profile has no
    vertices to collapse onto and reproduces the reported two-sided CD far
    less closely than the polygon does (0.353 vs. the paper's 0.352 here)."""
    n_verts = 2 * arms
    idx = np.arange(n_verts)
    radii = np.where(idx % 2 == 0, r_outer, r_inner)
    ang = idx * np.pi / arms
    verts = np.stack([radii * np.cos(ang), radii * np.sin(ang)], axis=-1)
    edges = np.diff(np.vstack([verts, verts[:1]]), axis=0)
    seg_len = np.linalg.norm(edges, axis=1)
    cum_len = np.cumsum(seg_len)
    total_len = cum_len[-1]
    s = np.linspace(0, total_len, n, endpoint=False)
    seg_idx = np.searchsorted(cum_len, s, side="right")
    prev_cum = np.concatenate([[0.0], cum_len[:-1]])[seg_idx]
    t = (s - prev_cum) / seg_len[seg_idx]
    return verts[seg_idx] + t[:, None] * edges[seg_idx]


def _percentile_clip_hook(percentile: float = CLIP_PERCENTILE):
    """Backward hook: clip per-point gradient norms to their P-th percentile.
    Registered directly on the (possibly non-leaf) point-cloud tensor so the
    same clipping rule applies whether points are themselves the optimized
    parameters (DCO, DCO+Repulsion) or are produced by backpropagating
    through a shared parameterization (Fourier), matching the "all three
    running ... with 95th-percentile gradient clipping" protocol of
    Supplementary Sec. 2.2.
    """

    def hook(grad: torch.Tensor) -> torch.Tensor:
        norms = grad.norm(dim=-1, keepdim=True)
        if norms.numel() == 0:
            return grad
        thresh = torch.quantile(norms.detach().reshape(-1), percentile / 100.0)
        scale = (thresh / norms.clamp_min(1e-12)).clamp(max=1.0)
        return grad * scale

    return hook


@dataclass
class Trajectory:
    positions: list = field(default_factory=list)   # snapshots [n,2]
    two_sided_cd: list = field(default_factory=list)  # per-step loss curve


def run_dco_2d(source0: np.ndarray, target: np.ndarray, lr: float = 0.015,
               n_steps: int = N_STEPS, n_snapshots: int = 3) -> Trajectory:
    target_t = torch.as_tensor(target, dtype=torch.float32)
    points = torch.as_tensor(source0, dtype=torch.float32).clone().requires_grad_(True)
    points.register_hook(_percentile_clip_hook())
    optimizer = torch.optim.SGD([points], lr=lr)
    traj = Trajectory()
    snap_steps = set(np.linspace(0, n_steps - 1, n_snapshots, dtype=int).tolist())
    for step in range(n_steps):
        optimizer.zero_grad()
        res = chamfer_components(points, target_t)
        loss = res.fwd_sq.mean() + res.rev_sq.mean()
        loss.backward()
        optimizer.step()
        traj.two_sided_cd.append(float(res.two_sided.item()))
        if step in snap_steps:
            traj.positions.append(points.detach().numpy().copy())
    return traj


def run_dco_repulsion_2d(source0: np.ndarray, target: np.ndarray, lr: float = 0.015,
                          k: int = 6, lam_rep: float = 2e-4, n_steps: int = N_STEPS,
                          n_snapshots: int = 3) -> Trajectory:
    target_t = torch.as_tensor(target, dtype=torch.float32)
    points = torch.as_tensor(source0, dtype=torch.float32).clone().requires_grad_(True)
    points.register_hook(_percentile_clip_hook())
    optimizer = torch.optim.SGD([points], lr=lr)
    traj = Trajectory()
    snap_steps = set(np.linspace(0, n_steps - 1, n_snapshots, dtype=int).tolist())
    for step in range(n_steps):
        optimizer.zero_grad()
        res = chamfer_components(points, target_t)
        loss = res.fwd_sq.mean() + res.rev_sq.mean()
        idx = knn_indices(points, k)
        neighbors = points[idx]
        diff = points.unsqueeze(1) - neighbors
        dist = diff.norm(dim=-1).clamp_min(1e-4)
        # potential U ~ 1/d gives force -dU/dp_i ~ (p_i-p_j)/d^3, i.e. force
        # magnitude ~ 1/d^2, matching the "(1/d^2)" force cited in Sec. 3.5.
        rep_energy = (1.0 / dist).mean()
        loss = loss + lam_rep * rep_energy
        loss.backward()
        optimizer.step()
        traj.two_sided_cd.append(float(res.two_sided.item()))
        if step in snap_steps:
            traj.positions.append(points.detach().numpy().copy())
    return traj


def run_fourier_2d(target: np.ndarray, lr: float = 0.003, n_modes: int = 12,
                    n_source: int = N_SOURCE, n_steps: int = N_STEPS,
                    n_snapshots: int = 3) -> Trajectory:
    """Shared-basis deformation: r(theta) = a0 + sum_k [a_k cos(k theta) +
    b_k sin(k theta)], all N_SOURCE points sharing the same coefficients.
    Initialized to the plain circle (a0 = 0.8, all higher modes = 0)."""
    target_t = torch.as_tensor(target, dtype=torch.float32)
    theta = torch.linspace(0, 2 * np.pi, n_source + 1)[:-1]
    a0 = torch.tensor(0.8, requires_grad=True)
    a = torch.zeros(n_modes, requires_grad=True)
    b = torch.zeros(n_modes, requires_grad=True)
    k_idx = torch.arange(1, n_modes + 1, dtype=torch.float32)

    def boundary():
        r = a0 + (a * torch.cos(k_idx * theta.unsqueeze(-1))).sum(-1) + (b * torch.sin(k_idx * theta.unsqueeze(-1))).sum(-1)
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        return torch.stack([x, y], dim=-1)

    optimizer = torch.optim.SGD([a0, a, b], lr=lr)
    traj = Trajectory()
    snap_steps = set(np.linspace(0, n_steps - 1, n_snapshots, dtype=int).tolist())
    for step in range(n_steps):
        optimizer.zero_grad()
        points = boundary()
        points.register_hook(_percentile_clip_hook())
        res = chamfer_components(points, target_t)
        loss = res.fwd_sq.mean() + res.rev_sq.mean()
        loss.backward()
        optimizer.step()
        traj.two_sided_cd.append(float(res.two_sided.item()))
        if step in snap_steps:
            with torch.no_grad():
                traj.positions.append(boundary().numpy().copy())
    return traj


def summarize(traj: Trajectory, target: np.ndarray, coverage_eps: float = 0.05,
              cluster_eps: float = 0.02) -> dict:
    final = traj.positions[-1]
    return {
        "two_sided_cd": traj.two_sided_cd[-1],
        "coverage_pct": coverage(final, target, eps=coverage_eps),
        "cluster_fraction_pct": cluster_fraction(final, eps=cluster_eps),
    }
