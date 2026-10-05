"""Differentiable MLS-MPM (Moving Least Squares Material Point Method) for
elastic solids, in pure PyTorch (no Taichi dependency, so it stays fully
differentiable through ``torch.autograd`` with no custom backward kernels
needed -- every operation below, including the P2G/G2P scatter/gather, is a
standard differentiable ``torch`` op).

This follows the well-known MLS-MPM formulation of Hu et al. 2018
("A Moving Least Squares Material Point Method...", cited by the main paper
as ``hu2018moving``) with a quadratic B-spline kernel, as popularized by the
Taichi "99-line" 2D/3D MPM reference implementations -- the same lineage
(ChainQueen / DiffTaichi / MLS-MPM) that Xu et al. 2025's differentiable MPM
morphing framework is built on. Material model: true Neo-Hookean elasticity
(not fixed-corotated), with the exact Lame parameters used in the paper
(Supplementary Table 1): mu=58333, lambda=38889 (E=140000, nu=0.20).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import torch

_OFFSETS = torch.tensor(list(product(range(3), repeat=3)), dtype=torch.long)  # [27, 3]


@dataclass
class MPMParams:
    n_grid: int = 32                 # nodes per axis (paper: 32)
    dx: float = 1.0                  # grid spacing (paper: 1.0)
    dt: float = 8.33e-3               # timestep (paper: 8.33e-3)
    mu: float = 58333.0                # shear modulus
    lam: float = 38889.0                # first Lame parameter
    density: float = 75.0             # material density (paper: 75.0)
    drag: float = 0.5                  # per-substep velocity damping (paper: 0.5)

    @property
    def origin(self) -> float:
        return -0.5 * self.n_grid * self.dx


def quadratic_bspline_weights(fx: torch.Tensor) -> torch.Tensor:
    """fx in [0,1)^3 (offset of the particle from its base grid node, in
    grid-cell units). Returns weights [N, 3, 3]: dim 1 indexes the spatial
    axis, dim 2 indexes the local offset in {0,1,2}."""
    w0 = 0.5 * (1.5 - fx) ** 2
    w1 = 0.75 - (fx - 1.0) ** 2
    w2 = 0.5 * (fx - 0.5) ** 2
    return torch.stack([w0, w1, w2], dim=-1)  # [N, 3, 3]


def neo_hookean_pk1(F: torch.Tensor, mu: float, lam: float) -> torch.Tensor:
    """First Piola-Kirchhoff stress for the true (non-corotated) Neo-Hookean
    model: Psi(F) = mu/2 (tr(F^T F) - 3) - mu log(J) + lambda/2 log(J)^2
    => P(F) = mu (F - F^-T) + lambda log(J) F^-T."""
    J = torch.linalg.det(F).clamp_min(1e-6)
    F_inv_T = torch.linalg.inv(F).transpose(-1, -2)
    return mu * (F - F_inv_T) + (lam * torch.log(J)).unsqueeze(-1).unsqueeze(-1) * F_inv_T


class MPMState:
    """Mutable particle state carried across substeps / frames. Positions,
    velocities and F are leaf-adjacent tensors that autograd differentiates
    through when the simulation is driven by a learnable per-frame control
    (see ``joint_opt.py``)."""

    def __init__(self, x: torch.Tensor, volume0: float, density: float, device: str = "cpu",
                 dtype: torch.dtype = torch.float32):
        n = x.shape[0]
        self.x = x.to(device=device, dtype=dtype)
        self.v = torch.zeros(n, 3, device=device, dtype=dtype)
        self.F = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(n, 1, 1)
        self.C = torch.zeros(n, 3, 3, device=device, dtype=dtype)
        self.volume0 = volume0
        self.mass = volume0 * density * torch.ones(n, device=device, dtype=dtype)


def _grid_shape(p: MPMParams):
    g = p.n_grid
    return (g, g, g)


def rasterize_mass(points: torch.Tensor, p: MPMParams) -> torch.Tensor:
    """Deposit unit point mass onto the grid via the same quadratic
    B-spline kernel used by P2G. Used once to build the fixed target mass
    field for the physics loss (Eq. 4)."""
    device = points.device
    g = p.n_grid
    grid_m = torch.zeros(g * g * g, device=device, dtype=points.dtype)
    rel = (points - p.origin) / p.dx
    base = (rel - 0.5).floor().long()
    fx = rel - base.to(rel.dtype)
    w = quadratic_bspline_weights(fx)  # [N,3,3]
    offsets = _OFFSETS.to(device)
    base_c = base.clamp(0, g - 1)
    for oi in range(27):
        off = offsets[oi]
        node = (base_c + off).clamp(0, g - 1)
        weight = w[:, 0, off[0]] * w[:, 1, off[1]] * w[:, 2, off[2]]
        flat = (node[:, 0] * g + node[:, 1]) * g + node[:, 2]
        grid_m.index_add_(0, flat, weight)
    return grid_m.view(g, g, g)


def substep(state: MPMState, p: MPMParams, external_velocity: torch.Tensor | None = None) -> MPMState:
    """One differentiable MLS-MPM substep. If ``external_velocity`` is
    given ([N,3]), it is added to particle velocities before P2G -- this is
    the control input the morphing optimizer learns (see ``joint_opt.py``).
    """
    device = state.x.device
    g = p.n_grid

    v = state.v
    if external_velocity is not None:
        v = v + external_velocity

    # F update using the affine velocity gradient C recovered at the end of
    # the previous substep's G2P (APIC / MLS-MPM update rule).
    F_new = (torch.eye(3, device=device, dtype=state.x.dtype).unsqueeze(0) + p.dt * state.C) @ state.F

    P = neo_hookean_pk1(F_new, p.mu, p.lam)
    stress_affine = (-p.dt * state.volume0 * 4.0 / (p.dx ** 2)) * (P @ F_new.transpose(-1, -2))
    affine = stress_affine + state.mass.view(-1, 1, 1) * state.C

    rel = (state.x - p.origin) / p.dx
    base = (rel - 0.5).floor().long()
    fx = rel - base.to(rel.dtype)
    w = quadratic_bspline_weights(fx)  # [N,3,3]
    base_c = base.clamp(0, g - 1)

    grid_v = torch.zeros(g * g * g, 3, device=device, dtype=state.x.dtype)
    grid_m = torch.zeros(g * g * g, device=device, dtype=state.x.dtype)

    offsets = _OFFSETS.to(device)
    mv = state.mass.view(-1, 1) * v  # [N,3]
    for oi in range(27):
        off = offsets[oi]
        node = (base_c + off).clamp(0, g - 1)
        weight = w[:, 0, off[0]] * w[:, 1, off[1]] * w[:, 2, off[2]]  # [N]
        dpos = (off.to(fx.dtype) - fx) * p.dx  # [N,3]
        contrib = mv + torch.einsum("nij,nj->ni", affine, dpos)
        flat = (node[:, 0] * g + node[:, 1]) * g + node[:, 2]
        grid_v.index_add_(0, flat, weight.unsqueeze(-1) * contrib)
        grid_m.index_add_(0, flat, weight * state.mass)

    grid_v = grid_v / grid_m.clamp_min(1e-10).unsqueeze(-1)
    grid_v = grid_v * (1.0 - p.drag)

    # G2P: gather back to particles.
    new_v = torch.zeros_like(state.v)
    new_C = torch.zeros_like(state.C)
    for oi in range(27):
        off = offsets[oi]
        node = (base_c + off).clamp(0, g - 1)
        weight = w[:, 0, off[0]] * w[:, 1, off[1]] * w[:, 2, off[2]]
        dpos = (off.to(fx.dtype) - fx) * p.dx
        flat = (node[:, 0] * g + node[:, 1]) * g + node[:, 2]
        gv = grid_v.index_select(0, flat)  # [N,3]
        new_v = new_v + weight.unsqueeze(-1) * gv
        new_C = new_C + (4.0 / (p.dx ** 2)) * weight.view(-1, 1, 1) * torch.einsum("ni,nj->nij", gv, dpos)

    new_x = state.x + p.dt * new_v

    out = MPMState.__new__(MPMState)
    out.x = new_x
    out.v = new_v
    out.F = F_new
    out.C = new_C
    out.volume0 = state.volume0
    out.mass = state.mass
    return out


def substep_checkpointed(state: MPMState, p: MPMParams,
                         external_velocity: torch.Tensor | None = None) -> MPMState:
    """``substep`` under gradient checkpointing: the substep's internal graph is not
    stored, it is recomputed during backward (identical values, ~1/3 more compute,
    far less memory -- needed at 20K+ particles where the stored graph exceeds RAM)."""
    from torch.utils.checkpoint import checkpoint

    def fn(x, v, F, C, ev):
        s = MPMState.__new__(MPMState)
        s.x, s.v, s.F, s.C = x, v, F, C
        s.volume0, s.mass = state.volume0, state.mass
        o = substep(s, p, external_velocity=ev)
        return o.x, o.v, o.F, o.C

    x, v, F, C = checkpoint(fn, state.x, state.v, state.F, state.C, external_velocity, use_reentrant=False)
    out = MPMState.__new__(MPMState)
    out.x, out.v, out.F, out.C = x, v, F, C
    out.volume0, out.mass = state.volume0, state.mass
    return out


def simulate_substeps(state: MPMState, p: MPMParams, n_substeps: int,
                       external_velocity: torch.Tensor | None = None,
                       checkpoint: bool = False) -> MPMState:
    """Run ``n_substeps`` MLS-MPM substeps. ``external_velocity`` (if given)
    is applied only on the first substep of the call (matches a per-frame
    velocity "kick" control input, see ``joint_opt.py``)."""
    for i in range(n_substeps):
        ev = external_velocity if i == 0 else None
        state = (substep_checkpointed if checkpoint else substep)(state, p, external_velocity=ev)
    return state
