"""Physics-only and joint physics+Chamfer morphing (Sec. 4-5 of the main
paper; schedule constants from Supplementary Sec. "Optimization Parameters").

Per frame, a learnable per-particle velocity "kick" ``v_ctrl`` is applied at
the start of the frame's substeps and optimized, by backpropagating through
the differentiable MLS-MPM simulator, against

    L_total(t) = w_phys(t) * L_physics + w_ch(t) * (L_fwd + w_rev(t) * L_rev)

with the coupled reverse schedule w_rev(t) = w_rev_base * w_phys(t)
(Eq. 6), reverse-gradient clamping at kappa times the mean active
magnitude (Sec. 4, "Reverse gradient clamping"), 3 forward-backward passes
per frame with up to 5 backtracking line-search steps each, and a step
scaled toward a target gradient norm of 2500 (Supplementary Table 2).
``v_ctrl`` starts at zero each frame and is refined in place across the 3
passes. Setting ``use_chamfer=False`` (w_ch == 0 for all t) reproduces the
"physics-only" baseline reported throughout Tables 1-5.

See the package-level docstring in ``chamfer_collapse/mpm/__init__.py`` for
why this outer loop is a reconstruction of Xu et al. 2025's optimizer from
the parameters the paper specifies, not a transcription of their code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..chamfer import chamfer_components
from ..progress import bar, log
from .sim import MPMParams, MPMState, rasterize_mass, simulate_substeps

PHYSICS_EPS = 1e-6


@dataclass
class MorphConfig:
    n_frames: int = 8
    substeps_per_frame: int = 5
    n_passes: int = 3
    line_search_iters: int = 5
    initial_step_size: float = 0.01
    target_grad_norm: float = 2500.0

    use_chamfer: bool = True
    w_chamfer: float = 10.0
    chamfer_start_frame: int = 2
    chamfer_ramp_frames: int = 3
    w_rev_base: float = 1.0
    reverse_clamp_kappa: float = 3.0
    #: 'mean' (per-point mean, grad ~ 1/N) or 'sum' (each particle feels its own
    #: Chamfer gradient at full strength). The paper does not specify; see README.
    chamfer_reduction: str = "mean"
    checkpoint: bool = False   # recompute substeps in backward: same result, far less memory

    physics_decay_start: int = 4
    physics_decay_ramp: int = 2
    w_phys_final: float = 0.1


#: Exact Supplementary Table 2 schedule (paper scale; combine with
#: MPMParams(n_grid=32, dx=1.0, ...) and ~37K/89K particles for a faithful
#: -- but computationally heavy, see README -- run).
PAPER_MORPH_CONFIG = MorphConfig(
    n_frames=40,
    substeps_per_frame=10,
    n_passes=3,
    line_search_iters=5,
    initial_step_size=0.01,
    target_grad_norm=2500.0,
    use_chamfer=True,
    w_chamfer=10.0,
    chamfer_start_frame=5,
    chamfer_ramp_frames=10,
    w_rev_base=1.0,
    reverse_clamp_kappa=3.0,
    physics_decay_start=15,
    physics_decay_ramp=5,
    w_phys_final=0.1,
)


def w_phys_schedule(t: int, cfg: MorphConfig) -> float:
    if t < cfg.physics_decay_start:
        return 1.0
    if t < cfg.physics_decay_start + cfg.physics_decay_ramp:
        frac = (t - cfg.physics_decay_start) / cfg.physics_decay_ramp
        return 1.0 + frac * (cfg.w_phys_final - 1.0)
    return cfg.w_phys_final


def w_chamfer_schedule(t: int, cfg: MorphConfig) -> float:
    if not cfg.use_chamfer:
        return 0.0
    if t < cfg.chamfer_start_frame:
        return 0.0
    frac = min(1.0, (t - cfg.chamfer_start_frame) / cfg.chamfer_ramp_frames)
    return cfg.w_chamfer * frac


def w_rev_schedule(t: int, cfg: MorphConfig) -> float:
    return cfg.w_rev_base * w_phys_schedule(t, cfg)


def physics_loss(current_mass: torch.Tensor, target_mass: torch.Tensor) -> torch.Tensor:
    """Eq. (4): scale-invariant log-mass matching on the Eulerian grid."""
    diff = torch.log(current_mass + 1 + PHYSICS_EPS) - torch.log(target_mass + 1 + PHYSICS_EPS)
    return 0.5 * (diff ** 2).sum()


def _clamp_reverse_grad(grad: torch.Tensor, kappa: float) -> torch.Tensor:
    mags = grad.norm(dim=-1)
    active = mags > 1e-12
    if active.sum() == 0:
        return grad
    mean_active = mags[active].mean()
    thresh = kappa * mean_active
    scale = torch.where(mags > thresh, thresh / mags.clamp_min(1e-12), torch.ones_like(mags))
    return grad * scale.unsqueeze(-1)


@dataclass
class FrameRecord:
    frame: int
    fwd_rms: float
    rev_rms: float
    two_sided: float


@dataclass
class MorphHistory:
    frames: list = field(default_factory=list)  # list[FrameRecord]


def _simulate_frame(state: MPMState, p: MPMParams, cfg: MorphConfig, v_ctrl: torch.Tensor) -> MPMState:
    return simulate_substeps(state, p, cfg.substeps_per_frame, external_velocity=v_ctrl, checkpoint=cfg.checkpoint)


def _frame_loss_and_grad(state0: MPMState, p: MPMParams, cfg: MorphConfig, v_ctrl: torch.Tensor,
                          target_mass: torch.Tensor, target_points: torch.Tensor, t: int):
    """Simulate one frame from ``v_ctrl`` and return (total_loss_value,
    grad_wrt_v_ctrl, end_state). The reverse Chamfer term's contribution to
    the gradient is clamped before being combined with the rest (Sec. 4)."""
    v_ctrl = v_ctrl.detach().requires_grad_(True)
    end_state = _simulate_frame(state0, p, cfg, v_ctrl)

    wp = w_phys_schedule(t, cfg)
    wc = w_chamfer_schedule(t, cfg)
    wr = w_rev_schedule(t, cfg)

    current_mass = rasterize_mass(end_state.x, p)
    l_phys = physics_loss(current_mass, target_mass)

    total_value = wp * l_phys.detach()
    non_rev_loss = wp * l_phys

    if wc > 0:
        res = chamfer_components(end_state.x, target_points)
        red = (lambda a: a.sum()) if cfg.chamfer_reduction == "sum" else (lambda a: a.mean())
        l_fwd = red(res.fwd_sq)
        non_rev_loss = non_rev_loss + wc * l_fwd
        total_value = total_value + wc * l_fwd.detach()

        if wr > 0:
            l_rev = red(res.rev_sq)
            total_value = total_value + wc * wr * l_rev.detach()
            (grad_rev_raw,) = torch.autograd.grad(wc * wr * l_rev, end_state.x, retain_graph=True)
            grad_rev_clamped = _clamp_reverse_grad(grad_rev_raw, cfg.reverse_clamp_kappa)
        else:
            grad_rev_clamped = None
    else:
        grad_rev_clamped = None

    v_ctrl_grad_parts = []
    if v_ctrl.grad is not None:
        v_ctrl.grad = None
    non_rev_loss.backward(retain_graph=grad_rev_clamped is not None)
    if v_ctrl.grad is not None:
        v_ctrl_grad_parts.append(v_ctrl.grad.clone())
    if grad_rev_clamped is not None:
        if v_ctrl.grad is not None:
            v_ctrl.grad = None
        end_state.x.backward(gradient=grad_rev_clamped)
        if v_ctrl.grad is not None:
            v_ctrl_grad_parts.append(v_ctrl.grad.clone())

    total_grad = sum(v_ctrl_grad_parts) if v_ctrl_grad_parts else torch.zeros_like(v_ctrl)
    return float(total_value.item()), total_grad, end_state


def optimize_frame(state0: MPMState, p: MPMParams, cfg: MorphConfig,
                    target_mass: torch.Tensor, target_points: torch.Tensor, t: int) -> MPMState:
    """3 forward-backward passes with backtracking line search, refining a
    per-frame control ``v_ctrl`` initialized to zero (Supplementary Table 2:
    "Forward-backward passes: 3", "GD iterations per pass: 1", "Line search
    iterations: 5")."""
    n = state0.x.shape[0]
    v_ctrl = torch.zeros(n, 3, dtype=state0.x.dtype, device=state0.x.device)
    best_state = None

    for _pass in range(cfg.n_passes):
        loss0, grad, end_state = _frame_loss_and_grad(state0, p, cfg, v_ctrl, target_mass, target_points, t)
        best_state = end_state
        grad_norm = grad.norm()
        if grad_norm < 1e-10:
            continue
        # rescale the (raw) update toward the target norm, then backtrack.
        direction = grad * (cfg.target_grad_norm / grad_norm)
        accepted = False
        for ls in range(cfg.line_search_iters):
            step = cfg.initial_step_size * (0.5 ** ls)
            trial_v_ctrl = v_ctrl - step * direction
            trial_loss, _, trial_state = _frame_loss_and_grad(
                state0, p, cfg, trial_v_ctrl, target_mass, target_points, t
            )
            if trial_loss <= loss0:
                v_ctrl = trial_v_ctrl.detach()
                best_state = trial_state
                accepted = True
                break
        if not accepted:
            break  # no improving step found; keep the pass's own end_state

    return best_state


def run_morph(source_points, target_points, cfg: MorphConfig, mpm_params: MPMParams,
              volume0: float, device: str = "cpu") -> tuple[MPMState, MorphHistory]:
    """Run the full ``cfg.n_frames``-frame morph from ``source_points`` to
    ``target_points`` (both [N,3] / [M,3] tensors or arrays)."""
    x0 = torch.as_tensor(source_points, dtype=torch.float32, device=device)
    target_t = torch.as_tensor(target_points, dtype=torch.float32, device=device)
    target_mass = rasterize_mass(target_t, mpm_params)

    state = MPMState(x0, volume0, mpm_params.density, device=device)
    history = MorphHistory()

    pbar = bar(cfg.n_frames, f"MPM {'joint' if cfg.use_chamfer else 'physics'} N={x0.shape[0]}")
    for t in range(cfg.n_frames):
        state = optimize_frame(state, mpm_params, cfg, target_mass, target_t, t)
        with torch.no_grad():
            res = chamfer_components(state.x.detach(), target_t)
            history.frames.append(FrameRecord(
                frame=t, fwd_rms=res.fwd_rms.item(), rev_rms=res.rev_rms.item(), two_sided=res.two_sided.item()
            ))
        state = _detach_state(state)
        pbar.update(1)
        pbar.set_postfix(cd=f"{history.frames[-1].two_sided:.4f}")
        log(f"frame {t + 1}/{cfg.n_frames} cd={history.frames[-1].two_sided:.4f}")
    pbar.close()
    return state, history


def _detach_state(state: MPMState) -> MPMState:
    out = MPMState.__new__(MPMState)
    out.x = state.x.detach()
    out.v = state.v.detach()
    out.F = state.F.detach()
    out.C = state.C.detach()
    out.volume0 = state.volume0
    out.mass = state.mass.detach()
    return out
