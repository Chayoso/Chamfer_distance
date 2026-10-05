"""Direct Chamfer Optimization (DCO): particle positions are optimized
directly against the Chamfer objective, with no physics simulation, no
morphing framework, and no temporal structure (Sec. 3.2). This isolates the
Chamfer gradient in its purest form and is the baseline behind Tables 1, 2
and 5, plus the ablation conditions ("+ Repulsion", "+ Rep. + Smooth",
"+ Rep. + Smooth + Vol.", "DCD") of Table 1.

Optimizer / schedule: AMSGrad, lr=0.01, 40 "frames" of 30 steps each
(Sec. 3.1) -- since DCO has no physics substeps, a "frame" is just a
bookkeeping unit; the run is 1200 total gradient steps, evaluated at the
final frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .chamfer import chamfer_components
from .progress import bar, log
from .dcd import dcd_forward_loss, dcd_reverse_loss
from .regularizers import laplacian_smoothness_loss, local_volume_proxy, repulsion_loss, volume_preservation_loss

N_FRAMES = 40
STEPS_PER_FRAME = 30
LR = 0.01


@dataclass
class DCOConfig:
    use_dcd: bool = False
    repulsion_weight: float = 0.0
    repulsion_k: int = 6
    smoothness_weight: float = 0.0
    smoothness_k: int = 6
    volume_weight: float = 0.0
    volume_k: int = 6
    n_frames: int = N_FRAMES
    steps_per_frame: int = STEPS_PER_FRAME
    lr: float = LR
    dcd_k: int = 8


@dataclass
class DCOHistory:
    fwd_rms: list = field(default_factory=list)
    rev_rms: list = field(default_factory=list)
    two_sided: list = field(default_factory=list)


def run_dco(source_init, target, cfg: DCOConfig, device: str = "cpu", record_every: int = 30):
    """Optimize source points against (regularized / density-aware) Chamfer
    loss. Returns (final_points [N,3] tensor, history)."""
    target_t = torch.as_tensor(target, dtype=torch.float32, device=device)
    points = torch.as_tensor(source_init, dtype=torch.float32, device=device).clone()
    points.requires_grad_(True)

    ref_volume = None
    if cfg.volume_weight > 0:
        with torch.no_grad():
            ref_volume = local_volume_proxy(points, k=cfg.volume_k).clone()

    optimizer = torch.optim.Adam([points], lr=cfg.lr, amsgrad=True)
    history = DCOHistory()

    total_steps = cfg.n_frames * cfg.steps_per_frame
    pbar = bar(total_steps, f"DCO{'/DCD' if cfg.use_dcd else ''} N={points.shape[0]}")
    for step in range(total_steps):
        optimizer.zero_grad()

        if cfg.use_dcd:
            fwd = dcd_forward_loss(points, target_t, k=cfg.dcd_k)
            rev = dcd_reverse_loss(points, target_t, k=cfg.dcd_k)
        else:
            res = chamfer_components(points, target_t)
            fwd = res.fwd_sq.mean()
            rev = res.rev_sq.mean()

        loss = fwd + rev
        if cfg.repulsion_weight > 0:
            loss = loss + cfg.repulsion_weight * repulsion_loss(points, k=cfg.repulsion_k)
        if cfg.smoothness_weight > 0:
            loss = loss + cfg.smoothness_weight * laplacian_smoothness_loss(points, k=cfg.smoothness_k)
        if cfg.volume_weight > 0:
            loss = loss + cfg.volume_weight * volume_preservation_loss(points, ref_volume, k=cfg.volume_k)

        loss.backward()
        optimizer.step()

        if (step + 1) % record_every == 0 or step == total_steps - 1:
            with torch.no_grad():
                res = chamfer_components(points.detach(), target_t)
                history.fwd_rms.append(res.fwd_rms.item())
                history.rev_rms.append(res.rev_rms.item())
                history.two_sided.append(res.two_sided.item())
            pbar.update(min(record_every, total_steps - pbar.n))
            pbar.set_postfix(cd=f"{res.two_sided.item():.4f}")
            log(f"DCO step {step + 1}/{total_steps} fwd={res.fwd_rms.item():.4f} rev={res.rev_rms.item():.4f} cd={res.two_sided.item():.4f}")

    pbar.close()
    return points.detach(), history
