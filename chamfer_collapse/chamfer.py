"""Chamfer distance, exactly as defined in Sec. 3.1 of the main paper and
Sec. 1 ("Setup") of the supplementary material.

    CD = sqrt( mean(d_fwd^2) + mean(d_rev^2) )

with d_fwd the source->target nearest-neighbor distance and d_rev the
target->source nearest-neighbor distance. The two directional components
reported throughout the paper's tables are the *per-direction RMS
distances* sqrt(mean(d_fwd^2)) and sqrt(mean(d_rev^2)) -- verified against
Table 1 of the main paper: sqrt(0.220^2 + 0.182^2) = 0.2855 =~ 0.286.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .geometry import nn_indices


@dataclass
class ChamferResult:
    fwd_sq: torch.Tensor   # [N] squared distances, source -> target
    rev_sq: torch.Tensor   # [M] squared distances, target -> source
    fwd_idx: torch.Tensor  # [N] index into target
    rev_idx: torch.Tensor  # [M] index into source

    @property
    def fwd_rms(self) -> torch.Tensor:
        return self.fwd_sq.mean().sqrt()

    @property
    def rev_rms(self) -> torch.Tensor:
        return self.rev_sq.mean().sqrt()

    @property
    def two_sided(self) -> torch.Tensor:
        return (self.fwd_sq.mean() + self.rev_sq.mean()).sqrt()


def chamfer_components(source: torch.Tensor, target: torch.Tensor) -> ChamferResult:
    """Per-point squared nearest-neighbor distances in both directions.

    ``source`` must require grad for the returned tensors to be
    differentiable w.r.t. particle positions; ``target`` is always treated
    as a constant (as in every experiment in the paper).
    """
    fwd_idx = nn_indices(source, target)
    rev_idx = nn_indices(target, source)
    fwd_nn = target.index_select(0, fwd_idx)
    rev_nn = source.index_select(0, rev_idx)
    fwd_sq = ((source - fwd_nn) ** 2).sum(-1)
    rev_sq = ((target - rev_nn) ** 2).sum(-1)
    return ChamferResult(fwd_sq=fwd_sq, rev_sq=rev_sq, fwd_idx=fwd_idx, rev_idx=rev_idx)


def forward_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L_fwd of Eq. (7) in the supplementary material: mean(||p_i - NN_T(p_i)||^2)."""
    fwd_idx = nn_indices(source, target)
    fwd_nn = target.index_select(0, fwd_idx)
    return ((source - fwd_nn) ** 2).sum(-1).mean()


def reverse_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L_rev of Eq. (2)/(8): mean(||q_j - NN_S(q_j)||^2)."""
    rev_idx = nn_indices(target, source)
    rev_nn = source.index_select(0, rev_idx)
    return ((target - rev_nn) ** 2).sum(-1).mean()


def two_sided_chamfer(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    res = chamfer_components(source, target)
    return res.two_sided


def chamfer_gradient_direction(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Explicit closed form of Eq. (1): (up to the 2/N scalar) p - NN_T(p).

    Provided so Proposition 1 can be checked against both the closed form
    and PyTorch's autograd result (see ``tests/test_propositions.py``).
    """
    fwd_idx = nn_indices(source, target)
    return source - target.index_select(0, fwd_idx)
