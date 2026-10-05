"""One source -> target morphing run (physics-only or joint), returning the
same metrics reported in Tables 3-5: s->t / t->s / two-sided Chamfer
distance, Hausdorff distance, and F1-score at the final frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..metrics import f1_score, hausdorff_distance, two_sided_chamfer_np
from ..shapes import load_shape_mesh
from .joint_opt import MorphHistory, run_morph
from .particles import mesh_volume, sample_volume_points
from .presets import ScalePreset, make_morph_config


@dataclass
class PairResult:
    source: str
    target: str
    use_chamfer: bool
    fwd_rms: float
    rev_rms: float
    two_sided: float
    hausdorff: float
    f1: float
    history: MorphHistory


def run_pair(source_name: str, target_name: str, preset: ScalePreset, use_chamfer: bool,
             tau: float = 0.2, seed: int = 0, device: str = "cpu") -> PairResult:
    src_mesh, _ = load_shape_mesh(source_name)
    tgt_mesh, _ = load_shape_mesh(target_name)

    n = preset.n_particles
    scale = preset.object_scale
    src_pts = sample_volume_points(src_mesh, n, seed=seed) * scale
    tgt_pts = sample_volume_points(tgt_mesh, n, seed=seed + 1) * scale
    volume0 = mesh_volume(src_mesh) * (scale ** 3) / n

    cfg = make_morph_config(preset, use_chamfer=use_chamfer)
    final_state, history = run_morph(src_pts, tgt_pts, cfg, preset.mpm_params, volume0, device=device)

    final_np = final_state.x.detach().cpu().numpy()
    fwd_rms, rev_rms, two_sided = two_sided_chamfer_np(final_np, tgt_pts)
    haus = hausdorff_distance(final_np, tgt_pts)
    f1 = f1_score(final_np, tgt_pts, tau=tau)

    return PairResult(
        source=source_name, target=target_name, use_chamfer=use_chamfer,
        fwd_rms=fwd_rms, rev_rms=rev_rms, two_sided=two_sided,
        hausdorff=haus, f1=f1, history=history,
    )
