"""Scale presets tying together particle count, grid resolution, and the
morphing schedule. Pure-PyTorch MLS-MPM on CPU cannot execute the paper's
exact particle counts / frame counts in reasonable time (see the README's
"Scale and runtime" section for the arithmetic): a single paper-scale run
(~37K particles, 32^3 grid, 40 frames, 3 passes/frame) is on the order of
hours on CPU. Three tiers are provided:

``QUICK``   -- smoke-test scale, seconds per run. Used by the test suite
               and as a sanity check before a longer run.
``DEFAULT`` -- what ``experiments/*.py`` run out of the box: small enough
               that the full pairwise sweep (Table 4, 20 pairs x 2
               conditions) finishes in single-digit minutes on a laptop
               CPU, while still being large enough to show the qualitative
               physics-only vs. joint-optimization gap.
``PAPER``   -- Supplementary Tables 1 & 2 exactly (32^3 grid, ~37K / ~89K
               particles, 40 frames). Pass ``--paper`` to any experiment
               script; expect it to need a GPU and a long time.
"""

from __future__ import annotations

from dataclasses import dataclass

from .joint_opt import MorphConfig
from .sim import MPMParams


@dataclass
class ScalePreset:
    name: str
    n_particles: int
    object_scale: float  # multiplies the unit-normalized mesh
    mpm_params: MPMParams
    morph_config_kwargs: dict


QUICK = ScalePreset(
    name="quick",
    n_particles=150,
    object_scale=3.0,
    mpm_params=MPMParams(n_grid=10, dx=1.0, dt=8.33e-3, drag=0.5),
    morph_config_kwargs=dict(
        n_frames=3, substeps_per_frame=3, n_passes=1, line_search_iters=1,
        chamfer_start_frame=1, chamfer_ramp_frames=1,
        physics_decay_start=1, physics_decay_ramp=1,
    ),
)

DEFAULT = ScalePreset(
    name="default",
    n_particles=800,
    object_scale=4.0,
    mpm_params=MPMParams(n_grid=16, dx=1.0, dt=8.33e-3, drag=0.5),
    morph_config_kwargs=dict(
        n_frames=8, substeps_per_frame=4, n_passes=2, line_search_iters=2,
        initial_step_size=0.6,  # calibrated on physics-only alone (step 0.01 barely moves anything)
        chamfer_start_frame=1, chamfer_ramp_frames=2,
        physics_decay_start=3, physics_decay_ramp=2,
    ),
)

PAPER = ScalePreset(
    name="paper",
    n_particles=37_000,  # 4 particles per cell -> ~89_000 for the dragon case study
    object_scale=16.0,   # matches the paper's [-16,16]^3 domain at dx=1
    mpm_params=MPMParams(n_grid=32, dx=1.0, dt=8.33e-3, drag=0.5),
    morph_config_kwargs=dict(
        n_frames=40, substeps_per_frame=10, n_passes=3, line_search_iters=5,
        chamfer_start_frame=5, chamfer_ramp_frames=10,
        physics_decay_start=15, physics_decay_ramp=5,
    ),
)

PRESETS = {"quick": QUICK, "default": DEFAULT, "paper": PAPER}


def make_morph_config(preset: ScalePreset, use_chamfer: bool) -> MorphConfig:
    return MorphConfig(use_chamfer=use_chamfer, **preset.morph_config_kwargs)
