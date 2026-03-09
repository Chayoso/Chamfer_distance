#!/usr/bin/env python3
"""Generate pairwise configs for 10 shapes × 2 modes (physics-only + coupled).

Shapes: sphere(isosphere), bunny, spot, C, dragon, E, teapot, V, armadilo, bob
Pairs:  10 × 9 = 90 directed pairs
Modes:  physics_only (no chamfer) + coupled_v2 (chamfer + coupled rev + early stop + pw_final=0.5)

Usage:
    python tools/gen_pairwise_configs_v2.py
"""
from pathlib import Path
from itertools import permutations

SHAPES = ['sphere', 'bunny', 'spot', 'C', 'dragon', 'E', 'teapot', 'V', 'armadilo', 'bob']
MESH_MAP = {s: f'assets/{s}.obj' for s in SHAPES}
MESH_MAP['sphere'] = 'assets/isosphere.obj'  # sphere uses isosphere.obj

# ── Physics-only template (no chamfer, full physics weight) ──
TEMPLATE_PHYSICS = """\
# Pairwise PHYSICS-ONLY: {src} -> {tgt}
# 40 episodes, no chamfer, full physics weight

input_mesh_path: "{src_mesh}"
target_mesh_path: "{tgt_mesh}"
output_dir: "output/pairwise_physics_only/{name}/"

simulation:
  grid_dx: 1
  points_per_cell_cuberoot: 4
  grid_min_point:
  - -16.0
  - -16.0
  - -16.0
  grid_max_point:
  - 16.0
  - 16.0
  - 16.0
  lam: 38888.89
  mu: 58333.3
  density: 75.0
  dt: 0.00833333333
  drag: 0.5
  external_force:
  - 0.0
  - 0.0
  - 0.0
  smoothing_factor: 0.955
optimization:
  num_animations: 40
  num_timesteps: 10
  control_stride: 1
  num_passes: 3
  max_gd_iters: 1
  max_ls_iters: 5
  initial_alpha: 0.01
  adaptive_alpha_enabled: true
  adaptive_alpha_target_norm: 2500.0
  adaptive_alpha_min_scale: 0.1
  gd_tol: 0.0
  use_session_mode: true
  use_pcgrad: false
  w_render_base: 0.0
  j_barrier_weight: 0.0
  j_barrier_target: 0.8
  render_adam_enabled: false
  w_chamfer: 0.0
  chamfer_in_callback: false
  loss:
    enabled: true
    w_alpha: 0.0
    w_depth: 0.0
    w_photo: 0.0
    w_edge: 0.0
    w_cov_align: 0.0
    w_cov_reg: 0.0
    w_det_barrier: 0.0
    schedule: constant
upsample:
  use_simple_pipeline: true
  render_loss_weight: 1.0
  debug:
    verbose: false
  covariance:
    use_curvature_for_target: true
    sigma_isotropic: 0.038
    sigma0: 0.25
    k_F: 32
    use_multiscale_F: true
    k_F_coarse: 64
    k_F_fine: 16
    multiscale_blend_mode: adaptive
    enable_subdivision: true
    subdivision_target: 100000
    subdivision_jitter: 0.15
    sv_min: 0.6
    sv_max: 1.3
camera:
  width: 640
  height: 360
  fx: 237.5
  fy: 237.5
  cx: 320.0
  cy: 180.0
  znear: 0.01
  zfar: 100.0
  lookat:
    eye:
    - 20.0
    - -25.0
    - 12.5
    target:
    - 0.0
    - 0.0
    - 0.0
    up:
    - 0.0
    - 0.0
    - 1.0
cameras:
- eye:
  - -25.0
  - 20.0
  - 12.5
  target:
  - 0.0
  - 0.0
  - 0.0
  up:
  - 0.0
  - 0.0
  - 1.0
render:
  num_frames: 1
  schedule: uniform
  bg:
  - 1.0
  - 1.0
  - 1.0
  training_resolution_scale: 0.5
  particle_color:
  - 0.27
  - 0.51
  - 0.71
  surface_mask_ratio: 0.2
  surface_mask_mode: last
"""

# ── Coupled v2 template (chamfer + coupled rev + higher physics_weight_final) ──
TEMPLATE_COUPLED = """\
# Pairwise COUPLED v2: {src} -> {tgt}
# coupled rev, rev_base=0.3, physics_final=0.5, early_stop_patience=8

input_mesh_path: "{src_mesh}"
target_mesh_path: "{tgt_mesh}"
output_dir: "output/pairwise_coupled_v2/{name}/"

simulation:
  grid_dx: 1
  points_per_cell_cuberoot: 4
  grid_min_point:
  - -16.0
  - -16.0
  - -16.0
  grid_max_point:
  - 16.0
  - 16.0
  - 16.0
  lam: 38888.89
  mu: 58333.3
  density: 75.0
  dt: 0.00833333333
  drag: 0.5
  external_force:
  - 0.0
  - 0.0
  - 0.0
  smoothing_factor: 0.955
optimization:
  num_animations: 40
  num_timesteps: 10
  control_stride: 1
  num_passes: 3
  max_gd_iters: 1
  max_ls_iters: 5
  initial_alpha: 0.01
  adaptive_alpha_enabled: true
  adaptive_alpha_target_norm: 2500.0
  adaptive_alpha_min_scale: 0.1
  gd_tol: 0.0
  use_session_mode: true
  use_pcgrad: false
  w_render_base: 0.0
  j_barrier_weight: 0.0
  j_barrier_target: 0.8
  render_adam_enabled: false
  w_chamfer: 10.0
  chamfer_start_ep: 5
  chamfer_ramp_ep: 10
  chamfer_rev_weight: 0.3
  chamfer_rev_mode: coupled
  physics_weight_start_ep: 15
  physics_weight_ramp_ep: 5
  physics_weight_final: 0.5
  smoothing_start_ep: 15
  smoothing_value: 0.7
  chamfer_in_callback: false
  chamfer_early_stop_patience: 8
  loss:
    enabled: true
    w_alpha: 0.0
    w_depth: 0.0
    w_photo: 0.0
    w_edge: 0.0
    w_cov_align: 0.0
    w_cov_reg: 0.0
    w_det_barrier: 0.0
    schedule: constant
upsample:
  use_simple_pipeline: true
  render_loss_weight: 1.0
  debug:
    verbose: false
  covariance:
    use_curvature_for_target: true
    sigma_isotropic: 0.038
    sigma0: 0.25
    k_F: 32
    use_multiscale_F: true
    k_F_coarse: 64
    k_F_fine: 16
    multiscale_blend_mode: adaptive
    enable_subdivision: true
    subdivision_target: 100000
    subdivision_jitter: 0.15
    sv_min: 0.6
    sv_max: 1.3
camera:
  width: 640
  height: 360
  fx: 237.5
  fy: 237.5
  cx: 320.0
  cy: 180.0
  znear: 0.01
  zfar: 100.0
  lookat:
    eye:
    - 20.0
    - -25.0
    - 12.5
    target:
    - 0.0
    - 0.0
    - 0.0
    up:
    - 0.0
    - 0.0
    - 1.0
cameras:
- eye:
  - -25.0
  - 20.0
  - 12.5
  target:
  - 0.0
  - 0.0
  - 0.0
  up:
  - 0.0
  - 0.0
  - 1.0
render:
  num_frames: 1
  schedule: uniform
  bg:
  - 1.0
  - 1.0
  - 1.0
  training_resolution_scale: 0.5
  particle_color:
  - 0.27
  - 0.51
  - 0.71
  surface_mask_ratio: 0.2
  surface_mask_mode: last
"""


def main():
    pairs = [(s, t) for s, t in permutations(SHAPES, 2)]
    print(f'Generating configs for {len(pairs)} directed pairs × 2 modes...\n')

    for mode, template, cfg_dir in [
        ('physics_only', TEMPLATE_PHYSICS, Path('configs/pairwise_physics_only')),
        ('coupled_v2',   TEMPLATE_COUPLED, Path('configs/pairwise_coupled_v2')),
    ]:
        cfg_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for src, tgt in pairs:
            name = f'{src}_to_{tgt}'
            cfg_path = cfg_dir / f'{name}.yaml'
            content = template.format(
                src=src, tgt=tgt, name=name,
                src_mesh=MESH_MAP[src], tgt_mesh=MESH_MAP[tgt],
            )
            cfg_path.write_text(content)
            count += 1
        print(f'  {mode}: {count} configs in {cfg_dir}/')

    print(f'\nTotal: {len(pairs) * 2} config files generated.')


if __name__ == '__main__':
    main()
