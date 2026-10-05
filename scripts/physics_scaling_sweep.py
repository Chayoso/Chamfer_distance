"""Does the physics-only baseline close the gap to DCO as resources grow toward the
paper's (37K particles, 40 frames)?  Sphere -> bunny at several (grid, scale, N, frames)
configs. For each: calibrate the control step on PHYSICS-ONLY alone, then run joint at the
same step, and DCO at the same N with scale-equivalent lr. CDs are reported divided by the
object scale so rows are comparable.

Console: progress bars -> stdout, log + tables -> stderr.
    python scripts/physics_scaling_sweep.py 2> sweep.log
    python scripts/physics_scaling_sweep.py --configs 0,1,2 --steps 0.3,0.6,1.2
"""
import argparse, dataclasses, time
import torch

from chamfer_collapse.chamfer import chamfer_components
from chamfer_collapse.dco import DCOConfig, run_dco
from chamfer_collapse.experiments.common import print_table, write_csv
from chamfer_collapse.mpm.joint_opt import MorphConfig, run_morph
from chamfer_collapse.mpm.particles import mesh_volume, sample_volume_points
from chamfer_collapse.mpm.presets import DEFAULT
from chamfer_collapse.mpm.sim import MPMParams
from chamfer_collapse.progress import bar, log
from chamfer_collapse.shapes import load_shape_mesh, load_shape_points

# (label, n_grid, object_scale, n_particles, n_frames)
CONFIGS = [
    ("800 / 8f / grid16", 16, 4.0, 800, 8),
    ("3K / 12f / grid16", 16, 4.0, 3000, 12),
    ("8K / 16f / grid24", 24, 8.0, 8000, 16),
    ("20K / 24f / grid32", 32, 12.0, 20000, 24),
    ("37K / 40f / grid32 (paper)", 32, 16.0, 37000, 40),
]


CHECKPOINT = False


def morph_cfg(n_frames, step, use_chamfer):
    kw = dict(DEFAULT.morph_config_kwargs)
    scale = n_frames / 8
    kw.update(n_frames=n_frames, initial_step_size=step, chamfer_start_frame=max(1, round(1 * scale)),
              chamfer_ramp_frames=max(2, round(2 * scale)), physics_decay_start=round(3 * scale),
              physics_decay_ramp=max(2, round(2 * scale)))
    return MorphConfig(use_chamfer=use_chamfer, checkpoint=CHECKPOINT, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="0,1,2", help="indices into CONFIGS (default 0,1,2)")
    ap.add_argument("--steps", default="0.3,0.6,1.2", help="control step sizes to calibrate on physics-only")
    ap.add_argument("--out", default="physics_scaling_sweep.csv", help="CSV name under results/")
    ap.add_argument("--checkpoint", action="store_true", help="gradient checkpointing (use for >=20K particles)")
    a = ap.parse_args()
    global CHECKPOINT
    CHECKPOINT = a.checkpoint
    steps = [float(x) for x in a.steps.split(",")]
    sm, sreal = load_shape_mesh("sphere"); tm, treal = load_shape_mesh("bunny")
    log(f"meshes: sphere real={sreal}, bunny real={treal}")
    rows = []
    outer = bar(len(a.configs.split(",")), "configs", leave=True)
    for ci in map(int, a.configs.split(",")):
        label, n_grid, scale, n, frames = CONFIGS[ci]
        log(f"=== config {label} ===")
        params = MPMParams(n_grid=n_grid, dx=1.0, dt=8.33e-3, drag=0.5)
        src = sample_volume_points(sm, n, 0) * scale; tgt = sample_volume_points(tm, n, 1) * scale
        vol = mesh_volume(sm) * scale ** 3 / n
        t0 = time.time()
        best = None
        for st in steps:                       # calibrate on physics-only alone
            _, h = run_morph(src, tgt, morph_cfg(frames, st, False), params, vol)
            cd = h.frames[-1].two_sided
            log(f"  calib step={st}: physics CD={cd / scale:.4f} (normalized)")
            if best is None or cd < best[1]:
                best = (st, cd)
        st = best[0]
        _, hj = run_morph(src, tgt, morph_cfg(frames, st, True), params, vol)
        s_dco, _ = load_shape_points("sphere", n, seed=0); t_dco, _ = load_shape_points("bunny", n, seed=1)
        pts, _ = run_dco(s_dco * scale, t_dco * scale, DCOConfig(lr=0.01 * scale / 16))
        cd_dco = chamfer_components(pts, torch.as_tensor(t_dco * scale, dtype=torch.float32)).two_sided.item()
        rows.append({"config": label, "step": st, "physics": best[1] / scale,
                     "joint": hj.frames[-1].two_sided / scale, "DCO": cd_dco / scale,
                     "phys/DCO": best[1] / cd_dco, "joint/phys": hj.frames[-1].two_sided / best[1],
                     "minutes": round((time.time() - t0) / 60, 1)})
        print_table(rows, "Physics-only vs joint vs DCO (CD / object scale)")
        write_csv(rows, a.out)
        outer.update(1)
    outer.close()


if __name__ == "__main__":
    main()
