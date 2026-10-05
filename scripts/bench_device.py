"""CPU vs GPU timing + agreement for the MPM morph and DCO.
    nix develop .#gpu --command python3 scripts/bench_device.py 2> results/logs/bench.log
Progress bars -> stdout, log -> stderr. Each (N, device) is timed on a short run (2 frames /
60 DCO steps) after a warm-up, so numbers are per-run costs, not full-experiment times."""
import argparse, time
import numpy as np
import torch

from chamfer_collapse.chamfer import chamfer_components
from chamfer_collapse.dco import DCOConfig, run_dco
from chamfer_collapse.experiments.common import print_table, write_csv
from chamfer_collapse.mpm.joint_opt import MorphConfig, run_morph
from chamfer_collapse.mpm.particles import mesh_volume, sample_volume_points
from chamfer_collapse.mpm.presets import DEFAULT
from chamfer_collapse.mpm.sim import MPMParams
from chamfer_collapse.progress import log
from chamfer_collapse.shapes import load_shape_mesh, load_shape_points

# (n_particles, n_grid, object_scale), matching physics_scaling_sweep.CONFIGS
SIZES = [(800, 16, 4.0), (3000, 16, 4.0), (8000, 24, 8.0), (20000, 32, 12.0), (37000, 32, 16.0)]


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def time_morph(n, grid, scale, dev, frames=2):
    sm, _ = load_shape_mesh("sphere"); tm, _ = load_shape_mesh("bunny")
    src = sample_volume_points(sm, n, 0) * scale; tgt = sample_volume_points(tm, n, 1) * scale
    vol = mesh_volume(sm) * scale ** 3 / n
    kw = dict(DEFAULT.morph_config_kwargs); kw.update(n_frames=frames, initial_step_size=0.6)
    cfg = MorphConfig(use_chamfer=True, chamfer_start_frame=0, chamfer_ramp_frames=1, **{k: v for k, v in kw.items() if k not in ("chamfer_start_frame", "chamfer_ramp_frames")})
    params = MPMParams(n_grid=grid, dx=1.0, dt=8.33e-3, drag=0.5)
    sync(dev); t = time.time()
    st, h = run_morph(src, tgt, cfg, params, vol, device=dev)
    sync(dev)
    return time.time() - t, h.frames[-1].two_sided / scale


def time_dco(n, dev, steps=60):
    s, _ = load_shape_points("sphere", n, seed=0); t, _ = load_shape_points("bunny", n, seed=1)
    cfg = DCOConfig(n_frames=2, steps_per_frame=steps // 2, lr=0.01 / 4)
    sync(dev); t0 = time.time()
    pts, _ = run_dco(s * 4, t * 4, cfg, device=dev)
    sync(dev)
    cd = chamfer_components(pts.cpu(), torch.as_tensor(t * 4, dtype=torch.float32)).two_sided.item() / 4
    return time.time() - t0, cd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-max", type=int, default=8000, help="skip CPU timing above this N (slow)")
    ap.add_argument("--sizes", default="0,1,2,3,4")
    a = ap.parse_args()
    has_cuda = torch.cuda.is_available()
    log(f"cuda={has_cuda} device={torch.cuda.get_device_name(0) if has_cuda else '-'}")
    if has_cuda:                                  # warm-up: cuBLAS/cdist init, allocator
        time_morph(800, 16, 4.0, "cuda", frames=1); time_dco(800, "cuda", steps=4)
    rows = []
    for i in map(int, a.sizes.split(",")):
        n, grid, scale = SIZES[i]
        for kind, fn in (("MPM 2 frames", lambda d: time_morph(n, grid, scale, d)), ("DCO 60 steps", lambda d: time_dco(n, d))):
            row = {"N": n, "task": kind}
            for dev in (["cpu"] if n <= a.cpu_max else []) + (["cuda"] if has_cuda else []):
                sec, cd = fn(dev)
                row[f"{dev}_s"] = round(sec, 1); row[f"{dev}_cd"] = round(cd, 5)
                log(f"N={n} {kind} {dev}: {sec:.1f}s cd={cd:.5f}")
            if "cpu_s" in row and "cuda_s" in row:
                row["speedup"] = round(row["cpu_s"] / row["cuda_s"], 2)
            rows.append(row)
        keys = sorted({k for r in rows for k in r}, key=lambda k: ["N", "task", "cpu_s", "cuda_s", "speedup", "cpu_cd", "cuda_cd"].index(k) if k in ["N", "task", "cpu_s", "cuda_s", "speedup", "cpu_cd", "cuda_cd"] else 99)
        print_table([{k: r.get(k, "-") for k in keys} for r in rows], "CPU vs GPU")
        write_csv([{k: r.get(k, "-") for k in keys} for r in rows], "bench_device.csv")


if __name__ == "__main__":
    main()
