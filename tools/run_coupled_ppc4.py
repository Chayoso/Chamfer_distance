#!/usr/bin/env python3
"""
Run coupled experiments for 5 shapes at PPC=4.
Generates configs and runs physics-only / coupled-L2 / coupled-Huber in parallel.

Usage:
    # Generate configs only
    python tools/run_coupled_ppc4.py --gen-only

    # Run all 15 experiments with max 3 parallel
    python tools/run_coupled_ppc4.py --parallel 3

    # Run specific condition
    python tools/run_coupled_ppc4.py --condition physics_only --parallel 5
"""
import os, sys, yaml, subprocess, argparse, time, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHAPES = ['bunny', 'dragon', 'bob', 'teapot', 'spot']
CONDITIONS = ['physics_only', 'coupled_l2', 'coupled_huber']

# Base simulation config (PPC=4)
BASE_SIM = {
    'grid_dx': 1,
    'points_per_cell_cuberoot': 4,
    'grid_min_point': [-16.0, -16.0, -16.0],
    'grid_max_point': [16.0, 16.0, 16.0],
    'lam': 38888.89,
    'mu': 58333.3,
    'density': 75.0,
    'dt': 0.00833333333,
    'drag': 0.5,
    'external_force': [0.0, 0.0, 0.0],
    'smoothing_factor': 0.955,
}

# Base optimization (shared)
BASE_OPT = {
    'num_animations': 40,
    'num_timesteps': 10,
    'control_stride': 1,
    'num_passes': 3,
    'max_gd_iters': 1,
    'max_ls_iters': 5,
    'initial_alpha': 0.01,
    'adaptive_alpha_enabled': True,
    'adaptive_alpha_target_norm': 2500.0,
    'adaptive_alpha_min_scale': 0.1,
    'gd_tol': 0.0,
    'use_session_mode': True,
    'use_pcgrad': False,
    'w_render_base': 0.0,
    'j_barrier_weight': 0.0,
    'j_barrier_target': 0.8,
    'render_adam_enabled': False,
    'chamfer_in_callback': False,
    'loss': {
        'enabled': True,
        'w_alpha': 0.0, 'w_depth': 0.0, 'w_photo': 0.0,
        'w_edge': 0.0, 'w_cov_align': 0.0, 'w_cov_reg': 0.0,
        'w_det_barrier': 0.0, 'schedule': 'constant',
    },
}

# Condition-specific overrides
CONDITION_OVERRIDES = {
    'physics_only': {
        'w_chamfer': 0.0,
    },
    'coupled_l2': {
        'w_chamfer': 10.0,
        'chamfer_start_ep': 5,
        'chamfer_ramp_ep': 10,
        'chamfer_rev_weight': 0.3,
        'chamfer_rev_mode': 'coupled',
        'physics_weight_start_ep': 15,
        'physics_weight_ramp_ep': 5,
        'physics_weight_final': 0.5,
        'smoothing_start_ep': 15,
        'smoothing_value': 0.7,
        'chamfer_early_stop_patience': 8,
    },
    'coupled_huber': {
        'w_chamfer': 10.0,
        'chamfer_start_ep': 5,
        'chamfer_ramp_ep': 10,
        'chamfer_huber_delta': 1.5,
        'chamfer_rev_weight': 0.3,
        'chamfer_rev_mode': 'coupled',
        'physics_weight_start_ep': 15,
        'physics_weight_ramp_ep': 5,
        'physics_weight_final': 0.5,
        'smoothing_start_ep': 15,
        'smoothing_value': 0.7,
        'chamfer_early_stop_patience': 8,
    },
}

BASE_UPSAMPLE = {
    'use_simple_pipeline': True,
    'render_loss_weight': 1.0,
    'debug': {'verbose': False},
    'covariance': {
        'use_curvature_for_target': True,
        'sigma_isotropic': 0.038,
        'sigma0': 0.25,
        'k_F': 32,
        'use_multiscale_F': True,
        'k_F_coarse': 64,
        'k_F_fine': 16,
        'multiscale_blend_mode': 'adaptive',
        'enable_subdivision': True,
        'subdivision_target': 100000,
        'subdivision_jitter': 0.15,
        'sv_min': 0.6,
        'sv_max': 1.3,
    },
}


def gen_config(shape, condition):
    """Generate a config dict for a shape+condition combo."""
    out_dir = f"output/coupled_ppc4/{condition}/sphere_to_{shape}/"
    cfg_path = ROOT / f"configs/coupled_ppc4/{condition}/sphere_to_{shape}.yaml"

    opt = dict(BASE_OPT)
    opt.update(CONDITION_OVERRIDES[condition])

    cfg = {
        'input_mesh_path': 'assets/isosphere.obj',
        'target_mesh_path': f'assets/{shape}.obj',
        'output_dir': out_dir,
        'simulation': dict(BASE_SIM),
        'optimization': opt,
        'upsample': dict(BASE_UPSAMPLE),
        'camera': {
            'width': 640, 'height': 360,
            'fx': 237.5, 'fy': 237.5,
            'cx': 320.0, 'cy': 180.0,
            'znear': 0.01, 'zfar': 100.0,
            'lookat': {
                'eye': [20.0, -25.0, 12.5],
                'target': [0.0, 0.0, 0.0],
                'up': [0.0, 0.0, 1.0],
            },
        },
        'cameras': [
            {'eye': [-25.0, 20.0, 12.5], 'target': [0.0, 0.0, 0.0], 'up': [0.0, 0.0, 1.0]},
        ],
        'render': {
            'num_frames': 1,
            'schedule': 'uniform',
            'bg': [1.0, 1.0, 1.0],
            'training_resolution_scale': 0.5,
            'particle_color': [0.27, 0.51, 0.71],
            'surface_mask_ratio': 0.2,
            'surface_mask_mode': 'last',
        },
    }
    return cfg, cfg_path


def write_configs():
    """Generate all 15 configs."""
    paths = {}
    for condition in CONDITIONS:
        for shape in SHAPES:
            cfg, cfg_path = gen_config(shape, condition)
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cfg_path, 'w') as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            key = f"{condition}/{shape}"
            paths[key] = str(cfg_path)
            print(f"  Generated: {cfg_path}")
    return paths


def run_one(cfg_path, label):
    """Run a single experiment via run.py."""
    env = os.environ.copy()
    torch_lib = ""
    for p in sys.path:
        candidate = os.path.join(p, "torch", "lib")
        if os.path.isdir(candidate):
            torch_lib = candidate
            break
    if not torch_lib:
        # fallback
        torch_lib = os.popen("python -c \"import torch; print(torch.__file__.replace('__init__.py','lib'))\"").read().strip()
    if torch_lib:
        env['LD_LIBRARY_PATH'] = torch_lib + ':' + env.get('LD_LIBRARY_PATH', '')

    cmd = [sys.executable, str(ROOT / "run.py"), "-c", str(cfg_path)]
    log_path = str(cfg_path).replace("configs/", "output/").replace(".yaml", ".log")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"[START] {label} -> {log_path}")
    t0 = time.time()
    with open(log_path, 'w') as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                              cwd=str(ROOT), env=env, timeout=7200)
    dt = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
    print(f"[{status}] {label} ({dt/60:.1f} min)")
    return label, proc.returncode, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-only", action="store_true", help="Only generate configs")
    ap.add_argument("--parallel", type=int, default=3, help="Max parallel runs")
    ap.add_argument("--condition", choices=CONDITIONS + ['all'], default='all',
                    help="Which condition(s) to run")
    ap.add_argument("--shapes", nargs='+', default=SHAPES, help="Which shapes")
    args = ap.parse_args()

    print(f"=== Coupled PPC=4 Experiment ===")
    print(f"Shapes: {args.shapes}")
    print(f"Conditions: {args.condition if args.condition != 'all' else CONDITIONS}")
    print(f"Parallel: {args.parallel}")
    print()

    # Generate configs
    print("Generating configs...")
    paths = write_configs()
    print(f"Generated {len(paths)} configs.\n")

    if args.gen_only:
        print("--gen-only: stopping here.")
        return

    # Build run list
    conditions = CONDITIONS if args.condition == 'all' else [args.condition]
    tasks = []
    for condition in conditions:
        for shape in args.shapes:
            key = f"{condition}/{shape}"
            if key in paths:
                tasks.append((paths[key], key))

    print(f"Running {len(tasks)} experiments (max {args.parallel} parallel)...\n")
    t0_all = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(run_one, p, l): l for p, l in tasks}
        for future in as_completed(futures):
            label, rc, dt = future.result()
            results.append({'label': label, 'returncode': rc, 'time_min': dt / 60})

    dt_all = time.time() - t0_all
    print(f"\n{'='*60}")
    print(f"ALL DONE in {dt_all/60:.1f} min")
    print(f"{'='*60}")
    for r in sorted(results, key=lambda x: x['label']):
        status = "OK" if r['returncode'] == 0 else "FAIL"
        print(f"  [{status}] {r['label']} ({r['time_min']:.1f} min)")

    # Save summary
    summary_path = ROOT / "output/coupled_ppc4/run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
