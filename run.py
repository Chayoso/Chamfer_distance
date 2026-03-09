"""
run.py - [CVPR 2026] PhysMorph-GS: Physics-guided Gaussian Splatting for Shape Morphing

Main entry point for shape morphing pipeline.

Usage:
    python run.py -c configs/sphere_to_spot.yaml --e2e
    python run.py -c configs/sphere_to_spot.yaml --e2e --png

Output Structure:
output/
    ├── target/          # Target renderings
    └── ep{:03d}/        # Episode results

Author: CHAYO
Version: 3.0 (Modularized - 2361 lines → 250 lines!)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import io

# Fix Windows console encoding issues
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass

import argparse
import numpy as np
import torch
import time
from pathlib import Path
from typing import Dict

# Core modules
from sampling import default_cfg
from sampling.utils.config_adapter import adapt_config
from loss import E2ELossManager

# Modularized utilities
from utils.physics_utils import (
    build_opt_input,
    initialize_point_clouds,
    initialize_grids,
    initialize_comp_graph,
    extract_target_point_cloud,
)
from utils.rendering_utils import (
    setup_renderer,
    create_target_render,
)
from utils.training_loop import run_e2e_episode, run_e2e_episode_session


# ============================================================================
# Configuration Management
# ============================================================================

def load_config(config_path: str) -> tuple:
    """Load YAML configuration."""
    import yaml
    
    config_path = Path(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    return cfg, config_path.parent


def _safe_deepcopy(obj):
    """
    Deep copy that safely handles torch tensors by skipping them.
    """
    import copy
    import torch
    
    if isinstance(obj, torch.Tensor):
        # Skip tensors (they shouldn't be in config anyway)
        return obj
    elif isinstance(obj, dict):
        return {k: _safe_deepcopy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_safe_deepcopy(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_safe_deepcopy(item) for item in obj)
    else:
        try:
            return copy.deepcopy(obj)
        except:
            # If deepcopy fails, return as-is
            return obj


def apply_episode_schedule(cfg: Dict, episode: int) -> Dict:
    """
    Apply episode-specific overrides from episode_schedule section.
    
    Args:
        cfg: Base configuration
        episode: Episode number (0-indexed, will be converted to 1-indexed for matching)
    
    Returns:
        Modified configuration with episode overrides
    """
    schedule = cfg.get('episode_schedule', {})
    if not schedule:
        return cfg
    
    # Find matching schedule entry (0-indexed)
    matched_key = None
    for key in schedule.keys():
        key_str = str(key)
        
        if key_str.isdigit() and int(key_str) == episode:
            matched_key = key
            break
        elif '-' in key_str:
            try:
                start, end = key_str.split('-')
                start, end = int(start.strip()), int(end.strip())
                if start <= episode <= end:
                    matched_key = key
                    break
            except ValueError:
                continue
        elif '+' in key_str:
            try:
                start = int(key_str.replace('+', '').strip())
                if episode >= start:
                    matched_key = key
            except ValueError:
                continue
    
    if matched_key is not None:
        cfg_out = _safe_deepcopy(cfg)
        overrides = schedule[matched_key]
        print(f"\n[Episode Schedule] Applying overrides for ep{episode:03d} (key: {matched_key})")
        _deep_update(cfg_out, overrides)
        return cfg_out
    
    return cfg


def _deep_update(base_dict: Dict, update_dict: Dict) -> None:
    """Recursively update nested dictionary."""
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value


def build_runtime_config(cfg: Dict, out_dir: Path) -> tuple[Dict, Dict]:
    """
    Normalize top-level config structure and produce the upsample runtime config.

    Returns:
        (cfg_runtime, rs) where cfg_runtime is a deep copy of the user config with
        required sections ensured, and rs is the prepared upsample config.
    """
    cfg_runtime = _safe_deepcopy(cfg)

    sim_cfg = cfg_runtime.setdefault("simulation", {})
    opt_cfg = cfg_runtime.setdefault("optimization", {})
    render_cfg = cfg_runtime.setdefault("render", {})

    # Fill basic defaults (minimal)
    sim_cfg.setdefault("grid_dx", 1.0)
    sim_cfg.setdefault("grid_min_point", [-16.0, -16.0, -16.0])
    sim_cfg.setdefault("grid_max_point", [16.0, 16.0, 16.0])

    render_cfg.setdefault("surface_mask_ratio", 0.2)
    render_cfg.setdefault("surface_mask_mode", "last")

    rs_default = default_cfg()
    user_upsample = _safe_deepcopy(cfg_runtime.get("upsample", {}) or {})

    legacy_mag = opt_cfg.get("magnitude_strategy")
    if legacy_mag is not None and 'magnitude_strategy' not in user_upsample:
        user_upsample['magnitude_strategy'] = legacy_mag

    render_loss_weight = user_upsample.get("render_loss_weight")
    if isinstance(render_loss_weight, str):
        try:
            user_upsample["render_loss_weight"] = float(render_loss_weight)
        except ValueError:
            pass

    adapted = adapt_config({'upsample': user_upsample})
    rs_default.update(adapted)

    rs_default.setdefault("debug", {})
    rs_default["debug"]["output_dir"] = str(out_dir)
    rs_default["output_dir"] = str(out_dir)

    rs_default["physics_grid"] = {
        "grid_min": sim_cfg.get("grid_min_point", [-16.0, -16.0, -16.0]),
        "grid_max": sim_cfg.get("grid_max_point", [16.0, 16.0, 16.0]),
        "grid_dx": sim_cfg.get("grid_dx", 1.0),
    }

    return cfg_runtime, rs_default


def validate_config(cfg: Dict) -> None:
    """Basic config validation."""
    required = ['input_mesh_path', 'target_mesh_path']
    for key in required:
        if key not in cfg:
            raise ValueError(f"Missing required config key: {key}")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main entry point."""
    # ════════════════════════════════════════════════════════════════════════════
    # Parse Arguments
    # ════════════════════════════════════════════════════════════════════════════
    parser = argparse.ArgumentParser(description='PhysMorph-GS Shape Morphing')
    parser.add_argument('-c', '--config', required=True, help='Path to config YAML')
    parser.add_argument('--png', action='store_true', help='Export PNG images')
    parser.add_argument('--png_dpi', type=int, default=160, help='PNG DPI')
    parser.add_argument('--png_ptsize', type=float, default=0.5, help='Point size')
    
    args = parser.parse_args()
    
    # ════════════════════════════════════════════════════════════════════════════
    # Load Configuration
    # ════════════════════════════════════════════════════════════════════════════
    print("="*80, flush=True)
    print("PhysMorph-GS: Physics-guided Gaussian Splatting for Shape Morphing", flush=True)
    print("="*80, flush=True)
    
    cfg, cfg_dir = load_config(args.config)
    validate_config(cfg)
    
    out_dir = Path(cfg.get("output_dir", "output/"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 🔥 NEW: Create loss log files (both TXT and JSON)
    loss_log = out_dir / "training_losses.txt"
    loss_log_json = out_dir / "training_losses.json"

    with open(loss_log, 'w') as f:
        f.write(f"# Training Losses Log\n")
        f.write(f"# Config: {args.config}\n")
        f.write(f"# Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # Initialize JSON log
    import json
    loss_history = {
        "config": args.config,
        "start_time": time.strftime('%Y-%m-%d %H:%M:%S'),
        "episodes": []
    }

    cfg_runtime, rs = build_runtime_config(cfg, out_dir)
    
    # ════════════════════════════════════════════════════════════════════════════
    # Initialize Physics
    # ════════════════════════════════════════════════════════════════════════════
    print("\n[Init] Building optimization input...", flush=True)
    opt = build_opt_input(cfg_runtime)

    print(f"[Init] Loading meshes...", flush=True)
    print(f"  Input:  {cfg_runtime['input_mesh_path']}", flush=True)
    print(f"  Target: {cfg_runtime['target_mesh_path']}", flush=True)
    
    input_pc, target_pc = initialize_point_clouds(opt)
    input_grid, target_grid = initialize_grids(opt)
    
    print("[Init] Calculating point cloud volumes...", flush=True)
    import diffmpm_bindings
    diffmpm_bindings.calculate_point_cloud_volumes(input_pc, input_grid)
    diffmpm_bindings.calculate_point_cloud_volumes(target_pc, target_grid)

    print("[Init] Creating computation graph...", flush=True)
    cg = initialize_comp_graph(input_pc, input_grid, target_grid)

    # 🔥 DEBUG: Check EndLayerMassLoss right after initialization
    # This isolates whether discrepancy is from init/normalization or rendering/session
    loss0 = cg.end_layer_mass_loss()
    print(f"[DEBUG] ⚠️  EndLayerMassLoss just after init: {loss0:.6f}")
    print(f"[DEBUG]     Compare this with pure C++ run using same initialization")
    print(f"[DEBUG]     If different → problem is init/normalization/references")
    print(f"[DEBUG]     If same → problem is rendering/session/gradients\n")

    tgt = extract_target_point_cloud(target_pc)[0]

    # ════════════════════════════════════════════════════════════════════════════
    # Grid Distance Field (for Chamfer-aware physics)
    # ════════════════════════════════════════════════════════════════════════════
    _w_distance = float(cfg.get("optimization", {}).get("w_distance", 0.0))
    if _w_distance > 0.0:
        from scipy.spatial import cKDTree
        print(f"[DistField] Computing grid distance field (w={_w_distance})...", flush=True)

        # Compute grid node positions from config
        sim_cfg = cfg.get("simulation", {})
        _dx = float(sim_cfg.get("grid_dx", 1.0))
        _gmin = sim_cfg.get("grid_min_point", [-16.0, -16.0, -16.0])
        _gmax = sim_cfg.get("grid_max_point", [16.0, 16.0, 16.0])
        _dim = [int((_gmax[d] - _gmin[d]) / _dx) for d in range(3)]
        print(f"[DistField] Grid: {_dim[0]}x{_dim[1]}x{_dim[2]} = {_dim[0]*_dim[1]*_dim[2]} nodes, dx={_dx}")

        # Build node positions: node(i,j,k).x = min_point + (i,j,k) * dx
        _ii = np.arange(_dim[0], dtype=np.float32)
        _jj = np.arange(_dim[1], dtype=np.float32)
        _kk = np.arange(_dim[2], dtype=np.float32)
        _gi, _gj, _gk = np.meshgrid(_ii, _jj, _kk, indexing='ij')
        grid_positions = np.stack([
            _gmin[0] + _gi.ravel() * _dx,
            _gmin[1] + _gj.ravel() * _dx,
            _gmin[2] + _gk.ravel() * _dx
        ], axis=1).astype(np.float32)  # (N_nodes, 3), order matches C++ i*dim_y*dim_z + j*dim_z + k

        # Target point cloud positions
        tgt_np = np.array(tgt, dtype=np.float32)

        # Build KDTree on target surface, query NN distance for each grid node
        tgt_tree = cKDTree(tgt_np)
        dist_to_target, _ = tgt_tree.query(grid_positions, k=1, workers=-1)
        dist_to_target = dist_to_target.astype(np.float32)

        print(f"[DistField] Grid nodes: {len(dist_to_target)}, "
              f"dist range: [{dist_to_target.min():.3f}, {dist_to_target.max():.3f}], "
              f"mean: {dist_to_target.mean():.3f}")

        # Inject into CompGraph
        cg.set_distance_field(dist_to_target, _w_distance)

        # Verify: re-compute loss with distance field
        loss_with_dist = cg.end_layer_mass_loss()
        print(f"[DistField] EndLayerMassLoss with distance field: {loss_with_dist:.6f}")

    # ════════════════════════════════════════════════════════════════════════════
    # Setup Upsampling
    # ════════════════════════════════════════════════════════════════════════════
    print("[Config] Setting up upsampling pipeline...", flush=True)
    
    if 'covariance' in rs and 'curvature_sigma' in rs['covariance']:
        cs = rs['covariance']['curvature_sigma']
        print(f"[Config] ✓ Target covariance (curvature-based):")
        print(f"          σ_n0={cs.get('sigma_n0', 0.020):.4f}, σ_t0={cs.get('sigma_t0', 0.030):.4f}")
        print(f"          a={cs.get('a', 3.0):.2f}, b={cs.get('b', 0.5):.2f}, u={cs.get('u', 0.4):.2f}")
    
    # ════════════════════════════════════════════════════════════════════════════
    # Setup Rendering
    # ════════════════════════════════════════════════════════════════════════════
    print("[Init] Setting up renderer...", flush=True)
    cam_cfg = cfg_runtime.get("camera", {}) or {}
    render_cfg = cfg_runtime.get("render", {}) or {}
    particle_color = render_cfg.get("particle_color", [0.27, 0.51, 0.71])

    # ════════════════════════════════════════════════════════════════════════════
    # Setup E2E Training (controlled by config only)
    # ════════════════════════════════════════════════════════════════════════════
    enable_e2e = cfg_runtime.get("optimization", {}).get("loss", {}).get("enabled", False)

    # 🚀 Use training mode (lower resolution) if E2E is enabled
    renderer, view_params = setup_renderer(cam_cfg, render_cfg, training_mode=enable_e2e)
    HAVE_3DGS = renderer is not None
    
    loss_manager = None
    target_render = None
    
    extra_views = []

    if enable_e2e and HAVE_3DGS:
        print("[E2E] Setting up rendering loss...", flush=True)

        loss_cfg = cfg.get("optimization", {}).get("loss", {})
        loss_manager = E2ELossManager(loss_cfg)

        print("[E2E] Creating target rendering...", flush=True)
        campos = view_params.get('campos')
        target_render = create_target_render(
            target_pc, renderer, rs, campos,
            render_cfg, particle_color, out_dir
        )

        if target_render is not None:
            print("✅ [E2E] E2E mode ENABLED", flush=True)

            # Multi-view: create extra renderers from cameras config
            extra_cam_lookats = cfg_runtime.get("cameras", [])
            if extra_cam_lookats:
                print(f"[Multi-view] Setting up {len(extra_cam_lookats)} extra views...", flush=True)
                for v_idx, lookat in enumerate(extra_cam_lookats):
                    cam_cfg_v = dict(cam_cfg)
                    cam_cfg_v = {**cam_cfg_v, 'lookat': lookat}
                    renderer_v, view_params_v = setup_renderer(cam_cfg_v, render_cfg, training_mode=enable_e2e)
                    if renderer_v is None:
                        print(f"[Multi-view] View {v_idx+1}: renderer setup failed, skipping")
                        continue
                    campos_v = view_params_v.get('campos')
                    target_render_v = create_target_render(
                        target_pc, renderer_v, rs, campos_v,
                        render_cfg, particle_color, out_dir
                    )
                    if target_render_v is not None:
                        extra_views.append({
                            'renderer': renderer_v,
                            'target_render': target_render_v,
                            'campos': campos_v
                        })
                        print(f"[Multi-view] View {v_idx+1}: OK (eye={lookat.get('eye')})")
                    else:
                        print(f"[Multi-view] View {v_idx+1}: target render failed, skipping")
                print(f"[Multi-view] {len(extra_views)}/{len(extra_cam_lookats)} extra views ready")
        else:
            print("⚠️ [E2E] Target rendering failed, E2E disabled", flush=True)
            enable_e2e = False
    else:
        if not HAVE_3DGS:
            print("[WARN] E2E mode requested but renderer not available")
        print("\n[Mode] Running in PHYSICS-ONLY mode")
    
    # ════════════════════════════════════════════════════════════════════════════
    # Training Loop
    # ════════════════════════════════════════════════════════════════════════════
    ema_state = {}

    # 🔥 NEW: Session Mode (10-15x faster!)
    use_session_mode = cfg.get("optimization", {}).get("use_session_mode", True)
    session = None

    if use_session_mode and enable_e2e and HAVE_3DGS:
        print("\n[Mode] 🔥 PERSISTENT SESSION MODE ENABLED 🔥", flush=True)
        print("  Expected speedup: 10-15x per episode!", flush=True)

        # Create E2ESession configuration
        import diffmpm_bindings as dmpm

        session_config = dmpm.E2EConfig()
        session_config.num_timesteps = int(opt.num_timesteps)
        session_config.control_stride = int(opt.control_stride)
        session_config.dt = float(opt.dt)
        session_config.drag = float(opt.drag)
        session_config.f_ext = opt.f_ext
        session_config.max_gd_iters = int(opt.max_gd_iters)
        session_config.max_ls_iters = int(opt.max_ls_iters)
        session_config.initial_alpha = float(opt.initial_alpha)
        session_config.gd_tol = float(opt.gd_tol)
        session_config.smoothing_factor = float(opt.smoothing_factor)
        session_config.adaptive_alpha_enabled = bool(opt.adaptive_alpha_enabled)
        session_config.adaptive_alpha_target_norm = float(opt.adaptive_alpha_target_norm)
        session_config.adaptive_alpha_min_scale = float(opt.adaptive_alpha_min_scale)
        # 🔥 NEW: Configurable num_passes (default 3 for backward compatibility)
        session_config.num_passes_per_episode = cfg.get("optimization", {}).get("num_passes", 3)
        session_config.enable_render_grads = True

        # Chamfer F-field: set gradient normalization alpha
        _chamfer_alpha = float(cfg.get("optimization", {}).get("chamfer_alpha", 1.0))
        cg.set_render_gain(_chamfer_alpha)
        print(f"  Render gain (α): {_chamfer_alpha}")

        # Create persistent session
        session = dmpm.E2ESession(cg, session_config)

        print(f"  Passes per episode: {session_config.num_passes_per_episode}")
        print(f"  Timesteps: {session_config.num_timesteps}")
        print(f"  Render gradients: {'enabled' if session_config.enable_render_grads else 'disabled'}")

    print(f"\n{'='*80}", flush=True)
    print(f"Starting training: {opt.num_animations} episodes", flush=True)
    print(f"{'='*80}\n", flush=True)

    for ep in range(int(opt.num_animations)):
        # Apply episode-specific config overrides
        cfg_ep = apply_episode_schedule(cfg, ep)
        
        # Build episode-specific upsampling config
        rs_ep = default_cfg()
        rs_ep_user = cfg_ep.get("upsample", {}) or {}
        
        adapted_cfg_ep = adapt_config({'upsample': rs_ep_user})
        rs_ep.update(adapted_cfg_ep)
        
        # 🔥 Pass episode number to covariance construction
        if "covariance" not in rs_ep:
            rs_ep["covariance"] = {}
        rs_ep["covariance"]["episode"] = ep

        # 🔥 Inject optimization config so training_loop can read use_pcgrad, w_render_base, etc.
        rs_ep["optimization"] = cfg_ep.get("optimization", {})
        
        # Debug: Show covariance config for this episode
        if "covariance" in rs_ep:
            cov_ep = rs_ep["covariance"]
            print(f"\n[Episode {ep}] Covariance config:")
            print(f"  sigma0: {cov_ep.get('sigma0', 'NOT SET')}")
            if "curvature_sigma" in cov_ep:
                cs_ep = cov_ep["curvature_sigma"]
                print(f"  curvature_sigma (target): σ_n0={cs_ep.get('sigma_n0'):.3f}, σ_t0={cs_ep.get('sigma_t0'):.3f}")
            else:
                print(f"  curvature_sigma: NOT SET (will use default)")
        
        # Fresh level set for each episode (no advection)
        external_levelset = None
        
        # Run episode
        if enable_e2e and loss_manager is not None and target_render is not None:
            # E2E mode with rendering loss

            # Define num_passes early for all code paths
            # 🔥 NEW: Configurable num_passes (default 3 for backward compatibility)
            num_passes = cfg.get("optimization", {}).get("num_passes", 3)
            num_timesteps = int(opt.num_timesteps)
            control_stride = int(opt.control_stride)
            campos = view_params.get('campos')

            # Create episode-specific output directory
            ep_dir = out_dir / f"ep{ep:03d}"
            ep_dir.mkdir(parents=True, exist_ok=True)

            if session is not None:
                # 🔥 SESSION MODE (10-15x faster!)
                print(f"\n⚠️  [SESSION MODE] Episode {ep} - PCGrad NOT available in session mode!")
                print(f"    To use PCGrad, add 'use_session_mode: false' to your config")

                ema_state, episode_losses = run_e2e_episode_session(
                    session, ep, num_timesteps,
                    rs_ep, ema_state, renderer, loss_manager, target_render,
                    view_params, campos, render_cfg, particle_color,
                    ep_dir, args.png, tgt,
                    cov_module=None,
                    external_levelset=external_levelset
                )
                
                print(f"[DEBUG] Session-based E2E episode {ep} completed")
            else:
                # Legacy pass-by-pass mode
                print(f"\n✅ [LEGACY MODE] Episode {ep} with {num_passes} passes - PCGrad available!")
                ema_state, episode_losses = run_e2e_episode(
                    ep, cg, opt, num_timesteps, control_stride, num_passes,
                    rs_ep, ema_state, renderer, loss_manager, target_render,
                    view_params, campos, render_cfg, particle_color,
                    ep_dir, args.png, tgt,
                    cov_module=None,
                    cov_optimizer=None,
                    external_levelset=external_levelset,
                    extra_views=extra_views
                )
                print(f"[DEBUG] E2E episode {ep} completed")

            # Save raw MPM particle positions (for fair ablation comparison)
            try:
                num_ts = int(opt.num_timesteps)
                pc_ep = cg.get_point_cloud(num_ts - 1)
                x_ep_np = np.array(pc_ep.get_positions(), dtype=np.float32)
                x_ep = torch.from_numpy(x_ep_np)
                save_path = ep_dir / f"ep{ep:03d}_particles.pt"
                torch.save(x_ep, save_path)
                print(f"[Save] {len(x_ep):,} raw particles → {save_path.name}")
            except Exception as e:
                print(f"[Save] Could not save raw particles: {e}")

            # ── Early stopping (Chamfer CD plateau detection) ──
            _es_patience = int(cfg.get("optimization", {}).get("chamfer_early_stop_patience", 0))
            if _es_patience > 0 and 'last_chamfer_dist' in ema_state:
                _cd_now = ema_state['last_chamfer_dist']
                if 'es_best_cd' not in ema_state:
                    ema_state['es_best_cd'] = _cd_now
                    ema_state['es_wait'] = 0
                    ema_state['es_best_ep'] = ep
                if _cd_now < ema_state['es_best_cd'] - 1e-4:
                    ema_state['es_best_cd'] = _cd_now
                    ema_state['es_wait'] = 0
                    ema_state['es_best_ep'] = ep
                else:
                    ema_state['es_wait'] = ema_state.get('es_wait', 0) + 1
                _es_min_ep = int(cfg.get("optimization", {}).get("chamfer_early_stop_min_ep", 30))
                if ema_state['es_wait'] >= _es_patience and ep >= _es_min_ep:
                    print(f"\n[EARLY_STOP] CD plateau for {_es_patience} eps "
                          f"(best={ema_state['es_best_cd']:.5f} at ep{ema_state['es_best_ep']}). Stopping.")
                    # Save final log before breaking
                    loss_history["early_stopped"] = True
                    loss_history["early_stop_ep"] = ep
                    loss_history["best_cd_ep"] = ema_state['es_best_ep']
                    loss_history["best_cd"] = ema_state['es_best_cd']
                    with open(loss_log_json, 'w') as f:
                        json.dump(loss_history, f, indent=2)
                    break

            # ✅ Carry over state to next episode (CRITICAL!)
            print(f"[CarryOver] Promoting final state to next episode...")
            cg.promote_last_as_initial(carry_grid=True)

            # Print episode summary
            print(f"\n[Summary] Episode {ep} losses:")
            for key, val in episode_losses.items():
                if isinstance(val, (int, float)):
                    print(f"  {key}: {val:.6f}")

            # 🔥 NEW: Save losses to both TXT and JSON
            # Text format (human-readable)
            with open(loss_log, 'a') as f:
                f.write(f"Episode {ep:03d}:\n")
                for key, val in episode_losses.items():
                    if isinstance(val, (int, float)):
                        f.write(f"  {key}: {val:.6f}\n")
                f.write("\n")
                f.flush()

            # JSON format (for plotting)
            episode_data = {"episode": ep}
            for key, val in episode_losses.items():
                if torch.is_tensor(val):
                    episode_data[key] = float(val.item())
                elif isinstance(val, (int, float)):
                    episode_data[key] = float(val)

            # Merge summary data from ep###_summary.json if it exists
            summary_path = ep_dir / f"ep{ep:03d}_summary.json"
            if summary_path.exists():
                try:
                    with open(summary_path, 'r') as sf:
                        summary_data = json.load(sf)
                        # Add summary fields to episode data
                        episode_data["J_min"] = summary_data.get("J_min")
                        episode_data["J_mean"] = summary_data.get("J_mean")
                        episode_data["num_surface_points"] = summary_data.get("num_surface_points")
                        # Merge render losses at top level
                        if "render_losses" in summary_data:
                            for k, v in summary_data["render_losses"].items():
                                episode_data[k] = v
                except Exception as e:
                    print(f"  ⚠️ Failed to merge summary data: {e}")

            loss_history["episodes"].append(episode_data)

            # Save JSON after each episode
            with open(loss_log_json, 'w') as f:
                json.dump(loss_history, f, indent=2)
                f.flush()
        else:
            # Physics-only mode
            print(f"\n[Episode {ep}] Running standard physics optimization...")
            print(f"[DEBUG] Physics-only mode (enable_e2e={enable_e2e}, loss_manager={loss_manager is not None}, target_render={target_render is not None})")
            opt.current_episodes = ep
            cg.run_optimization(opt)

            # Save final particle positions for ablation comparison
            try:
                ep_dir = out_dir / f"ep{ep:03d}"
                ep_dir.mkdir(parents=True, exist_ok=True)
                num_ts = int(opt.num_timesteps)
                pc_final = cg.get_point_cloud(num_ts - 1)
                x_final_np = np.array(pc_final.get_positions(), dtype=np.float32)
                x_final = torch.from_numpy(x_final_np)
                save_path = ep_dir / f"ep{ep:03d}_particles.pt"
                torch.save(x_final, save_path)
                print(f"[Save] {len(x_final):,} particles → {save_path.name}")
            except Exception as e:
                print(f"[Save] Could not save particles: {e}")

            # ✅ Carry over state to next episode
            print(f"[CarryOver] Promoting final state to next episode...")
            cg.promote_last_as_initial(carry_grid=True)
    
    # ════════════════════════════════════════════════════════════════════════════
    # Finalization
    # ════════════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("✅ Training Complete!")
    print(f"   Output directory: {out_dir}")

    # Print session statistics if using session mode
    if session is not None:
        stats = session.get_statistics()
        print("\n[Session Statistics]")
        print(f"  Total episodes: {stats.total_episodes}")
        print(f"  Total passes: {stats.total_passes}")
        print(f"  Total time: {stats.total_wall_time:.1f}s")
        print(f"  Avg time/episode: {stats.total_wall_time/max(1, stats.total_episodes):.1f}s")
        print(f"  Best loss: {stats.best_loss:.2f} (episode {stats.best_episode})")

    # 🔥 NEW: Plot losses if E2E mode was used
    if enable_e2e and loss_log_json.exists():
        print("\n[Plotting] Generating loss curves...")
        try:
            from utils.plotting_utils import plot_training_losses
            plot_training_losses(loss_log_json, out_dir / "loss_curves.png")
            print(f"✅ Loss curves saved to: {out_dir / 'loss_curves.png'}")
        except Exception as e:
            print(f"⚠️  Failed to generate loss plots: {e}")
            print(f"   You can manually plot using: python plot_losses.py {loss_log_json}")

    print("="*80)


if __name__ == "__main__":
    main()
