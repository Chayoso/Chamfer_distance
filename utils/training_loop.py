"""
Training Loop - E2E Episode Training

Main E2E training loop with multi-pass refinement.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import time

from utils.physics_utils import run_physics_optimization, run_physics_optimization_batched, extract_point_cloud_state
from utils.rendering_utils import compute_render_loss_pass, extract_render_gradients, upsample_current_state
from visualization.utils import visualize_episode
from utils.gradient_utils import (
    compute_gradient_statistics,
    compute_gradient_cosine_similarity,
    pcgrad_projection,
    ema_gradient_scaling,
    adaptive_target_ratio_schedule,
    diagnose_gradient_health,
    normalize_and_combine_gradients
)


# ============================================================================
# E2E Training Episode (Session Mode - MAXIMUM PERFORMANCE)
# ============================================================================

def run_e2e_episode_session(
    session: Any,
    ep: int,
    num_timesteps: int,
    rs_full: Dict,
    ema_state: Dict,
    renderer: Any,
    loss_manager: Any,
    target_render: Dict,
    view_params: Dict,
    campos: np.ndarray,
    render_cfg: Dict,
    particle_color: list,
    out_dir: Path,
    png_enabled: bool,
    tgt: np.ndarray,
    cov_module=None,
    external_levelset=None
) -> Tuple[Dict, Dict]:
    """
    🔥 MAXIMUM PERFORMANCE: Run E2E episode using persistent session.

    This is ~10-15x faster than run_e2e_episode() because:
      - Single Python→C++ transition per episode (vs 50-100)
      - All physics runs with GIL released
      - Persistent buffer reuse across episodes

    Args:
        session: E2ESession instance (C++)
        ep: Episode number
        num_timesteps: Number of simulation timesteps
        rs_full: Upsampling configuration
        ema_state: EMA state dict (updated in-place)
        renderer: 3DGS renderer
        loss_manager: Loss manager
        target_render: Target rendering dict
        view_params: Camera parameters
        campos: Camera position
        render_cfg: Rendering configuration
        particle_color: Base particle color
        out_dir: Output directory for this episode
        png_enabled: Whether to save PNG visualizations
        tgt: Target point cloud positions
        cov_module: Optional learnable covariance module
        external_levelset: Optional pre-computed level set

    Returns:
        Tuple of (updated_ema_state, final_losses)
    """
    print(f"\n{'='*70}")
    print(f"🔥 Session Mode: Episode {ep} START")
    print(f"{'='*70}")

    # Track render loss components
    last_render_loss_components = {}

    # Define render gradient callback
    def compute_render_grads_callback(episode_num: int, pass_idx: int):
        """
        Called by C++ to get render gradients for a pass.
        This is the ONLY Python code that runs during the episode!
        """
        nonlocal last_render_loss_components

        import time

        try:
            t_start = time.time()
            print(f"\n[Render Callback] Episode {episode_num}, Pass {pass_idx+1}", flush=True)

            # Get final state from comp graph (after physics pass completed)
            t0 = time.time()
            pc = session.get_final_point_cloud()
            t_extract_pc = time.time() - t0
            if pc is None:
                print("  ⚠️  No point cloud available")
                return None

            # Extract state with zero-copy views
            t0 = time.time()
            try:
                x = pc.get_positions_torch_view().clone().requires_grad_(True)
                F = pc.get_def_grads_total_torch_view().clone().requires_grad_(True)
            except:
                print("  ⚠️  Failed to extract positions/gradients")
                return None
            t_extract_state = time.time() - t0

            print(f"  ├─ Extracted state: {len(x)} particles")
            print(f"  ├─ x.requires_grad: {x.requires_grad}, F.requires_grad: {F.requires_grad}")

            # ── Python-side x_late: render-only position correction ─────────────
            # Pure Python — physics NEVER sees this correction (no EndLayerMassLoss conflict).
            # x_late_py persists across episodes via ema_state['py_xlate_tensor'].
            _opt_cfg_cb = rs_full.get('optimization', {}) if rs_full else {}
            _py_xlate_enabled = bool(_opt_cfg_cb.get('py_xlate_enabled', False))
            _alpha_py_xlate   = float(_opt_cfg_cb.get('alpha_py_xlate', 1e-2))
            _x_physics_leaf = x  # keep leaf ref for dLdx_render (x.grad after backward)
            if _py_xlate_enabled:
                import torch as _torch_cb
                if 'py_xlate_tensor' not in ema_state:
                    ema_state['py_xlate_tensor'] = _torch_cb.zeros_like(x, requires_grad=False)
                    ema_state['py_xlate_tensor'].requires_grad_(True)
                    ema_state['py_xlate_optim']  = _torch_cb.optim.Adam(
                        [ema_state['py_xlate_tensor']], lr=_alpha_py_xlate,
                        betas=(0.9, 0.999), eps=1e-3, amsgrad=True
                    )
                    print(f"  ├─ [py_xlate] Initialized {x.shape[0]}×3 correction tensor", flush=True)
                _xlt = ema_state['py_xlate_tensor']
                # Resize if shape changes (shouldn't happen, but be safe)
                if _xlt.shape != x.shape:
                    ema_state['py_xlate_tensor'] = _torch_cb.zeros_like(x, requires_grad=True)
                    ema_state['py_xlate_optim']  = _torch_cb.optim.Adam(
                        [ema_state['py_xlate_tensor']], lr=_alpha_py_xlate,
                        betas=(0.9, 0.999), eps=1e-3, amsgrad=True
                    )
                    _xlt = ema_state['py_xlate_tensor']
                # Apply correction: x_render = x_physics + x_late_py
                # After backward: _x_physics_leaf.grad = dL/dx_phys (leaf, for C++ return)
                #                 _xlt.grad            = dL/dx_late  (leaf, for py_xlate Adam)
                # Both equal dL/dx_render since x_render = x_physics + x_late_py.
                x = x + _xlt  # x is now a non-leaf; _x_physics_leaf keeps the leaf ref
                _norm_xlt = float(_xlt.detach().norm())
                print(f"  ├─ [py_xlate] ||x_late_py||={_norm_xlt:.4f}", flush=True)

            # Upsample and compute render loss
            seed = 9999 + episode_num*1000 + pass_idx

            # ✅ CRITICAL: Pass x, F directly to maintain gradient connection!
            from sampling import upsample
            t0 = time.time()
            result = upsample(
                x, F,
                cfg=rs_full,
                state=ema_state,
                seed=seed,
                return_torch=True,
                export_stages=True,  # 🔥 DEBUG: Enable subdivision debug output
                learnable_cov_module=cov_module,
                current_episode=episode_num,
                external_levelset=external_levelset,
                use_simple_pipeline=rs_full.get('upsample', {}).get('use_simple_pipeline', True)
            )
            t_upsample = time.time() - t0

            mu = result.get('points')
            cov = result.get('cov')

            if mu is None:
                print("  ⚠️  Upsampling failed")
                return None

            print(f"  ├─ Upsampled: {len(mu)} points")

            # Prepare rendering inputs
            from utils.rendering_utils import prepare_rendering_inputs
            t0 = time.time()
            rgb = prepare_rendering_inputs(mu, result, campos, render_cfg, particle_color)
            t_prep_render = time.time() - t0

            # Render
            t0 = time.time()
            out_pred = renderer.render(mu, cov, rgb=rgb)
            t_render = time.time() - t0

            # ✅ Convert render outputs to torch tensors and ensure same device
            device = mu.device if torch.is_tensor(mu) else 'cuda'
            
            # Convert pred to torch and move to device
            pred_dict = {}
            for key in ['image', 'alpha', 'depth']:
                if key in out_pred and out_pred[key] is not None:
                    val = out_pred[key]
                    if not torch.is_tensor(val):
                        val = torch.from_numpy(val)
                    val = val.to(device)
                    pred_dict[key] = val
            
            # ✅ Convert target to torch and move to SAME device
            target_dict = {}
            for key in ['image', 'alpha', 'depth']:
                if key in target_render and target_render[key] is not None:
                    val = target_render[key]
                    if not torch.is_tensor(val):
                        val = torch.from_numpy(val)
                    val = val.to(device)
                    target_dict[key] = val

            # Extract F_interp from upsampling result (or use original F as fallback)
            F_interp = result.get('F_interp')
            if F_interp is None:
                print(f"  ⚠️  F_interp not in result, using original F")
                F_interp = F

            # ✅ Get cov_target from target_render (computed during target rendering)
            cov_target = target_render.get('cov_target')
            if cov_target is not None:
                print(f"  ├─ Using cov_target for spectral alignment loss")

            # 🔥 NEW: Create opacity tensor for shrinkage regularization
            # Start with all particles fully opaque (α=1.0)
            # Shrinkage loss will gradually reduce interior particle opacity
            opacity = torch.ones(mu.shape[0], dtype=torch.float32, device=mu.device, requires_grad=True)

            # Compute loss
            t0 = time.time()
            loss_components = loss_manager.compute_render_loss(
                pred=pred_dict,
                target=target_dict,  # ✅ Use converted target
                cov=cov,
                mu=mu,
                view_params=view_params,
                cov_target=cov_target,  # ✅ Pass target covariance from target_render
                F=F_interp,
                opacity=opacity  # 🔥 NEW: Per-Gaussian opacity for shrinkage regularization
            )
            t_loss = time.time() - t0

            loss_total = loss_components.get('loss_render_total', loss_components.get('loss_total', torch.tensor(0.0)))
            print(f"  ├─ Render loss total: {loss_total.item():.6f}", flush=True)

            # Print detailed loss components
            print(f"  │  ├─ Alpha:     {loss_components.get('loss_alpha', torch.tensor(0.0)).item():.6f}", flush=True)

            # Depth with MAE (depth difference metric)
            depth_loss = loss_components.get('loss_depth', torch.tensor(0.0)).item()
            depth_mae = loss_components.get('loss_depth_unweighted', torch.tensor(0.0)).item()
            if depth_mae > 0:
                print(f"  │  ├─ Depth:     {depth_loss:.6f} (MAE: {depth_mae:.4f})", flush=True)
            else:
                print(f"  │  ├─ Depth:     {depth_loss:.6f}", flush=True)

            print(f"  │  ├─ Photo:     {loss_components.get('loss_photo', torch.tensor(0.0)).item():.6f}", flush=True)
            print(f"  │  ├─ Edge:      {loss_components.get('loss_edge', torch.tensor(0.0)).item():.6f}", flush=True)
            print(f"  │  ├─ Cov align: {loss_components.get('loss_cov_align', torch.tensor(0.0)).item():.6f}", flush=True)
            print(f"  │  ├─ Cov reg:   {loss_components.get('loss_cov_reg', torch.tensor(0.0)).item():.6f}", flush=True)
            print(f"  │  ├─ Det barrier: {loss_components.get('loss_det_barrier', torch.tensor(0.0)).item():.6f}", flush=True)
            print(f"  │  └─ Opacity shrink: {loss_components.get('loss_opacity_shrink', torch.tensor(0.0)).item():.6f} (interior: {loss_components.get('opacity_shrink_num_interior', 0)})", flush=True)

            # ── Chamfer loss through F-field backprop ─────────────────────────
            # Adds two-sided Chamfer distance on RAW particle positions to loss_total.
            # After backward(), the Chamfer gradient flows into dLdx → C++ backward
            # → Back_Timestep → dLdF at control layer → Adam updates dFc.
            # This STEERS physics via F-field rather than fighting elastic restoring forces.
            _chamfer_cb_enabled = bool(_opt_cfg_cb.get('chamfer_in_callback', False))
            _w_chamfer = float(_opt_cfg_cb.get('w_chamfer', 0.5))
            print(f"  ├─ [DEBUG] chamfer_in_callback={_chamfer_cb_enabled}, w_chamfer={_w_chamfer}, tgt={'OK' if tgt is not None else 'None'}", flush=True)

            if _chamfer_cb_enabled and tgt is not None:
                import numpy as _np_cb
                from scipy.spatial import cKDTree as _cKDTree_cb

                # Cache target tensor + KDTree (built once, reused across episodes)
                if 'chamfer_cb_tgt_tensor' not in ema_state:
                    _tgt_np_init = _np_cb.array(tgt, dtype=_np_cb.float32)
                    ema_state['chamfer_cb_tgt_tensor'] = torch.from_numpy(_tgt_np_init).to(device)
                    ema_state['chamfer_cb_tgt_np'] = _tgt_np_init
                    ema_state['chamfer_cb_tgt_tree'] = _cKDTree_cb(_tgt_np_init)
                    print(f"  ├─ [Chamfer CB] Initialized target tree ({len(_tgt_np_init)} pts)", flush=True)

                _x_cb = _x_physics_leaf  # raw particle positions, requires_grad=True
                _x_cb_np = _x_cb.detach().cpu().numpy().astype(_np_cb.float32)
                _tgt_t = ema_state['chamfer_cb_tgt_tensor']
                _tgt_np_cb = ema_state['chamfer_cb_tgt_np']

                # Source → Target direction
                _, _nn_s2t = ema_state['chamfer_cb_tgt_tree'].query(_x_cb_np, k=1, workers=4)
                _x_nn_s2t = _tgt_t[torch.from_numpy(_nn_s2t).long().to(device)]
                _loss_s2t = (_x_cb.to(device) - _x_nn_s2t).pow(2).sum(dim=1).mean()

                # Target → Source direction
                _tree_src_cb = _cKDTree_cb(_x_cb_np)
                _, _nn_t2s = _tree_src_cb.query(_tgt_np_cb, k=1, workers=4)
                _x_nn_t2s = _x_cb.to(device)[torch.from_numpy(_nn_t2s).long().to(device)]
                _loss_t2s = (_tgt_t - _x_nn_t2s).pow(2).sum(dim=1).mean()

                _loss_chamfer_cb = 0.5 * (_loss_s2t + _loss_t2s)
                loss_total = loss_total + _w_chamfer * _loss_chamfer_cb

                _cd_metric = float(torch.sqrt(_loss_chamfer_cb).item())
                print(f"  ├─ Chamfer (F-field): loss={_loss_chamfer_cb.item():.6f}  CD={_cd_metric:.5f}  (w={_w_chamfer})", flush=True)

            # Store for final reporting
            last_render_loss_components = {k: v.item() if torch.is_tensor(v) else v
                                          for k, v in loss_components.items()}
            if _chamfer_cb_enabled and '_loss_chamfer_cb' in locals():
                last_render_loss_components['loss_chamfer_cb'] = _loss_chamfer_cb.item()
                last_render_loss_components['chamfer_cb_cd'] = _cd_metric

            if not torch.isfinite(loss_total):
                print("  └─ ❌ Render loss produced NaN/Inf (session mode)")
                return None

            # Backward
            print(f"  ├─ Running backward()...")
            print(f"  │  loss_total.requires_grad: {loss_total.requires_grad}")
            print(f"  │  loss_total.device: {loss_total.device}")

            t0 = time.time()
            try:
                loss_total.backward()
                print(f"  ├─ Backward completed ✅")
            except Exception as e:
                print(f"  └─ ❌ Backward failed: {e}")
                return None
            t_backward = time.time() - t0

            # Extract gradients
            t0 = time.time()
            print(f"  ├─ Checking gradients...")
            print(f"  │  F.grad is None: {F.grad is None}")
            # For py_xlate: x is non-leaf; check the leaf ref instead
            _x_grad_check = _x_physics_leaf.grad if _py_xlate_enabled else x.grad
            print(f"  │  x.grad is None: {_x_grad_check is None}")

            # Chamfer-only mode: F.grad may be None (Chamfer depends on x only, not F).
            # In this case, use zeros — C++ backward will still propagate dLdx → dLdF.
            if F.grad is None and _chamfer_cb_enabled:
                F_grad_zeros = torch.zeros_like(F)
                print(f"  ├─ [Chamfer-only] F.grad=None → using zeros (expected: Chamfer has no F dependence)")
            elif F.grad is None or _x_grad_check is None:
                print("  └─ ⚠️  No gradients computed")
                print(f"     Possible cause: Computational graph disconnected during upsampling")
                return None

            if _x_grad_check is None:
                print("  └─ ⚠️  x.grad is None — no gradient signal")
                return None

            _F_grad = F.grad if F.grad is not None else F_grad_zeros if 'F_grad_zeros' in locals() else torch.zeros_like(F)

            if (not torch.isfinite(_F_grad).all()) or (not torch.isfinite(_x_grad_check).all()):
                print("  └─ ❌ Render gradients contain NaN/Inf (session mode)")
                return None

            # 🔍 DEBUG: Check gradient magnitude
            F_grad_norm = torch.linalg.norm(_F_grad).item()
            x_grad_norm = torch.linalg.norm(_x_grad_check).item()
            print(f"  ├─ [DEBUG BACKWARD] F.grad norm: {F_grad_norm:.12e}")
            print(f"  ├─ [DEBUG BACKWARD] x.grad norm: {x_grad_norm:.12e}")
            if F_grad_norm > 0:
                print(f"  ├─ [DEBUG BACKWARD] F.grad range: [{_F_grad.min().item():.6e}, {_F_grad.max().item():.6e}]")

            # ✅ Import numpy inside callback
            import numpy as np

            # ── Python-side x_late: Adam step using x.grad ─────────────────────
            # If chamfer_xlate_enabled, skip render-based update (Chamfer handles it after episode).
            _chamfer_xlate_cb = bool(_opt_cfg_cb.get('chamfer_xlate_enabled', False))
            if _py_xlate_enabled and 'py_xlate_tensor' in ema_state and not _chamfer_xlate_cb:
                _xlt_t = ema_state['py_xlate_tensor']
                _xlt_o = ema_state['py_xlate_optim']
                if _xlt_t.grad is not None:
                    _grad_norm_xlt = float(_xlt_t.grad.norm())
                    _xlt_o.step()
                    _xlt_o.zero_grad()
                    _new_norm_xlt = float(_xlt_t.detach().norm())
                    print(f"  ├─ [py_xlate] Adam step: ||grad||={_grad_norm_xlt:.3e}  ||x_late||={_new_norm_xlt:.4f}", flush=True)
            elif _py_xlate_enabled and _chamfer_xlate_cb and 'py_xlate_tensor' in ema_state:
                # Chamfer mode: clear render grad (Chamfer handles update after episode)
                ema_state['py_xlate_tensor'].grad = None

            dLdF_render = _F_grad.detach().cpu().numpy()
            # For py_xlate: x is a non-leaf (x_physics + x_late_py), use the leaf ref
            _x_grad_leaf = _x_physics_leaf.grad if _py_xlate_enabled else x.grad
            dLdx_render = _x_grad_leaf.detach().cpu().numpy()

            # Ensure contiguous
            if not dLdF_render.flags['C_CONTIGUOUS']:
                dLdF_render = np.ascontiguousarray(dLdF_render)
            if not dLdx_render.flags['C_CONTIGUOUS']:
                dLdx_render = np.ascontiguousarray(dLdx_render)

            grad_F_norm_raw = np.linalg.norm(dLdF_render)
            grad_x_norm_raw = np.linalg.norm(dLdx_render)
            print(f"  ├─ Raw render gradients: ||∂L/∂F||={grad_F_norm_raw:.3e}, ||∂L/∂x||={grad_x_norm_raw:.3e}", flush=True)

            t_grad_extract = time.time() - t0
            t_total = time.time() - t_start

            # Print timing summary
            print(f"\n  [Timing Breakdown]:", flush=True)
            print(f"    Extract PC:    {t_extract_pc*1000:6.2f}ms ({t_extract_pc/t_total*100:4.1f}%)", flush=True)
            print(f"    Extract state: {t_extract_state*1000:6.2f}ms ({t_extract_state/t_total*100:4.1f}%)", flush=True)
            print(f"    Upsample:      {t_upsample*1000:6.2f}ms ({t_upsample/t_total*100:4.1f}%)", flush=True)
            print(f"    Prep render:   {t_prep_render*1000:6.2f}ms ({t_prep_render/t_total*100:4.1f}%)", flush=True)
            print(f"    Render:        {t_render*1000:6.2f}ms ({t_render/t_total*100:4.1f}%)", flush=True)
            print(f"    Loss compute:  {t_loss*1000:6.2f}ms ({t_loss/t_total*100:4.1f}%)", flush=True)
            print(f"    Backward:      {t_backward*1000:6.2f}ms ({t_backward/t_total*100:4.1f}%)", flush=True)
            print(f"    Grad extract:  {t_grad_extract*1000:6.2f}ms ({t_grad_extract/t_total*100:4.1f}%)", flush=True)
            print(f"    TOTAL:         {t_total*1000:6.2f}ms\n", flush=True)

            # Return raw gradients as tuple
            return (dLdF_render, dLdx_render)

        except Exception as e:
            print(f"  ❌ Render callback error: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ── Parse schedule params (needed by both Chamfer and physics blocks) ──
    _opt_chamfer = rs_full.get('optimization', {}) if rs_full else {}
    _pw_start_ep = int(_opt_chamfer.get('physics_weight_start_ep', 999))
    _pw_ramp_ep  = int(_opt_chamfer.get('physics_weight_ramp_ep', 5))
    _pw_final    = float(_opt_chamfer.get('physics_weight_final', 0.1))

    # ── Native Chamfer: set target + weight schedule ─────────────────────
    _w_chamfer_cfg = float(_opt_chamfer.get('w_chamfer', 0.0))
    _chamfer_start_ep = int(_opt_chamfer.get('chamfer_start_ep', 0))
    _chamfer_ramp_ep = int(_opt_chamfer.get('chamfer_ramp_ep', 10))

    _chamfer_rev_base = float(_opt_chamfer.get('chamfer_rev_weight', 1.0))
    _chamfer_rev_mode = str(_opt_chamfer.get('chamfer_rev_mode', 'fixed'))  # 'fixed' or 'coupled'
    _chamfer_huber_delta = float(_opt_chamfer.get('chamfer_huber_delta', 0.0))

    if _w_chamfer_cfg > 0 and tgt is not None:
        # Schedule: 0 until start_ep, linear ramp, then constant
        if ep < _chamfer_start_ep:
            _w_chamfer_eff = 0.0
        elif _chamfer_ramp_ep > 0 and ep < _chamfer_start_ep + _chamfer_ramp_ep:
            _w_chamfer_eff = _w_chamfer_cfg * (ep - _chamfer_start_ep) / _chamfer_ramp_ep
        else:
            _w_chamfer_eff = _w_chamfer_cfg

        # Compute effective rev_weight based on mode
        if _chamfer_rev_mode == 'coupled':
            # rev_weight = rev_base × current_physics_weight
            # (physics weight computed below, but we need it here — compute inline)
            if ep >= _pw_start_ep + _pw_ramp_ep:
                _pw_now = _pw_final
            elif ep >= _pw_start_ep:
                _t_pw = (ep - _pw_start_ep) / max(_pw_ramp_ep, 1)
                _pw_now = 1.0 * (1.0 - _t_pw) + _pw_final * _t_pw
            else:
                _pw_now = 1.0
            _chamfer_rev_weight = _chamfer_rev_base * _pw_now
        else:
            _chamfer_rev_weight = _chamfer_rev_base

        session.set_chamfer_rev_weight(_chamfer_rev_weight)

        # Huber delta for robust CD (0 = disabled = pure L2)
        if _chamfer_huber_delta > 0:
            session.set_chamfer_huber_delta(_chamfer_huber_delta)

        # Reverse gradient smoothing (clamp + spatial Gaussian smooth)
        _rev_smooth = bool(_opt_chamfer.get('chamfer_rev_smooth', False))
        _rev_smooth_radius = float(_opt_chamfer.get('chamfer_rev_smooth_radius', 2.0))
        _rev_clamp_ratio = float(_opt_chamfer.get('chamfer_rev_clamp_ratio', 3.0))
        if _rev_smooth:
            session.set_chamfer_rev_smoothing(_rev_smooth_radius, _rev_clamp_ratio, True)
            if ep == 0 or ep == _chamfer_start_ep:
                print(f"[Chamfer] rev_smooth enabled: radius={_rev_smooth_radius}, clamp={_rev_clamp_ratio}")

        if _w_chamfer_eff > 0:
            _tgt_np = np.array(tgt, dtype=np.float32)
            if _tgt_np.ndim == 1:
                _tgt_np = _tgt_np.reshape(-1, 3)
            session.set_chamfer_target(_tgt_np, _w_chamfer_eff)
            _mode_str = f"coupled(base={_chamfer_rev_base:.2f})" if _chamfer_rev_mode == 'coupled' else "fixed"
            _huber_str = f" huber_d={_chamfer_huber_delta:.2f}" if _chamfer_huber_delta > 0 else ""
            print(f"[Chamfer] ep={ep}: w_eff={_w_chamfer_eff:.4f} rev_w={_chamfer_rev_weight:.3f} mode={_mode_str}{_huber_str}")
        else:
            session.set_chamfer_target(np.zeros((1, 3), dtype=np.float32), 0.0)
    # ───────────────────────────────────────────────────────────────────────

    # ── Physics weight + smoothing schedule ──────────────────────────────
    # (_pw_start_ep, _pw_ramp_ep, _pw_final already parsed above)
    _sm_start_ep = int(_opt_chamfer.get('smoothing_start_ep', 999))
    _sm_value    = float(_opt_chamfer.get('smoothing_value', 0.7))

    # Physics weight ramp: 1.0 → _pw_final over _pw_ramp_ep episodes
    if ep >= _pw_start_ep + _pw_ramp_ep:
        _pw_eff = _pw_final
    elif ep >= _pw_start_ep:
        _t = (ep - _pw_start_ep) / max(_pw_ramp_ep, 1)
        _pw_eff = 1.0 * (1.0 - _t) + _pw_final * _t
    else:
        _pw_eff = 1.0
    session.set_physics_weight(_pw_eff)

    # Smoothing override
    if ep >= _sm_start_ep:
        session.set_smoothing_override(_sm_value)
        if ep == _sm_start_ep:
            print(f"[Smoothing] Override activated: {_sm_value} (was episode-based schedule)")
    else:
        session.set_smoothing_override(-1.0)  # use default schedule

    if _pw_eff < 1.0 or ep >= _sm_start_ep:
        print(f"[Schedule] ep={ep}: physics_w={_pw_eff:.3f}  smoothing={'override=' + str(_sm_value) if ep >= _sm_start_ep else 'default'}")
    # ───────────────────────────────────────────────────────────────────────

    # 🔥 RUN EPISODE (SINGLE C++ CALL!)
    result = session.run_episode(ep, compute_render_grads_callback)

    print(f"\n[Episode {ep}] Session Results:")
    print(f"  Loss (physics): {result.loss_physics:.2f}")
    print(f"  Passes executed: {result.num_passes_executed}/{session.get_statistics().total_passes}")
    print(f"  Wall time: {result.wall_time_seconds:.1f}s")
    print(f"  Success: {'✅' if result.success else '❌'}")

    # ── Chamfer distance measurement + optional x_late update ───────────────────
    # Always measures 3D Chamfer distance (for logging/comparison).
    # When chamfer_xlate_enabled=True: also updates x_late_py after ep>=chamfer_start_ep.
    # KEY IMPROVEMENTS:
    #   1. chamfer_start_ep: skip updates until physics mostly converged (reduces noise)
    #   2. Two-sided Chamfer: source→target + target→source (better shape coverage)
    #   3. True 3D Chamfer distance logging (sqrt of mean sq dist, in world units)
    # ─────────────────────────────────────────────────────────────────────────────
    _opt_cfg_ch = rs_full.get('optimization', {}) if rs_full else {}
    _chamfer_xlate_enabled = bool(_opt_cfg_ch.get('chamfer_xlate_enabled', False))
    _alpha_chamfer_xlate   = float(_opt_cfg_ch.get('alpha_chamfer_xlate', 1e-2))
    _chamfer_start_ep      = int(_opt_cfg_ch.get('chamfer_start_ep', 0))
    _chamfer_two_sided     = bool(_opt_cfg_ch.get('chamfer_two_sided', True))
    _use_wasserstein       = bool(_opt_cfg_ch.get('use_wasserstein', False))
    _wasserstein_blur      = float(_opt_cfg_ch.get('wasserstein_blur', 0.05))

    # Always compute true 3D Chamfer distance for logging (works for ANY experiment)
    if tgt is not None and result.success:
        try:
            import numpy as _np_cd_log
            from scipy.spatial import cKDTree as _cKDTree_log
            _pc_log = session.get_final_point_cloud()
            _pos_raw = _pc_log.get_positions() if _pc_log is not None else None
            if _pos_raw is not None:
                # get_positions() may return list or ndarray — normalize to float32 ndarray
                _pos_log = _np_cd_log.array(_pos_raw, dtype=_np_cd_log.float32)
                if 'chamfer_log_tree' not in ema_state:
                    _tgt_log = _np_cd_log.array(tgt, dtype=_np_cd_log.float32)
                    ema_state['chamfer_log_tree'] = _cKDTree_log(_tgt_log)
                    ema_state['chamfer_log_tgt'] = _tgt_log
                _, _nn_idx_log = ema_state['chamfer_log_tree'].query(_pos_log, k=1, workers=4)
                _sq_d = ((_pos_log - ema_state['chamfer_log_tgt'][_nn_idx_log])**2).sum(axis=1)
                _cd_log = float(_np_cd_log.sqrt(_sq_d.mean()))
                ema_state['last_chamfer_dist'] = _cd_log
                print(f"[Chamfer3D] ep={ep}: source→target = {_cd_log:.5f} units", flush=True)
        except Exception as _e_cd:
            print(f"[Chamfer3D] Measurement failed: {_e_cd}", flush=True)

    if _chamfer_xlate_enabled and tgt is not None and result.success:
        import torch as _torch_ch
        import numpy as _np_ch
        from scipy.spatial import cKDTree as _cKDTree_ch
        try:
            _pc_f = session.get_final_point_cloud()
            _pos_raw_ch = _pc_f.get_positions() if _pc_f is not None else None
            if _pos_raw_ch is not None:
                # Normalize to float32 ndarray (get_positions may return list)
                _pos_np = _np_ch.array(_pos_raw_ch, dtype=_np_ch.float32)
                _N_ch = len(_pos_np)
                _dev_ch = _torch_ch.device('cuda' if _torch_ch.cuda.is_available() else 'cpu')

                # Initialize x_late_py if not yet done (shared with py_xlate)
                if 'py_xlate_tensor' not in ema_state:
                    ema_state['py_xlate_tensor'] = _torch_ch.zeros(
                        (_N_ch, 3), dtype=_torch_ch.float32, device=_dev_ch)
                    ema_state['py_xlate_tensor'].requires_grad_(True)
                    ema_state['py_xlate_optim'] = _torch_ch.optim.Adam(
                        [ema_state['py_xlate_tensor']], lr=_alpha_chamfer_xlate,
                        betas=(0.9, 0.999), eps=1e-3, amsgrad=True)
                    print(f"[chamfer_xlate] Init {_N_ch}×3 x_late_py (ep{ep})", flush=True)

                # Cache NN tree (or rebuild if not cached)
                if 'chamfer_tree' not in ema_state:
                    _tgt_np_ch = _np_ch.array(tgt, dtype=_np_ch.float32)
                    ema_state['chamfer_tree'] = _cKDTree_ch(_tgt_np_ch)
                    ema_state['chamfer_tgt_tensor'] = _torch_ch.from_numpy(_tgt_np_ch).to(_dev_ch)
                    # Also build source→target tree for two-sided Chamfer
                    print(f"[chamfer_xlate] Built NN tree ({len(_tgt_np_ch)} target pts)", flush=True)

                _xlt_ch = ema_state['py_xlate_tensor']
                _xlt_o_ch = ema_state['py_xlate_optim']
                _tree_ch = ema_state['chamfer_tree']
                # Use same device as py_xlate_tensor (may have been inited on CPU by render callback)
                _dev_ch = _xlt_ch.device
                _x_tgt_all = ema_state['chamfer_tgt_tensor'].to(_dev_ch)

                # x_render = x_physics + x_late_py  (only x_late_py has grad)
                _x_phys = _torch_ch.from_numpy(_pos_np).to(_dev_ch)
                _x_render_ch = _x_phys.detach() + _xlt_ch

                # ── Compute true 3D Chamfer distance (for logging) ──────────────
                _x_render_np = _x_render_ch.detach().cpu().numpy()
                _, _nn_src2tgt_idx = _tree_ch.query(_x_render_np, k=1, workers=4)
                _sq_dist_s2t = ((_x_render_np - _np_ch.array(tgt)[_nn_src2tgt_idx])**2).sum(axis=1)
                _chamfer_s2t = float(_np_ch.sqrt(_sq_dist_s2t.mean()))

                if _chamfer_two_sided:
                    # target→source direction
                    if 'chamfer_src_tree' not in ema_state:
                        ema_state['chamfer_src_tree'] = _cKDTree_ch(_x_render_np)
                        _src_tree = ema_state['chamfer_src_tree']
                    else:
                        # Update source tree with current x_render positions
                        ema_state['chamfer_src_tree'] = _cKDTree_ch(_x_render_np)
                        _src_tree = ema_state['chamfer_src_tree']
                    _tgt_np_ch = _np_ch.array(tgt, dtype=_np_ch.float32)
                    _, _nn_tgt2src_idx = _src_tree.query(_tgt_np_ch, k=1, workers=4)
                    _sq_dist_t2s = ((_tgt_np_ch - _x_render_np[_nn_tgt2src_idx])**2).sum(axis=1)
                    _chamfer_t2s = float(_np_ch.sqrt(_sq_dist_t2s.mean()))
                    _chamfer_dist = 0.5 * (_chamfer_s2t + _chamfer_t2s)
                else:
                    _chamfer_dist = _chamfer_s2t

                print(f"[chamfer_xlate] ep={ep}: 3D_chamfer={_chamfer_dist:.4f}  "
                      f"(s→t={_chamfer_s2t:.4f})", flush=True)

                # ── Chamfer x_late update (only after chamfer_start_ep) ──────────
                _chamfer_adam_steps = int(_opt_cfg_ch.get('chamfer_adam_steps', 1))
                if ep >= _chamfer_start_ep:
                    _tgt_np_optim = _np_ch.array(tgt, dtype=_np_ch.float32)
                    _total_loss_ch = 0.0

                    if _use_wasserstein:
                        # ── Sinkhorn OT loss (geomloss) ────────────────────────
                        try:
                            from geomloss import SamplesLoss as _SamplesLoss
                            # Create OT loss fn once (tensorized: pure PyTorch, no KeOps needed)
                            # Subsample for memory: tensorized is O(N*M); 5000×5000 ≈ 100MB
                            _ot_n_sub = int(_opt_cfg_ch.get('wasserstein_n_subsample', 5000))
                            _ot_loss_fn = _SamplesLoss("sinkhorn", p=2,
                                                        blur=_wasserstein_blur,
                                                        backend="tensorized",
                                                        scaling=0.5)
                            print(f"[chamfer_xlate] Using Sinkhorn OT (tensorized, N_sub={_ot_n_sub})", flush=True)
                        except ImportError:
                            print("[chamfer_xlate] WARNING: geomloss not installed, falling back to Chamfer", flush=True)
                            _use_wasserstein = False

                    for _step_ch in range(_chamfer_adam_steps):
                        _x_render_step = _x_phys.detach() + _xlt_ch
                        _xlt_o_ch.zero_grad()

                        if _use_wasserstein:
                            # Sinkhorn Wasserstein-2: inherently two-sided (bijective soft matching)
                            # Subsample fresh random subsets each step → stochastic OT gradient
                            _src_idx_ot = _torch_ch.randperm(_x_render_step.shape[0],
                                                              device=_dev_ch)[:_ot_n_sub]
                            _tgt_idx_ot = _torch_ch.randperm(_x_tgt_all.shape[0],
                                                              device=_dev_ch)[:_ot_n_sub]
                            _x_src_sub = _x_render_step[_src_idx_ot]   # (N_sub,3) grad via _xlt_ch
                            _x_tgt_sub = _x_tgt_all[_tgt_idx_ot]       # (N_sub,3) no grad
                            _loss_step = _ot_loss_fn(_x_src_sub, _x_tgt_sub)
                        else:
                            _x_np_step = _x_render_step.detach().cpu().numpy().astype(_np_ch.float32)
                            # s→t
                            _, _nn_s2t = _tree_ch.query(_x_np_step, k=1, workers=4)
                            _x_nn_s2t = _x_tgt_all[_torch_ch.from_numpy(_nn_s2t).long()]
                            _loss_s2t = (_x_render_step - _x_nn_s2t).pow(2).sum(dim=1).mean()
                            if _chamfer_two_sided:
                                _src_tree_step = _cKDTree_ch(_x_np_step)
                                _, _nn_t2s = _src_tree_step.query(_tgt_np_optim, k=1, workers=4)
                                _x_nn_t2s = _x_render_step[_torch_ch.from_numpy(_nn_t2s).long()]
                                _loss_t2s = (_x_tgt_all - _x_nn_t2s).pow(2).sum(dim=1).mean()
                                _loss_step = 0.5 * (_loss_s2t + _loss_t2s)
                            else:
                                _loss_step = _loss_s2t

                        _loss_step.backward()
                        _xlt_o_ch.step()
                        _total_loss_ch += _loss_step.item()

                    _mode_str = "Wasserstein" if _use_wasserstein else f"Chamfer({'2-sided' if _chamfer_two_sided else '1-sided'})"
                    _gnorm_ch = float(_xlt_ch.grad.norm()) if _xlt_ch.grad is not None else 0.0
                    _xnorm_ch = float(_xlt_ch.detach().norm())
                    print(f"[chamfer_xlate] UPDATED [{_mode_str}] ({_chamfer_adam_steps} steps): "
                          f"avg_loss={_total_loss_ch/_chamfer_adam_steps:.5f}  "
                          f"||grad||={_gnorm_ch:.3e}  ||x_late||={_xnorm_ch:.4f}", flush=True)
                else:
                    print(f"[chamfer_xlate] Warmup (ep{ep} < start_ep{_chamfer_start_ep}): "
                          f"measuring only, no update", flush=True)

                # Store Chamfer dist for episode summary
                ema_state['last_chamfer_dist'] = _chamfer_dist

        except Exception as _e_ch:
            print(f"[chamfer_xlate] Error: {_e_ch}", flush=True)
            import traceback; traceback.print_exc()

    # 🔥 DEBUG: Check visualization conditions
    print(f"\n[DEBUG] Visualization check:")
    print(f"  png_enabled = {png_enabled}")
    print(f"  result.success = {result.success}")
    print(f"  out_dir = {out_dir}")
    print(f"  Episode number (ep) = {ep}")

    # Visualization (last pass)
    if png_enabled and result.success:
        print(f"\n[Visualization] Saving results...")
        print(f"  Output directory: {out_dir}")
        try:
            # Get final point cloud for visualization
            print(f"  [1/7] Getting final point cloud from session...")
            pc_final = session.get_final_point_cloud()
            if pc_final is None:
                print(f"  ⚠️  No final point cloud available")
            else:
                print(f"  [2/7] Point cloud retrieved successfully")
                # Session mode: Do simplified visualization without cg
                from utils.physics_utils import extract_point_cloud_state

                # Extract state
                print(f"  [3/7] Extracting state (x, v, F)...")
                x, v, F = extract_point_cloud_state(pc_final, requires_grad=False)
                print(f"    Extracted {len(x)} particles")

                # Upsample for visualization
                seed = 9999 + ep*1000 + (result.num_passes_executed - 1)
                from sampling import upsample

                print(f"  [4/7] Upsampling for visualization (seed={seed})...")
                result_viz = upsample(
                    x, F,
                    cfg=rs_full,
                    state=ema_state,
                    seed=seed,
                    return_torch=True,
                    export_stages=True,  # 🔥 DEBUG: Enable subdivision debug output
                    learnable_cov_module=cov_module,
                    current_episode=ep,
                    external_levelset=external_levelset,
                    use_simple_pipeline=rs_full.get('upsample', {}).get('use_simple_pipeline', True)
                )

                mu_viz = result_viz.get('points')
                cov_viz = result_viz.get('cov')
                print(f"    Upsampled to {len(mu_viz)} points")

                # Save stage progression if available
                if "stage_outputs" in result_viz:
                    from sampling.io.export import save_stage_progression
                    print(f"    Saving stage progression...")
                    save_stage_progression(out_dir, -1, result_viz["stage_outputs"])

                # Render and save
                from utils.rendering_utils import prepare_rendering_inputs
                print(f"  [5/7] Preparing rendering inputs...")
                rgb = prepare_rendering_inputs(mu_viz, result_viz, campos, render_cfg, particle_color)
                normals_viz = result_viz.get('normals')

                # 🔥 Orient normals toward camera (for correct normal map visualization)
                if normals_viz is not None:
                    # Convert to numpy if torch
                    if hasattr(normals_viz, 'cpu'):
                        normals_viz = normals_viz.detach().cpu().numpy()
                    if hasattr(mu_viz, 'cpu'):
                        mu_viz_np = mu_viz.detach().cpu().numpy()
                    else:
                        mu_viz_np = mu_viz

                    # Orient normals toward camera
                    view_dir = campos - mu_viz_np  # (N, 3) vector from particle to camera
                    view_dir_norm = view_dir / (np.linalg.norm(view_dir, axis=1, keepdims=True) + 1e-8)
                    dot_product = np.sum(normals_viz * view_dir_norm, axis=1)  # (N,)
                    flip_mask = dot_product < 0
                    normals_viz = normals_viz.copy()
                    normals_viz[flip_mask] = -normals_viz[flip_mask]

                print(f"  [6/7] Rendering...")
                out_render = renderer.render(mu_viz, cov_viz, rgb=rgb, normals=normals_viz, render_normal_map=True)

                # Save images
                from utils.io_utils import save_image_png, save_depth_png

                # Convert to numpy if needed
                def to_np(x):
                    if x is None:
                        return None
                    if torch.is_tensor(x):
                        return x.detach().cpu().numpy()
                    return x

                img_np = to_np(out_render.get('image'))
                alpha_np = to_np(out_render.get('alpha'))
                depth_np = to_np(out_render.get('depth'))
                normal_np = to_np(out_render.get('normal_map'))

                # Save renders
                print(f"  [7/7] Saving PNG files...")
                saved_files = []
                if img_np is not None:
                    fpath = out_dir / "render.png"
                    save_image_png(fpath, img_np)
                    saved_files.append(str(fpath))
                if alpha_np is not None:
                    fpath = out_dir / "alpha.png"
                    save_image_png(fpath, alpha_np)
                    saved_files.append(str(fpath))
                if depth_np is not None:
                    fpath = out_dir / "depth.png"
                    save_depth_png(fpath, depth_np)
                    saved_files.append(str(fpath))
                if normal_np is not None:
                    fpath = out_dir / "normal.png"
                    save_image_png(fpath, normal_np)
                    saved_files.append(str(fpath))

                # Save gaussians.npz for render_comparison.py
                try:
                    import numpy as _np_viz
                    from utils.io_utils import save_gaussians_npz as _save_npz
                    _mu_np = mu_viz.detach().cpu().numpy() if torch.is_tensor(mu_viz) else _np_viz.array(mu_viz)
                    _cov_np = cov_viz.detach().cpu().numpy() if torch.is_tensor(cov_viz) else _np_viz.array(cov_viz)
                    _rgb_np = to_np(rgb) if rgb is not None else _np_viz.tile(
                        _np_viz.array(particle_color, dtype=_np_viz.float32), (len(_mu_np), 1))
                    _save_npz(out_dir / f"ep{ep:03d}_gaussians.npz", _mu_np, _cov_np, _rgb_np)
                    saved_files.append(str(out_dir / f"ep{ep:03d}_gaussians.npz"))
                except Exception as _e_npz:
                    print(f"  ⚠️  Could not save gaussians.npz: {_e_npz}", flush=True)

                print(f"  ✅ Visualization saved ({len(saved_files)} files)")
                for f in saved_files:
                    print(f"      - {f}")
                
        except Exception as e:
            print(f"  ⚠️  Visualization failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n[DEBUG] Visualization SKIPPED:")
        if not png_enabled:
            print(f"  Reason: png_enabled={png_enabled}")
        if not result.success:
            print(f"  Reason: result.success={result.success}")

    print(f"\n{'='*70}")
    print(f"🔥 Session Mode: Episode {ep} COMPLETE")
    print(f"{'='*70}\n")

    # Prepare final losses
    final_losses = {
        'loss_physics': result.loss_physics,
    }
    final_losses.update(last_render_loss_components)
    # Include 3D Chamfer distance if computed
    if 'last_chamfer_dist' in ema_state:
        final_losses['chamfer_3d'] = ema_state['last_chamfer_dist']

    return ema_state, final_losses


# ============================================================================
# E2E Training Episode (Legacy Pass-by-Pass Mode)
# ============================================================================

def run_e2e_episode(
    ep: int,
    cg: Any,
    opt: Any,
    num_timesteps: int,
    control_stride: int,
    num_passes: int,
    rs_full: Dict,
    ema_state: Dict,
    renderer: Any,
    loss_manager: Any,
    target_render: Dict,
    view_params: Dict,
    campos: np.ndarray,
    render_cfg: Dict,
    particle_color: list,
    out_dir: Path,
    png_enabled: bool,
    tgt: np.ndarray,
    cov_module=None,
    cov_optimizer=None,
    external_levelset=None,
    extra_views=None  # Multi-view: list of {renderer, target_render, campos}
) -> Tuple[Dict, Dict]:
    """
    Run one complete E2E training episode with multi-pass refinement.
    
    ════════════════════════════════════════════════════════════════════════════
    E2E (End-to-End) Training Architecture
    ════════════════════════════════════════════════════════════════════════════
    
    This function implements the core training loop that jointly optimizes:
      1. Physics simulation (MPM)
      2. Surface synthesis (Upsampling)
      3. Rendering (3D Gaussian Splatting)
    
    Training Flow (Multi-Pass Refinement):
    ────────────────────────────────────────────────────────────────────────────
    For pass in [1, 2, 3]:
      
      Phase 1: Inject Render Gradients (if pass > 1)
        • Take ∂L_render/∂F, ∂L_render/∂x from previous pass
        • Inject into C++ MPM backend
        • L_total = L_physics + L_render
      
      Phase 2: Physics Optimization
        • Run forward simulation: x(0) → x(1) → ... → x(T)
        • Compute physics loss: L_physics = ||x(T) - x_target||²
        • Backward: compute gradients ∂L_total/∂controls
        • Update: Adam step on control forces
        • 🔥 Advect level set using final velocity (for next pass)
      
      Phase 3: Render Loss Computation
        • Get final state: x_final, F_final
        • Upsample: (x, F) → (μ, Σ, normals) [400k points]
        • Compute shading: RGB from normals + lighting
        • Render: (μ, Σ, RGB) → {image, alpha, depth}
        • Compare with target: L_render = f(pred, target)
        • Backward: L_render.backward() → ∂L/∂F, ∂L/∂x
        • Store gradients for next pass
      
      Phase 4: Visualization (last pass only)
        • Save rendered images (PNG)
        • Export Gaussians (NPZ)
        • Export point cloud (PLY)
        • Generate comparison images
    
    ════════════════════════════════════════════════════════════════════════════
    """
    # ════════════════════════════════════════════════════════════════════════════
    # Episode Initialization
    # ════════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"Episode {ep} START")
    print(f"{'='*70}")
    
    # Reset EMA threshold at episode boundary
    if ep > 0 and "ema_thr" in ema_state:
        print(f"\n[Episode Boundary] Resetting EMA threshold")
        ema_state["ema_thr"] = None
    
    # Setup computation graph
    print(f"\n[Setup] Creating {num_timesteps} timestep layers...")
    cg.set_up_comp_graph(num_timesteps)

    # LSRC: initialize Fc_late once per episode (EnableFcLate is idempotent: no-op if already init'd)
    _opt_cfg_init = rs_full.get('optimization', {}) if rs_full else {}
    try:
        _pc0 = cg.get_point_cloud_at_timestep(0)
        _pos0 = _pc0.get_positions()          # (N, 3) numpy array
        _N_particles = len(_pos0) if _pos0 is not None else 0
    except Exception as _e_pc:
        _N_particles = 0
        print(f"[Init] Could not read particle count: {_e_pc}")
    if _N_particles > 0:
        if bool(_opt_cfg_init.get('lsrc_enabled', False)):
            try:
                cg.enable_fc_late(_N_particles)
            except Exception as _e_lsrc:
                print(f"[LSRC] enable_fc_late failed: {_e_lsrc}")
        if bool(_opt_cfg_init.get('xlate_enabled', False)):
            try:
                cg.enable_x_late(_N_particles)
            except Exception as _e_xlate:
                print(f"[x_late] enable_x_late failed: {_e_xlate}")

    print(f"[Setup] Running initial forward simulation...")
    cg.compute_forward_pass(0, ep)
    
    try:
        loss_initial = cg.end_layer_mass_loss()
        print(f"[Setup] Initial physics loss: {loss_initial:.2f}")
    except Exception as e:
        print(f"[Setup] Loss computation failed: {e}")
        loss_initial = 0.0
    
    print(f"\n{'='*70}")
    print(f"E2E Training - {num_passes} Passes")
    print(f"{'='*70}")
    
    accumulated_render_grads = None
    final_loss_components = None
    
    # ════════════════════════════════════════════════════════════════════════════
    # Multi-Pass Refinement Loop
    # ════════════════════════════════════════════════════════════════════════════
    for pass_idx in range(num_passes):
        print(f"\n{'─'*70}")
        print(f"Pass {pass_idx+1}/{num_passes}")
        print(f"{'─'*70}")
        
        # 🔥 Physics weight: DISABLED (we now use normalized gradient combination in Python)
        # Old approach scaled physics gradients in C++ (backwards logic).
        # New approach: normalize and combine gradients in Python before injection.
        phys_w = 1.0  # Keep physics at full scale; balance is handled in Python

        try:
            cg.set_physics_weight(phys_w)
            print(f"[Physics Weight] Fixed at {phys_w:.2f} (balance handled by gradient normalization)")
        except Exception as e:
            print(f"[Physics Weight] Failed: {e}")

        # J-min barrier: prevent over-compression of deformation gradient.
        # Read from config: optimization.j_barrier_weight (default 0 = disabled)
        #                   optimization.j_barrier_target (default 0.8)
        _opt_cfg_for_barrier = rs_full.get('optimization', {}) if rs_full else {}
        j_barrier_weight = float(_opt_cfg_for_barrier.get('j_barrier_weight', 0.0))
        j_barrier_target = float(_opt_cfg_for_barrier.get('j_barrier_target', 0.8))
        try:
            cg.set_j_barrier(j_barrier_target, j_barrier_weight)
            if j_barrier_weight > 0:
                print(f"[J-barrier] Enabled: target={j_barrier_target:.2f}, weight={j_barrier_weight:.3f}")
            else:
                print(f"[J-barrier] Disabled (weight=0)")
        except Exception as e:
            print(f"[J-barrier] set_j_barrier failed: {e}")

        torch.cuda.empty_cache()
        
        # ────────────────────────────────────────────────────────────────────────
        # Phase 1+2: Physics Optimization with Batched Gradient Injection (OPTIMIZED)
        # ────────────────────────────────────────────────────────────────────────
        # 🔥 NEW: Use batched E2E pass (combines gradient injection + physics optimization)
        use_batched = True  # Set to False to use old method

        if use_batched:
            # Prepare render gradients (if available)
            render_grads_dict = None
            if accumulated_render_grads is not None and renderer is not None and loss_manager is not None:
                render_grads_dict = {
                    'dLdF': accumulated_render_grads['dLdF'],
                    'dLdx': accumulated_render_grads['dLdx']
                }
                grad_F_norm = np.linalg.norm(render_grads_dict['dLdF'])
                grad_x_norm = np.linalg.norm(render_grads_dict['dLdx'])
                print(f"\n[Batched E2E] Pass {pass_idx+1} with render gradients")
                print(f"├─ Points: {len(render_grads_dict['dLdF'])}")
                print(f"├─ ||∂L_render/∂F|| = {grad_F_norm:.6e}")
                print(f"└─ ||∂L_render/∂x|| = {grad_x_norm:.6e}")
            else:

                    print(f"\n[Inject] No render grads available\n")
            
            # ────────────────────────────────────────────────────────────────────────
            # Phase 2: Physics Optimization
            # ────────────────────────────────────────────────────────────────────────
            t0_physics = time.time()
            
            # 🔍 Check physics optimization parameters
            print(f"\n🔍 [Physics Params]")
            print(f"├─ Learning rate (alpha): {opt.initial_alpha}")
            print(f"├─ Max GD iters: {opt.max_gd_iters}")
            print(f"├─ Max LS iters: {opt.max_ls_iters}")
            print(f"└─ Timesteps: {num_timesteps}")

            # 🔥 Single batched call (2-3x faster than separate calls!)
            # Skip SetUpCompGraph for pass 2+ to preserve simulation state
            skip_setup = (pass_idx > 0)
            loss_physics = run_physics_optimization_batched(
                cg, opt, render_grads_dict, pass_idx, skip_setup=skip_setup
            )

        else:
            # Old method (kept for fallback/debugging)
            if accumulated_render_grads is not None and renderer is not None and loss_manager is not None:
                dLdF = accumulated_render_grads['dLdF']
                dLdx = accumulated_render_grads['dLdx']

                grad_F_norm = np.linalg.norm(dLdF)
                grad_x_norm = np.linalg.norm(dLdx)

                print(f"\n[Inject] Applying render gradients from Pass {pass_idx}")
                print(f"├─ Points: {len(dLdF)}")
                print(f"├─ ||∂L_render/∂F|| = {grad_F_norm:.6e}")
                print(f"└─ ||∂L_render/∂x|| = {grad_x_norm:.6e}")

                try:
                    cg.set_render_gradients(dLdF, dLdx)
                    print(f"   ✅ Gradients injected successfully\n")
                except Exception as e:
                    print(f"   ❌ Gradient injection failed: {e}\n")
            else:
                if pass_idx == 0:
                    print(f"\n[Inject] No previous render grads (first pass)\n")
                elif renderer is None or loss_manager is None:
                    print(f"\n[Inject] Skipped - Physics-only mode\n")
                else:
                    print(f"\n[Inject] No render grads available\n")

            loss_physics = run_physics_optimization(
                cg, opt, num_timesteps, control_stride, ep, pass_idx
            )
        
        t_physics = time.time() - t0_physics
        print(f"⏱️  [Physics Optimization] {t_physics:.2f}s")
        
        # ────────────────────────────────────────────────────────────────────────
        # Phase 3: Compute Render Loss
        # ────────────────────────────────────────────────────────────────────────
        # Pass 1: Compute render loss (no render grads were injected during physics)
        # Pass 2/3: Compute render loss (render grads from previous pass were injected)

        t0_render = time.time()
        seed = 9999 + ep*1000 + pass_idx

        print(f"\n[Render] Computing loss for Pass {pass_idx+1}...")

        try:
            result = compute_render_loss_pass(
                cg, num_timesteps, rs_full, ema_state, renderer,
                loss_manager, target_render, view_params, campos,
                render_cfg, particle_color, seed, cov_module,
                external_levelset=None,
                current_episode=ep,
                extra_views=extra_views
            )
        except FloatingPointError as err:
            print(f"└─ ❌ Render loss computation failed: {err}")
            print(f"   ↳ Skipping remaining passes for this episode.")
            break
        
        if result and result[0] is not None:
            ema_state_new, F, x, loss_components = result
            
            # Store loss components from last pass
            if pass_idx == num_passes - 1:
                final_loss_components = loss_components
            
            # Update learnable covariance
            if cov_optimizer is not None and cov_module is not None:
                cov_optimizer.step()
                cov_optimizer.zero_grad()
            
            # 🔥 Track render loss change
            if 'loss_render_total' in loss_components:
                current_render_loss = loss_components['loss_render_total']
                if hasattr(current_render_loss, 'item'):
                    current_render_loss = current_render_loss.item()
                
                if pass_idx > 0 and final_loss_components is not None:
                    prev_render_loss = final_loss_components.get('loss_render_total', 0)
                    if hasattr(prev_render_loss, 'item'):
                        prev_render_loss = prev_render_loss.item()
                    
                    render_change = current_render_loss - prev_render_loss
                    print(f"\n🔍 [Render Loss Tracking]")
                    print(f"├─ Previous: {prev_render_loss:.6f}")
                    print(f"├─ Current:  {current_render_loss:.6f}")
                    print(f"└─ Change:   {render_change:+.6f} {'⚠️ INCREASING!' if render_change > 0 else '✅ Decreasing'}")
            
            # Extract gradients (E2E mode only)
            render_grads = extract_render_gradients(F, x)
            if render_grads is not None and renderer is not None and loss_manager is not None:
                dLdF_render = render_grads['dLdF']
                dLdx_render = render_grads['dLdx']
                
                # ═══════════════════════════════════════════════════════════════
                # 🔥 NEW: Normalized Gradient Combination (prevents magnitude mismatch)
                # ═══════════════════════════════════════════════════════════════

                # 1. Diagnose gradient health
                grad_health = diagnose_gradient_health(dLdF_render, dLdx_render, grad_type="render")
                if not grad_health['is_healthy']:
                    print(f"├─ ⚠️  Skipping unhealthy render gradients")
                    accumulated_render_grads = None
                    continue

                # ── Independent Render Adam Step ──────────────────────────────────────
                # Runs HERE — before the w_render_base check — so it fires even when
                # render-into-physics injection is disabled (w_render_base=0.0).
                # Uses raw render grads (dLdF_render, dLdx_render) for a clean signal.
                # ─────────────────────────────────────────────────────────────────────
                _opt_cfg_r = rs_full.get('optimization', {}) if rs_full else {}
                _render_adam_enabled = bool(_opt_cfg_r.get('render_adam_enabled', False))
                _alpha_render = float(_opt_cfg_r.get('alpha_render', 1e-3))
                if _render_adam_enabled:
                    try:
                        # Store RAW render grads in C++ for render-only backward
                        cg.set_render_gradients(dLdF_render, dLdx_render)
                        # Backward with physics terminal zeroed, only render grads injected
                        cg.compute_render_backward_only(0)
                        _r_norm = cg.get_render_grad_norm()
                        if _r_norm > 1e-12:
                            cg.apply_render_adam_step(
                                alpha_r=_alpha_render,
                                beta1=0.9, beta2=0.999, epsilon=1e-3,
                                timestep=ep + 1, control_layer=0
                            )
                            print(f"[Render Adam] ep={ep}, pass={pass_idx}, "
                                  f"||dLdF||_ctrl={_r_norm:.3e}, alpha={_alpha_render:.3e}")
                        else:
                            print(f"[Render Adam] skipped — zero grad norm at control layer")
                    except Exception as _e:
                        print(f"[Render Adam] Error: {_e}")

                # ── LSRC: Late-Stage Render Control ───────────────────────────────────
                # Fc_late is applied to F at the FINAL layer AFTER ForwardTimeStep.
                # Gradient = direct render grad at final layer (0 backward steps, ~47 norm).
                # 188,000× attenuation completely bypassed.
                # ─────────────────────────────────────────────────────────────────────
                _lsrc_enabled = bool(_opt_cfg_r.get('lsrc_enabled', False))
                _alpha_lsrc   = float(_opt_cfg_r.get('alpha_lsrc', 1e-2))
                if _lsrc_enabled:
                    try:
                        cg.set_render_gradients(dLdF_render, dLdx_render)
                        # 0 backward steps: render grad stays in final layer (no attenuation)
                        _num_layers = cg.get_num_layers()
                        cg.compute_render_backward_only(_num_layers - 1)
                        _fc_late_norm = cg.get_fc_late_grad_norm()
                        if _fc_late_norm > 1e-12:
                            cg.apply_fc_late_adam_step(
                                alpha_late=_alpha_lsrc,
                                beta1=0.9, beta2=0.999, epsilon=1e-3,
                                timestep=ep + 1
                            )
                            print(f"[LSRC] ep={ep}, pass={pass_idx}, "
                                  f"||dL/dFc_late||={_fc_late_norm:.3e}, alpha={_alpha_lsrc:.3e}")
                        else:
                            print(f"[LSRC] skipped — zero grad norm at final layer")
                    except Exception as _e:
                        print(f"[LSRC] Error: {_e}")

                # ── x_late: Late-Stage Position Correction ─────────────────────────────
                # x_late is added to particle positions at the FINAL layer.
                # Gradient = dL/dx_final (~2e-3 norm) — dominant render signal, 0 attenuation.
                # ─────────────────────────────────────────────────────────────────────────
                _xlate_enabled = bool(_opt_cfg_r.get('xlate_enabled', False))
                _alpha_xlate   = float(_opt_cfg_r.get('alpha_xlate', 1e-2))
                if _xlate_enabled:
                    try:
                        cg.set_render_gradients(dLdF_render, dLdx_render)
                        _num_layers = cg.get_num_layers()
                        cg.compute_render_backward_only(_num_layers - 1)
                        _x_late_norm = cg.get_x_late_grad_norm()
                        if _x_late_norm > 1e-12:
                            cg.apply_x_late_adam_step(
                                alpha_late=_alpha_xlate,
                                beta1=0.9, beta2=0.999, epsilon=1e-3,
                                timestep=ep + 1
                            )
                            print(f"[x_late] ep={ep}, pass={pass_idx}, "
                                  f"||dL/dx_late||={_x_late_norm:.3e}, alpha={_alpha_xlate:.3e}")
                        else:
                            print(f"[x_late] skipped — zero grad norm at final layer")
                    except Exception as _e:
                        print(f"[x_late] Error: {_e}")

                try:
                    # 🔥 NEW: Get physics gradients from C++ backend
                    dLdF_phys_np, dLdx_phys_np = cg.get_last_layer_phys_gradients()

                    if dLdF_phys_np is None or dLdx_phys_np is None:
                        print(f"\n⚠️  [PCGrad] Physics gradients not available")
                        accumulated_render_grads = render_grads
                        continue

                    # Keep as numpy arrays (gradient functions expect numpy)
                    dLdF_phys = dLdF_phys_np
                    dLdx_phys = dLdx_phys_np

                    # 2. Compute gradient statistics (before combination)
                    render_stats = compute_gradient_statistics(dLdF_render, dLdx_render)
                    phys_stats = compute_gradient_statistics(dLdF_phys, dLdx_phys)

                    g_render = render_stats['grad_total_norm']
                    g_phys = phys_stats['grad_total_norm']

                    # Compute cosine similarity (conflict detection)
                    cosine = compute_gradient_cosine_similarity(
                        dLdF_phys, dLdx_phys, dLdF_render, dLdx_render
                    )
                    conflict_status = (
                        '⚠️ CONFLICT' if cosine < -0.3 else
                        '✓ aligned' if cosine > 0.3 else
                        '~ neutral'
                    )

                    # 3. 🔥 FIXED: Use render_loss_weight from config
                    # This is the proper place to apply render_loss_weight!

                    # DEBUG: Check what's in the config
                    print(f"\n[DEBUG CONFIG READ]")
                    print(f"  rs_full keys: {list(rs_full.keys())}")

                    # Check if there's a nested 'upsample' key
                    if 'upsample' in rs_full:
                        print(f"  'upsample' key exists, contents: {rs_full['upsample']}")

                    # Try reading from multiple locations
                    render_loss_weight = rs_full.get('render_loss_weight', None)
                    if render_loss_weight is None and 'upsample' in rs_full:
                        render_loss_weight = rs_full['upsample'].get('render_loss_weight', None)
                    if render_loss_weight is None:
                        render_loss_weight = 1.0  # Default

                    # 🔥 CRITICAL: Convert to float (YAML may parse 1e5 as string!)
                    render_loss_weight = float(render_loss_weight)

                    print(f"  render_loss_weight READ: {render_loss_weight}")

                    # Adaptive weight scheduling based on episode progress
                    total_animations = rs_full.get('optimization', {}).get('num_animations', 50)
                    progress = ep / max(1, total_animations)

                    # 🔥 w_render_base: config override or adaptive schedule
                    # Check optimization.w_render_base first (from YAML optimization section),
                    # then upsample.w_render_base, then rs_full top-level.
                    _w_render_base_override = None
                    for _loc in [
                        rs_full.get('optimization', {}),
                        rs_full.get('upsample', {}),
                        rs_full,
                    ]:
                        if isinstance(_loc, dict) and 'w_render_base' in _loc:
                            _w_render_base_override = float(_loc['w_render_base'])
                            break

                    if _w_render_base_override is not None:
                        w_render_base = _w_render_base_override
                        print(f"\n├─ [w_render_base] Config override: {w_render_base:.3f}")
                    elif ep < 5:
                        # Early episodes: Strong render weight from the start
                        w_render_base = 0.50
                        print(f"\n├─ [Early Training] Episode {ep} < 5: Strong render weight ({w_render_base})")
                    elif ep < 15:
                        # Ramp-up: Maintain high render weight
                        w_render_base = 0.50 + 0.20 * ((ep - 5) / 10)  # 0.50 → 0.70
                    elif ep < 30:
                        # Mid-training: High balanced weight
                        w_render_base = 0.70
                    else:
                        # Late: Full render emphasis for refinement
                        w_render_base = 0.80

                    # 🔥 Apply render_loss_weight from config DIRECTLY (no scaling!)
                    # User controls exact weight via config
                    w_render = w_render_base * render_loss_weight  # Direct multiplication!
                    w_physics = 1.0

                    if w_render <= 0:
                        print(f"\n├─ [Render Weight] Disabled (w_render={w_render:.3f}), skipping gradient injection")
                        accumulated_render_grads = None
                        # Update ema_state and run visualization even when not injecting
                        ema_state = ema_state_new
                        del F, x
                        if pass_idx == num_passes - 1:
                            seed = 9999 + ep*1000 + pass_idx
                            render_losses = final_loss_components if final_loss_components is not None else None
                            visualize_episode(
                                ep, out_dir, cg, num_timesteps, rs_full, ema_state,
                                renderer, campos, render_cfg, particle_color,
                                png_enabled, tgt, loss_physics, seed, cov_module,
                                external_levelset=external_levelset,
                                render_losses=render_losses
                            )
                        continue

                    print(f"\n├─ [Weight Calculation]")
                    print(f"│  ├─ render_loss_weight (config): {render_loss_weight}")
                    print(f"│  ├─ w_render_base (schedule): {w_render_base:.3f}")
                    print(f"│  ├─ w_render (final): {w_render:.3f}")
                    print(f"│  └─ w_physics: {w_physics:.3f}")

                    # ══════════════════════════════════════════════════════════
                    # 4. PCGrad: Gradient Conflict Resolution
                    # ══════════════════════════════════════════════════════════
                    # PCGrad (Gradient Surgery) projects out conflicting gradient
                    # components to prevent render and physics from fighting.
                    # Config: optimization.use_pcgrad (default: true)
                    # ══════════════════════════════════════════════════════════

                    # Read PCGrad configuration
                    optimization_cfg = rs_full.get('optimization', {})
                    use_pcgrad = optimization_cfg.get('use_pcgrad', True)  # Default: enabled
                    pcgrad_threshold = optimization_cfg.get('pcgrad_threshold', -0.1)  # Conflict threshold

                    # Determine if conflict exists
                    has_conflict = (cosine < pcgrad_threshold)
                    should_apply_pcgrad = use_pcgrad and has_conflict

                    # Status logging
                    print(f"\n├─ [PCGrad Status]")
                    print(f"│  ├─ Config: use_pcgrad = {use_pcgrad}")
                    print(f"│  ├─ Threshold: {pcgrad_threshold:.2f}")
                    print(f"│  │")
                    print(f"│  ├─ 🎯 GRADIENT SIMILARITY:")
                    print(f"│  │   ├─ Cosine: {cosine:+.4f} {conflict_status}")
                    print(f"│  │   ├─ Interpretation:")
                    if cosine < -0.3:
                        print(f"│  │   │  └─ ⚠️  Strong conflict (gradients oppose)")
                    elif cosine < -0.1:
                        print(f"│  │   │  └─ ⚠️  Mild conflict (gradients diverge)")
                    elif cosine < 0.3:
                        print(f"│  │   │  └─ ~ Neutral (gradients independent)")
                    else:
                        print(f"│  │   │  └─ ✅ Aligned (gradients cooperate)")
                    print(f"│  │   └─ Range: -1.0 (opposite) → 0.0 (orthogonal) → +1.0 (aligned)")
                    print(f"│  │")
                    print(f"│  └─ Action: {'✅ APPLYING PCGrad' if should_apply_pcgrad else '⏭️  Skipping (no conflict)' if use_pcgrad else '❌ DISABLED'}")

                    # Apply PCGrad if needed
                    if should_apply_pcgrad:
                        print(f"\n🔥 [PCGrad] Conflict detected! Projecting render gradients...")
                        print(f"    ├─ Cosine: {cosine:.3f} (threshold: {pcgrad_threshold:.2f})")
                        print(f"    └─ Removing conflicting components from render gradient")

                        # Project render gradients to remove conflict
                        dLdF_render_proj, dLdx_render_proj, pcgrad_info = pcgrad_projection(
                            dLdF_render=dLdF_render,
                            dLdx_render=dLdx_render,
                            dLdF_physics=dLdF_phys,
                            dLdx_physics=dLdx_phys,
                            conflict_threshold=pcgrad_threshold
                        )

                        # Use projected gradients
                        dLdF_render_final = dLdF_render_proj
                        dLdx_render_final = dLdx_render_proj
                        pcgrad_applied = True

                        print(f"    ✅ PCGrad projection complete")
                        print(f"       ├─ Projection scale: {pcgrad_info.get('pcgrad_projection_scale', 0.0):.3f}")
                        print(f"       └─ Render gradient adjusted to avoid conflict")
                    else:
                        # Use original gradients (no projection needed)
                        dLdF_render_final = dLdF_render
                        dLdx_render_final = dLdx_render
                        pcgrad_applied = False
                        pcgrad_info = {}

                        if not use_pcgrad:
                            print(f"    ⚠️  PCGrad disabled in config")

                    # ══════════════════════════════════════════════════════════
                    # [FIX] Inject render-only gradient (not combined)
                    #
                    # ROOT CAUSE: C++ ComputeBackwardPass already computes physics
                    # gradients from EndLayerMassLoss. Sending combined (phys+render)
                    # caused physics to be double-counted (~1.7x).
                    #
                    # CORRECT design:
                    #   C++ final layer = physics_grad (from its own backward)
                    #                   + render_inject (from Python)
                    #   render_inject = render_direction × (w_render_base × physics_mag)
                    # ══════════════════════════════════════════════════════════

                    # [DIAGNOSTIC ONLY] normalize_and_combine for logging
                    magnitude_strategy = rs_full.get('magnitude_strategy', None)
                    if magnitude_strategy is None and 'upsample' in rs_full:
                        magnitude_strategy = rs_full['upsample'].get('magnitude_strategy', 'physics')
                    if magnitude_strategy is None:
                        magnitude_strategy = 'physics'
                    dLdF_combined, dLdx_combined, norm_info = normalize_and_combine_gradients(
                        dLdF_physics=dLdF_phys,
                        dLdx_physics=dLdx_phys,
                        dLdF_render=dLdF_render_final,
                        dLdx_render=dLdx_render_final,
                        w_physics=w_physics,
                        w_render=w_render,
                        magnitude_strategy=magnitude_strategy
                    )
                    norm_info['pcgrad_enabled'] = use_pcgrad
                    norm_info['pcgrad_applied'] = pcgrad_applied
                    norm_info['pcgrad_cosine'] = cosine
                    if pcgrad_applied:
                        norm_info.update(pcgrad_info)

                    # Scale render-only gradient to w_render_base × physics_magnitude
                    # ──────────────────────────────────────────────────────────
                    # EndLayerMassLoss depends on x (positions) but NOT directly
                    # on F (deformation gradients) at the final layer.
                    # → g_F_p = 0 structurally; use cross-space normalization for F:
                    #   dLdF_inject = render_F_dir × g_x_p × (g_F_r/g_x_r) × w_render_base
                    # This preserves the F:x ratio from the render gradient while
                    # anchoring the overall scale to the physics x magnitude.
                    # ──────────────────────────────────────────────────────────
                    _eps = 1e-12
                    g_F_r = np.linalg.norm(dLdF_render_final)
                    g_x_r = np.linalg.norm(dLdx_render_final)
                    g_F_p = phys_stats['grad_F_norm']
                    g_x_p = phys_stats['grad_x_norm']

                    # x inject: scale render x to w_render_base × physics_x_mag
                    if g_x_r > _eps:
                        dLdx_inject = (dLdx_render_final / g_x_r) * g_x_p * w_render_base
                    else:
                        dLdx_inject = np.zeros_like(dLdx_render_final)

                    # F inject: scale by physics_F_mag if available,
                    # else use cross-space normalization via physics_x_mag
                    if g_F_r > _eps:
                        if g_F_p > _eps:
                            # Physics has F grads: direct scaling
                            dLdF_inject = (dLdF_render_final / g_F_r) * g_F_p * w_render_base
                        elif g_x_p > _eps and g_x_r > _eps:
                            # Physics has no F grads (structural: EndLayerMassLoss → x only)
                            # Preserve render F:x ratio, anchor to physics x magnitude
                            Fx_ratio = g_F_r / g_x_r
                            dLdF_inject = (dLdF_render_final / g_F_r) * g_x_p * Fx_ratio * w_render_base
                        else:
                            dLdF_inject = np.zeros_like(dLdF_render_final)
                    else:
                        dLdF_inject = np.zeros_like(dLdF_render_final)

                    render_grads['dLdF'] = dLdF_inject
                    render_grads['dLdx'] = dLdx_inject

                    # ══════════════════════════════════════════════════════════
                    # Diagnostic Logging
                    # ══════════════════════════════════════════════════════════
                    print(f"\n[Gradient Injection Summary] Pass {pass_idx+1}")
                    print(f"├─ Physics grad norms:  ||F||={g_F_p:.3e}, ||x||={g_x_p:.3e}")
                    print(f"├─ Render grad norms:   ||F||={g_F_r:.3e}, ||x||={g_x_r:.3e}")
                    print(f"├─ Ratio (render/phys): {g_render / (g_phys + _eps):.3e} (raw)")
                    print(f"├─ Cosine similarity:   {cosine:+.4f} {conflict_status}")
                    print(f"├─ PCGrad applied:      {'YES' if pcgrad_applied else 'NO'}")
                    print(f"├─ w_render_base:       {w_render_base:.2f} (target render/physics ratio)")
                    _Fx_ratio = g_F_r / (g_x_r + _eps)
                    _F_target = (g_F_p if g_F_p > _eps else g_x_p * _Fx_ratio) * w_render_base
                    _F_mode = "direct" if g_F_p > _eps else "cross-space(via x)"
                    print(f"├─ Injecting render-only (not combined) — C++ adds physics separately")
                    print(f"│  ├─ F mode:  {_F_mode}  (phys_F={'0' if g_F_p < _eps else f'{g_F_p:.2e}'})")
                    print(f"│  ├─ ||dLdF_inject|| = {np.linalg.norm(dLdF_inject):.3e}  (target {_F_target:.3e})")
                    print(f"│  └─ ||dLdx_inject|| = {np.linalg.norm(dLdx_inject):.3e}  (target {g_x_p * w_render_base:.3e} = {w_render_base:.2f}×phys_x)")
                    print(f"└─ Injection complete")

                except Exception as e:
                    print(f"├─ [Gradient Combination] Error: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"├─ [Fallback] Using render gradients only")
                    # Keep render_grads as-is (fallback)
                
                accumulated_render_grads = render_grads
                print(f"└─ ✅ Render grads processed and saved for Pass {pass_idx+2}\n")
            else:
                if renderer is None or loss_manager is None:
                    print(f"├─ ⚠️  Physics-only mode - no render grads")

            del F, x
            ema_state = ema_state_new
        elif result is not None:
            print(f"└─ ⚠️ compute_render_loss_pass returned None\n")
        
        del result
        torch.cuda.empty_cache()
        
        # ────────────────────────────────────────────────────────────────────────
        # Phase 4: Visualization (last pass only)
        # ────────────────────────────────────────────────────────────────────────
        if pass_idx == num_passes - 1:
            print(f"\n[Visualization] Saving final results...")
            seed = 9999 + ep*1000 + pass_idx

            # Prepare render losses for saving
            render_losses = final_loss_components if final_loss_components is not None else None

            visualize_episode(
                ep, out_dir, cg, num_timesteps, rs_full, ema_state,
                renderer, campos, render_cfg, particle_color,
                png_enabled, tgt, loss_physics, seed, cov_module,
                external_levelset=external_levelset,
                render_losses=render_losses
            )
    
    # ════════════════════════════════════════════════════════════════════════════
    # Episode Finalization
    # ════════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"Episode {ep} COMPLETE")
    print(f"{'='*70}\n")
    
    print(f"[Cleanup] Final memory cleanup...")
    accumulated_render_grads = None
    
    if cg.has_render_gradients():
        cg.clear_render_gradients()
    
    torch.cuda.empty_cache()
    
    # Prepare final losses
    final_losses = {
        'loss_physics': loss_physics if 'loss_physics' in locals() else 0.0
    }
    
    if final_loss_components is not None:
        final_losses.update(final_loss_components)
    
    return ema_state, final_losses


__all__ = [
    'run_e2e_episode',
    'run_e2e_episode_session',  # 🔥 NEW: Session mode (10-15x faster!)
]
