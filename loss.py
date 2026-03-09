"""
E2E Loss Manager with Silhouette Alignment & Curvature-based Target Covariance
All operations are differentiable and support autograd.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Callable, Union
import numpy as np


# ============================================================================
# Constants
# ============================================================================
DEFAULT_WEIGHTS = {
    'w_alpha': 0.3,
    'w_depth': 0.1,
    'w_photo': 0.0,
    'w_normal': 0.0,
    'w_cov_reg': 0.01,
    'w_normal_smooth': 0.0,
    'w_edge': 0.1,
    'w_cov_align': 0.05,
    'w_coverage': 0.0,  # 🔥 NEW: Coverage loss (hole penalty)
    'w_det_barrier': 0.1,  # 🔥 NEW: det(F) barrier loss (compression/sinking prevention)
}

SCHEDULE_PARAMS = {
    'linear': {'alpha_range': (0.3, 0.7), 'edge_range': (0.05, 0.20)},
    'cosine': {'alpha_range': (0.3, 0.7)},
}

# Numerical stability constants
EPS_SAFE = 1e-8
EPS_NORMALIZE = 1e-9
CLAMP_MIN_DEPTH = 0.01
CLAMP_GRAD_NORM = (-3.0, 3.0)
CLAMP_CURVATURE = (0.0, 10.0)
# 🔥 Scale clamping removed - trust curvature-based calculation!
# CLAMP_SCALE and CLAMP_NORMAL_SCALE no longer used


# ============================================================================
# Helper Functions
# ============================================================================
def _zero_tensor(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Create a gradient-safe zero tensor with explicit device and dtype.

    Args:
        device: Target device
        dtype: Target dtype (default: float32)

    Returns:
        Zero scalar tensor (no gradients)
    """
    z = torch.zeros((), device=device, dtype=dtype)
    z.requires_grad_(False)  # Explicit: no gradient tracking needed
    return z


def _safe_normalize(tensor: torch.Tensor, dim: int = -1, eps: float = EPS_NORMALIZE) -> torch.Tensor:
    """
    Safely normalize a tensor along the given dimension.
    
    Args:
        tensor: Input tensor to normalize
        dim: Dimension along which to normalize
        eps: Small epsilon for numerical stability
    
    Returns:
        Normalized tensor (differentiable)
    """
    norm = torch.norm(tensor, dim=dim, keepdim=True)
    return tensor / torch.clamp(norm, min=eps)


def _align_tensor_shapes(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Align shapes of prediction and target tensors by removing batch dimension if needed.
    
    Args:
        pred: Predicted tensor
        target: Target tensor
    
    Returns:
        Tuple of (pred, target) with aligned shapes
    """
    if pred.ndim != target.ndim:
        if pred.ndim == 3 and target.ndim == 2:
            pred = pred.squeeze(0) if pred.shape[0] == 1 else pred[0]
        elif pred.ndim == 2 and target.ndim == 3:
            target = target.squeeze(0) if target.shape[0] == 1 else target[0]
    return pred, target


def _safe_device_transfer(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Safely transfer tensor to device if needed (no-op if already on device).
    
    Args:
        tensor: Input tensor
        device: Target device
    
    Returns:
        Tensor on target device
    """
    if tensor.device != device:
        return tensor.to(device)
    return tensor


def _validate_tensor(tensor: torch.Tensor, name: str) -> bool:
    """
    Check if tensor contains NaN or Inf values.
    
    Args:
        tensor: Tensor to validate
        name: Name for logging
    
    Returns:
        True if valid, False if contains NaN/Inf
    """
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"[WARN] {name} contains NaN/Inf values")
        return False
    return True


def _convert_to_torch(array: Union[np.ndarray, torch.Tensor], device: torch.device) -> torch.Tensor:
    """
    Convert numpy array or torch tensor to torch tensor on specified device.
    
    Args:
        array: Input array (numpy or torch)
        device: Target device
    
    Returns:
        Torch tensor on target device
    """
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array.astype(np.float32)).to(device)
    elif isinstance(array, torch.Tensor):
        return array.to(device)
    else:
        raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(array)}")


# ============================================================================
# Loss Manager
# ============================================================================
class E2ELossManager:
    """
    End-to-End Training Loss Manager with comprehensive loss components.
    
    Manages multiple loss terms including:
    - Alpha (silhouette) loss
    - Depth loss
    - Photometric loss
    - Edge alignment loss
    - Covariance alignment loss
    - Covariance regularization
    
    All loss computations are differentiable and support autograd.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize loss manager with configuration.

        Args:
            config: Configuration dictionary containing loss weights and parameters
        """

        # [🔥 DEBUG]
        print("="*50)
        print("[DEBUG] E2ELossManager received these weights from YAML config:")
        print(f"  w_alpha:       {config.get('w_alpha', 'DEFAULT')}")
        print(f"  w_depth:       {config.get('w_depth', 'DEFAULT')}")
        print(f"  w_edge:        {config.get('w_edge', 'DEFAULT')}")
        print(f"  w_photo:       {config.get('w_photo', 'DEFAULT')}")
        print(f"  w_cov_align:   {config.get('w_cov_align', 'DEFAULT')}")
        print(f"  w_cov_reg:     {config.get('w_cov_reg', 'DEFAULT')}")
        print(f"  w_cov_spd:     {config.get('w_cov_spd', 'DEFAULT')}")
        print(f"  w_coverage:    {config.get('w_coverage', 'DEFAULT')}")
        print(f"  w_det_barrier: {config.get('w_det_barrier', 'DEFAULT')}")
        print(f"  w_normal_smooth: {config.get('w_normal_smooth', 'DEFAULT')}")
        print("="*50)

        self.config = config
        self.weights = self._initialize_weights(config)
        self.schedule = config.get('schedule', 'constant')
        self.total_steps = config.get('total_steps', 100)
        self.current_step = 0
    
    def _initialize_weights(self, config: Dict) -> Dict[str, float]:
        """
        Initialize loss weights from config with defaults.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            Dictionary of loss weights
        """
        weights = {}
        for key, default_val in DEFAULT_WEIGHTS.items():
            weights[key] = float(config.get(key, default_val))
        return weights
    
    def get_weights(self) -> Dict[str, float]:
        """
        Get a copy of current weights.
        
        Returns:
            Copy of current weight dictionary
        """
        return self.weights.copy()
    
    def update_weights(self, step: Optional[int] = None):
        """
        Update weights based on schedule (linear, cosine, or constant).
        
        Args:
            step: Optional step number to set current step
        """
        if step is not None:
            self.current_step = step
        
        if self.schedule == 'constant':
            return
        
        progress = min(1.0, self.current_step / self.total_steps)
        
        if self.schedule == 'linear':
            self._update_linear_schedule(progress)
        elif self.schedule == 'cosine':
            self._update_cosine_schedule(progress)
    
    def _update_linear_schedule(self, progress: float):
        """
        Apply linear schedule to weights with progressive refinement strategy.
        
        Schedule (recommended):
        - Early (0-10%):  High cov_align (0.10), low edge/coverage (spectrum alignment)
        - Mid (10-60%):   Balanced cov_align (0.05), rising edge (0.10)
        - Late (60-100%): Low cov_align (0.02), high edge (0.12) (detail refinement)
        
        Args:
            progress: Training progress [0, 1]
        """
        # 🔥 Progressive refinement strategy
        if progress < 0.1:
            # Early: Spectrum alignment
            cov_align_mult = 2.0
            edge_mult = 0.5
            coverage_mult = 0.5
        elif progress < 0.6:
            # Mid: Balanced
            cov_align_mult = 1.0
            edge_mult = 1.0
            coverage_mult = 1.0
        else:
            # Late: Detail refinement
            cov_align_mult = 0.4
            edge_mult = 1.2
            coverage_mult = 0.5

        # 🔥 FIX: Actually apply multipliers to base weights
        params = SCHEDULE_PARAMS['linear']
        alpha_min, alpha_max = params['alpha_range']
        edge_min, edge_max = params['edge_range']

        self.weights['w_alpha'] = alpha_min + (alpha_max - alpha_min) * progress
        self.weights['w_edge'] = (edge_min + (edge_max - edge_min) * progress) * edge_mult

        # Apply multipliers to other weights
        base_cov_align = self.config.get('w_cov_align', DEFAULT_WEIGHTS['w_cov_align'])
        base_coverage = self.config.get('w_coverage', DEFAULT_WEIGHTS['w_coverage'])

        self.weights['w_cov_align'] = base_cov_align * cov_align_mult
        self.weights['w_coverage'] = base_coverage * coverage_mult
    
    def _update_cosine_schedule(self, progress: float):
        """
        Apply cosine schedule to weights.
        
        Args:
            progress: Training progress in [0, 1]
        """
        import math
        params = SCHEDULE_PARAMS['cosine']
        alpha_min, alpha_max = params['alpha_range']
        
        factor = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
        self.weights['w_alpha'] = alpha_min + (alpha_max - alpha_min) * (1 - factor)
    
    def compute_render_loss(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        cov: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        view_params: Optional[Dict] = None,
        cov_target: Optional[torch.Tensor] = None,
        F: Optional[torch.Tensor] = None,  # 🔥 NEW: Deformation gradient for det(F) barrier
        surface_mask: Optional[torch.Tensor] = None,  # 🔥 NEW: (N,) boolean mask for surface particles
        opacity: Optional[torch.Tensor] = None,  # 🔥 NEW: (N,) per-Gaussian opacity for shrinkage regularization
        visibility_ratio: Optional[float] = None  # 🔥 NEW: Scale image losses when only subset visible
    ) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive render loss (DIFFERENTIABLE).

        Args:
            pred: Dictionary with keys {'image', 'alpha', 'depth'} - predicted renders
            target: Dictionary with keys {'image', 'alpha', 'depth', 'cov_target'} - ground truth
            cov: (N, 3, 3) covariance matrices
            mu: (N, 3) 3D positions for edge alignment
            view_params: Dictionary with keys {'view_T', 'W', 'H', 'tanfovx', 'tanfovy'}
            cov_target: (N, 3, 3) target covariances from curvature
            F: (N, 3, 3) deformation gradients for det(F) barrier
            surface_mask: (N,) boolean tensor indicating surface particles (optional)
            opacity: (N,) per-Gaussian opacity values for shrinkage regularization (optional)

        Returns:
            Dictionary of loss components including 'loss_render_total'
        """
        losses = {}
        device = self._get_device(pred, cov, mu)
        total = _zero_tensor(device)

        # 🔥 CRITICAL FIX: Handle surface mask for geometric losses
        # If no mask provided, warn and use all particles (old behavior)
        if surface_mask is None and mu is not None:
            print("[WARN] No surface_mask provided! Geometric losses (edge, cov_align) will be diluted by volume particles.")
            print("  → Recommend: Pass surface_mask to compute_render_loss()")
            surface_mask = torch.ones(mu.shape[0], dtype=torch.bool, device=device)

        # Log surface statistics
        if surface_mask is not None:
            num_surface = surface_mask.sum().item()
            num_total = surface_mask.shape[0]
            losses['num_surface_particles'] = num_surface
            losses['num_total_particles'] = num_total
            losses['surface_ratio'] = num_surface / num_total if num_total > 0 else 0.0

        # Compute individual loss components (all differentiable)
        vis_scale = max(float(visibility_ratio) if visibility_ratio is not None else 1.0, 1e-3)

        total += self._compute_alpha_loss(pred, target, losses, vis_scale)
        total += self._compute_depth_loss(pred, target, losses, vis_scale)
        total += self._compute_photo_loss(pred, target, losses, vis_scale)

        # 🔥 SURFACE-ONLY LOSSES: Pass surface mask to filter contributions
        total += self._compute_edge_loss(pred, target, mu, cov, view_params, losses, surface_mask)
        total += self._compute_cov_align_loss(cov, cov_target, target, losses, surface_mask)

        # VOLUME LOSSES: Apply to ALL particles (no mask)
        total += self._compute_cov_reg_loss(cov, losses)
        total += self._compute_coverage_loss(pred, target, losses, vis_scale)
        total += self._compute_cov_spd_regularization(cov, losses)
        total += self._compute_det_barrier_loss(F, losses)

        # 🔥 NEW: Opacity shrinkage (interior particles only)
        total += self._compute_opacity_shrinkage_loss(opacity, surface_mask, losses)

        # 🔥 FIXED: Do NOT apply render_loss_weight here!
        # It should be applied during gradient combination in training_loop.py
        # Applying it here causes double-weighting since gradients are extracted after backward()
        losses['loss_render_total'] = total
        losses['render_loss_weight_configured'] = self.config.get('render_loss_weight', 1.0)  # For logging
        return losses
    
    def _get_device(self, pred: Dict, cov: Optional[torch.Tensor], mu: Optional[torch.Tensor]) -> torch.device:
        """
        Get device from available tensors.
        
        Args:
            pred: Prediction dictionary
            cov: Covariance tensor
            mu: Position tensor
        
        Returns:
            Device object
        """
        if 'alpha' in pred:
            return pred['alpha'].device
        if cov is not None:
            return cov.device
        if mu is not None:
            return mu.device
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _compute_alpha_loss(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        losses: Dict[str, torch.Tensor],
        vis_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Compute alpha channel loss (DIFFERENTIABLE).

        Args:
            pred: Prediction dictionary
            target: Target dictionary
            losses: Dictionary to store loss component

        Returns:
            Weighted alpha loss
        """
        if self.weights['w_alpha'] <= 0 or 'alpha' not in pred or 'alpha' not in target:
            device = pred.get('alpha', next(iter(pred.values()))).device if pred else torch.device('cuda')
            zero = _zero_tensor(device)
            losses['loss_alpha'] = zero
            return zero
        
        pred_alpha, target_alpha = _align_tensor_shapes(pred['alpha'], target['alpha'])

        # 🔥 FIX: Use sum/area normalization instead of mean
        # This makes gradients resolution-invariant
        area = pred_alpha.numel()
        loss_alpha_unweighted = F.l1_loss(pred_alpha, target_alpha, reduction='sum') / area

        # Store weighted loss (consistent with logging)
        loss_alpha_weighted = self.weights['w_alpha'] * loss_alpha_unweighted * vis_scale
        losses['loss_alpha'] = loss_alpha_weighted
        losses['loss_alpha_unweighted'] = loss_alpha_unweighted
        return loss_alpha_weighted
    
    def _compute_coverage_loss(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        losses: Dict[str, torch.Tensor],
        vis_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Compute coverage loss to prevent holes (DIFFERENTIABLE).
        
        Penalizes low alpha values in regions where target has high alpha.
        This prevents sparse particles from creating holes in the rendered surface.
        
        Args:
            pred: Prediction dictionary with 'alpha' key
            target: Target dictionary with 'alpha' key
            losses: Dictionary to store loss component
        
        Returns:
            Weighted coverage loss
        """
        if self.weights.get('w_coverage', 0.0) <= 0 or 'alpha' not in pred or 'alpha' not in target:
            device = pred.get('alpha', next(iter(pred.values()))).device if pred else torch.device('cuda')
            zero = _zero_tensor(device)
            losses['loss_coverage'] = zero
            return zero
        
        pred_alpha, target_alpha = _align_tensor_shapes(pred['alpha'], target['alpha'])

        # 🔥 FIX: Focus on hole penalty only, remove variance penalty
        # Global variance penalty pushes alpha toward mid-values → blur!

        # Mask: Where target expects opacity (alpha > 0.5)
        target_mask = (target_alpha > 0.5).float()

        # Hole penalty: In target regions, pred_alpha should be high
        hole_penalty = target_mask * (1.0 - pred_alpha)  # High when target=1 but pred=0

        # Optional: Small leak penalty (pred opaque where target transparent)
        background_mask = (target_alpha < 0.1).float()
        leak_penalty = background_mask * pred_alpha

        # 🔥 FIX: Use sum/area normalization for resolution invariance
        area = pred_alpha.numel()
        loss_coverage_unweighted = (hole_penalty.sum() + 0.5 * leak_penalty.sum()) / area

        # Store weighted loss (consistent with logging)
        w_coverage = self.weights.get('w_coverage', 0.0)
        loss_coverage_weighted = w_coverage * loss_coverage_unweighted * vis_scale
        losses['loss_coverage'] = loss_coverage_weighted
        losses['loss_coverage_unweighted'] = loss_coverage_unweighted
        return loss_coverage_weighted
    
    def _compute_depth_loss(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        losses: Dict[str, torch.Tensor],
        vis_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Compute depth loss with validity masking (DIFFERENTIABLE).
        
        Args:
            pred: Prediction dictionary
            target: Target dictionary
            losses: Dictionary to store loss component
        
        Returns:
            Weighted depth loss
        """
        if self.weights['w_depth'] <= 0 or 'depth' not in pred or target.get('depth') is None:
            device = pred.get('depth', next(iter(pred.values()))).device if pred else torch.device('cuda')
            zero = _zero_tensor(device)
            losses['loss_depth'] = zero
            return zero
        
        pred_depth = pred['depth']
        target_depth = target['depth']
        device = pred_depth.device
        
        valid_mask = (target_depth > 0) & (pred_depth > 0)

        if valid_mask.sum() > 0:
            # 🔥 FIX: Use sum/valid_count for mask-invariant scaling
            # Loss magnitude should not depend on number of valid pixels
            valid_count = valid_mask.float().sum().clamp_min(1.0)
            loss_depth_unweighted = (pred_depth[valid_mask] - target_depth[valid_mask]).abs().sum() / valid_count

            # Store weighted loss (consistent with logging)
            loss_depth_weighted = self.weights['w_depth'] * loss_depth_unweighted * vis_scale
            losses['loss_depth'] = loss_depth_weighted
            losses['loss_depth_unweighted'] = loss_depth_unweighted
            return loss_depth_weighted
        else:
            zero = _zero_tensor(device)
            losses['loss_depth'] = zero
            return zero
    
    def _compute_photo_loss(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        losses: Dict[str, torch.Tensor],
        vis_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Compute photometric loss (DIFFERENTIABLE).
        
        Args:
            pred: Prediction dictionary
            target: Target dictionary
            losses: Dictionary to store loss component
        
        Returns:
            Weighted photometric loss
        """
        if self.weights['w_photo'] <= 0 or 'image' not in pred or 'image' not in target:
            device = pred.get('image', next(iter(pred.values()))).device if pred else torch.device('cuda')
            zero = _zero_tensor(device)
            losses['loss_photo'] = zero
            return zero
        
        # 🔥 FIX: Use sum/area normalization instead of mean
        area = pred['image'].numel()
        loss_photo_unweighted = F.l1_loss(pred['image'], target['image'], reduction='sum') / area

        # Store weighted loss (consistent with logging)
        loss_photo_weighted = self.weights['w_photo'] * loss_photo_unweighted * vis_scale
        losses['loss_photo'] = loss_photo_weighted
        losses['loss_photo_unweighted'] = loss_photo_unweighted
        return loss_photo_weighted
    
    def _compute_edge_loss(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        mu: Optional[torch.Tensor],
        cov: Optional[torch.Tensor],
        view_params: Optional[Dict],
        losses: Dict[str, torch.Tensor],
        surface_mask: Optional[torch.Tensor] = None  # 🔥 NEW: Surface mask
    ) -> torch.Tensor:
        """
        Compute silhouette edge alignment loss (DIFFERENTIABLE).

        Args:
            pred: Prediction dictionary
            target: Target dictionary
            mu: (N, 3) 3D positions
            cov: (N, 3, 3) covariances
            view_params: View parameters dictionary
            losses: Dictionary to store loss component
            surface_mask: (N,) boolean mask for surface particles

        Returns:
            Weighted edge alignment loss
        """
        if self.weights['w_edge'] <= 0 or mu is None or cov is None or view_params is None:
            device = mu.device if mu is not None else (cov.device if cov is not None else torch.device('cuda'))
            zero = _zero_tensor(device)
            losses['loss_edge'] = zero
            return zero

        try:
            alpha_target = target.get('alpha', pred.get('alpha'))

            # Get edge loss configuration
            edge_loss_mode = self.config.get('edge_loss_mode', 'detach_position')
            gradient_boost_factor = self.config.get('gradient_boost_factor', 1000.0)
            debug_edge_gradients = self.config.get('debug_edge_gradients', False)

            loss_edge_unweighted, edge_info = edge_align_loss(
                mu, cov, alpha_target,
                view_params['view_T'],
                view_params['W'],
                view_params['H'],
                view_params['tanfovx'],
                view_params['tanfovy'],
                edge_loss_mode=edge_loss_mode,
                gradient_boost_factor=gradient_boost_factor,
                debug_gradients=debug_edge_gradients,
                surface_mask=surface_mask  # 🔥 PASS MASK
            )

            # Store weighted loss (consistent with logging)
            loss_edge_weighted = self.weights['w_edge'] * loss_edge_unweighted
            losses['loss_edge'] = loss_edge_weighted
            losses['loss_edge_unweighted'] = loss_edge_unweighted
            losses.update(edge_info)
            return loss_edge_weighted
        except Exception as e:
            print(f"[WARN] Edge alignment failed: {e}")
            device = mu.device
            zero = _zero_tensor(device)
            losses['loss_edge'] = zero
            return zero
    
    def _compute_cov_align_loss(
        self,
        cov: Optional[torch.Tensor],
        cov_target: Optional[torch.Tensor],
        target: Dict[str, torch.Tensor],
        losses: Dict[str, torch.Tensor],
        surface_mask: Optional[torch.Tensor] = None  # 🔥 NEW: Surface mask
    ) -> torch.Tensor:
        """
        Compute covariance spectral alignment loss (DIFFERENTIABLE).

        Args:
            cov: (N, 3, 3) predicted covariances
            cov_target: (N, 3, 3) target covariances
            target: Target dictionary (fallback for cov_target)
            losses: Dictionary to store loss component
            surface_mask: (N,) boolean mask for surface particles

        Returns:
            Weighted covariance alignment loss
        """
        if self.weights['w_cov_align'] <= 0 or cov is None:
            device = cov.device if cov is not None else torch.device('cuda')
            zero = _zero_tensor(device)
            losses['loss_cov_align'] = zero
            return zero

        # Get cov_target from argument or target dict
        if cov_target is None and 'cov_target' in target:
            cov_target = target['cov_target']

        # 🔥 Guard: Check if cov_target is available
        if cov_target is None:
            print("[WARN] cov_target missing → spectral loss skipped")
            print("  → Make sure target covariance Σ★ is computed in pipeline.py STAGE 6")
            device = cov.device if cov is not None else torch.device('cuda')
            zero = _zero_tensor(device)
            losses['loss_cov_align'] = zero
            return zero

        # Compute spectral alignment loss
        loss_cov_align_unweighted = covariance_spectral_loss(cov, cov_target, surface_mask=surface_mask)

        # Store weighted loss (consistent with logging)
        loss_cov_align_weighted = self.weights['w_cov_align'] * loss_cov_align_unweighted
        losses['loss_cov_align'] = loss_cov_align_weighted
        losses['loss_cov_align_unweighted'] = loss_cov_align_unweighted
        return loss_cov_align_weighted
    
    def _compute_cov_reg_loss(
        self,
        cov: Optional[torch.Tensor],
        losses: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute covariance regularization loss (DIFFERENTIABLE).
        
        Args:
            cov: (N, 3, 3) covariances
            losses: Dictionary to store loss component
        
        Returns:
            Weighted covariance regularization loss
        """
        if self.weights['w_cov_reg'] <= 0 or cov is None:
            device = cov.device if cov is not None else torch.device('cuda')
            zero = _zero_tensor(device)
            losses['loss_cov_reg'] = zero
            return zero
        
        loss_cov_reg_unweighted = self._covariance_regularization(cov)

        # Store weighted loss (consistent with logging)
        loss_cov_reg_weighted = self.weights['w_cov_reg'] * loss_cov_reg_unweighted
        losses['loss_cov_reg'] = loss_cov_reg_weighted
        losses['loss_cov_reg_unweighted'] = loss_cov_reg_unweighted
        return loss_cov_reg_weighted
    
    def _compute_cov_spd_regularization(
        self,
        cov: Optional[torch.Tensor],
        losses: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute SPD regularization to prevent covariance collapse (DIFFERENTIABLE).
        
        Args:
            cov: (N, 3, 3) covariances
            losses: Dictionary to store loss component
        
        Returns:
            Weighted SPD regularization loss
        """
        # SPD 정규화 가중치 (config에서 가져오거나 기본값)
        w_spd = self.config.get('w_cov_spd', 1e-6)
        
        if w_spd <= 0 or cov is None:
            device = cov.device if cov is not None else torch.device('cuda')
            zero = _zero_tensor(device)
            losses['loss_cov_spd'] = zero
            return zero
        
        from utils.covariance_utils import covariance_regularization_loss

        # 로그 스케일 정규화: 극단적 스케일 방지
        loss_spd_unweighted = covariance_regularization_loss(
            cov,
            lambda_scale=1.0,  # 내부 람다 (외부 w_spd로 조정)
            target_trace=None   # 목표 대각합 없음 (극단값만 페널티)
        )

        # 🔥 FIX: Store weighted loss (consistent with other losses)
        loss_spd_weighted = w_spd * loss_spd_unweighted
        losses['loss_cov_spd'] = loss_spd_weighted
        losses['loss_cov_spd_unweighted'] = loss_spd_unweighted  # For debugging

        return loss_spd_weighted
    
    def _compute_det_barrier_loss(
        self,
        F: Optional[torch.Tensor],
        losses: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute det(F) barrier loss to prevent compression/sinking (DIFFERENTIABLE).
        
        과압축 및 침몰 방지를 위한 바리어 손실.
        det(F) < 0.5에 강한 페널티를 주어 물리 에너지 감소만으로 버티지 못하게 함.
        
        Args:
            F: (N, 3, 3) deformation gradients
            losses: Dictionary to store loss component
        
        Returns:
            Weighted det(F) barrier loss
        """
        w_det = self.weights.get('w_det_barrier', 0.0)
        
        if w_det <= 0 or F is None:
            device = F.device if F is not None else torch.device('cuda')
            zero = _zero_tensor(device)
            losses['loss_det_barrier'] = zero
            return zero
        
        loss_det_unweighted = deformation_gradient_barrier_loss(F, w_det=1.0)

        # 🔥 FIX: Store weighted loss (consistent with other losses)
        loss_det_weighted = w_det * loss_det_unweighted
        losses['loss_det_barrier'] = loss_det_weighted
        losses['loss_det_barrier_unweighted'] = loss_det_unweighted  # For debugging

        return loss_det_weighted

    def _compute_opacity_shrinkage_loss(
        self,
        opacity: Optional[torch.Tensor],
        surface_mask: Optional[torch.Tensor],
        losses: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute opacity shrinkage regularization for interior particles (DIFFERENTIABLE).

        Penalize opacity of NON-surface particles to encourage them to fade away.
        This allows natural pruning of interior particles without affecting surface rendering.

        Args:
            opacity: (N,) opacity values from renderer
            surface_mask: (N,) boolean mask indicating surface particles
            losses: Dictionary to store loss component

        Returns:
            Weighted opacity shrinkage loss
        """
        w_shrink = self.weights.get('w_opacity_shrink', 0.0)

        if w_shrink <= 0 or opacity is None:
            device = opacity.device if opacity is not None else torch.device('cuda')
            zero = _zero_tensor(device)
            losses['loss_opacity_shrink'] = zero
            losses['loss_opacity_shrink_unweighted'] = 0.0
            return zero

        device = opacity.device

        # Only penalize NON-surface particles (interior particles)
        if surface_mask is not None:
            interior_mask = ~surface_mask
            num_interior = interior_mask.sum().item()

            if num_interior > 0:
                # Sum opacity of interior particles only
                loss_unweighted = opacity[interior_mask].sum()

                # Store statistics
                losses['opacity_shrink_num_interior'] = num_interior
                losses['opacity_shrink_mean_interior_opacity'] = opacity[interior_mask].mean().item()
            else:
                # No interior particles to penalize
                loss_unweighted = _zero_tensor(device)
                losses['opacity_shrink_num_interior'] = 0
                losses['opacity_shrink_mean_interior_opacity'] = 0.0
        else:
            # Fallback: penalize ALL particles (not recommended - will fight with render loss)
            loss_unweighted = opacity.sum()
            losses['opacity_shrink_num_interior'] = opacity.shape[0]
            losses['opacity_shrink_mean_interior_opacity'] = opacity.mean().item()
            print("[WARN] No surface_mask provided for opacity shrinkage! Penalizing ALL particles.")

        # Apply weight
        loss_weighted = w_shrink * loss_unweighted
        losses['loss_opacity_shrink'] = loss_weighted
        losses['loss_opacity_shrink_unweighted'] = loss_unweighted.item() if isinstance(loss_unweighted, torch.Tensor) else loss_unweighted

        return loss_weighted

    def _covariance_regularization(self, cov: torch.Tensor) -> torch.Tensor:
        """
        Regularize covariance matrices (DIFFERENTIABLE).
        
        Args:
            cov: (N, 3, 3) covariances
        
        Returns:
            Regularization loss
        """
        mode = self.config.get('cov_reg_mode', 'frobenius')
        target_scale = self.config.get('target_cov_scale', 0.02)
        device = cov.device
        
        if mode == 'frobenius':
            return self._frobenius_regularization(cov, target_scale, device)
        elif mode == 'trace':
            return self._trace_regularization(cov, target_scale)
        elif mode == 'eigenvalue':
            return self._eigenvalue_regularization(cov, target_scale)
        else:
            print(f"[WARN] Unknown cov_reg_mode '{mode}', returning zero loss")
            return _zero_tensor(device)
    
    def _frobenius_regularization(
        self, 
        cov: torch.Tensor, 
        target_scale: float,
        device: torch.device
    ) -> torch.Tensor:
        """
        Frobenius norm regularization (DIFFERENTIABLE).
        
        Args:
            cov: (N, 3, 3) covariances
            target_scale: Target scale parameter
            device: Device
        
        Returns:
            Frobenius regularization loss
        """
        target_cov = (target_scale ** 2) * torch.eye(3, device=device).unsqueeze(0).expand_as(cov)
        return F.mse_loss(cov, target_cov)
    
    def _trace_regularization(self, cov: torch.Tensor, target_scale: float) -> torch.Tensor:
        """
        Trace regularization (DIFFERENTIABLE).
        
        Args:
            cov: (N, 3, 3) covariances
            target_scale: Target scale parameter
        
        Returns:
            Trace regularization loss
        """
        trace = torch.diagonal(cov, dim1=-2, dim2=-1).sum(-1)
        target_trace = 3 * (target_scale ** 2)
        return F.l1_loss(trace, torch.full_like(trace, target_trace))
    
    def _eigenvalue_regularization(self, cov: torch.Tensor, target_scale: float) -> torch.Tensor:
        """
        Eigenvalue regularization (DIFFERENTIABLE).
        
        Args:
            cov: (N, 3, 3) covariances
            target_scale: Target scale parameter
        
        Returns:
            Eigenvalue regularization loss
        """
        eigvals = torch.linalg.eigvalsh(cov)
        target_eig = target_scale ** 2
        return F.mse_loss(eigvals, torch.full_like(eigvals, target_eig))


# ============================================================================
# Silhouette Edge Alignment Loss
# ============================================================================
def compute_projection_jacobian(
    mu: torch.Tensor,
    view_T: Union[np.ndarray, torch.Tensor],
    tanfovx: float,
    tanfovy: float,
    W: int,
    H: int
) -> torch.Tensor:
    """
    Compute projection Jacobian J = ∂(screen_xy)/∂(world_xyz) (DIFFERENTIABLE).
    
    Args:
        mu: (N, 3) world-space positions
        view_T: (4, 4) view matrix (world to camera) - numpy or torch
        tanfovx: Tangent of half horizontal FOV
        tanfovy: Tangent of half vertical FOV
        W: Image width in pixels
        H: Image height in pixels
    
    Returns:
        J: (N, 2, 3) Jacobian matrices
    """
    N = mu.shape[0]
    device = mu.device
    
    # Convert view matrix to torch with type safety
    view_T_torch = _convert_to_torch(view_T, device)
    
    # Transform to camera space (differentiable)
    mu_hom = torch.cat([mu, torch.ones(N, 1, device=device)], dim=1)  # [N, 4]
    mu_cam = (view_T_torch @ mu_hom.T).T  # [N, 4]
    
    x, y, z = mu_cam[:, 0], mu_cam[:, 1], mu_cam[:, 2]
    z_safe = torch.clamp(z, min=CLAMP_MIN_DEPTH)
    
    # Compute Jacobian components (differentiable)
    z2 = z_safe * z_safe
    
    J_screen_cam = torch.zeros(N, 2, 3, device=device)
    J_screen_cam[:, 0, 0] = W / (2 * tanfovx * z_safe)
    J_screen_cam[:, 0, 2] = -W * x / (2 * tanfovx * z2)
    J_screen_cam[:, 1, 1] = -H / (2 * tanfovy * z_safe)
    J_screen_cam[:, 1, 2] = H * y / (2 * tanfovy * z2)
    
    # Chain rule: J = J_screen_cam @ R (differentiable)
    R = view_T_torch[:3, :3]  # [3, 3]
    J = torch.bmm(J_screen_cam, R.unsqueeze(0).expand(N, -1, -1))  # [N, 2, 3]
    
    return J

def _compute_sobel_gradients(
    alpha_target: torch.Tensor,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Sobel gradients of alpha map (DIFFERENTIABLE).
    
    Args:
        alpha_target: (H, W) or (1, H, W) or (H, W, 1) alpha channel
        device: Target device
    
    Returns:
        Tuple of (grad_x, grad_y, grad_norm) each of shape (H, W)
    """
    # Ensure 2D shape
    if alpha_target.ndim == 3:
        if alpha_target.shape[0] == 1:
            alpha_target = alpha_target[0]
        elif alpha_target.shape[-1] == 1:
            alpha_target = alpha_target.squeeze(-1)
    
    alpha_target = _safe_device_transfer(alpha_target, device)
    alpha_target_expanded = alpha_target.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # Create Sobel filters
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
        dtype=torch.float32, 
        device=device
    ).view(1, 1, 3, 3)
    
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
        dtype=torch.float32, 
        device=device
    ).view(1, 1, 3, 3)
    
    # Compute gradients (differentiable)
    grad_x = F.conv2d(alpha_target_expanded, sobel_x, padding=1).squeeze()
    grad_y = F.conv2d(alpha_target_expanded, sobel_y, padding=1).squeeze()
    grad_norm = torch.sqrt(grad_x**2 + grad_y**2 + EPS_SAFE)
    
    return grad_x, grad_y, grad_norm


def _compute_silhouette_tangents(
    grad_x: torch.Tensor,
    grad_y: torch.Tensor,
    grad_norm: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute silhouette tangent vectors from gradients (DIFFERENTIABLE).
    
    Args:
        grad_x: (H, W) horizontal gradients
        grad_y: (H, W) vertical gradients
        grad_norm: (H, W) gradient magnitudes
    
    Returns:
        Tuple of (tangent_x, tangent_y) each of shape (H, W)
    """
    tangent_x = -grad_y / (grad_norm + EPS_SAFE)
    tangent_y = grad_x / (grad_norm + EPS_SAFE)
    return tangent_x, tangent_y


def _project_points_to_screen(
    mu: torch.Tensor,
    view_T: Union[np.ndarray, torch.Tensor],
    tanfovx: float,
    tanfovy: float,
    W: int,
    H: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project 3D points to screen space (DIFFERENTIABLE).
    
    Args:
        mu: (N, 3) 3D positions
        view_T: (4, 4) view matrix
        tanfovx: Tangent of half horizontal FOV
        tanfovy: Tangent of half vertical FOV
        W: Image width
        H: Image height
        device: Device
    
    Returns:
        Tuple of (screen_x, screen_y) each of shape (N,)
    """
    N = mu.shape[0]
    mu_hom = torch.cat([mu, torch.ones(N, 1, device=device)], dim=1)
    view_T_torch = _convert_to_torch(view_T, device)
    mu_cam = (view_T_torch @ mu_hom.T).T
    
    z = torch.clamp(mu_cam[:, 2], min=CLAMP_MIN_DEPTH)
    ndc_x = mu_cam[:, 0] / z
    ndc_y = mu_cam[:, 1] / z
    
    screen_x = (ndc_x / tanfovx * 0.5 + 0.5) * W
    screen_y = (-ndc_y / tanfovy * 0.5 + 0.5) * H
    
    return screen_x, screen_y

def _sample_tangents_at_points(
    tangent_x: torch.Tensor,
    tangent_y: torch.Tensor,
    grad_norm: torch.Tensor,
    screen_x: torch.Tensor,
    screen_y: torch.Tensor,
    W: int,
    H: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample tangent vectors at projected point locations (DIFFERENTIABLE).
    
    Args:
        tangent_x: (H, W) horizontal tangent components
        tangent_y: (H, W) vertical tangent components
        grad_norm: (H, W) gradient magnitudes
        screen_x: (N,) screen x-coordinates
        screen_y: (N,) screen y-coordinates
        W: Image width
        H: Image height
    
    Returns:
        Tuple of (t_hat, grad_sampled)
            - t_hat: (N, 2) sampled tangent vectors
            - grad_sampled: (N,) sampled gradient magnitudes
    """
    grid_x = 2.0 * screen_x / W - 1.0
    grid_y = 2.0 * screen_y / H - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0)
    
    tangent_x_exp = tangent_x.unsqueeze(0).unsqueeze(0)
    tangent_y_exp = tangent_y.unsqueeze(0).unsqueeze(0)
    grad_norm_exp = grad_norm.unsqueeze(0).unsqueeze(0)
    
    # 🔥 FIX: Use explicit view() instead of squeeze() for shape safety
    # grid_sample output: [1, 1, 1, N] -> view(-1) ensures [N]
    t_x_sampled = F.grid_sample(tangent_x_exp, grid, align_corners=False).view(-1)
    t_y_sampled = F.grid_sample(tangent_y_exp, grid, align_corners=False).view(-1)
    grad_sampled = F.grid_sample(grad_norm_exp, grid, align_corners=False).view(-1)
    
    t_hat = torch.stack([t_x_sampled, t_y_sampled], dim=-1)
    return t_hat, grad_sampled

def edge_align_loss(
    mu: torch.Tensor,
    cov: torch.Tensor,
    alpha_target: torch.Tensor,
    view_T: Union[np.ndarray, torch.Tensor],
    W: int,
    H: int,
    tanfovx: float,
    tanfovy: float,
    edge_loss_mode: str = 'detach_position',  # 'detach_position', 'gradient_boost', 'original'
    gradient_boost_factor: float = 1000.0,
    debug_gradients: bool = False,
    surface_mask: Optional[torch.Tensor] = None  # 🔥 NEW: (N,) boolean mask for surface particles
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute silhouette edge alignment loss (DIFFERENTIABLE).

    Aligns 2D projected covariance principal axes with silhouette edges
    detected from the alpha channel. All operations support autograd.

    Args:
        mu: (N, 3) 3D positions
        cov: (N, 3, 3) 3D covariances
        alpha_target: (H, W) target alpha channel
        view_T: (4, 4) view matrix (world to camera)
        W: Image width
        H: Image height
        tanfovx: Tangent of half horizontal FOV
        tanfovy: Tangent of half vertical FOV
        edge_loss_mode: Strategy to handle gradient paths
            - 'detach_position': Block dL/dx by detaching mu (RECOMMENDED)
            - 'gradient_boost': Boost dL/dF path by scaling cov gradient
            - 'original': Original implementation (has spurious dL/dx)
        gradient_boost_factor: Boost factor for 'gradient_boost' mode (default: 1000)
        debug_gradients: Print gradient magnitude diagnostics

    Returns:
        Tuple of (loss, info_dict)
            - loss: Scalar edge alignment loss
            - info_dict: Dictionary with alignment statistics
    """
    device = mu.device
    N = mu.shape[0]

    # [🔥 OPTION 2] Detach position to block spurious dL/dx path
    if edge_loss_mode == 'detach_position':
        mu_for_jacobian = mu.detach()
        if debug_gradients:
            print("[DEBUG EDGE] Using 'detach_position' mode - blocking dL/dx through Jacobian")
    else:
        mu_for_jacobian = mu

    # 1. Compute projection Jacobian (differentiable)
    J = compute_projection_jacobian(mu_for_jacobian, view_T, tanfovx, tanfovy, W, H)

    # 2. Project covariance to screen space (differentiable)
    cov_2d = torch.bmm(torch.bmm(J, cov), J.transpose(1, 2))  # [N, 2, 2]

    # [🔥 OPTION 1] Gradient boost: Re-balance dL/dF vs dL/dx
    if edge_loss_mode == 'gradient_boost':
        # Separate the gradient paths
        cov_2d_detached_from_J = torch.bmm(torch.bmm(J.detach(), cov), J.detach().transpose(1, 2))

        # Re-assemble with boosted cov gradient
        cov_2d = cov_2d - cov_2d_detached_from_J + (cov_2d_detached_from_J * gradient_boost_factor)

        if debug_gradients:
            print(f"[DEBUG EDGE] Using 'gradient_boost' mode - boost factor = {gradient_boost_factor}")

    # 🔥 FIX: Symmetrize cov_2d before eigendecomposition (numerical stability)
    cov_2d = 0.5 * (cov_2d + cov_2d.transpose(1, 2))

    # Add small regularization for SPD guarantee
    eps_spd = 1e-6
    eye_2d = torch.eye(2, device=device).unsqueeze(0).expand(N, -1, -1)
    cov_2d = cov_2d + eps_spd * eye_2d

    # 3. Extract principal axis via eigen-decomposition (differentiable)
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_2d)
        v_max = eigenvectors[:, :, -1]  # [N, 2] - principal axis
    except Exception as e:
        print(f"[WARN] Eigendecomposition failed: {e}")
        return _zero_tensor(device), {}

    # 4. Compute silhouette tangents from alpha gradients (differentiable)
    grad_x, grad_y, grad_norm = _compute_sobel_gradients(alpha_target, device)
    tangent_x, tangent_y = _compute_silhouette_tangents(grad_x, grad_y, grad_norm)

    # [🔥 OPTION 3] Diagnostics: Check edge strength
    if debug_gradients:
        print(f"[DEBUG EDGE] Alpha edge statistics:")
        print(f"  grad_norm mean: {grad_norm.mean().item():.6e}")
        print(f"  grad_norm max: {grad_norm.max().item():.6e}")
        print(f"  grad_norm min: {grad_norm.min().item():.6e}")
        print(f"  Percentage of pixels with strong edges (>0.1): {(grad_norm > 0.1).float().mean().item()*100:.2f}%")

    # 5. Project points to screen and sample tangents (differentiable)
    screen_x, screen_y = _project_points_to_screen(mu, view_T, tanfovx, tanfovy, W, H, device)
    t_hat, grad_sampled = _sample_tangents_at_points(
        tangent_x, tangent_y, grad_norm, screen_x, screen_y, W, H
    )
    # Detach tangent sampling to prevent another spurious dL/dx path
    t_hat = t_hat.detach()
    grad_sampled = grad_sampled.detach()

    # 6. Compute alignment score (differentiable)
    alignment = torch.abs((v_max * t_hat).sum(dim=-1))

    # 7. Compute edge weighting (differentiable)
    edge_weight = grad_sampled / (grad_sampled.mean() + EPS_SAFE)
    edge_weight = torch.clamp(edge_weight, 0.0, 10.0)

    # 🔥 CRITICAL FIX: Apply surface mask using WEIGHTED AVERAGING (not indexing!)
    # This prevents gradient dilution by volume particles while preserving gradient flow
    if surface_mask is not None:
        # Ensure surface_mask is on same device as other tensors
        surface_mask = _safe_device_transfer(surface_mask, device)
        # Convert boolean mask to float weights: 1.0 for surface, 0.0 for volume
        particle_weights = surface_mask.float()  # [N]

        # Normalize weights so they sum to 1.0 (prevents loss scaling issues)
        weight_sum = particle_weights.sum().clamp_min(1.0)
        particle_weights_norm = particle_weights / weight_sum

        # Weighted loss: Only surface particles contribute
        # CRITICAL: Use particle_weights (not particle_weights_norm) to scale loss properly
        loss_per_particle = edge_weight * (1.0 - alignment)  # [N]
        loss = (particle_weights * loss_per_particle).sum() / particle_weights.sum().clamp_min(1.0)

        # Diagnostics: Compute on surface particles only
        surface_indices = surface_mask.nonzero(as_tuple=True)[0]
        if len(surface_indices) > 0:
            alignment_surface = alignment[surface_indices]
            edge_weight_surface = edge_weight[surface_indices]
            grad_sampled_surface = grad_sampled[surface_indices]
            v_max_surface = v_max[surface_indices]

            info = {
                'edge_alignment_mean': alignment_surface.mean().item(),
                'edge_alignment_mean_all': alignment.mean().item(),  # For comparison
                'edge_weight_mean': edge_weight_surface.mean().item(),
                'edge_grad_norm_mean': grad_sampled_surface.mean().item(),
                'edge_alignment_max': alignment_surface.max().item(),
                'edge_alignment_min': alignment_surface.min().item(),
                'v_max_norm_mean': torch.norm(v_max_surface, dim=-1).mean().item(),
                'num_surface_for_edge': len(surface_indices),
            }
        else:
            # No surface particles - fallback
            info = {
                'edge_alignment_mean': 0.0,
                'edge_alignment_mean_all': alignment.mean().item(),
                'edge_weight_mean': 0.0,
                'edge_grad_norm_mean': 0.0,
                'edge_alignment_max': 0.0,
                'edge_alignment_min': 0.0,
                'v_max_norm_mean': 0.0,
                'num_surface_for_edge': 0,
            }
    else:
        # No mask provided - use all particles (old behavior)
        loss = (edge_weight * (1.0 - alignment)).mean()

        info = {
            'edge_alignment_mean': alignment.mean().item(),
            'edge_alignment_mean_all': alignment.mean().item(),
            'edge_weight_mean': edge_weight.mean().item(),
            'edge_grad_norm_mean': grad_sampled.mean().item(),
            'edge_alignment_max': alignment.max().item(),
            'edge_alignment_min': alignment.min().item(),
            'v_max_norm_mean': torch.norm(v_max, dim=-1).mean().item(),
            'num_surface_for_edge': N,
        }

    if debug_gradients:
        print(f"[DEBUG EDGE] Alignment statistics (surface only):")
        print(f"  alignment mean (surface): {info['edge_alignment_mean']:.6f}")
        print(f"  alignment mean (all): {info['edge_alignment_mean_all']:.6f}")
        print(f"  alignment max: {info['edge_alignment_max']:.6f}")
        print(f"  alignment min: {info['edge_alignment_min']:.6f}")
        print(f"  num surface particles: {info['num_surface_for_edge']}")

    return loss, info

def covariance_spectral_loss(
    cov_pred: torch.Tensor,
    cov_target: torch.Tensor,
    mode: str = 'eigenvalue',
    surface_mask: Optional[torch.Tensor] = None  # 🔥 NEW: (N,) boolean mask for surface particles
) -> torch.Tensor:
    """
    Align predicted and target covariance spectra (DIFFERENTIABLE).

    Compares eigenvalue spectra or Frobenius norm between predicted
    and target covariance matrices.

    Args:
        cov_pred: (N, 3, 3) predicted covariances
        cov_target: (N, 3, 3) target covariances
        mode: 'eigenvalue' or 'frobenius'
        surface_mask: (N,) boolean mask for surface particles (optional)

    Returns:
        loss: Scalar loss
    """
    device = cov_pred.device

    # Ensure device alignment
    cov_target = _safe_device_transfer(cov_target, device)

    # 🔥 CRITICAL: Detach target to prevent gradient flow
    # Target covariance (from curvature) should guide, not be optimized
    cov_target = cov_target.detach()

    # Validate inputs
    if not _validate_tensor(cov_target, "cov_target") or not _validate_tensor(cov_pred, "cov_pred"):
        print(f"[WARN] Skipping spectral loss due to invalid inputs")
        return _zero_tensor(device)

    # 🔥 Prefer strict masking (subset) when possible
    particle_weights = None
    num_surface = cov_pred.shape[0]
    if surface_mask is not None:
        mask_tensor = _safe_device_transfer(surface_mask, device).bool()
        if mask_tensor.shape[0] != cov_pred.shape[0]:
            print(f"[WARN] surface_mask length mismatch in spectral loss (mask={mask_tensor.shape[0]}, cov={cov_pred.shape[0]})")
        elif not mask_tensor.any():
            print(f"[WARN] No surface particles in mask! Returning zero loss.")
            return _zero_tensor(device)
        else:
            num_surface = mask_tensor.sum().item()
            if cov_target.shape[0] == num_surface:
                cov_pred = cov_pred[mask_tensor]
            else:
                # Fallback to weighted averaging if sizes differ
                particle_weights = mask_tensor.float()

    if particle_weights is not None:
        num_surface = particle_weights.sum().item()
        if num_surface < 1:
            print(f"[WARN] Surface weights sum to zero! Returning zero loss.")
            return _zero_tensor(device)

    # ✅ Handle size mismatch with global statistics instead of truncation
    use_global_statistics = (cov_pred.shape[0] != cov_target.shape[0])
    
    if use_global_statistics:
        print(f"[INFO] Size mismatch → using global eigenvalue distribution")
        print(f"  pred={cov_pred.shape[0]}, target={cov_target.shape[0]}")
    
    # Add regularization for numerical stability
    eps = 1e-6
    eye = torch.eye(3, device=device).unsqueeze(0)
    cov_pred_reg = cov_pred + eps * eye
    cov_target_reg = cov_target + eps * eye
    
    if mode == 'eigenvalue':
        try:
            # Compute eigenvalues (differentiable)
            eig_pred = torch.linalg.eigvalsh(cov_pred_reg)  # [N_pred, 3]
            eig_target = torch.linalg.eigvalsh(cov_target_reg)  # [N_target, 3]
            
            if not _validate_tensor(eig_pred, "eig_pred") or not _validate_tensor(eig_target, "eig_target"):
                print(f"[WARN] Invalid eigenvalues, using Frobenius fallback")
                return F.mse_loss(cov_pred[:min(cov_pred.shape[0], cov_target.shape[0])], 
                                 cov_target[:min(cov_pred.shape[0], cov_target.shape[0])])
            
            if use_global_statistics:
                # ✅ Compare global eigenvalue distributions (size-invariant!)
                # 🔥 CRITICAL FIX: Apply surface mask to PRED statistics only
                # (Target already filtered to surface particles during rendering)
                if particle_weights is not None:
                    # Weighted statistics for PRED (surface particles only)
                    weight_sum = particle_weights.sum().clamp_min(1.0)
                    weights_expanded = particle_weights.unsqueeze(1)  # [N_pred, 1]

                    eig_pred_mean = (weights_expanded * eig_pred).sum(dim=0) / weight_sum  # [3]
                    eig_pred_centered = eig_pred - eig_pred_mean.unsqueeze(0)
                    eig_pred_std = torch.sqrt((weights_expanded * eig_pred_centered ** 2).sum(dim=0) / weight_sum)

                    # Target: Use ALL particles (already filtered to surface by phi-mask)
                    eig_target_mean = eig_target.mean(dim=0)  # [3]
                    eig_target_std = eig_target.std(dim=0)    # [3]
                else:
                    # Compute statistics: mean, std of each eigenvalue channel
                    eig_pred_mean = eig_pred.mean(dim=0)    # [3]
                    eig_target_mean = eig_target.mean(dim=0)  # [3]
                    eig_pred_std = eig_pred.std(dim=0)      # [3]
                    eig_target_std = eig_target.std(dim=0)  # [3]

                # Loss: Match mean and std of eigenvalue distributions
                loss_mean = F.l1_loss(eig_pred_mean, eig_target_mean)
                loss_std = F.l1_loss(eig_pred_std, eig_target_std)
                loss = loss_mean + 0.5 * loss_std
            else:
                # Original: Point-wise comparison (when sizes match)
                # Normalize for scale invariance (differentiable)
                eig_pred_norm = eig_pred / (eig_pred.sum(dim=-1, keepdim=True) + EPS_NORMALIZE)
                eig_target_norm = eig_target / (eig_target.sum(dim=-1, keepdim=True) + EPS_NORMALIZE)

                # 🔥 CRITICAL FIX: Apply surface mask to point-wise loss
                if particle_weights is not None:
                    # Weighted loss: Only surface particles contribute
                    loss_per_particle = torch.abs(eig_pred_norm - eig_target_norm).sum(dim=-1)  # [N]
                    loss = (particle_weights * loss_per_particle).sum() / particle_weights.sum().clamp_min(1.0)
                else:
                    loss = F.l1_loss(eig_pred_norm, eig_target_norm)
        except Exception as e:
            print(f"[WARN] Eigenvalue computation failed: {e}, using Frobenius fallback")
            # 🔥 FIX: Handle size mismatch in fallback
            if cov_pred.shape[0] != cov_target.shape[0]:
                # Use global statistics fallback for size mismatch
                min_size = min(cov_pred.shape[0], cov_target.shape[0])
                print(f"[WARN] Size mismatch in Frobenius fallback, using first {min_size} samples")
                loss = F.mse_loss(cov_pred[:min_size], cov_target[:min_size])
            else:
                # Apply surface mask if available
                if particle_weights is not None:
                    diff_squared = ((cov_pred - cov_target) ** 2).sum(dim=(1, 2))  # [N]
                    loss = (particle_weights * diff_squared).sum() / particle_weights.sum().clamp_min(1.0)
                else:
                    loss = F.mse_loss(cov_pred, cov_target)

    elif mode == 'frobenius':
        # 🔥 FIX: Handle size mismatch and surface mask
        if cov_pred.shape[0] != cov_target.shape[0]:
            min_size = min(cov_pred.shape[0], cov_target.shape[0])
            print(f"[WARN] Size mismatch in Frobenius mode, using first {min_size} samples")
            loss = F.mse_loss(cov_pred[:min_size], cov_target[:min_size])
        else:
            # Apply surface mask if available
            if particle_weights is not None:
                diff_squared = ((cov_pred - cov_target) ** 2).sum(dim=(1, 2))  # [N]
                loss = (particle_weights * diff_squared).sum() / particle_weights.sum().clamp_min(1.0)
            else:
                loss = F.mse_loss(cov_pred, cov_target)
    
    else:
        print(f"[WARN] Unknown mode '{mode}', returning zero loss")
        loss = _zero_tensor(device)
    
    return loss


# ============================================================================
# Advanced Regularization
# ============================================================================
def deformation_gradient_barrier_loss(
    F: torch.Tensor,
    w_det: float = 0.1
) -> torch.Tensor:
    """
    det(F) 바리어 손실: 과압축 및 침몰 방지 (DIFFERENTIABLE).
    
    손실: w_d * [-log(det F) + (det F - 1)²]
    
    목적:
    - det(F) < 1: 압축 → log 바리어로 강한 페널티
    - det(F) ≈ 1: 체적 보존 목표
    - det(F) > 1: 팽창 → 제곱 페널티
    
    Args:
        F: (N, 3, 3) 변형 그래디언트
        w_det: 바리어 가중치 (기본: 0.1)
    
    Returns:
        loss: 스칼라 바리어 손실
    """
    if F is None or w_det <= 0:
        device = F.device if F is not None else torch.device('cuda')
        return _zero_tensor(device)
    
    device = F.device
    
    # det(F) 계산
    det_F = torch.det(F)  # (N,)

    # 🔥 FIX: Piecewise barrier (different treatment for compression vs expansion)
    # Compression (det < 1): -log(det) barrier + quadratic
    # Expansion (det > 1): only quadratic penalty

    compression_mask = (det_F < 1.0)

    # Compression loss: -log(det) + (det - 1)²
    det_F_safe = torch.clamp(det_F, min=EPS_SAFE)
    log_barrier = torch.where(
        compression_mask,
        -torch.log(det_F_safe),  # Strong penalty for compression
        torch.zeros_like(det_F)   # No log barrier for expansion
    )

    # Volume preservation: (det - 1)² for both compression and expansion
    vol_penalty = (det_F - 1.0) ** 2

    loss_barrier = w_det * (log_barrier + vol_penalty).mean()
    
    return loss_barrier # differentiable
