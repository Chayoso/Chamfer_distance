"""Numerical verification of Propositions 1-3 and their consequences,
exactly as stated and proved in Supplementary Material Sec. 1.

These tests check the *mathematical claims themselves* (closed-form
gradient, Hessian positive-definiteness, gradient sparsity, translational
invariance / centroid-drift independence of regularizer strength) -- not
any specific numerical result from a Table, which is the one part of the
paper that is exactly, not just qualitatively, reproducible.
"""

from __future__ import annotations

import pytest
import torch

from chamfer_collapse.chamfer import forward_loss, reverse_loss
from chamfer_collapse.regularizers import (
    laplacian_smoothness_loss,
    local_volume_proxy,
    repulsion_loss,
    volume_preservation_loss,
)

torch.manual_seed(0)


def _fixed_assignment_fwd_loss(points: torch.Tensor, assigned_targets: torch.Tensor, N: int) -> torch.Tensor:
    """L_fwd restricted to a cluster whose nearest-neighbor assignment is
    fixed at ``assigned_targets`` (Eq. 3 of the supplementary material),
    with ``N`` the size of the *full* source point set the cluster sits
    inside of."""
    return ((points - assigned_targets) ** 2).sum(-1).sum() / N


# --------------------------------------------------------------------
# Proposition 1: many-to-one collapse is the unique attractor
# --------------------------------------------------------------------

def test_proposition1_gradient_matches_closed_form():
    N, k = 12, 5
    q_star = torch.tensor([0.3, -1.1, 2.0])
    cluster = (q_star + 0.5 * torch.randn(k, 3)).requires_grad_(True)
    assigned = q_star.unsqueeze(0).expand(k, 3)

    loss = _fixed_assignment_fwd_loss(cluster, assigned, N)
    (grad,) = torch.autograd.grad(loss, cluster)

    expected_grad = (2.0 / N) * (cluster.detach() - assigned)
    assert torch.allclose(grad, expected_grad, atol=1e-5)


def test_proposition1_hessian_is_positive_definite_and_matches_closed_form():
    N, k = 12, 5
    q_star = torch.tensor([0.3, -1.1, 2.0])
    cluster = q_star + 0.5 * torch.randn(k, 3)
    assigned = q_star.unsqueeze(0).expand(k, 3)
    flat0 = cluster.reshape(-1).detach()

    def loss_flat(x: torch.Tensor) -> torch.Tensor:
        return _fixed_assignment_fwd_loss(x.reshape(k, 3), assigned, N)

    hess = torch.autograd.functional.hessian(loss_flat, flat0)
    expected_hess = (2.0 / N) * torch.eye(3 * k)
    assert torch.allclose(hess, expected_hess, atol=1e-4)

    eigvals = torch.linalg.eigvalsh(hess)
    assert torch.all(eigvals > 0), "collapse equilibrium must be a stable minimum (positive-definite Hessian)"


def test_proposition1_unique_zero_is_the_collapse_state():
    # the closed-form gradient (2/N)(p_i - q*) is an invertible affine map
    # of p_i for N > 0, so its zero set is exactly {p_i = q*}; verify this
    # both at the claimed zero and away from it.
    N = 8
    q_star = torch.tensor([1.0, 2.0, -3.0])
    # gradient formula evaluated at p=q* is exactly 0, and away from q* it
    # is proportional to (p - q*) != 0.
    p_at = q_star.clone()
    p_away = q_star + torch.tensor([0.1, 0.0, 0.0])
    assert torch.allclose((2.0 / N) * (p_at - q_star), torch.zeros(3))
    assert not torch.allclose((2.0 / N) * (p_away - q_star), torch.zeros(3))


# --------------------------------------------------------------------
# Proposition 2: the reverse term cannot separate collapsed points
# --------------------------------------------------------------------

def test_proposition2_reverse_gradient_nonzero_for_at_most_one_point():
    k = 6
    q_star = torch.zeros(3)
    # exact collapse: all k source points coincide at q*
    source = q_star.unsqueeze(0).repeat(k, 1).clone().requires_grad_(True)
    # M target points near (but not at) q*, giving genuine reverse pressure
    target = q_star.unsqueeze(0) + 0.3 * torch.randn(5, 3)

    loss = reverse_loss(source, target)
    (grad,) = torch.autograd.grad(loss, source)
    nonzero = grad.norm(dim=-1) > 1e-8
    assert nonzero.sum().item() <= 1, "at most one of the k collapsed points may receive reverse gradient"


def test_proposition2_forward_gradient_vanishes_at_exact_collapse():
    k = 6
    q_star = torch.zeros(3)
    source = q_star.unsqueeze(0).repeat(k, 1).clone().requires_grad_(True)
    target = q_star.unsqueeze(0)

    loss = forward_loss(source, target)
    (grad,) = torch.autograd.grad(loss, source)
    assert torch.allclose(grad, torch.zeros_like(grad), atol=1e-6)


def test_proposition2_deadlock_kminus1_points_have_zero_total_gradient():
    """Combining Prop. 1 (forward term at equilibrium) and Prop. 2 (reverse
    term singles out at most one point): under the bidirectional objective,
    k-1 of the k collapsed points have exactly zero total gradient."""
    k = 7
    q_star = torch.zeros(3)
    source = q_star.unsqueeze(0).repeat(k, 1).clone().requires_grad_(True)
    target = q_star.unsqueeze(0) + 0.4 * torch.randn(4, 3)

    total = forward_loss(source, q_star.unsqueeze(0)) + reverse_loss(source, target)
    (grad,) = torch.autograd.grad(total, source)
    n_zero = int((grad.norm(dim=-1) < 1e-8).sum().item())
    assert n_zero >= k - 1


# --------------------------------------------------------------------
# Proposition 3 / Corollary 1: local regularizers cannot alter cluster
# centroid drift, regardless of strength
# --------------------------------------------------------------------

REGULARIZERS = {
    "repulsion": lambda pts: repulsion_loss(pts, k=5, alpha=1.0),
    "smoothness": lambda pts: laplacian_smoothness_loss(pts, k=5),
    "volume": lambda pts: volume_preservation_loss(pts, local_volume_proxy(pts, k=5).detach(), k=5),
}


@pytest.mark.parametrize("name", REGULARIZERS)
def test_proposition3_regularizer_is_translationally_invariant(name):
    torch.manual_seed(3)
    pts = torch.randn(15, 3)
    v = torch.randn(3) * 5.0
    reg = REGULARIZERS[name]
    assert torch.allclose(reg(pts), reg(pts + v), atol=1e-4)


@pytest.mark.parametrize("name", REGULARIZERS)
def test_proposition3_regularizer_gradients_sum_to_zero(name):
    """The structural claim underlying Proposition 3: for a translationally
    invariant R, d/dv R({p_i + v})|_{v=0} = sum_i dR/dp_i = 0."""
    torch.manual_seed(4)
    pts = torch.randn(15, 3, requires_grad=True)
    reg = REGULARIZERS[name]
    loss = reg(pts)
    (grad,) = torch.autograd.grad(loss, pts)
    total = grad.sum(dim=0)
    assert torch.allclose(total, torch.zeros(3), atol=1e-3)


@pytest.mark.parametrize("name", REGULARIZERS)
def test_proposition3_centroid_drift_independent_of_regularizer_strength(name):
    """Corollary 1's core empirical prediction (Table 1: regularizer
    strength changes CD but never fixes t->s coverage): the cluster
    centroid's gradient under L_fwd + lambda*R is the same for every
    lambda, because the regularizer's contribution to the centroid
    gradient is exactly zero (previous test)."""
    torch.manual_seed(5)
    k = 6
    q_star = torch.tensor([2.0, -1.0, 0.5])
    cluster0 = q_star + 0.3 * torch.randn(k, 3)
    reg = REGULARIZERS[name]

    centroid_grads = []
    for lam in (0.0, 0.01, 1.0, 100.0):
        pts = cluster0.clone().requires_grad_(True)
        fwd = _fixed_assignment_fwd_loss(pts, q_star.unsqueeze(0).expand(k, 3), N=k)
        loss = fwd + lam * reg(pts)
        (grad,) = torch.autograd.grad(loss, pts)
        centroid_grads.append(grad.mean(dim=0))

    reference = centroid_grads[0]
    for g in centroid_grads[1:]:
        assert torch.allclose(g, reference, atol=1e-3), (
            f"{name}: centroid drift changed with regularizer strength, "
            "contradicting Proposition 3 / Corollary 1"
        )

    expected_centroid_grad = (2.0 / k) * (cluster0.mean(dim=0) - q_star)
    assert torch.allclose(reference, expected_centroid_grad, atol=1e-3)
