"""Ground-truth checks for the differentiable MLS-MPM simulator and the
joint-optimization gradient plumbing. These are the validation gates for
treating any MPM-based table as evidence about the paper."""
import numpy as np
import pytest
import torch

from chamfer_collapse.chamfer import chamfer_components
from chamfer_collapse.mpm.joint_opt import (MorphConfig, _clamp_reverse_grad, _frame_loss_and_grad,
                                            physics_loss, w_chamfer_schedule, w_phys_schedule,
                                            w_rev_schedule)
from chamfer_collapse.mpm.sim import (MPMParams, MPMState, neo_hookean_pk1, quadratic_bspline_weights,
                                      rasterize_mass, simulate_substeps, substep)

P = MPMParams(n_grid=12, dx=1.0, dt=8.33e-3, drag=0.0)


def ball(n, radius=2.0, seed=0, center=(0, 0, 0), dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    pts = torch.randn(n, 3, generator=g, dtype=torch.float64)
    pts = pts / pts.norm(dim=1, keepdim=True) * radius * torch.rand(n, 1, generator=g, dtype=torch.float64) ** (1 / 3)
    return (pts + torch.tensor(center, dtype=torch.float64)).to(dtype)


def make_state(n=150, dtype=torch.float32, **kw):
    x = ball(n, dtype=dtype, **kw)
    return MPMState(x, volume0=(4 / 3 * np.pi * 8) / n, density=P.density, dtype=dtype)


def test_bspline_weights_partition_of_unity():
    fx = torch.rand(500, 3)
    w = quadratic_bspline_weights(fx)
    assert torch.allclose(w.sum(-1), torch.ones(500, 3), atol=1e-6)


def test_rasterized_mass_is_conserved():
    x = ball(300)
    assert rasterize_mass(x, P).sum().item() == pytest.approx(300.0, rel=1e-5)


def test_neo_hookean_pk1_matches_autograd_of_energy():
    # Ground truth: P = dPsi/dF with Psi = mu/2 (tr(F^T F) - 3) - mu log J + lam/2 log(J)^2.
    torch.manual_seed(0)
    F = (torch.eye(3) + 0.1 * torch.randn(8, 3, 3)).double().requires_grad_(True)
    mu, lam = 58333.0, 38889.0
    J = torch.linalg.det(F)
    psi = 0.5 * mu * ((F * F).sum((-1, -2)) - 3) - mu * torch.log(J) + 0.5 * lam * torch.log(J) ** 2
    (ref,) = torch.autograd.grad(psi.sum(), F)
    assert torch.allclose(neo_hookean_pk1(F.detach(), mu, lam), ref, rtol=1e-8, atol=1e-6)


def test_rest_state_stays_at_rest():
    s = make_state()
    out = simulate_substeps(s, P, 5)
    assert out.v.abs().max().item() < 1e-6
    assert torch.allclose(out.x, s.x, atol=1e-6)
    assert torch.allclose(out.F, s.F, atol=1e-6)


def test_uniform_translation_conserves_momentum_and_moves_rigidly():
    s = make_state()
    v0 = torch.tensor([0.3, -0.2, 0.1])
    ext = v0.expand_as(s.x)
    out = substep(s, P, external_velocity=ext)
    assert torch.allclose(out.v, ext, atol=1e-4)  # uniform field is reproduced exactly by the kernel
    mom = (out.mass.unsqueeze(-1) * out.v).sum(0)
    assert torch.allclose(mom, s.mass.sum() * v0, rtol=1e-4)
    assert torch.allclose(out.x - s.x, P.dt * v0.expand_as(s.x), atol=1e-5)


def test_pre_stretched_body_contracts():
    # Elastic restoring force: a body with F = 1.05 I and no velocity must start moving inward.
    s = make_state()
    s.F = 1.05 * s.F
    out = substep(s, P)
    radial = ((out.x - s.x) * (s.x - s.x.mean(0))).sum(-1)
    assert radial.mean().item() < 0


def test_drag_damps_energy():
    s = make_state()
    ext = ball(150, radius=0.3, seed=3)
    p0 = MPMParams(n_grid=12, drag=0.0)
    p1 = MPMParams(n_grid=12, drag=0.5)
    ke = lambda st: (0.5 * st.mass.unsqueeze(-1) * st.v ** 2).sum().item()
    assert ke(substep(s, p1, external_velocity=ext)) < ke(substep(s, p0, external_velocity=ext))


def test_simulation_gradient_matches_finite_differences():
    # float64 gradcheck of d(final positions)/d(control velocity) through 3 substeps.
    s = make_state(n=40, dtype=torch.float64)
    ctrl = (0.05 * torch.randn(40, 3, dtype=torch.float64)).requires_grad_(True)
    target = ball(40, seed=9, center=(0.5, 0, 0), dtype=torch.float64)

    def f(c):
        out = simulate_substeps(s, P, 3, external_velocity=c)
        return chamfer_components(out.x, target).fwd_sq.mean() + (out.x ** 2).sum() * 1e-3

    assert torch.autograd.gradcheck(f, (ctrl,), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_reverse_clamp_is_identity_for_large_kappa_and_bounds_otherwise():
    g = torch.randn(50, 3)
    g[0] *= 100
    assert torch.equal(_clamp_reverse_grad(g, 1e9), g)
    clamped = _clamp_reverse_grad(g, 3.0)
    mean_active = g.norm(dim=-1).mean()
    assert clamped.norm(dim=-1).max().item() <= 3.0 * mean_active.item() * (1 + 1e-5)
    # direction preserved
    cos = torch.nn.functional.cosine_similarity(clamped, g, dim=-1)
    assert torch.all(cos > 0.999999)


def test_schedules_follow_paper_table():
    cfg = MorphConfig(n_frames=40, chamfer_start_frame=5, chamfer_ramp_frames=10,
                      physics_decay_start=15, physics_decay_ramp=5)
    assert w_chamfer_schedule(4, cfg) == 0 and w_chamfer_schedule(15, cfg) == cfg.w_chamfer
    assert w_phys_schedule(14, cfg) == 1.0 and w_phys_schedule(20, cfg) == pytest.approx(cfg.w_phys_final)
    assert w_rev_schedule(30, cfg) == pytest.approx(cfg.w_rev_base * cfg.w_phys_final)  # coupled, Eq. 6


def _frame_setup():
    s = make_state(n=80)
    target = ball(80, seed=5, center=(1.0, 0.5, 0), radius=2.2)
    tmass = rasterize_mass(target, P)
    cfg = MorphConfig(use_chamfer=True, chamfer_start_frame=0, chamfer_ramp_frames=1,
                      physics_decay_start=0, physics_decay_ramp=1, substeps_per_frame=3,
                      reverse_clamp_kappa=1e9)
    return s, target, tmass, cfg


def test_injected_gradient_equals_direct_autograd_when_clamp_inactive():
    s, target, tmass, cfg = _frame_setup()
    t = 3  # wc = w_chamfer, w_phys = w_phys_final, wr = coupled
    ctrl = 0.02 * torch.randn(80, 3)
    _, grad, _ = _frame_loss_and_grad(s, P, cfg, ctrl, tmass, target, t)

    v = ctrl.clone().requires_grad_(True)
    end = simulate_substeps(s, P, cfg.substeps_per_frame, external_velocity=v)
    wp, wc, wr = w_phys_schedule(t, cfg), w_chamfer_schedule(t, cfg), w_rev_schedule(t, cfg)
    res = chamfer_components(end.x, target)
    loss = wp * physics_loss(rasterize_mass(end.x, P), tmass) + wc * (res.fwd_sq.mean() + wr * res.rev_sq.mean())
    (ref,) = torch.autograd.grad(loss, v)
    assert torch.allclose(grad, ref, rtol=1e-3, atol=1e-6 * ref.abs().max().item())


def test_chamfer_term_contributes_to_control_gradient():
    # Not inert: switching the Chamfer term on must change the control gradient.
    s, target, tmass, cfg = _frame_setup()
    ctrl = torch.zeros(80, 3)
    _, g_on, _ = _frame_loss_and_grad(s, P, cfg, ctrl, tmass, target, 3)
    cfg_off = MorphConfig(**{**cfg.__dict__, "use_chamfer": False})
    _, g_off, _ = _frame_loss_and_grad(s, P, cfg_off, ctrl, tmass, target, 3)
    rel = (g_on - g_off).norm() / g_off.norm().clamp_min(1e-12)
    print("relative Chamfer contribution to control gradient:", rel.item())
    assert rel.item() > 1e-2


def test_checkpointed_simulation_matches_plain_values_and_gradients():
    s = make_state(n=60)
    target = ball(60, seed=4, center=(0.5, 0, 0))
    out = {}
    for ck in (False, True):
        ctrl = (0.05 * torch.randn(60, 3, generator=torch.Generator().manual_seed(0))).requires_grad_(True)
        end = simulate_substeps(s, P, 4, external_velocity=ctrl, checkpoint=ck)
        chamfer_components(end.x, target).fwd_sq.mean().backward()
        out[ck] = (end.x.detach(), ctrl.grad.clone())
    assert torch.allclose(out[False][0], out[True][0], atol=1e-6)
    assert torch.allclose(out[False][1], out[True][1], rtol=1e-4, atol=1e-8)
