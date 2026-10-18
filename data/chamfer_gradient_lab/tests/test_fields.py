import torch
from pipeline.fields import robust_nn, rff_objective, gaussian_force, gaussian_objective, geometry_scales
from pipeline.field_steps import scale_direction, finite_trials
from pipeline.controlled import cd


def clouds():
    generator = torch.Generator().manual_seed(72)
    return [torch.randn(n, 3, generator=generator, dtype=torch.float64) for n in (7, 9)]


def test_exact_gaussian_force_matches_independent_autograd():
    p, q = clouds()
    p.requires_grad_()
    scale = .8
    kernel = lambda a, b: (-(a[:, None]-b[None, :]).square().sum(2)/(2*scale**2)).exp().mean()
    loss = kernel(p, p)+kernel(q, q)-2*kernel(p, q)
    gradient = torch.autograd.grad(loss, p)[0]
    torch.testing.assert_close(gaussian_force(p.detach(), q, scale), -gradient, atol=1e-14, rtol=1e-12)
    exact = gaussian_objective(q, scale)
    torch.testing.assert_close(exact(p), loss.detach(), atol=1e-14, rtol=1e-12)
    torch.testing.assert_close(torch.autograd.grad(exact(p), p)[0], gradient, atol=1e-14, rtol=1e-12)


def test_rff_preserves_rng_and_translation_permutation_invariance():
    p, q = clouds()
    state = torch.get_rng_state().clone()
    objective = rff_objective(q, .7, 81)
    assert torch.equal(state, torch.get_rng_state())
    torch.testing.assert_close(objective(q), torch.zeros((), dtype=q.dtype), atol=1e-30, rtol=0)
    torch.testing.assert_close(objective(p.flip(0)), objective(p), atol=1e-14, rtol=1e-12)
    shift = p.new_tensor([7., -3., 2.])
    moved = rff_objective(q+shift, .7, 81)
    torch.testing.assert_close(moved(p+shift), objective(p), atol=1e-14, rtol=1e-12)
    assert torch.autograd.gradcheck(objective, (p.requires_grad_(),))


def test_robust_gradient_continuous_and_downweights_far_residuals():
    q = torch.zeros(1, 1, dtype=torch.float64)
    magnitudes = []
    for distance in (.199999, .200001, 4.):
        p = torch.tensor([[distance]], dtype=torch.float64, requires_grad=True)
        value = robust_nn(p, q, 1.)
        magnitudes.append(float(torch.autograd.grad(value, p)[0]))
    assert abs(magnitudes[0]-magnitudes[1]) < 4e-6
    assert magnitudes[2] < .03*magnitudes[0]


def test_observed_scale_and_direction_ignore_zero_adam_speed():
    p, q = clouds()
    h, radius = geometry_scales(p, q)
    moved_h, moved_radius = geometry_scales(3*p+7, 3*q+7)
    assert abs(moved_h-3*h) < 1e-12 and abs(moved_radius-3*radius) < 1e-12
    forward = lambda delta: delta[0]
    delta, velocity = scale_direction([p], forward, len(p)**.5*h)
    assert float(velocity.norm()) > 0
    torch.testing.assert_close(velocity.norm(), p.new_tensor(len(p)**.5*h))
    zero, velocity = scale_direction([torch.zeros_like(p)], forward, 1.)
    assert not velocity.any() and not zero[0].any()


def test_finite_controller_preserves_original_and_uses_objective_only():
    class Shift(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.offset = torch.nn.Parameter(torch.zeros(1, 1, dtype=torch.float64))
        def forward(self, x):
            return x+self.offset, None
    layer = Shift()
    p = torch.tensor([[0.]], dtype=torch.float64)
    q = torch.tensor([[1.]], dtype=torch.float64)
    velocity = torch.ones_like(p)*.25
    rows = list(finite_trials(layer, p, p, q, [velocity], velocity,
                             lambda z: cd(z, q), -2*torch.ones_like(p), -2*torch.ones_like(p)))
    assert len(rows) == 9 and all(row['accepted'] for row, _ in rows)
    assert all(row['cd_change'] < 0 and row['linearization_error'] < 1e-12 for row, _ in rows)
    assert not layer.offset.any()
