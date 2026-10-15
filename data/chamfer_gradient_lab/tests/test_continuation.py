"""Optimizer cloning, budget boundaries, and transfer interpretation checks."""
import copy
import pytest
import torch
import pipeline.continuation as continuation
from pipeline.continuation import clone_adam, transfer_error, adam_path, budget_bracket
from pipeline.controlled import cd
from pipeline.capacity import frozen_change
from pipeline.trust import select_step


class Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(2))
    def forward(self, x):
        return x@self.weight.T, None


def history(layer, optimizer):
    result = []
    for p in layer.parameters():
        h = optimizer.state[p]
        result.append((h.get('exp_avg', torch.zeros_like(p)).double().clone(),
                       h.get('exp_avg_sq', torch.zeros_like(p)).double().clone(),
                       int(h.get('step', 0))))
    return result


@pytest.mark.parametrize('warmup', [0, 4])
def test_cloned_history_reproduces_native_adam_without_aliasing(warmup):
    layer = Linear()
    x = torch.tensor([[.2, .7], [-.9, .4], [.5, -.8]])
    target = .7*x+.03
    optimizer = torch.optim.Adam(layer.parameters(), lr=.013, betas=(.81, .97))
    def step(model, opt):
        opt.zero_grad(); cd(model(x)[0], target).backward(); opt.step()
    for _ in range(warmup):
        step(layer, optimizer)
    clone, other = clone_adam(layer, history(layer, optimizer), optimizer.param_groups[0])
    assert clone.weight.data_ptr() != layer.weight.data_ptr()
    for _ in range(3):
        step(layer, optimizer); step(clone, other)
        torch.testing.assert_close(layer.weight, clone.weight, rtol=0, atol=0)
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(optimizer.state[layer.weight][key],
                                       other.state[clone.weight][key], rtol=0, atol=0)


def test_ray_error_is_scale_invariant_and_does_not_flip_direction():
    target = torch.tensor([1., 2.], dtype=torch.float64)
    velocity = torch.tensor([3., 1.], dtype=torch.float64)
    a, b = transfer_error(target, velocity), transfer_error(target, 9*velocity)
    assert a['ray_error'] == pytest.approx(b['ray_error'])
    assert a['relative_error'] != pytest.approx(b['relative_error'])
    assert transfer_error(target, -target)['ray_error'] == 1.
    assert transfer_error(target, torch.zeros_like(target))['ray_error'] == 1.
    assert transfer_error(torch.zeros_like(target), velocity)['ray_error'] is None


def test_ridge_residual_separates_null_space_and_damping():
    j = torch.diag(torch.tensor([2., .1, 0.], dtype=torch.float64))
    f = torch.tensor([1., 3., 4.], dtype=j.dtype)
    residuals = []
    for gamma in (1., .1, .001):
        output = j@j.T@torch.linalg.solve(j@j.T+gamma*torch.eye(3), f)
        expected = f[2]**2+(gamma/(4+gamma)*f[0])**2+(gamma/(.01+gamma)*f[1])**2
        torch.testing.assert_close((f-output).square().sum(), expected)
        residuals.append(float((f-output).norm()))
    assert residuals[0] > residuals[1] > residuals[2] > 4.
    f = torch.tensor([0., 1., 0.], dtype=j.dtype)
    output = j@j.T@torch.linalg.solve(j@j.T+torch.eye(3), f)
    error = transfer_error(f, output)
    assert error['relative_error'] > .99 and error['ray_error'] < 1e-14


def test_budget_selection_includes_first_overrun_and_uses_only_cd():
    rows = [dict(step=i, seconds=t, cd=v) for i, (t, v) in
            enumerate([(.2, 3.), (.4, 2.), (.6, 2.5), (.8, 1.), (1., .1)])]
    result = budget_bracket(rows, .7)
    assert result['last']['step'] == 2 and result['after']['step'] == 3
    assert result['best']['step'] == 3
    assert budget_bracket(rows, .1)['last'] is None


def test_continuation_keeps_original_model_and_brackets_budget(monkeypatch):
    layer = Linear(); original = copy.deepcopy(layer.state_dict())
    x = torch.tensor([[.2, .7], [-.9, .4], [.5, -.8]])
    target = .7*x+.03
    optimizer = torch.optim.Adam(layer.parameters(), lr=.013)
    counter = iter(i/10 for i in range(100))
    monkeypatch.setattr(continuation, 'tick', lambda tensor: next(counter))
    rows = adam_path(layer, x, target, history(layer, optimizer), optimizer.param_groups[0], .65)
    assert [r['step'] for r in rows] == [0, 1, 2, 3]
    assert rows[-2]['seconds'] <= .65 < rows[-1]['seconds']
    assert rows[-1]['cd'] < rows[0]['cd']
    assert torch.equal(layer.weight, original['weight'])
    assert all(not h for h in optimizer.state.values())


def test_zero_cd_slope_can_have_a_finite_assignment_crossing():
    theta = torch.tensor(-.5, dtype=torch.float64, requires_grad=True)
    x = torch.tensor([[0.], [2.]], dtype=theta.dtype)
    target = torch.tensor([[1.], [3.]], dtype=theta.dtype)
    start = (x+theta).detach(); velocity = torch.full_like(start, 1.5)
    value = cd(x+theta, target)
    gradient, = torch.autograd.grad(value, theta)
    assert float(gradient) == 0.
    assert float(cd(start+.1*velocity, target)) == pytest.approx(float(value))
    mass = torch.full((2,), .5, dtype=theta.dtype); force = .5*(target-start)
    q = lambda h: (start+h*velocity, float(frozen_change(start+h*velocity, start, force, mass)))
    h, _ = select_step(q, start, velocity, 0., -float((force*velocity).sum()))
    assert h == 1. and float(cd(start+h*velocity, target)) == 0.
    h, _ = select_step(lambda h: (start+h*velocity, float(cd(start+h*velocity, target))),
                       start, velocity, float(value), 0.)
    assert h == 0.
