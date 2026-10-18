import pytest
import torch
from pipeline.joint import correspondence, fixed_lift, joint_path
from pipeline.controlled import cd


def test_nn_majorizer_matches_gradient_and_upper_bounds_actual_loss():
    p = torch.tensor([[0., .2], [1.3, -.4], [2.1, 1.]], dtype=torch.float64)
    q = p+torch.tensor([.23, .17], dtype=p.dtype)
    variable = p.clone().requires_grad_()
    gradient = torch.autograd.grad(cd(variable, q), variable)[0]
    force, mass, value, _ = correspondence(p, q, 'cd')(p)
    torch.testing.assert_close(force, -gradient)
    for shift in (p*.03, p*-.05, torch.ones_like(p)*.4):
        upper = value+float((.5*mass[:, None]*shift.square()-force*shift).sum())
        assert float(cd(p+shift, q)) <= upper+1e-12


def test_fixed_weighted_solve_matches_explicit_parameter_normal_equations():
    j = torch.tensor([[1., .2], [.1, .3], [2., -.4]], dtype=torch.float64)
    force = torch.tensor([[.2], [-.4], [.6]], dtype=j.dtype)
    mass = torch.tensor([.2, .7, .1], dtype=j.dtype)
    forward = lambda delta: (j@delta[0])[:, None]
    transpose = lambda v: [j.T@v[:, 0]]
    delta, info = fixed_lift(forward, transpose, force, mass, .03)
    expected = torch.linalg.solve(j.T@(mass[:, None]*j)+.03*torch.eye(2, dtype=j.dtype), j.T@force[:, 0])
    assert info['converged']; torch.testing.assert_close(delta[0], expected)


def test_joint_path_decreases_regularized_objective_without_task_labels():
    p = torch.tensor([[0., .2], [1.3, -.4], [2.1, 1.]], dtype=torch.float64)
    q = p+torch.tensor([[.23, .17], [.13, -.2], [.1, .12]], dtype=p.dtype)
    # Shared translation: changing correspondences cannot add output degrees of freedom.
    forward = lambda delta: delta[0].expand_as(p)
    transpose = lambda v: [v.sum(0, keepdim=True)]
    for kind in ('cd', 'uot'):
        rows = list(joint_path(p, q, forward, transpose, 71, kind))
        assert [r[0] for r in rows] == [1, 4, 0]
        history = rows[-1][-1]
        assert len(history) == 5
        assert len({r['solve']['damping'] for r in history[:-1]}) == 1
        assert all(r['majorizer_change'] <= 1e-10 for r in history[:-1])
        assert all(r['reassignment_change'] is None or r['reassignment_change'] <= 1e-10 for r in history[:-1])
        assert history[-1]['final_reassignment_change'] <= 1e-10


def test_unsquared_zero_distance_is_reported_not_silently_smoothed():
    p = torch.tensor([[0.], [1.]], dtype=torch.float64)
    with pytest.raises(RuntimeError, match='Zero NN distance'):
        correspondence(p, p, 'cd')(p)
