"""Weighted normal equations, invariances, and a GT-risk counterexample."""
import pytest
import torch
from pipeline.lifting import ridge_lift, weighted_lift


def operators(j, shape):
    return lambda d:(j@d[0]).reshape(shape), lambda v:[j.T@v.flatten()]


def test_weighted_lift_matches_parameter_normal_equations_with_zero_mass():
    gen=torch.Generator().manual_seed(902)
    j=torch.randn(8,5,generator=gen,dtype=torch.float64)
    mass=torch.tensor([0.,.03,.4,2.],dtype=j.dtype)
    target=torch.randn(4,2,generator=gen,dtype=j.dtype);force=mass[:,None]*target
    forward,transpose=operators(j,force.shape)
    delta,info=weighted_lift(forward,transpose,force,mass,31)
    weights=mass.repeat_interleave(2);a=j.T@(weights[:,None]*j)
    expected=torch.linalg.solve(a+info['damping']*torch.eye(5),j.T@force.flatten())
    assert info['converged'];torch.testing.assert_close(delta[0],expected,rtol=2e-7,atol=1e-9)
    value=.5*(weights*(j@delta[0]).square()).sum()-(force.flatten()*(j@delta[0])).sum()
    identity=-.5*delta[0]@a@delta[0]-info['damping']*delta[0].square().sum()
    torch.testing.assert_close(value,identity,rtol=1e-7,atol=1e-10)
    assert value<0


def test_uniform_mass_preserves_normalized_force_ridge_direction():
    gen=torch.Generator().manual_seed(903);j=torch.randn(9,4,generator=gen,dtype=torch.float64)
    force=torch.randn(3,3,generator=gen,dtype=j.dtype);mass=torch.full((3,),.2,dtype=j.dtype)
    forward,transpose=operators(j,force.shape)
    a,ia=ridge_lift(forward,transpose,-force,32)
    b,ib=weighted_lift(forward,transpose,force,mass,32)
    torch.testing.assert_close(b[0]*.2,a[0],rtol=2e-7,atol=1e-9)
    assert ib['damping']==pytest.approx(.2*ia['damping'],rel=1e-12)


def test_global_mass_scaling_preserves_weighted_parameter_solution():
    gen=torch.Generator().manual_seed(904);j=torch.randn(9,4,generator=gen,dtype=torch.float64)
    mass=torch.tensor([.01,.2,.9],dtype=j.dtype)
    force=mass[:,None]*torch.randn(3,3,generator=gen,dtype=j.dtype)
    forward,transpose=operators(j,force.shape)
    a,_=weighted_lift(forward,transpose,force,mass,33)
    b,_=weighted_lift(forward,transpose,7*force,7*mass,33)
    torch.testing.assert_close(a[0],b[0],rtol=2e-7,atol=1e-9)


def test_correct_targets_can_increase_uniform_gt_risk_under_weighted_projection():
    j=torch.tensor([[1.],[2.]],dtype=torch.float64);error=torch.tensor([[3.],[-1.]],dtype=j.dtype)
    mass=torch.tensor([.01,1.],dtype=j.dtype);force=-mass[:,None]*error
    forward,transpose=operators(j,force.shape)
    delta,info=weighted_lift(forward,transpose,force,mass,34);motion=forward(delta)
    assert info['damping']==pytest.approx(.0401)
    assert float((error*motion).sum())>0
    assert float(.5*(mass[:,None]*motion.square()).sum()-(force*motion).sum())<0
    assert (error+motion).square().mean()>error.square().mean()


def test_invalid_mass_and_inconsistent_zero_rows_are_rejected():
    j=torch.eye(2,dtype=torch.float64);force=torch.ones(2,1,dtype=j.dtype)
    forward,transpose=operators(j,force.shape)
    for mass in [torch.tensor([-1.,1.]),torch.zeros(2),torch.tensor([0.,1.]),torch.ones(3)]:
        with pytest.raises(ValueError):weighted_lift(forward,transpose,force,mass,35)
