"""Compare implicit derivatives/solves with independent explicit dense answers."""
import numpy as np
import torch
from pipeline.lifting import jacobian_products, cg, ridge_lift


def test_nonlinear_jacobian_and_ridge_against_dense_solve():
    gen = torch.Generator().manual_seed(37)
    theta = torch.randn(7,generator=gen,dtype=torch.float64,requires_grad=True)
    a = torch.randn(9,7,generator=gen,dtype=theta.dtype)

    def model(t): return (a@t).sin()+.1*(a@t).square()

    output = model(theta)
    j = torch.autograd.functional.jacobian(model,theta)
    forward,transpose = jacobian_products(output,[theta])
    d = torch.randn(7,generator=gen,dtype=theta.dtype)
    z = torch.randn(9,generator=gen,dtype=theta.dtype)
    torch.testing.assert_close(forward([d]),j@d,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(transpose(z)[0],j.T@z,rtol=1e-12,atol=1e-12)
    state = torch.get_rng_state().clone()
    delta,info = ridge_lift(forward,transpose,z,83)
    assert torch.equal(state,torch.get_rng_state()) and info['converged']
    exact = -torch.linalg.solve(j.T@j+info['damping']*torch.eye(7),j.T@z)
    torch.testing.assert_close(delta[0],exact,rtol=1e-6,atol=1e-8)
    assert float(z@forward(delta)) < 0


def test_rank_deficient_dual_residual_bounds_output_error():
    gen = torch.Generator().manual_seed(42)
    j = torch.randn(8,3,generator=gen,dtype=torch.float64)
    g = torch.randn(8,generator=gen,dtype=j.dtype)
    k = j@j.T
    a = k+.2*torch.eye(8)
    z,info = cg(lambda v:a@v,g,rtol=1e-13,maxiter=2)
    assert not info['converged']  # Deliberately stop early; still test the bound.
    exact = torch.linalg.solve(a,g)
    assert float((k@(z-exact)).norm()) <= info['residual_norm']+1e-12
    z,info = cg(lambda v:a@v,g,rtol=1e-11)
    assert info['converged']
    torch.testing.assert_close(k@z,k@exact,rtol=1e-10,atol=1e-10)


def test_ridge_spectral_filter_and_unit_direction_error_bound():
    k = torch.diag(torch.tensor([100.,1.,0.],dtype=torch.float64))
    g = torch.tensor([-.1,1.,.7],dtype=k.dtype)
    exact = -k@torch.linalg.solve(k+torch.eye(3),g)
    torch.testing.assert_close(exact,-torch.tensor([100/101,1/2,0.])*g)
    task = torch.tensor([1.,1.,0.],dtype=k.dtype)
    assert float(task@(-k@g)) > 0 and float(task@exact) < 0
    trial = exact+torch.tensor([.02,-.01,.01],dtype=k.dtype)
    error = (trial-exact).norm()
    assert float((trial/trial.norm()-exact/exact.norm()).norm()) <= 2*error/trial.norm()
    zero,info = cg(lambda v:v,torch.zeros(3,dtype=k.dtype))
    assert info['converged'] and info['iterations']==0 and not zero.any()
