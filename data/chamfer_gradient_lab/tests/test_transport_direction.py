"""Check mass constraints, analytic scalar solution, and envelope derivative."""
import numpy as np
import torch
from scipy.optimize import linprog
from transport_direction import transport


def test_scalar_kl_transport_exact_value_and_mass():
    p=torch.tensor([[.8]],dtype=torch.float64,requires_grad=True); q=torch.zeros_like(p)
    for tau in [.1,1.]:
        value, info=transport(p,q,tau)
        expected=np.exp(-.64/(.05+2*tau))
        assert abs(info['mass']-expected)<1e-10
        assert abs(float(value)-(.05+2*tau)*(1-expected))<1e-10
        grad,=torch.autograd.grad(value,p)
        assert abs(float(grad)-1.6*expected)<1e-10


def test_transport_envelope_derivative_and_balanced_mass():
    torch.manual_seed(343)
    p=torch.randn(4,2,dtype=torch.float64,requires_grad=True)*.2
    q=torch.randn(5,2,dtype=torch.float64)*.2
    direction=torch.randn_like(p); direction/=direction.norm()
    for tau in [.1,1.,np.inf]:
        value, info=transport(p,q,tau)
        gradient,=torch.autograd.grad(value,p)
        plus,_=transport(p+1e-5*direction,q,tau); minus,_=transport(p-1e-5*direction,q,tau)
        assert abs(float((plus-minus)/2e-5)-float((gradient*direction).sum()))<2e-7
        assert info['residual']<1e-8
        if np.isinf(tau): assert abs(info['mass']-1)<1e-10


def test_partial_transport_closed_form_against_linear_program():
    x=np.array([-1.,0.,1.]); q=np.array([-1.,0.])
    capacities=np.concatenate([np.repeat(np.eye(3),2,axis=1),np.tile(np.eye(2),(1,3))])
    for theta in [-.49,-.3,-1/6,0.,.2,.49]:
        costs=((x+theta)[:,None]-q[None,:])**2
        result=linprog(costs.ravel(),A_ub=capacities,b_ub=[1/3]*3+[1/2]*2,
                       A_eq=np.ones((1,6)),b_eq=[2/3],bounds=(0,None),method='highs')
        assert result.success and abs(result.fun-(2/3)*theta**2)<1e-12
    values=[(6+2*w)**-2 for w in [0.,.1,1.,10.]]
    assert np.all(np.diff(values)<0)
