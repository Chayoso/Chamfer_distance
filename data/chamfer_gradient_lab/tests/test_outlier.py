"""Explicit mixture weights, fixed-scale likelihood gradient, and mass controls."""
import numpy as np
import torch
from pipeline.posterior import correspondence_field
from pipeline.graph import propagate
from pipeline.trust import select_step


def test_outlier_weights_match_explicit_mixture_and_gradient():
    p=torch.tensor([[0.,.1],[1.,1.]],dtype=torch.float64,requires_grad=True)
    q=torch.tensor([[.3,.2],[1.1,.9],[5.,4.]],dtype=p.dtype)
    b,mass,info=correspondence_field(p,q,True,weight=.1)
    s=info['variance'];c=(2*np.pi*s)**(p.shape[1]/2)*(.1/.9)*len(p)/len(q)
    distance=(p[:,None,:]-q[None,:,:]).square().sum(2)
    kernel=np.exp(-distance.detach().numpy()/(2*s))
    prob=kernel/(kernel.sum(0)+c)
    np.testing.assert_allclose(mass.numpy(),prob.sum(1),rtol=1e-12)
    np.testing.assert_allclose(b.numpy(),prob@q.numpy()-prob.sum(1)[:,None]*p.detach().numpy(),rtol=1e-12)
    objective=-s*torch.log(torch.exp(-distance/(2*s)).sum(0)+c).sum()
    gradient,=torch.autograd.grad(objective,p)
    torch.testing.assert_close(gradient,-b,rtol=1e-12,atol=1e-12)
    assert np.isclose(info['retained_fraction'],prob.sum()/len(q))


def test_uniform_attenuation_preserves_force_direction_but_changes_graph():
    p=torch.tensor([[0.],[1.],[4.]],dtype=torch.float64)
    q=torch.tensor([[.2],[.4],[3.],[9.]],dtype=p.dtype)
    b0,a0,_=correspondence_field(p,q,True)
    bu,au,iu=correspondence_field(p,q,True,weight=.1,uniform=True)
    bo,ao,io=correspondence_field(p,q,True,weight=.1)
    t=iu['retained_fraction']
    torch.testing.assert_close(bu,t*b0,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(au,t*a0,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(au.sum(),ao.sum(),rtol=1e-12,atol=1e-12)
    assert iu['target_weight_std']<1e-15 and io['target_weight_std']>0
    u0,_=propagate(p,b0,a0,maxiter=512);uu,_=propagate(p,bu,au,maxiter=512)
    assert (u0/u0.norm()-uu/uu.norm()).norm()>1e-4


def test_single_target_outlier_does_not_change_conditional_assignment():
    p=torch.tensor([[0.],[3.]],dtype=torch.float64);q=torch.tensor([[1.]],dtype=p.dtype)
    b0,a0,_=correspondence_field(p,q,True)
    b,a,info=correspondence_field(p,q,True,weight=.1)
    torch.testing.assert_close(a/a.sum(),a0/a0.sum())
    torch.testing.assert_close(b/b.norm(),b0/b0.norm())
    truth=p-2
    assert ((p-truth)*b).sum()>0 and 0<info['retained_fraction']<1


def test_coordinate_rescaling_changes_conventional_background_weights():
    p=torch.tensor([[0.,.1],[1.,1.]],dtype=torch.float64)
    q=torch.tensor([[.3,.2],[1.1,.9],[5.,4.]],dtype=p.dtype)
    b0,a0,_=correspondence_field(p,q,True)
    bs,as_,_=correspondence_field(10*p,10*q,True)
    torch.testing.assert_close(bs,10*b0,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(as_,a0,rtol=1e-12,atol=1e-12)
    _,_,i=correspondence_field(p,q,True,weight=.1)
    _,_,j=correspondence_field(10*p,10*q,True,weight=.1)
    assert j['retained_fraction']<i['retained_fraction']/10


def test_uniform_goal_scaling_changes_controller_despite_identical_velocity():
    p=torch.zeros(1,dtype=torch.float64);v=torch.ones_like(p)
    fractions=[]
    for goal in (1.,.25):
        loss=lambda z:float((z-goal).square().sum()/2)
        evaluate=lambda h:(p+h*v,loss(p+h*v))
        h,_=select_step(evaluate,p,v,loss(p),-goal)
        fractions.append(h)
    assert fractions==[1.,.25]
