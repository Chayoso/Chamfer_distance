"""Explicit rigidity matrix, independent parameter solve and noise-filter bound."""
import numpy as np
import torch
from pipeline.graph import graph_edges,laplacian
from pipeline.metric import strain_operator,parameter_metric


def test_strain_matches_dense_rigidity_matrix_and_preserves_infinitesimal_rigid_motion():
    gen=torch.Generator().manual_seed(51)
    p=torch.randn(10,3,generator=gen,dtype=torch.float64)
    inputs=p+.1*torch.randn(10,3,generator=gen,dtype=p.dtype)
    apply,info=strain_operator(inputs,p);row,col,w,_=graph_edges(inputs)
    rigidity=torch.zeros(len(row),p.numel(),dtype=p.dtype)
    for e,(i,j) in enumerate(zip(row,col)):
        direction=p[i]-p[j];direction=direction/direction.norm()
        rigidity[e,3*i:3*i+3]=w[e].sqrt()*direction
        rigidity[e,3*j:3*j+3]=-w[e].sqrt()*direction
    dense=rigidity.T@rigidity;dense*=p.numel()/dense.trace()
    v=torch.randn(10,3,generator=gen,dtype=p.dtype)
    torch.testing.assert_close(apply(v),(dense@v.flatten()).reshape_as(v),rtol=1e-12,atol=1e-12)
    omega=torch.tensor([[0.,-3.,2.],[3.,0.,-1.],[-2.,1.,0.]],dtype=p.dtype)
    rotation=p@omega.T
    assert apply(rotation+torch.ones_like(p)).norm()<1e-12 and np.isclose(info['mean_diagonal'],1)
    isotropic,_=laplacian(inputs)
    assert (rotation*isotropic(rotation)).sum()>1
    finite_rotation=p@torch.matrix_exp(.5*omega).T-p
    assert (finite_rotation*apply(finite_rotation)).sum()>1e-3


def test_parameter_metric_matches_independent_dense_solve_and_decreases_observed_loss():
    gen=torch.Generator().manual_seed(52)
    j=torch.randn(12,7,generator=gen,dtype=torch.float64)
    a=torch.randn(12,12,generator=gen,dtype=j.dtype);a=a.T@a+torch.eye(12)
    g=torch.randn(12,generator=gen,dtype=j.dtype);gamma=.2
    forward=lambda delta:j@delta[0]
    transpose=lambda v:[j.T@v]
    delta,info=parameter_metric(forward,transpose,g,lambda v:a@v,gamma)
    normal=j.T@a@j+gamma*torch.eye(7,dtype=j.dtype)
    exact=-torch.linalg.solve(normal,j.T@g)
    assert info['converged'] and float(g@forward(delta))<0
    torch.testing.assert_close(delta[0],exact,rtol=1e-8,atol=1e-10)
    np.testing.assert_allclose(float(g@forward(delta)),-float(exact@normal@exact),rtol=1e-8)


def test_nullspace_signal_and_noise_bound_for_direct_point_metric():
    s=torch.diag(torch.tensor([0.,1.,4.],dtype=torch.float64))
    c=torch.linalg.inv(torch.eye(3)+s);signal=torch.tensor([2.,0.,0.],dtype=s.dtype)
    covariance=torch.tensor([[2.,.2,0.],[.2,1.,.3],[0.,.3,3.]],dtype=s.dtype)
    torch.testing.assert_close(c@signal,signal)
    assert torch.trace(c@covariance@c.T)<torch.trace(covariance)
    h=.25
    base=(1-h)**2*signal.square().sum()+h*h*torch.trace(covariance)
    filtered=(1-h)**2*signal.square().sum()+h*h*torch.trace(c@covariance@c.T)
    assert filtered<base


def test_coincident_edges_are_reported_and_do_not_make_nan():
    p=torch.zeros(4,3,dtype=torch.float64)
    apply,info=strain_operator(p,p)
    assert info['valid_edges']==0 and info['omitted_edges']==12 and info['mean_diagonal']==0
    assert not apply(torch.ones_like(p)).any()


def test_noncommuting_positive_operators_can_increase_observed_loss():
    k=torch.tensor([[1.,2.],[2.,5.]],dtype=torch.float64)
    c=torch.diag(k.new_tensor([1.,.01]));g=k.new_tensor([-1.,1.])
    assert torch.linalg.eigvalsh(k).min()>0 and torch.linalg.eigvalsh(c).min()>0
    torch.testing.assert_close(g@(-k@c@g),g.new_tensor(.97))


def test_wrong_translation_survives_all_metric_strengths_and_cd_controller():
    from pipeline.controlled import cd
    from pipeline.trust import select_step
    p=torch.tensor([[0.,0.,0.],[3.,0.,0.]],dtype=torch.float64,requires_grad=True)
    q=p.new_tensor([[1.,0.,0.]]);truth=p.detach()-p.new_tensor([2.,0.,0.])
    loss=cd(p,q);g,=torch.autograd.grad(loss,p)
    j=p.new_tensor([[1.,0.,0.],[1.,0.,0.]])
    forward=lambda delta:delta[0]*j
    transpose=lambda v:[(j*v).sum().reshape(1)]
    for operator in (laplacian(p.detach())[0],strain_operator(p.detach(),p.detach())[0]):
        for strength in (0.,1.,1e6):
            delta,info=parameter_metric(forward,transpose,g,lambda v:v+strength*operator(v),.2)
            torch.testing.assert_close(delta[0],p.new_tensor([1/2.2]))
            assert info['converged']
            velocity=forward(delta);slope=float((g*velocity).sum())
            evaluate=lambda h:(p.detach()+h*velocity,float(cd(p.detach()+h*velocity,q)))
            h,_=select_step(evaluate,p.detach(),velocity,float(loss.detach()),slope)
            output,value=evaluate(h)
            assert h==1 and value<float(loss.detach())
            assert (output-truth).square().sum()>(p.detach()-truth).square().sum()
