"""Responsibilities against explicit mixtures and objective derivatives."""
import numpy as np
import torch
from pipeline.posterior import correspondence_field
from pipeline.graph import propagate,laplacian


def test_soft_force_equals_frozen_scale_likelihood_gradient_and_masses():
    p=torch.tensor([[0.,.4],[.2,-.3],[2.,1.]],dtype=torch.float64,requires_grad=True)
    q=torch.tensor([[.1,.6],[.3,.9],[1.8,.5],[2.,.7]],dtype=p.dtype)
    b,mass,info=correspondence_field(p,q,True)
    distance=(p[:,None,:]-q[None,:,:]).square().sum(2)
    energy=-info['variance']*torch.logsumexp(-distance/(2*info['variance']),dim=0).sum()
    gradient,=torch.autograd.grad(energy,p)
    torch.testing.assert_close(gradient,-b,rtol=1e-12,atol=1e-12)
    raw=np.exp(-distance.detach().numpy()/(2*info['variance']))
    prob=raw/raw.sum(0)
    np.testing.assert_allclose(mass.numpy(),prob.sum(1),rtol=1e-12)
    np.testing.assert_allclose(b.numpy(),prob@q.numpy()-prob.sum(1)[:,None]*p.detach().numpy(),rtol=1e-12,atol=1e-14)
    assert info['mass_error']<1e-12


def test_hard_reverse_force_is_kmeans_gradient_with_empty_cell():
    p=torch.tensor([[0.],[4.],[100.]],dtype=torch.float64,requires_grad=True)
    q=torch.tensor([[1.],[2.1],[3.]],dtype=p.dtype)
    b,mass,info=correspondence_field(p,q,False)
    loss=.5*(p[:,None,:]-q[None,:,:]).square().sum(2).min(0).values.sum()
    gradient,=torch.autograd.grad(loss,p)
    torch.testing.assert_close(gradient,-b)
    assert mass.tolist()==[1.,2.,0.] and b[-1]==0 and info['entropy']==0


def test_weighted_graph_dense_solve_and_coincident_uniform_posterior():
    p=torch.arange(10,dtype=torch.float64)[:,None]
    q=torch.tensor([[.1],[.2],[.3],[2.],[7.]],dtype=p.dtype)
    force,mass,_=correspondence_field(p,q,True)
    apply,_=laplacian(p)
    matrix=apply(torch.eye(10,dtype=p.dtype))+torch.diag(mass)+.01*torch.eye(10,dtype=p.dtype)
    field,info=propagate(p,force,mass)
    assert info['converged']
    torch.testing.assert_close(field,torch.linalg.solve(matrix,force),rtol=1e-7,atol=1e-9)
    zero=torch.zeros(3,2,dtype=p.dtype)
    force,mass,info=correspondence_field(zero,zero[:2],True)
    assert not force.any() and np.isclose(info['entropy'],np.log(3))
    torch.testing.assert_close(mass,torch.full((3,),2/3,dtype=p.dtype))


def test_soft_hessian_contains_negative_assignment_covariance():
    p=torch.tensor([[-2.,0.],[2.,0.]],dtype=torch.float64)
    q=torch.zeros(1,2,dtype=p.dtype)
    _,mass,info=correspondence_field(p,q,True);s=info['variance']
    objective=lambda z:-s*torch.logsumexp(-(z-q).square().sum(1)/(2*s),dim=0)
    hessian=torch.autograd.functional.hessian(objective,p).reshape(4,4)
    pair_gradients=torch.tensor([[-2.,0.,0.,0.],[0.,0.,2.,0.]],dtype=p.dtype)
    mean=pair_gradients.mean(0)
    covariance=pair_gradients.T@pair_gradients/2-torch.outer(mean,mean)
    expected=torch.diag(mass.repeat_interleave(2))-covariance/s
    torch.testing.assert_close(hessian,expected,rtol=1e-12,atol=1e-12)
    assert torch.linalg.eigvalsh(hessian).min()<0


def test_soft_and_graph_fields_can_both_increase_true_translation_error():
    p=torch.tensor([[0.],[3.]],dtype=torch.float64);q=torch.tensor([[1.]],dtype=p.dtype)
    truth=p-2
    force,mass,info=correspondence_field(p,q,True)
    alpha=1/(1+np.exp(-1.5))
    assert info['variance']==1 and np.isclose(float(mass[0]),alpha)
    assert float((2*(p-truth)*force).sum()/len(p))>0
    graph,info=propagate(p,force,mass,maxiter=512)
    assert info['converged'] and (graph>0).all()
    cd=lambda z:(z-q).abs().mean()+(z-q).abs().min()
    assert cd(p+graph)<cd(p)
    assert (p+graph-truth).square().mean()>(p-truth).square().mean()


def test_recomputed_scale_field_has_nonzero_curl_without_nn_switch():
    p=torch.tensor([[1.],[2.]],dtype=torch.float64);q=torch.zeros(1,1,dtype=p.dtype)
    h=1e-5;derivatives=[]
    for column,row in [(1,0),(0,1)]:
        perturb=torch.zeros_like(p);perturb[column]=h
        plus=-correspondence_field(p+perturb,q,True)[0]
        minus=-correspondence_field(p-perturb,q,True)[0]
        derivatives.append(float((plus[row]-minus[row])/(2*h)))
    alpha=1/(1+np.exp(-1.5))
    np.testing.assert_allclose(derivatives,np.array([2.,8.])*alpha*(1-alpha),rtol=1e-8)
    assert abs(derivatives[0]-derivatives[1])>.8
