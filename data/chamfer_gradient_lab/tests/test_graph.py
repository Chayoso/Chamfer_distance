"""Independent dense solves, missing anchors, and a false-reciprocity example."""
import numpy as np
import torch
from pipeline.graph import laplacian,propagate,reciprocal_field
from pipeline.trust import select_step


def dense_laplacian(points,k):
    x=points.numpy();distance=np.linalg.norm(x[:,None]-x[None,:],axis=2)
    np.fill_diagonal(distance,np.inf)
    indices=np.argsort(distance,axis=1)[:,:k]
    values=np.take_along_axis(distance,indices,axis=1)
    bandwidth=np.sort(values[:,-1])[(len(x)-1)//2]
    w=np.zeros((len(x),len(x)))
    for i in range(len(x)):
        for j in indices[i]:w[i,j]=np.exp(-.5*(distance[i,j]/bandwidth)**2)
    w=(w+w.T)/2
    return torch.from_numpy((np.diag(w.sum(1))-w)/w.sum(1).mean())


def test_sparse_graph_against_independent_dense_solution_and_error_identity():
    gen=torch.Generator().manual_seed(197)
    points=torch.randn(15,3,generator=gen,dtype=torch.float64)
    exact_l=dense_laplacian(points,8);apply,_=laplacian(points)
    target=torch.randn(15,3,generator=gen,dtype=points.dtype)
    torch.testing.assert_close(apply(target),exact_l@target,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(apply(torch.ones_like(target)),torch.zeros_like(target))
    mask=torch.arange(15)%3==0
    noise=.1*torch.randn(15,3,generator=gen,dtype=points.dtype)
    observed=mask[:,None]*(target+noise)
    h=torch.diag(mask.double())+exact_l+.01*torch.eye(15,dtype=points.dtype)
    actual,info=propagate(points,observed,mask)
    exact=torch.linalg.solve(h,observed)
    assert info['converged']
    torch.testing.assert_close(actual,exact,rtol=1e-7,atol=1e-8)
    error=torch.linalg.solve(h,mask[:,None]*noise-exact_l@target-.01*target)
    torch.testing.assert_close(exact-target,error,rtol=1e-11,atol=1e-11)


def test_unanchored_components_zero_and_constant_field_shrinkage():
    points=torch.cat([torch.arange(9),10000+torch.arange(9)]).double()[:,None]
    mask=torch.arange(18)<9
    anchor=mask[:,None]*torch.tensor([[2.,-3.]],dtype=points.dtype)
    field,info=propagate(points,anchor,mask)
    assert info['converged']
    torch.testing.assert_close(field[:9],anchor[:9]/1.01,rtol=1e-8,atol=1e-8)
    assert not field[9:].any()
    empty,info=propagate(points,torch.zeros_like(anchor),torch.zeros_like(mask))
    assert not empty.any() and info['iterations']==0 and info['converged']


def test_mutual_neighbors_can_move_away_from_true_complete_correspondence():
    # The observation is the true target of source1; source0's target is hidden.
    p=torch.tensor([[0.],[3.]],dtype=torch.float64)
    q=torch.tensor([[1.]],dtype=p.dtype)
    truth=torch.tensor([[-2.],[1.]],dtype=p.dtype)
    field,mask=reciprocal_field(p,q)
    assert mask.tolist()==[True,False]
    task_gradient=2*(p-truth)/len(p)
    assert float((task_gradient*field).sum())>0
    graph,info=propagate(p,field,mask)
    assert info['converged'] and (graph>0).all()
    assert float((task_gradient*graph).sum())>0
    apply,_=laplacian(p)
    assert not apply(truth-p).any()  # Even exactly constant true motion fails.
    cd=lambda z:(z-q).abs().mean()+(z-q).abs().min()
    evaluate=lambda h:(p+h*graph,float(cd(p+h*graph)))
    slope=float(-1.5*graph[0,0]+.5*graph[1,0])
    h,_=select_step(evaluate,p,graph,float(cd(p)),slope)
    assert h==1 and cd(p+graph)<cd(p)
    assert (p+graph-truth).square().mean()>(p-truth).square().mean()
    torch.testing.assert_close(cd(p+graph)-cd(p),torch.tensor(-1.015/1.0301,dtype=p.dtype))


def test_sharp_parameter_alignment_bound_and_strict_descent_margin():
    c=torch.diag(torch.tensor([7.,2.,0.],dtype=torch.float64))
    r=torch.tensor([1.,2.,3.],dtype=c.dtype)
    cr=c@r;threshold=(r@cr)/cr.norm()
    for factor in (.99,1.01):
        error=-factor*threshold*cr/cr.norm()
        alignment=r@(c@(r+error))
        bound=r@cr-cr.norm()*error.norm()
        torch.testing.assert_close(alignment,bound,rtol=1e-12,atol=1e-12)
        assert (alignment>0)==(factor<1)
