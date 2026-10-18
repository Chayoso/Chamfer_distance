"""Explicit spectral controls for damping versus irreducible transfer error."""
import torch
from pipeline.lifting import weighted_lift
from pipeline.continuation import transfer_error


def test_weighted_path_matches_dense_and_separates_null_residual():
    a=torch.tensor([[10.,0.],[0.,1.],[0.,0.]],dtype=torch.float64)
    mass=torch.tensor([.2,2.,1.],dtype=a.dtype)
    j=a/mass.sqrt()[:,None]
    b=torch.tensor([1.,1.,.3],dtype=a.dtype)
    forward=lambda d:(j@d[0])[:,None]
    transpose=lambda v:[j.T@v[:,0]]
    force=(mass.sqrt()*b)[:,None]
    residuals=[];norms=[];rays=[]
    for ratio in (.01,.001,.0001,.00001):
        delta,info=weighted_lift(forward,transpose,force,mass,37,ratio,maxiter=512)
        exact=torch.linalg.solve(a.T@a+info['damping']*torch.eye(2,dtype=a.dtype),a.T@b)
        assert info['converged']
        torch.testing.assert_close(delta[0],exact,rtol=1e-9,atol=1e-10)
        residuals.append(float((a@delta[0]-b).norm()))
        norms.append(float(delta[0].norm()))
        rays.append(transfer_error(b,a@delta[0])['ray_error'])
    assert residuals==sorted(residuals,reverse=True) and min(residuals)>=.3
    assert norms==sorted(norms)
    assert rays==sorted(rays,reverse=True)


def test_better_raw_transfer_can_reverse_gt_descent():
    k=torch.diag(torch.tensor([100.,1.],dtype=torch.float64))
    b=torch.ones(2,dtype=k.dtype);e=torch.tensor([-1.,2.],dtype=k.dtype)
    outputs=[k@torch.linalg.solve(k+lam*torch.eye(2,dtype=k.dtype),b) for lam in (10.,.1)]
    assert (b-outputs[1]).norm()<(b-outputs[0]).norm()
    assert e@outputs[0]<0<e@outputs[1]
