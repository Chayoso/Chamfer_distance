"""Independent kernel/inverse checks and unchanged full-system solutions."""
import pytest
import torch
from pipeline.model import official
from pipeline.head import head_preconditioner
from pipeline.lifting import jacobian_products,cg
from pipeline.native_mix import directions


def test_head_inverse_matches_dense_matrix_and_actual_ndp_jacobian():
    torch.manual_seed(90)
    layer=official()['NDPLayer'](depth=2,width=8,k0=-2,m=1,rotation_format='axis_angle').double()
    x=torch.randn(17,3,dtype=torch.float64);out=layer(x)[0];v=torch.randn_like(x)
    features=layer.mlp(layer.input(layer.posenc(x)))
    design=layer.mlp_scale*torch.cat([features,torch.ones_like(features[:,:1])],dim=1)
    damping=float(torch.linalg.eigvalsh(design.T@design)[-1].detach())*1e-4
    forward,transpose=jacobian_products(out,list(layer.parameters()))
    raw=transpose(v)
    head=[r if name.startswith('trn_branch.') else torch.zeros_like(r) for (name,_),r in zip(layer.named_parameters(),raw)]
    kernel=forward(head)
    torch.testing.assert_close(kernel,design@(design.T@v),rtol=1e-11,atol=1e-12)
    inverse=head_preconditioner(layer,x,damping);answer=inverse(v)
    matrix=design@design.T+damping*torch.eye(len(x),dtype=x.dtype)
    torch.testing.assert_close(answer,torch.linalg.solve(matrix,v),rtol=1e-9,atol=1e-9)
    assert float((v*answer).sum())>0
    torch.testing.assert_close(matrix@answer,v,rtol=1e-9,atol=1e-9)


def test_pcg_reports_true_residual_and_rejects_nonpositive_preconditioner():
    matrix=torch.diag(torch.tensor([1.,2.,8.],dtype=torch.float64));rhs=torch.tensor([1.,2.,3.],dtype=matrix.dtype)
    one,info=cg(lambda v:matrix@v,rhs,maxiter=1,preconditioner=lambda v:v)
    assert not info['converged'] and info['residual_norm']==float((rhs-matrix@one).norm())
    answer,info=cg(lambda v:matrix@v,rhs,preconditioner=lambda v:v/matrix.diag())
    assert info['converged'] and info['iterations']==1
    torch.testing.assert_close(answer,torch.linalg.solve(matrix,rhs))
    with pytest.raises(RuntimeError,match='preconditioned residual'):
        cg(lambda v:matrix@v,rhs,preconditioner=lambda v:-v)
    zero,info=cg(lambda v:matrix@v,torch.zeros_like(rhs),preconditioner=lambda v:-v)
    assert info['converged'] and not zero.any()


def test_preconditioned_native_proposal_preserves_full_ridge_solution_and_rng():
    torch.manual_seed(90)
    layer=official()['NDPLayer'](depth=2,width=8,k0=-2,m=1,rotation_format='axis_angle')
    x=torch.randn(17,3);q=x+.1;anchor=x.double()+torch.tensor([.4,-.2,.1])
    before=torch.get_rng_state();a=directions(layer,x,q,anchor,19,method='fgr_ridgemix')
    b=directions(layer,x,q,anchor,19,method='fgr_pcgmix')
    assert torch.equal(before,torch.get_rng_state())
    assert a['solvers']['fgr']['damping']==b['solvers']['fgr']['damping']
    assert all(r['converged'] and r['relative_residual']<=1e-8 for r in (a['solvers']['fgr'],b['solvers']['fgr']))
    for m,n in (('fgr_ridge','fgr_pcg'),('fgr_ridgemix','fgr_pcgmix')):
        torch.testing.assert_close(a['directions'][m]['velocity'],b['directions'][n]['velocity'],rtol=1e-5,atol=1e-7)
