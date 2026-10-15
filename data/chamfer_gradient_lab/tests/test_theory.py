"""Tests of the scientific claims, including counterexamples to the old proofs."""
import pytest
import torch
from experiment import chamfer, density_chamfer, distances, operators, sliced_w2

torch.set_num_threads(2)
DT = torch.float64


def test_bidirectional_gradient_hessian_and_finite_difference():
    x = torch.tensor([[.1],[.8],[2.3]],dtype=DT,requires_grad=True)
    y = torch.tensor([[0.],[1.],[1.4],[3.]],dtype=DT)
    d = distances(x,y); f,r = d.argmin(1),d.argmin(0)
    gradient = 2*(x-y[f])/len(x) + torch.zeros_like(x).index_add(0,r,2*(x[r]-y)/len(y))
    assert torch.allclose(torch.autograd.grad(chamfer(x,y),x)[0],gradient)
    hessian = torch.autograd.functional.hessian(lambda z:chamfer(z,y),x).reshape(3,3)
    assert torch.allclose(hessian,torch.diag(2/len(x)+2*torch.bincount(r,minlength=3).to(DT)/len(y)))
    assert torch.autograd.gradcheck(lambda z:chamfer(z,y),(x,))


def test_reverse_reactivates_a_previously_collapsed_point():
    x = torch.zeros(3,1,dtype=DT,requires_grad=True)
    y = torch.tensor([[0.],[.1],[1.],[2.]],dtype=DT)
    g = torch.autograd.grad(chamfer(x,y),x)[0]
    assert g[1] == 0
    x = (x-.5*g).detach().requires_grad_(True)
    assert torch.autograd.grad(chamfer(x,y),x)[0][1].abs() > 0


def test_tie_rule_changes_selected_reverse_gradient():
    x = torch.zeros(3,1,dtype=DT,requires_grad=True); y = torch.tensor([[1.]],dtype=DT)
    d = distances(x,y)
    first = torch.autograd.grad(d.min(0).values.mean(),x,retain_graph=True)[0]
    shared = torch.autograd.grad(d.amin(0).mean(),x)[0]
    assert int((first.abs()>0).sum()) == 1
    assert int((shared.abs()>0).sum()) == 3


def test_cluster_boundary_force_does_not_cancel():
    x = torch.tensor([0.,.1,2.],dtype=DT,requires_grad=True)
    gradient = torch.autograd.grad((x[0]-x[2]).square(),x)[0]
    assert gradient.sum() == 0 and gradient[:2].sum() != 0


def test_local_repulsion_has_noncollapsed_strict_minimum_for_full_cd():
    lam=.04; b=(lam/96)**.25
    x = torch.tensor([[b],[-3*b]],dtype=DT,requires_grad=True); y=torch.zeros(1,1,dtype=DT)
    objective=lambda z:chamfer(z,y)+lam/(z[0]-z[1]).square().sum()
    assert torch.autograd.grad(objective(x),x)[0].abs().max() < 1e-12
    hessian=torch.autograd.functional.hessian(objective,x).reshape(2,2)
    assert torch.linalg.eigvalsh(hessian).min() > 0
    assert chamfer(x,y) < chamfer(torch.tensor([[.5],[-1.5]],dtype=DT),y)


def test_coupling_is_psd_and_cannot_move_a_zero_gradient():
    torch.manual_seed(7); x=torch.randn(20,3,dtype=DT)
    kernels,_,_=operators(x,rank=6)
    for matrix in kernels.values():
        assert torch.allclose(matrix,matrix.T,atol=1e-12)
        assert torch.linalg.eigvalsh(matrix).min() > -1e-12
        assert torch.equal(matrix@torch.zeros_like(x),torch.zeros_like(x))


def test_full_rank_identity_and_disconnected_propagation():
    x=torch.tensor([[0.],[.1],[10.],[10.1]],dtype=DT)
    kernels,_,_=operators(x,neighbors=1,rank=4)
    assert (kernels['spectral']-torch.eye(4,dtype=DT)).abs().max() < 1e-14
    assert kernels['sobolev'][0,1] > 0
    assert kernels['sobolev'][0,2] == 0


def test_shared_basis_updates_equal_jjt_gradient():
    torch.manual_seed(5); basis=torch.randn(7,3,dtype=DT)
    theta=torch.zeros(3,2,dtype=DT,requires_grad=True)
    points=torch.randn(7,2,dtype=DT)+basis@theta; target=torch.randn(6,2,dtype=DT)
    loss=chamfer(points,target)
    g,gt=torch.autograd.grad(loss,[points,theta])
    assert torch.allclose(basis@gt,basis@basis.T@g)


def test_sliced_transport_matches_enumerated_1d_assignments():
    import itertools
    x=torch.tensor([[.2],[1.4],[2.7]],dtype=DT,requires_grad=True)
    y=torch.tensor([[3.],[0.],[1.]],dtype=DT)
    loss=sliced_w2(x,y,torch.ones(1,1,dtype=DT))
    expected=min(float((x.detach()-y[list(p)]).square().mean()) for p in itertools.permutations(range(3)))
    assert abs(float(loss.detach())-expected) < 1e-12
    assert torch.autograd.gradcheck(lambda z:sliced_w2(z,y,torch.ones(1,1,dtype=DT)),(x,))


def test_strict_bad_local_minimum_and_exact_loss_increment():
    x=torch.tensor([[-1.],[-1.],[5/6]],dtype=DT,requires_grad=True)
    y=torch.tensor([[-1.],[.5],[1.5]],dtype=DT)
    assert abs(float(chamfer(x,y).detach())-2/9) < 1e-14
    assert torch.autograd.grad(chamfer(x,y),x)[0].norm() < 1e-14
    torch.manual_seed(20)
    for _ in range(100):
        delta=(torch.rand(3,1,dtype=DT)-.5)/6
        expected=(delta[0].square()+delta[1].square()+torch.minimum(delta[0].square(),delta[1].square()))/3+delta[2].square()
        assert torch.allclose(chamfer(x+delta,y)-chamfer(x,y),expected.squeeze(),atol=1e-14)
        assert expected > 0
    assert chamfer(y,y) == 0


def test_bad_minimum_has_an_attracting_neighborhood():
    center=torch.tensor([[-1.],[-1.],[5/6]],dtype=DT)
    y=torch.tensor([[-1.],[.5],[1.5]],dtype=DT)
    torch.manual_seed(21)
    for _ in range(10):
        x=(center+(torch.rand_like(center)-.5)/6).requires_grad_(True)
        for _ in range(80):
            g=torch.autograd.grad(chamfer(x,y),x)[0]
            x=(x-.3*g).detach().requires_grad_(True)
            assert (x-center).abs().max() < 1/12
        assert (x-center).abs().max() < 2e-9


def test_global_translation_cannot_split_but_distinct_rows_can():
    g=torch.tensor([[1.],[0.],[0.]],dtype=DT)
    translation=torch.ones(3,3,dtype=DT)/3
    assert (translation@g)[0] == (translation@g)[1]
    k=torch.tensor([[2.,1.,1.],[1.,2.,1.],[1.,1.,2.]],dtype=DT)
    assert (k@g)[0] != (k@g)[1]
    assert float((g*(k@g)).sum()) > 0


def test_unequal_sliced_transport_matches_repeated_uniform_masses():
    x=torch.tensor([[.2],[1.4]],dtype=DT,requires_grad=True)
    y=torch.tensor([[0.],[1.],[3.]],dtype=DT)
    directions=torch.ones(1,1,dtype=DT)
    expected=(x.repeat_interleave(3,0)-y.repeat_interleave(2,0)).square().mean()
    assert torch.allclose(sliced_w2(x,y,directions),expected)
    assert torch.autograd.gradcheck(lambda z:sliced_w2(z,y,directions),(x,))


def test_dcd_occupancy_formula_and_gradient():
    x=torch.tensor([[.1],[.2],[2.]],dtype=DT,requires_grad=True)
    y=torch.tensor([[0.],[2.5]],dtype=DT); alpha=2.
    forward=(1-(3/2)*torch.exp(-alpha*torch.tensor([.01,.04,.25],dtype=DT))/(torch.tensor([2.,2.,1.],dtype=DT)+1e-6)).mean()
    reverse=(1-(2/3)*torch.exp(-alpha*torch.tensor([.01,.25],dtype=DT))/(1+1e-6)).mean()
    assert torch.allclose(density_chamfer(x,y,alpha),(forward+reverse)/2)
    assert torch.autograd.gradcheck(lambda z:density_chamfer(z,y,alpha),(x,))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cpu_cuda_gradient_equivalence():
    torch.manual_seed(8); x=torch.randn(11,3,dtype=DT,requires_grad=True); y=torch.randn(13,3,dtype=DT)
    xc=x.detach().cuda().requires_grad_(True)
    cpu,cuda=chamfer(x,y),chamfer(xc,y.cuda())
    assert torch.allclose(cpu,cuda.cpu(),atol=1e-12)
    assert torch.allclose(torch.autograd.grad(cpu,x)[0],torch.autograd.grad(cuda,xc)[0].cpu(),atol=1e-12)
