"""Position-free matching: expected invariance, benefit, and ambiguity limits."""
import pytest
import torch
from pipeline.capacity import plan_field,transport_plan,frozen_change


@pytest.mark.parametrize('kind',('soft','unbalanced'))
def test_feature_only_plan_is_independent_of_spatial_pose(kind):
    gen=torch.Generator().manual_seed(90)
    p=torch.randn(5,3,generator=gen,dtype=torch.float64)
    q=torch.randn(7,3,generator=gen,dtype=p.dtype)
    cost=torch.rand(5,7,generator=gen,dtype=p.dtype)*3
    plan,info=transport_plan(cost,1.,kind);assert info['converged']
    for z in (p,20*p+p.new_tensor([17.,-9.,4.])):
        force,mass,info=plan_field(z,q,kind,penalty=cost,feature_only=True)
        assert info['converged'] and info['epsilon']==1.
        torch.testing.assert_close(mass,plan.sum(1),rtol=0,atol=0)
        torch.testing.assert_close(force,plan@q-mass[:,None]*z,rtol=0,atol=0)


def test_spatial_cost_can_reverse_correct_observed_feature_direction():
    # Observed labels identify source 1; this is not a claim about FPFH accuracy.
    p=torch.tensor([[0.,0.,0.],[3.,0.,0.]],dtype=torch.float64)
    q=p.new_tensor([[1.,0.,0.]]);penalty=p.new_tensor([[2.],[0.]])
    mixed,m,info=plan_field(p,q,'soft',penalty=penalty)
    free,r,_=plan_field(p,q,'soft',penalty=penalty,feature_only=True)
    gap=1.5/info['epsilon']
    assert float(m[1]/m[0])==pytest.approx(float(torch.exp(p.new_tensor(2-gap))))
    assert float(r[1]/r[0])==pytest.approx(float(torch.exp(p.new_tensor(2.))))
    # Common translation force has wrong/right sign for the same ideal shift -2.
    assert float(mixed.sum(0)[0])>0 and float(free.sum(0)[0])<0
    error=p-(p+p.new_tensor([-2.,0.,0.]))
    assert float((error*mixed).sum())>0 and float((error*free).sum())<0


def test_frozen_feature_objective_is_not_spatial_force_potential():
    p=torch.tensor([[0.,0.,0.],[3.,0.,0.]],dtype=torch.float64)
    q=p.new_tensor([[1.,0.,0.]]);cost=p.new_tensor([[2.],[0.]])
    plan,_=transport_plan(cost,1.,'soft')
    force,mass,_=plan_field(p,q,'soft',penalty=cost,feature_only=True)
    z=p+.1*force;other,_=transport_plan(cost,1.,'soft')
    # Fixed features give the identical transport objective at both positions.
    torch.testing.assert_close(plan,other,rtol=0,atol=0)
    assert float(frozen_change(z,p,force,mass))<0
    numeric=(frozen_change(p+1e-5*force,p,force,mass)-frozen_change(p-1e-5*force,p,force,mass))/2e-5
    torch.testing.assert_close(numeric,-force.square().sum(),rtol=1e-9,atol=1e-10)


def test_indistinguishable_features_propose_centroid_collapse():
    p=torch.tensor([[-2.,0.,0.],[2.,0.,0.]],dtype=torch.float64)
    q=p.clone();force,mass,_=plan_field(p,q,'soft',penalty=p.new_zeros(2,2),feature_only=True)
    torch.testing.assert_close(force/mass[:,None],-p,rtol=0,atol=0)
    assert float(frozen_change(p+.1*force,p,force,mass))<0
    # Starting at exact GT, any such nonzero move worsens uniform MSE.
    assert float((.1*force).square().mean())>0


def test_feature_only_requires_an_observed_cost():
    p=torch.zeros(2,3,dtype=torch.float64)
    with pytest.raises(ValueError,match='requires a feature cost'):
        plan_field(p,p,'soft',feature_only=True)
