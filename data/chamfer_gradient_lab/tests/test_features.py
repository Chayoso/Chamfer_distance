"""Feature likelihood mathematics and actual FPFH transformation checks."""
import numpy as np
import torch
from pipeline.capacity import plan_field,transport_plan,frozen_change
from pipeline.features import descriptor


def test_zero_feature_penalty_preserves_existing_geometric_fields():
    gen=torch.Generator().manual_seed(81)
    p=torch.randn(7,3,generator=gen,dtype=torch.float64);q=torch.randn(8,3,generator=gen,dtype=p.dtype)
    for kind in ('soft','unbalanced'):
        a,r,info=plan_field(p,q,kind)
        b,s,other=plan_field(p,q,kind,penalty=p.new_zeros(len(p),len(q)))
        torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(r,s,rtol=0,atol=0)
        assert info==other and info['converged']


def test_frozen_feature_envelope_and_majorizer_against_resolved_values():
    gen=torch.Generator().manual_seed(82);eps=.2
    p=.2*torch.randn(4,3,generator=gen,dtype=torch.float64);q=.2*torch.randn(5,3,generator=gen,dtype=p.dtype)
    penalty=torch.rand(4,5,generator=gen,dtype=p.dtype)
    direction=torch.randn(4,3,generator=gen,dtype=p.dtype);direction/=direction.norm()
    cost=lambda z:.5*torch.cdist(z,q).square()+eps*penalty
    def value(z,kind):
        c=cost(z);plan,info=transport_plan(c,eps,kind,tolerance=1e-12);assert info['converged']
        kl=lambda a,b:(a*(a/b).log()-a+b).sum()
        result=(plan*c).sum()+eps*kl(plan,1/(len(p)*len(q)))
        if kind=='unbalanced':result+=2*eps*(kl(plan.sum(1),1/len(p))+kl(plan.sum(0),1/len(q)))
        return result
    for kind in ('soft','unbalanced'):
        plan,info=transport_plan(cost(p),eps,kind,tolerance=1e-12);assert info['converged']
        mass=plan.sum(1);force=plan@q-mass[:,None]*p
        derivative=(value(p+1e-5*direction,kind)-value(p-1e-5*direction,kind))/2e-5
        assert abs(float(derivative+(force*direction).sum()))<1e-8
        z=p+.1*direction
        assert value(z,kind)-value(p,kind)<=frozen_change(z,p,force,mass)+1e-10


def test_soft_matching_margin_bound():
    n,m=5,7;gap=.8;eps=.2
    cost=torch.full((n,m),gap,dtype=torch.float64);correct=torch.arange(m)%n
    cost[correct,torch.arange(m)]=0
    plan,_=transport_plan(cost,eps,'soft');wrong=1-m*plan[correct,torch.arange(m)]
    ratio=(n-1)*np.exp(-gap/eps);bound=ratio/(1+ratio)
    torch.testing.assert_close(wrong,torch.full_like(wrong,bound),atol=1e-14,rtol=1e-12)


def test_observed_label_likelihood_can_resolve_partial_translation_ambiguity():
    # Synthetic observed labels distinguish the second source; not an FPFH claim.
    p=torch.tensor([[0.,0.,0.],[3.,0.,0.]],dtype=torch.float64);q=p.new_tensor([[1.,0.,0.]])
    base,_,_=plan_field(p,q,'soft')
    feature,_,_=plan_field(p,q,'soft',penalty=p.new_tensor([[100.],[0.]]))
    assert float(base.sum(0)[0])>0 and float(feature.sum(0)[0])<0


def test_actual_fpfh_permutation_and_rigid_fixture_match_identity():
    gen=np.random.default_rng(83);points=gen.normal(size=(320,3))
    points/=np.linalg.norm(points,axis=1,keepdims=True);points*=np.array([.6,.9,1.2])
    p=torch.tensor(points,dtype=torch.float64);a,az=descriptor(p,.2)
    angle=.63;c,s=np.cos(angle),np.sin(angle);rotation=p.new_tensor([[c,-s,0],[s,c,0],[0,0,1]])
    permutation=torch.tensor(gen.permutation(len(p)))
    q=3*(p@rotation.T)+p.new_tensor([.7,-.2,1.3]);b,bz=descriptor(q[permutation],.6)
    assert az==bz==0 and torch.isfinite(a).all()
    shuffled,_=descriptor(p[permutation],.2)
    torch.testing.assert_close(a[permutation],shuffled,rtol=0,atol=0)
    # Histogram bin boundaries prevent a generic numerical equivariance claim.
    distances=torch.cdist(a,b,compute_mode='donot_use_mm_for_euclid_dist')
    correct=distances[permutation,torch.arange(len(p))].clone()
    distances[permutation,torch.arange(len(p))]=float('inf')
    assert (correct<distances.min(0).values).all()
