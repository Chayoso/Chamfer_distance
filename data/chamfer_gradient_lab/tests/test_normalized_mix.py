import numpy as np
import pytest
import torch
from scipy.optimize import brentq
from pipeline.direction_mix import mix_fraction, normalized_mix_fraction
from pipeline.native_mix import directions


def test_normalized_constraint_matches_independent_vector_root():
    rng=np.random.default_rng(10854);changed=0
    for _ in range(256):
        g,a,c=rng.normal(size=(3,5));a/=np.linalg.norm(a);c/=np.linalg.norm(c)
        if g@c>=0:c=-c
        sa,sc=g@a,g@c;rho=a@c;old=mix_fraction(sa,sc)
        beta=normalized_mix_fraction(sa,sc,rho)
        def residual(b):
            v=(1-b)*a+b*c
            return g@v/np.linalg.norm(v)-.5*sc
        expected=0. if old==0 else brentq(residual,0.,1.,xtol=1e-14)
        assert beta==pytest.approx(expected,abs=2e-13) and 0<=beta<=old
        assert residual(beta)<=1e-12
        if beta>1e-8:assert residual(beta-1e-8)>0
        changed+=beta<old-1e-8
    assert changed>50


def test_nonmonotone_slope_and_degenerate_rays():
    # Slope first increases, then decreases; endpoint bracketing still works.
    sa,sc,rho=3.,-1.,-.9
    assert sc-rho*sa>0
    beta=normalized_mix_fraction(sa,sc,rho)
    assert 0<beta<mix_fraction(sa,sc)
    assert normalized_mix_fraction(1.,-1.,-1.)==.75
    assert normalized_mix_fraction(1.,-1.,-1.+5e-13)==.75
    assert normalized_mix_fraction(-1.,-1.,1.)==0.
    for cosine in (float('nan'),-1.01,1.01):
        with pytest.raises(ValueError):normalized_mix_fraction(1.,-1.,cosine)
    with pytest.raises(ValueError):normalized_mix_fraction(1.,0.,0.)


def test_actual_neural_jvp_and_native_inputs_are_preserved():
    from pipeline.model import official
    torch.manual_seed(10854)
    layer=official()['NDPLayer'](depth=2,width=8,k0=-2,m=1,rotation_format='axis_angle')
    x=torch.randn(19,3);q=x+.1;anchor=x.double()+torch.tensor([.4,-.2,.1])
    saved=[p.detach().clone() for p in layer.parameters()];rng=torch.get_rng_state().clone()
    old=directions(layer,x,q,anchor,19,method='fgr_pcgmix')
    new=directions(layer,x,q,anchor,19,method='fgr_normmix')
    again=directions(layer,x,q,anchor,19,method='fgr_pcgmix')
    for name in ('fgr_pcg','cd_gd'):
        torch.testing.assert_close(old['directions'][name]['velocity'],new['directions'][name]['velocity'],rtol=0,atol=0)
    torch.testing.assert_close(old['directions']['fgr_pcgmix']['velocity'],again['directions']['fgr_pcgmix']['velocity'],rtol=0,atol=0)
    v=new['directions']['fgr_normmix'];g=new['gradient'];sc=float((g*new['directions']['cd_gd']['velocity']).sum())
    assert v['beta']<=old['directions']['fgr_pcgmix']['beta']
    assert float((g*v['velocity']).sum())<=.5*sc+1e-12
    assert float(v['velocity'].norm())==pytest.approx(new['bound'],rel=1e-12)
    assert torch.equal(rng,torch.get_rng_state())
    for p,prior in zip(layer.parameters(),saved):assert torch.equal(p,prior)
