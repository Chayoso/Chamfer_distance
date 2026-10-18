"""Verify exact readout realizability on the official nonlinear NDP layer."""
import copy
import numpy as np
import torch
import pytest
from pipeline.model import official
from pipeline.lifting import jacobian_products
from pipeline.head import translation_lift


@pytest.mark.parametrize('ratio',[.01,.0001])
def test_official_translation_head_is_exact_affine(ratio):
    env=official();torch.manual_seed(90)
    layer=env['NDPLayer'](depth=2,width=8,k0=-2,m=1,rotation_format='axis_angle').double()
    x=torch.randn(17,3,dtype=torch.float64)
    out=layer(x)[0];g=torch.randn_like(out)
    delta,info=translation_lift(layer,x,g,damping_ratio=ratio)
    forward,_=jacobian_products(out,list(layer.parameters()))
    velocity=forward(delta)
    torch.testing.assert_close(velocity,info['velocity'],rtol=1e-11,atol=1e-12)
    assert info['converged'] and float((g*velocity).sum())<0
    for h in (.01,1.,10.):
        trial=copy.deepcopy(layer)
        with torch.no_grad():
            for p,a,d in zip(trial.parameters(),layer.parameters(),delta):p.copy_(a+h*d)
        torch.testing.assert_close(trial(x)[0]-out,h*velocity,rtol=1e-10,atol=1e-12)
