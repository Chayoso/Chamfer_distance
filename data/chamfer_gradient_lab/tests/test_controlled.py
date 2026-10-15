"""Native rounding, finite acceptance, and rejection restoration."""
import numpy as np
import torch
import pytest
import pipeline.controlled as controlled


class Linear(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.eye(3))
    def forward(self,x):return x@self.weight.T,None


@pytest.mark.parametrize('kind',['adam_trust','adaptive'])
def test_native_control_accepts_and_restores(kind,monkeypatch):
    layer=Linear();x=torch.tensor([[1.,.2,.3],[-1.,.4,-.2],[.2,-.3,1.]])
    target=x*.8+.01
    optimizer=torch.optim.Adam(layer.parameters(),lr=.01)
    initial=float(controlled.cd(layer(x)[0],target).detach())
    controlled.cd(layer(x)[0],target).backward()
    saved=[p.detach().clone() for p in layer.parameters()];optimizer.step()
    actual=[p.detach().double()-a.double() for p,a in zip(layer.parameters(),saved)]
    info=controlled.controlled_step(layer,x,target,saved,actual,kind,123)
    assert info['fraction']>0 and info['converged'] and info['linearization_error']<=.1
    assert float(controlled.cd(layer(x)[0],target).detach())<initial
    assert next(layer.parameters()).dtype==torch.float32
    monkeypatch.setattr(controlled,'select_step',lambda *a,**kw:(0.,0))
    controlled.controlled_step(layer,x,target,saved,actual,kind,123)
    assert all(torch.equal(p,a) for p,a in zip(layer.parameters(),saved))
