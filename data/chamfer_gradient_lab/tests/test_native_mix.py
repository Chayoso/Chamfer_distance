import torch
from pipeline.controlled import cd
from pipeline.native_mix import native_trials, directions


class Translation(torch.nn.Module):
    def __init__(self):
        super().__init__();self.shift=torch.nn.Parameter(torch.zeros(3))
    def forward(self,x):return x+self.shift,None


def test_native_controller_evaluates_rounded_parameters_and_preserves_original():
    model=Translation();x=torch.tensor([[0.,0,0],[3.,0,0],[0.,3,0]])
    q=x+.1;start=model(x)[0].detach();gradient=torch.full_like(start,-1.)
    delta=[torch.full((3,),.1,dtype=torch.float64)];velocity=torch.full_like(start,.1).double()
    rows=list(native_trials(model,x,q,start.double(),float(cd(start,q)),delta,velocity,gradient.double()))
    assert len(rows)==6 and torch.equal(model.shift,torch.zeros(3))
    for row,output in rows:
        assert row['fraction']<=row['cap'] and row['controller_seconds']>0
        if row['policy']=='trust':
            assert row['accepted'] and row['cd_change']<0 and row['linearization_error']<1e-5
        assert torch.equal(output,output.float().double())


def test_native_direction_builder_preserves_observed_progress_constraint():
    model=Translation();x=torch.tensor([[0.,0,0],[3.,0,0],[0.,3,0]])
    result=directions(model,x,x+.1,x.double()+torch.tensor([.4,-.2,.1]),19)
    g=result['gradient'];d=result['directions'];sc=float((g*d['cd_gd']['velocity']).sum())
    assert sc<0 and all(r['converged'] for r in result['solvers'].values())
    for method in ('fgr_gdmix','fgr_ridgemix'):
        assert float((g*d[method]['velocity']).sum())<=.5*sc+1e-12
        assert abs(float(d[method]['velocity'].norm())-result['bound'])<1e-10
