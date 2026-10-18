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


def test_restricted_direction_and_single_cap_match_diagnostic_path(monkeypatch):
    import pipeline.native_mix as native
    model=Translation();x=torch.tensor([[0.,0,0],[3.,0,0],[0.,3,0]]);q=x+.1
    anchor=x.double()+torch.tensor([.4,-.2,.1]);full=directions(model,x,q,anchor,19)
    calls=[];solve=native.ridge_lift
    def counted(*args,**kwargs):
        calls.append(1);return solve(*args,**kwargs)
    monkeypatch.setattr(native,'ridge_lift',counted)
    for method,expected_calls in [('cd_ridge',1),('fgr_gdmix',0),('fgr_ridgemix',1)]:
        calls.clear();limited=directions(model,x,q,anchor,19,method=method)
        assert len(calls)==expected_calls
        for a,b in zip(limited['directions'][method]['delta'],full['directions'][method]['delta']):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        a=limited['directions'][method]
        args=(model,x,q,limited['native'],limited['initial'],a['delta'],a['velocity'],limited['gradient'])
        all_caps=list(native_trials(*args));single=list(native_trials(*args,caps=(1.,)))
        assert len(single)==2
        for old,new in zip(all_caps[-2:],single):
            assert {k:v for k,v in old[0].items() if k!='controller_seconds'}=={k:v for k,v in new[0].items() if k!='controller_seconds'}
            assert torch.equal(old[1],new[1])
