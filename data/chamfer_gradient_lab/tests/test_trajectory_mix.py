from types import SimpleNamespace
import pytest
import torch
from pipeline.model import config
from pipeline.optim import fixed_official
from pipeline.trajectory_mix import trajectory_factory
from pipeline.controlled import cd


def run(factory,updates):
    cfg=config();cfg.m=2;cfg.width=8;cfg.depth=1;cfg.samples=8;cfg.budget=[updates]*2
    x=torch.arange(24,dtype=torch.float32).reshape(8,3)/10
    torch.manual_seed(817)
    env=factory();env['compute_truncated_chamfer_distance']=lambda p,q,**kw:cd(p[0],q[0])
    instance=SimpleNamespace(config=cfg,device='cpu',src_pcd=x,tgt_pcd=x+.1,landmarks=None)
    result=env['optimize_deformation_pyramid'](instance)[0]
    return result,torch.get_rng_state()


def test_plain_wrapper_preserves_sampling_and_original_trajectory():
    expected,rng=run(fixed_official,50);records=[]
    actual,after=run(lambda:trajectory_factory('adam',records,100),50)
    assert torch.equal(actual,expected) and torch.equal(after,rng)
    assert len(records)==100 and all(not r['controlled'] for r in records)
    assert [r['moment_step'] for r in records]==list(range(1,51))*2


def test_rejected_last_update_restores_parameters_but_advances_moments(monkeypatch):
    import pipeline.trajectory_mix as trajectory
    def reject(layer,inputs,target,native,initial,delta,velocity,gradient,caps):
        row=dict(fraction=0.,accepted=False,attempts=1,cd_change=0.,linearization_error=0.)
        yield dict(row,policy='raw'),native
        yield dict(row,policy='trust'),native
    monkeypatch.setattr(trajectory,'native_trials',reject)
    expected,_=run(fixed_official,49);records=[]
    actual,_=run(lambda:trajectory_factory('adam_trust',records,100),50)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    controlled=[r for r in records if r['controlled']]
    assert len(controlled)==2 and all(r['step']==49 and r['moment_step']==50 and not r['accepted'] for r in controlled)


def test_frozen_anchor_uses_original_samples_across_own_trajectory(monkeypatch):
    import pipeline.trajectory_mix as trajectory
    captured=[];original=trajectory.directions
    def capture(layer,inputs,target,anchor,*args,**kwargs):
        captured.append((inputs.clone(),anchor.clone()))
        return original(layer,inputs,target,anchor,*args,**kwargs)
    monkeypatch.setattr(trajectory,'directions',capture)
    pose=torch.eye(4,dtype=torch.float64);pose[:3,3]=.1;records=[]
    result,_=run(lambda:trajectory_factory('fgr_ridgemix',records,100,pose),50)
    assert torch.isfinite(result).all() and len(captured)==2
    assert torch.equal(captured[0][1],captured[1][1])
    assert not torch.equal(captured[0][0],captured[1][0])
    x=torch.arange(24,dtype=torch.float32).reshape(8,3)/10
    expected=x.double()+.1-(x+.1).mean(0).double()
    torch.testing.assert_close(captured[0][1].sort(dim=0).values,expected.sort(dim=0).values,rtol=0,atol=0)
    assert all(r['valid'] and r['applied_exact'] for r in records if r['controlled'])


def test_short_plain_schedule_preserves_original_updates_and_sampling():
    expected,rng=run(fixed_official,16);records=[]
    actual,after=run(lambda:trajectory_factory('adam',records,100,updates_per_level=16,event_steps=(7,15)),16)
    assert torch.equal(actual,expected) and torch.equal(after,rng)
    assert len(records)==32 and all(not r['controlled'] for r in records)
    assert [r['moment_step'] for r in records]==list(range(1,17))*2


@pytest.mark.parametrize('kind',('fgr_pcgmix','fgr_normmix'))
def test_two_pcg_events_keep_original_anchor_and_own_moment_progress(monkeypatch,kind):
    import pipeline.trajectory_mix as trajectory
    captured=[];original=trajectory.directions
    def capture(layer,inputs,target,anchor,*args,**kwargs):
        assert kwargs['method']==kind
        captured.append((inputs.clone(),anchor.clone()))
        return original(layer,inputs,target,anchor,*args,**kwargs)
    monkeypatch.setattr(trajectory,'directions',capture)
    pose=torch.eye(4,dtype=torch.float64);pose[:3,3]=.1;records=[]
    result,_=run(lambda:trajectory_factory(kind,records,100,pose,updates_per_level=16,event_steps=(7,15)),16)
    assert torch.isfinite(result).all() and len(captured)==4
    assert all(torch.equal(captured[0][1],x[1]) for x in captured)
    assert torch.equal(captured[0][0],captured[1][0]) and not torch.equal(captured[1][0],captured[2][0])
    controlled=[r for r in records if r['controlled']]
    assert [(r['level'],r['step']) for r in controlled]==[(0,7),(0,15),(1,7),(1,15)]
    assert all(r['moment_step']==r['step']+1 for r in records)
    assert all(r['valid'] and r['applied_exact'] for r in controlled)
