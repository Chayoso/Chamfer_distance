import torch
import pytest
from pipeline.model import config
from pipeline.runner import fit
from pipeline.trajectory_mix import trajectory_factory
from pipeline.prefix import prefix_observer
from pipeline.update_geometry import geometry_observer,risk_terms
from ndp_mix_trajectory import deterministic


def test_identity_uses_actual_motion_and_separates_overshoot():
    a=torch.tensor([[-.1,0.,0.]],dtype=torch.float64,requires_grad=True)
    b=torch.tensor([[.2,0.,0.]],dtype=torch.float64);truth=torch.zeros_like(b)
    loss=(a-truth).square().sum(1).mean();gradient=torch.autograd.grad(loss,a)[0]
    terms=risk_terms(a,b,truth)
    assert terms['linear']==pytest.approx(float((gradient*(b-a.detach())).sum()),abs=1e-14)
    assert terms['linear']<0<terms['change'] and terms['quadratic']>-terms['linear']
    assert terms['identity_error']<1e-14 and risk_terms(b,b,truth)['change']==0


def test_geometry_observations_and_truth_do_not_change_path(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    cfg=config();cfg.m=2;cfg.width=8;cfg.depth=1;cfg.samples=6;cfg.budget=[50]*2
    x=torch.arange(30,dtype=torch.float32).reshape(10,3)/10;pose=torch.eye(4,dtype=torch.float64);pose[:3,3]=.1
    runs=[]
    for index,(observed,shift) in enumerate(((False,0.),(True,0.),(True,2.))):
        folder=tmp_path/str(index);folder.mkdir();updates=[];snapshots=[];movements=[];metadata={};outputs=[];truth=x+.1+shift
        def base():
            env=trajectory_factory('fgr_pcgmix',updates,100,pose,updates_per_level=50,event_steps=(7,15))
            optimize=env['optimize_deformation_pyramid']
            def capture(instance):
                result=optimize(instance);outputs.append(result[0].detach().clone());return result
            env['optimize_deformation_pyramid']=capture;return env
        if observed:
            factory,loss=geometry_observer(base,truth,pose,updates,snapshots,movements)
            factory,loss=prefix_observer(factory,metadata,loss_fn=loss)
        else:factory,loss=prefix_observer(base,metadata)
        result=fit(100,0,'official',0,folder,cfg=cfg,factory=factory,loss_fn=loss,
            loader=lambda pair:(x,x+.1,truth,torch.ones(len(x),dtype=torch.bool)))
        runs.append(dict(result=result,updates=deterministic(updates),prefix=metadata,output=outputs[0],
            rng=torch.get_rng_state(),snapshots=snapshots,movements=movements,
            trace=(folder/'p100_s0_official_w0_trace.txt').read_bytes()))
    assert all(torch.equal(runs[0]['output'],r['output']) and torch.equal(runs[0]['rng'],r['rng']) for r in runs)
    assert runs[0]['updates']==runs[1]['updates']==runs[2]['updates'] and runs[0]['prefix']==runs[1]['prefix']==runs[2]['prefix']
    assert deterministic(runs[0]['result'])==deterministic(runs[1]['result']) and runs[0]['trace']==runs[1]['trace']
    assert runs[1]['snapshots']!=runs[2]['snapshots']
    for r in runs[1:]:
        assert len(r['snapshots'])==102 and len(r['movements'])==100
        assert [(s['level'],s['step']) for s in r['snapshots'] if 'held_epe' in s]==[(0,16),(0,50),(1,16),(1,50)]
        assert sum(m['kind']=='late_adam' for m in r['movements'])==68
        assert sum(m['kind']=='early_adam' for m in r['movements'])==28
        for level in (0,1):
            s=[v for v in r['snapshots'] if v['level']==level]
            for key in ('task','anchor'):
                assert sum(m[key]['change'] for m in r['movements'] if m['level']==level)==pytest.approx(s[-1][key+'_mse']-s[0][key+'_mse'],abs=1e-12)
