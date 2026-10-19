import pytest
import torch
from pipeline.model import config
from pipeline.runner import fit
from pipeline.trajectory_mix import trajectory_factory
from pipeline.prefix import prefix_observer
from ndp_mix_trajectory import deterministic


def run(folder,steps,observed):
    folder.mkdir();cfg=config();cfg.m=2;cfg.width=8;cfg.depth=1;cfg.samples=6;cfg.budget=[steps]*2
    x=torch.arange(30,dtype=torch.float32).reshape(10,3)/10;pose=torch.eye(4,dtype=torch.float64);pose[:3,3]=.1
    updates=[];metadata={};outputs=[]
    def base():
        env=trajectory_factory('fgr_pcgmix',updates,100,pose,updates_per_level=steps,event_steps=(7,15))
        optimize=env['optimize_deformation_pyramid']
        def capture(instance):
            result=optimize(instance);outputs.append(result[0].detach().clone());return result
        env['optimize_deformation_pyramid']=capture
        return env
    factory,loss=prefix_observer(base,metadata) if observed else (base,None)
    result=fit(100,0,'official',0,folder,cfg=cfg,factory=factory,loss_fn=loss,
        trace_steps=None if steps==50 else (0,8,16),
        loader=lambda pair:(x,x+.1,x+.1,torch.ones(len(x),dtype=torch.bool)))
    return dict(result=result,updates=updates,metadata=metadata,output=outputs[0],rng=torch.get_rng_state(),
        trace=(folder/'p100_s0_official_w0_trace.txt').read_bytes())


@pytest.mark.parametrize('steps',[16,50])
def test_prefix_observer_does_not_change_trajectory(tmp_path,monkeypatch,steps):
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    plain=run(tmp_path/'plain',steps,False);observed=run(tmp_path/'observed',steps,True)
    assert torch.equal(plain['output'],observed['output']) and torch.equal(plain['rng'],observed['rng'])
    assert plain['trace']==observed['trace']
    assert deterministic(plain['updates'])==deterministic(observed['updates'])
    assert deterministic(plain['result'])==deterministic(observed['result'])
    assert len(observed['metadata']['outputs'])==17


def test_restoration_preserves_first_sixteen_updates(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    short=run(tmp_path/'short',16,True);restored=run(tmp_path/'restored',50,True)
    assert short['metadata']==restored['metadata']
    assert deterministic(short['updates'][:16])==deterministic(restored['updates'][:16])
    assert len(short['updates'])==32 and len(restored['updates'])==100
    assert [(r['level'],r['step']) for r in restored['updates'] if r['controlled']]==[(0,7),(0,15),(1,7),(1,15)]
    assert all(r['moment_step']==r['step']+1 for r in restored['updates'])
