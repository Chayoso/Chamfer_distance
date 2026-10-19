import torch
from pipeline.model import config
from pipeline.runner import fit
from pipeline.trajectory_mix import trajectory_factory
from pipeline.prefix import prefix_observer
from pipeline.event_chain import event_chain_factory
from ndp_mix_trajectory import deterministic


def test_after_decision_event_observation_and_truth_do_not_change_updates(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    cfg=config();cfg.m=2;cfg.width=8;cfg.depth=1;cfg.samples=6;cfg.budget=[50]*2
    x=torch.arange(30,dtype=torch.float32).reshape(10,3)/10;pose=torch.eye(4,dtype=torch.float64);pose[:3,3]=.1
    runs=[]
    for index,(observed,shift) in enumerate(((False,0.),(True,0.),(True,2.))):
        folder=tmp_path/str(index);folder.mkdir();updates=[];events=[];metadata={};outputs=[];truth=x+.1+shift
        def base():
            env=event_chain_factory(updates,events,100,pose,truth) if observed else trajectory_factory(
                'fgr_pcgmix',updates,100,pose,updates_per_level=50,event_steps=(7,15))
            optimize=env['optimize_deformation_pyramid']
            def capture(instance):
                result=optimize(instance);outputs.append(result[0].detach().clone());return result
            env['optimize_deformation_pyramid']=capture;return env
        factory,loss=prefix_observer(base,metadata)
        result=fit(100,0,'official',0,folder,cfg=cfg,factory=factory,loss_fn=loss,
            loader=lambda pair:(x,x+.1,truth,torch.ones(len(x),dtype=torch.bool)))
        runs.append(dict(result=result,updates=deterministic(updates),prefix=metadata,output=outputs[0],
            rng=torch.get_rng_state(),events=events,trace=(folder/'p100_s0_official_w0_trace.txt').read_bytes()))
    assert all(torch.equal(runs[0]['output'],r['output']) and torch.equal(runs[0]['rng'],r['rng']) for r in runs)
    assert runs[0]['updates']==runs[1]['updates']==runs[2]['updates'] and runs[0]['prefix']==runs[1]['prefix']==runs[2]['prefix']
    assert deterministic(runs[0]['result'])==deterministic(runs[1]['result']) and runs[0]['trace']==runs[1]['trace']
    assert [(r['level'],r['step']) for r in runs[1]['events']]==[(0,7),(0,15),(1,7),(1,15)]
    assert any(a['products']['force']['task_native']!=b['products']['force']['task_native'] for a,b in zip(runs[1]['events'],runs[2]['events']))
    for a,b in zip(runs[1]['events'],runs[2]['events']):
        assert a['fraction']==b['fraction'] and a['beta']==b['beta'] and a['anchor']==b['anchor']
        assert all(a['products'][k]['cd_double']==b['products'][k]['cd_double'] for k in a['products'])
        assert max(a['reconstruction_errors'].values())<1e-12 and a['actual_task_dot_error']<1e-12
