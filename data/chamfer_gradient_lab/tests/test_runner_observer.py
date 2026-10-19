"""Changing evaluation truth or trace density must not change native updates."""
import numpy as np
import torch
from pipeline.model import config
from pipeline.runner import fit
from pipeline.trajectory_mix import trajectory_factory
from ndp_mix_trajectory import deterministic


def test_trace_sampling_and_truth_are_evaluation_only(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    cfg=config();cfg.m=2;cfg.width=8;cfg.depth=1;cfg.samples=6;cfg.budget=[4]*2
    x=torch.arange(24,dtype=torch.float32).reshape(8,3)/10;y=x+.1
    pose=torch.eye(4,dtype=torch.float64);pose[:3,3]=.1
    outputs=[];histories=[];states=[];scores=[]
    for index,(shift,steps) in enumerate(((0.,(0,2,4)),(17.,(0,2,4)),(0.,(0,4)))):
        records=[];folder=tmp_path/str(index);folder.mkdir()
        def factory():
            env=trajectory_factory('fgr_pcgmix',records,100,pose,updates_per_level=4,event_steps=(1,3))
            optimize=env['optimize_deformation_pyramid']
            def observed(instance):
                result=optimize(instance);outputs.append(result[0].detach().clone());return result
            env['optimize_deformation_pyramid']=observed
            return env
        result=fit(100,0,'official',0,folder,cfg=cfg,factory=factory,trace_steps=steps,
            loader=lambda pair:(x,y,y+shift,torch.ones(len(x),dtype=torch.bool)))
        trace=np.loadtxt(folder/'p100_s0_official_w0_trace.txt',delimiter=',',skiprows=1)
        assert np.array_equal(trace[:,:2],np.asarray([(l,s) for l in range(2) for s in steps]))
        assert result['calls_by_level']==[5]*2
        histories.append(deterministic(records));states.append(torch.get_rng_state());scores.append(result['full_epe'])
    assert all(torch.equal(outputs[0],v) for v in outputs)
    assert all(torch.equal(states[0],v) for v in states) and histories[0]==histories[1]==histories[2]
    assert scores[0]!=scores[1] and scores[0]==scores[2]
