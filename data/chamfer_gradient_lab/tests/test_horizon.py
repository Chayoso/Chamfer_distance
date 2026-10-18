from types import SimpleNamespace
import torch
from pipeline.model import config
from pipeline.controlled import cd
from pipeline.horizon import prefix_output,observer_factory
from pipeline.optim import fixed_official
from pipeline.trajectory_mix import trajectory_factory


def run(factory):
    cfg=config();cfg.m=2;cfg.width=8;cfg.depth=1;cfg.samples=6;cfg.budget=[50]*2
    x=torch.arange(24,dtype=torch.float32).reshape(8,3)/10
    torch.manual_seed(817);env=factory()
    env['compute_truncated_chamfer_distance']=lambda p,q,**kw:cd(p[0],q[0])
    instance=SimpleNamespace(config=cfg,device='cpu',src_pcd=x,tgt_pcd=x+.1,landmarks=None)
    return env['optimize_deformation_pyramid'](instance)[0],torch.get_rng_state()


def test_empty_intervention_schedule_preserves_plain_path_and_rng():
    expected,rng=run(fixed_official);updates=[]
    output,after=run(lambda:trajectory_factory('fgr_ridgemix',updates,100,torch.eye(4),levels=()))
    assert torch.equal(output,expected) and torch.equal(rng,after)
    assert len(updates)==100 and not any(r['controlled'] for r in updates)


def test_observations_and_changed_truth_cannot_change_selected_updates():
    results=[];records=[]
    for truth in (torch.zeros(8,3),torch.full((8,3),17.)):
        updates=[];observations=[];metadata={}
        results.append(run(lambda:observer_factory('fgr_ridgemix',100,1,torch.eye(4),truth,updates,observations,metadata)))
        assert [(r['level'],r['step']) for r in updates if r['controlled']]==[(1,49)]
        assert len(observations)==102 and len(metadata)==4
        records.append((updates,observations))
    assert all(torch.equal(a,b) for a,b in zip(results[0],results[1]))
    for a,b in zip(records[0][0],records[1][0]):
        assert {k:v for k,v in a.items() if 'seconds' not in k}=={k:v for k,v in b.items() if 'seconds' not in k}
    assert records[0][1][-1]['train_mse']!=records[1][1][-1]['train_mse']
    expected,_=run(lambda:trajectory_factory('fgr_ridgemix',[],100,torch.eye(4),levels=(1,)))
    assert torch.equal(results[0][0],expected)


def test_full_prefix_evaluation_does_not_overwrite_training_input_hooks():
    class Shift(torch.nn.Module):
        def forward(self,x):return x+1,None
    calls=[];layers=[Shift(),Shift()]
    for layer in layers:layer.register_forward_pre_hook(lambda module,inputs:calls.append(inputs[0]))
    x=torch.zeros(3,3)
    assert torch.equal(prefix_output(SimpleNamespace(pyramid=layers),x,1),x+2)
    assert not calls and torch.equal(x,torch.zeros_like(x))
