"""Observe the already committed global event; GT never enters its decision."""
from .model import torch
from .trajectory_mix import trajectory_factory
from .update_geometry import risk_terms


@torch.no_grad()
def describe_event(record,result,anchor,applied,truth):
    z,native=result['start'],result['native'];n=len(z);bound=float(result['bound'])
    force=anchor-z;force_norm=float(force.norm())
    desired=force*(bound/force_norm) if force_norm else torch.zeros_like(force)
    rays=result['directions'];a=rays['fgr_pcg']['velocity'];c=rays['cd_gd']['velocity'];v=rays['fgr_pcgmix']['velocity']
    beta=rays['fgr_pcgmix']['beta'];raw=(1-beta)*a+beta*c;raw_norm=float(raw.norm())
    assert bound>0 and raw_norm>0
    reconstructed=raw*(bound/raw_norm);delta=applied-native
    g=2*(z-truth)/n;gn=2*(native-truth)/n;cd=result['gradient']
    products={}
    for name,d in (('force',desired),('pcg',a),('cd_reference',c),('mixed',v),('actual',delta)):
        products[name]=dict(norm=float(d.norm()),task_double=float((g*d).sum()),
            task_native=float((gn*d).sum()),cd_double=float((cd*d).sum()))
    task=risk_terms(native,applied,truth);anchor_terms=risk_terms(native,applied,anchor)
    errors=dict(velocity=float((reconstructed-v).norm()),
        task_double=abs(float((g*reconstructed).sum())-products['mixed']['task_double']),
        task_native=abs(float((gn*reconstructed).sum())-products['mixed']['task_native']),
        observed=abs(float((cd*reconstructed).sum())-products['mixed']['cd_double']))
    return dict(level=record['level'],step=record['step'],points=n,accepted=record['accepted'],fraction=record['fraction'],
        beta=beta,bound=bound,force_norm=force_norm,force_zero=force_norm==0,mixed_raw_norm=raw_norm,
        normalized_force_pcg_cosine=float((desired*a).sum()/(desired.norm()*a.norm())) if force_norm and float(a.norm()) else None,
        normalized_force_pcg_error=float((desired-a).norm()/bound) if force_norm else None,
        native_double_start_gap=float((native-z).norm()),products=products,reconstruction_errors=errors,
        task=task,anchor=anchor_terms,actual_task_dot_error=abs(products['actual']['task_native']-task['linear']),
        cd_change=record['cd_change'],linearization_error=record['linearization_error'],solvers=result['solvers'])


def event_chain_factory(updates,events,pair,pose,truth,updates_per_level=50):
    state={}
    def observe(record,result,anchor,applied):
        # Evaluation truth is first read after native acceptance/application.
        reference=(truth[state['ids']]-state['mean']).double()
        row=describe_event(record,result,anchor,applied,reference);events.append(row)
        assert max(row['reconstruction_errors'].values())<1e-12
        assert max(row['task']['identity_error'],row['anchor']['identity_error'],row['actual_task_dot_error'])<1e-12
    env=trajectory_factory('fgr_pcgmix',updates,pair,pose,updates_per_level=updates_per_level,
        event_steps=(7,15),event_observer=observe)
    build=env['Deformation_Pyramid'];optimize=env['optimize_deformation_pyramid']
    def constructor(**kwargs):
        net=build(**kwargs);instance=state['instance']
        generator=torch.Generator().set_state(torch.get_rng_state())
        ids=torch.randperm(len(instance.src_pcd),generator=generator)[:instance.config.samples].to(instance.src_pcd.device)
        state.update(ids=ids,mean=instance.tgt_pcd.mean(0))
        return net
    def run(instance):
        state['instance']=instance
        return optimize(instance)
    env.update(Deformation_Pyramid=constructor,optimize_deformation_pyramid=run)
    return env
