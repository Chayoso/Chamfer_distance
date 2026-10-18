"""Evaluation-only observation of a single intervention and its later path."""
import hashlib
from types import SimpleNamespace
from .model import torch
from .trajectory_mix import trajectory_factory


def digest(tensors):
    h=hashlib.sha256()
    for value in tensors:
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def prefix_output(net,source,level):
    """Bypass root input-observation hooks; queries cannot change live inputs."""
    output=source
    for layer in net.pyramid[:level+1]:output=layer.forward(output)[0]
    return output


def observer_factory(kind,pair,site,pose,truth,updates,observations,metadata):
    """Truth is used only after the ordinary loss has been computed."""
    env=trajectory_factory(kind,updates,pair,pose,levels=(site,))
    build=env['Deformation_Pyramid'];optimize=env['optimize_deformation_pyramid'];state={}
    def constructor(**kwargs):
        net=build(**kwargs);state['net']=net
        instance=state['instance'];x,y=instance.src_pcd,instance.tgt_pcd
        generator=torch.Generator().set_state(torch.get_rng_state())
        ids=torch.randperm(len(x),generator=generator)[:instance.config.samples].to(x.device)
        held=torch.ones(len(x),device=x.device,dtype=torch.bool);held[ids]=False
        state.update(source=x-x.mean(0),mean=y.mean(0),truth=(truth[ids]-y.mean(0)).double(),held=held)
        metadata.update(source_ids_sha256=digest([ids]),initial_parameters_sha256=digest(
            [p for layer in net.pyramid for p in layer.parameters()]))
        return net
    class ObservedAdam(env['optim'].Adam):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);state['optimizer']=self
    def run(instance):
        state['instance']=instance;original=env['compute_truncated_chamfer_distance']
        def observed(p,q,**kwargs):
            value=original(p,q,**kwargs)
            level=next(i for i,layer in enumerate(state['net'].pyramid) if next(layer.parameters()).requires_grad)
            step=state.get('step',-1)+1 if state.get('level')==level else 0
            state.update(level=level,step=step)
            with torch.no_grad():
                error=p[0].double()-state['truth']
                row=dict(level=level,step=step,cd=float(value.detach()),
                    train_mse=float(error.square().sum(1).mean()),sample_sha256=digest([p]))
                if step in (0,50) or (level==site and step==49):
                    output=prefix_output(state['net'],state['source'],level)+state['mean']
                    residual=(output.double()-truth.double()).norm(dim=1)
                    row.update(full_epe=float(residual.mean()),full_mse=float(residual.square().mean()),
                        held_epe=float(residual[state['held']].mean()) if state['held'].any() else None,
                        held_mse=float(residual[state['held']].square().mean()) if state['held'].any() else None,
                        full_output_sha256=digest([output]))
                if level==site and step==49:
                    opt=state['optimizer'];parameters=[p for layer in state['net'].pyramid for p in layer.parameters()]
                    moments=[opt.state[p][key] for p in state['net'].pyramid[level].parameters()
                             for key in ('step','exp_avg','exp_avg_sq')]
                    metadata.update(before_parameters_sha256=digest(parameters),before_moments_sha256=digest(moments))
                observations.append(row)
            return value
        env['compute_truncated_chamfer_distance']=observed
        return optimize(instance)
    env.update(Deformation_Pyramid=constructor,optimize_deformation_pyramid=run,
               optim=SimpleNamespace(Adam=ObservedAdam))
    return env
