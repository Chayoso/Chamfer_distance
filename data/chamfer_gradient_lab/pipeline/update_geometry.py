"""Evaluation-only geometry of actual movements on an unchanged native path."""
from .model import torch
from .losses import objective
from .horizon import digest,prefix_output


def risk_terms(start,finish,reference):
    a,b,r=[x.detach().double() for x in (start,finish,reference)]
    d=b-a;n=len(a);before=float((a-r).square().sum()/n);after=float((b-r).square().sum()/n)
    linear=float(2*((a-r)*d).sum()/n);quadratic=float(d.square().sum()/n);change=after-before
    return dict(before=before,after=after,change=change,linear=linear,quadratic=quadratic,
        identity_error=abs(change-linear-quadratic))


def geometry_observer(base_factory,truth,pose,updates,snapshots,movements):
    state={}
    def factory():
        env=base_factory();build=env['Deformation_Pyramid'];optimize=env['optimize_deformation_pyramid']
        def constructor(**kwargs):
            net=build(**kwargs);instance=state['instance'];x,y=instance.src_pcd,instance.tgt_pcd
            generator=torch.Generator().set_state(torch.get_rng_state())
            ids=torch.randperm(len(x),generator=generator)[:instance.config.samples].to(x.device)
            mean=y.mean(0);transform=torch.as_tensor(pose,device=x.device,dtype=torch.float64)
            held=torch.ones(len(x),device=x.device,dtype=torch.bool);held[ids]=False
            state.update(net=net,source=x-x.mean(0),mean=mean,truth=(truth[ids]-mean).double(),held=held,
                anchor=x[ids].double()@transform[:3,:3].T+transform[:3,3]-mean.double())
            return net
        def run(instance):
            state['instance']=instance
            return optimize(instance)
        env.update(Deformation_Pyramid=constructor,optimize_deformation_pyramid=run)
        return env
    def loss(p,q,*args):
        value=objective(p,q,*args)
        gradient=torch.autograd.grad(value,p,retain_graph=True)[0].detach().double()
        level=next(i for i,layer in enumerate(state['net'].pyramid) if next(layer.parameters()).requires_grad)
        step=state['step']+1 if state.get('level')==level else 0
        z=p.detach().double();error=z-state['truth']
        row=dict(level=level,step=step,cd=float(value.detach()),sample_sha256=digest([p]),
            task_mse=float(error.square().sum(1).mean()),task_epe=float(error.norm(dim=1).mean()),
            anchor_mse=float((z-state['anchor']).square().sum(1).mean()))
        if step in (16,50):
            with torch.no_grad():
                output=prefix_output(state['net'],state['source'],level)+state['mean']
                residual=(output.double()-truth.double()).norm(dim=1);held=residual[state['held']]
                row.update(full_epe=float(residual.mean()),full_mse=float(residual.square().mean()),
                    held_epe=float(held.mean()) if len(held) else None,held_mse=float(held.square().mean()) if len(held) else None,
                    full_output_sha256=digest([output]))
        snapshots.append(row)
        if step:
            previous=state['previous'];delta=z-previous['output'];update=updates[-1]
            assert (update['level'],update['step'])==(level,step-1)
            kind=f'global_{step-1}' if update['controlled'] else ('late_adam' if step>16 else 'early_adam')
            task=risk_terms(previous['output'],z,state['truth']);anchor=risk_terms(previous['output'],z,state['anchor'])
            movements.append(dict(level=level,step=step-1,kind=kind,task=task,anchor=anchor,
                cd_change=row['cd']-previous['row']['cd'],cd_gradient_dot=float((previous['gradient']*delta).sum()),
                task_epe_change=row['task_epe']-previous['row']['task_epe'],motion_norm=float(delta.norm()),
                fraction=update.get('fraction'),beta=update.get('beta'),accepted=update.get('accepted'),
                linearization_error=update.get('linearization_error')))
            assert max(task['identity_error'],anchor['identity_error'])<1e-12
        state.update(level=level,step=step,previous=dict(output=z,gradient=gradient,row=row))
        return value
    return factory,loss
