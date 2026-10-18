"""GT-free mixed directions and native float32 trials with explicit timing."""
import copy
from .model import torch
from .controlled import cd
from .continuation import tick
from .fields import geometry_scales
from .lifting import jacobian_products, ridge_lift
from .field_steps import scale_direction
from .direction_mix import mix_fraction
from .trust import select_step


def directions(layer, inputs, target, anchor, seed):
    begin=tick(inputs)
    native=layer(inputs)[0].detach();initial=float(cd(native,target))
    model=copy.deepcopy(layer).double();x,q=inputs.double(),target.double()
    p=model(x)[0];loss=cd(p,q)
    gradient=torch.autograd.grad(loss,p,retain_graph=True)[0].detach()
    forward,transpose=jacobian_products(p,list(model.parameters()))
    start=p.detach();spacing,_=geometry_scales(start,q);bound=len(start)**.5*spacing
    setup=tick(inputs)-begin
    result={};solvers={}
    forces={'cd':-gradient,'fgr':anchor-start}
    for field,transfer in (('cd','gd'),('cd','ridge'),('fgr','gd'),('fgr','ridge')):
        begin=tick(inputs);force=forces[field]
        if transfer=='gd':raw=transpose(force)
        else:
            raw,solver=ridge_lift(forward,transpose,-force,seed,damping_ratio=.0001,maxiter=2048)
            solvers[field]=solver
            if not solver['converged']:raise RuntimeError(f'Unconverged {field} solve: {solver}')
        delta,velocity=scale_direction(raw,forward,bound)
        result[field+'_'+transfer]=dict(delta=delta,velocity=velocity,seconds=tick(inputs)-begin)
    c=result['cd_gd'];sc=float((gradient*c['velocity']).sum())
    for transfer in ('gd','ridge'):
        begin=tick(inputs);a=result['fgr_'+transfer]
        sa=float((gradient*a['velocity']).sum());beta=mix_fraction(sa,sc)
        raw=[(1-beta)*v+beta*w for v,w in zip(a['delta'],c['delta'])]
        delta,velocity=scale_direction(raw,forward,bound)
        result['fgr_'+transfer+'mix']=dict(delta=delta,velocity=velocity,beta=beta,
            seconds=tick(inputs)-begin+a['seconds']+c['seconds'])
    return dict(native=native.double(),initial=initial,start=start,gradient=gradient,bound=bound,
                setup_seconds=setup,directions=result,solvers=solvers)


def native_trials(layer,inputs,target,native,initial,delta,velocity,gradient):
    """Each cap is an independent timed controller, with no cached-cost discount."""
    slope=float((gradient*velocity).sum());speed=float(velocity.norm())
    for cap in (1/16,.25,1.):
        begin=tick(inputs);trial=copy.deepcopy(layer)
        live=list(trial.parameters());saved=[v.detach().double().clone() for v in layer.parameters()]
        cache={}
        def evaluate(fraction):
            if fraction not in cache:
                with torch.no_grad():
                    for parameter,start,step in zip(live,saved,delta):parameter.copy_(start+fraction*step)
                    output=trial(inputs)[0].detach()
                    cache[fraction]=(output.double(),float(cd(output,target)))
            return cache[fraction]
        raw_output,raw_value=evaluate(cap)
        fraction,attempts=select_step(evaluate,native,velocity,initial,slope,cap)
        accepted,value=evaluate(fraction) if fraction else (native,initial)
        seconds=tick(inputs)-begin
        for policy,f,z,v,count in (('raw',cap,raw_output,raw_value,1),('trust',fraction,accepted,value,attempts)):
            error=float((z-native-f*velocity).norm())/(f*speed) if f*speed else 0.
            yield dict(policy=policy,cap=cap,fraction=f,attempts=count,accepted=f>0,cd_change=v-initial,
                linearization_error=error,output_norm=float((z-native).norm()),cd_slope=slope,
                controller_seconds=seconds),z
