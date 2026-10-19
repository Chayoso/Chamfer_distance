"""GT-free mixed directions and native float32 trials with explicit timing."""
import copy
from .model import torch
from .controlled import cd
from .continuation import tick
from .fields import geometry_scales
from .lifting import jacobian_products, ridge_lift
from .field_steps import scale_direction
from .direction_mix import mix_fraction, normalized_mix_fraction
from .trust import select_step
from .head import translation_lift, head_preconditioner


def directions(layer, inputs, target, anchor, seed, method=None, adam_delta=None):
    if method not in (None,'adam','cd_ridge','fgr_gdmix','fgr_ridgemix','fgr_headmix','fgr_pcgmix','fgr_normmix','fgr_search'):
        raise ValueError('Unknown production direction')
    total_begin=begin=tick(inputs)
    native=layer(inputs)[0].detach();initial=float(cd(native,target))
    model=copy.deepcopy(layer).double();model._forward_pre_hooks.clear()
    x,q=inputs.double(),target.double()
    p=model(x)[0];loss=cd(p,q)
    gradient=torch.autograd.grad(loss,p,retain_graph=True)[0].detach()
    forward,transpose=jacobian_products(p,list(model.parameters()))
    start=p.detach();spacing,_=geometry_scales(start,q);bound=len(start)**.5*spacing
    setup=tick(inputs)-begin
    result={};solvers={}
    forces={'cd':-gradient}
    if anchor is not None:forces['fgr']=anchor-start
    needed={None:(('cd','gd'),('cd','ridge'),('fgr','gd'),('fgr','ridge')),
        'adam':(), 'cd_ridge':(('cd','ridge'),),
        'fgr_gdmix':(('cd','gd'),('fgr','gd')), 'fgr_ridgemix':(('cd','gd'),('fgr','ridge')),
        'fgr_headmix':(('cd','gd'),('fgr','head')),'fgr_pcgmix':(('cd','gd'),('fgr','pcg')),
        'fgr_normmix':(('cd','gd'),('fgr','pcg')),'fgr_search':(('cd','gd'),('fgr','pcg'))}
    for field,transfer in needed[method]:
        begin=tick(inputs);force=forces[field]
        if transfer=='gd':raw=transpose(force)
        elif transfer=='head':
            raw,solver=translation_lift(model,x,-force,damping_ratio=.0001)
            predicted=solver.pop('velocity');actual=forward(raw)
            solver['jvp_relative_error']=float((actual-predicted).norm()/predicted.norm().clamp_min(1e-30))
            solver['transfer_relative_error']=float((actual-force).norm()/force.norm().clamp_min(1e-30))
            solvers[field]=solver
            if not solver['converged'] or solver['jvp_relative_error']>1e-10:
                raise RuntimeError(f'Invalid affine-head solve: {solver}')
        else:
            factory=(lambda damping:head_preconditioner(model,x,damping)) if transfer=='pcg' else None
            raw,solver=ridge_lift(forward,transpose,-force,seed,damping_ratio=.0001,maxiter=2048,preconditioner_factory=factory)
            solvers[field]=solver
            if not solver['converged']:raise RuntimeError(f'Unconverged {field} solve: {solver}')
        delta,velocity=scale_direction(raw,forward,bound)
        result[field+'_'+transfer]=dict(delta=delta,velocity=velocity,seconds=tick(inputs)-begin)
    if method=='adam':
        if adam_delta is None:raise ValueError('An actual Adam delta is required')
        begin=tick(inputs);delta,velocity=scale_direction(adam_delta,forward,bound)
        result['adam']=dict(delta=delta,velocity=velocity,seconds=tick(inputs)-begin)
    transfers={None:('gd','ridge'),'fgr_gdmix':('gd',),'fgr_ridgemix':('ridge',),'fgr_headmix':('head',),'fgr_pcgmix':('pcg',),'fgr_normmix':('pcg',)}
    for transfer in transfers.get(method,()):
        c=result['cd_gd'];sc=float((gradient*c['velocity']).sum())
        begin=tick(inputs);a=result['fgr_'+transfer]
        sa=float((gradient*a['velocity']).sum());beta=mix_fraction(sa,sc)
        if method=='fgr_normmix':
            cosine=float((a['velocity']*c['velocity']).sum())/(bound*bound)
            beta=normalized_mix_fraction(sa,sc,cosine)
        raw=[(1-beta)*v+beta*w for v,w in zip(a['delta'],c['delta'])]
        delta,velocity=scale_direction(raw,forward,bound)
        if method=='fgr_normmix' and float((gradient*velocity).sum())>.5*sc+1e-12:
            raise RuntimeError('Normalized mixture lost its observed descent certificate')
        name='fgr_normmix' if method=='fgr_normmix' else 'fgr_'+transfer+'mix'
        result[name]=dict(delta=delta,velocity=velocity,beta=beta,
            seconds=tick(inputs)-begin+a['seconds']+c['seconds'])
    native=native.double();production_seconds=tick(inputs)-total_begin
    value=dict(native=native,initial=initial,start=start,gradient=gradient,bound=bound,
                setup_seconds=setup,production_seconds=production_seconds,directions=result,solvers=solvers)
    if method=='fgr_search':value['forward']=forward
    return value


def native_trials(layer,inputs,target,native,initial,delta,velocity,gradient,caps=(1/16,.25,1.)):
    """Each cap is an independent timed controller, with no cached-cost discount."""
    slope=float((gradient*velocity).sum());speed=float(velocity.norm())
    for cap in caps:
        begin=tick(inputs);trial=copy.deepcopy(layer)
        trial._forward_pre_hooks.clear()
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
