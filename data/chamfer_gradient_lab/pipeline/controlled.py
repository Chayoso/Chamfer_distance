"""Native-parameter step control with this trajectory's own Adam history."""
import copy
from types import SimpleNamespace
from .model import torch
from .optim import fixed_official
from .losses import components
from .lifting import jacobian_products,ridge_lift
from .trust import select_step


def cd(p,q):
    a,b=components(p,q)
    return a.mean()+b.mean()


def controlled_step(layer,inputs,target,original,actual,kind,seed):
    """GT is intentionally absent. Mutates only this layer's parameters."""
    if kind not in ('adam_trust','adaptive'):raise ValueError(kind)
    before=copy.deepcopy(layer);before._forward_pre_hooks.clear();before=before.double()
    params=list(before.parameters());live=list(layer.parameters())
    with torch.no_grad():
        for p,a in zip(params,original):p.copy_(a)
        for p,a in zip(live,original):p.copy_(a)
        start=layer(inputs)[0].detach().double()
        initial=float(cd(start.float(),target))
    output=before(inputs.double())[0];loss=cd(output,target.double())
    gradient,=torch.autograd.grad(loss,output,retain_graph=True)
    forward,transpose=jacobian_products(output,params)
    velocity=forward(actual);speed=velocity.norm()
    info=dict(alpha=1.,fraction=0.,attempts=0,converged=True,iterations=0,
              relative_residual=0.,cd_change=0.,linearization_error=0.,merit=0.)
    if speed<=1e-20:return info
    delta=actual
    if kind=='adaptive':
        lift,solver=ridge_lift(forward,transpose,gradient,seed)
        info.update({k:solver[k] for k in ('converged','iterations','relative_residual')})
        if not solver['converged']:return info
        norm=forward(lift).norm()
        if norm<=1e-20:return info
        lift=[p*speed/norm for p in lift]
        merit=float((gradient*velocity).sum()/(gradient*forward(lift)).sum())
        alpha=1. if merit>=.8 else .4
        delta=[alpha*a+(1-alpha)*b for a,b in zip(actual,lift)]
        norm=forward(delta).norm()
        if norm<=1e-20:return info
        delta=[p*speed/norm for p in delta]
        velocity=forward(delta);info.update(alpha=alpha,merit=merit)
    slope=float((gradient*velocity).sum())
    cache={}
    def evaluate(h):
        with torch.no_grad():
            for p,a,d in zip(live,original,delta):p.copy_(a.double()+h*d)
            p=layer(inputs)[0]
            cache[h]=(p.detach().double(),float(cd(p,target)))
        return cache[h]
    h,attempts=select_step(evaluate,start,velocity,initial,slope)
    with torch.no_grad():
        for p,a,d in zip(live,original,delta):p.copy_(a.double()+h*d)
    info.update(fraction=h,attempts=attempts)
    if h:
        p,value=cache[h]
        info.update(cd_change=value-initial,
                    linearization_error=float((p-start-h*velocity).norm()/(h*speed)))
    return info


def controlled_factory(kind,records,periodic=False,restart=False):
    env=fixed_official()
    build,optimize=env['Deformation_Pyramid'],env['optimize_deformation_pyramid']
    state={}
    def constructor(**kwargs):
        net=build(**kwargs);state['net']=net
        for i,layer in enumerate(net.pyramid):
            layer.register_forward_pre_hook(lambda module,inputs,i=i:state.update({i:inputs[0].detach()}))
        return net
    def optimizer(instance):
        original=env['compute_truncated_chamfer_distance']
        def observed(p,q,**kwargs):
            level=next(i for i,m in enumerate(state['net'].pyramid) if next(m.parameters()).requires_grad)
            step=state.get('step',-1)+1 if state.get('level')==level else 0
            state.update(level=level,step=step,target=q[0].detach())
            return original(p,q,**kwargs)
        env['compute_truncated_chamfer_distance']=observed
        return optimize(instance)
    class ControlledAdam(torch.optim.Adam):
        def step(self,closure=None):
            level,step=state['level'],state['step'];layer=state['net'].pyramid[level]
            reset=restart and step in (25,49)
            if kind=='adam' or (periodic and step not in (25,49)):
                value=super().step(closure)
                records.append(dict(level=level,step=step,alpha=1.,fraction=1.,attempts=0,
                    converged=True,iterations=0,relative_residual=0.,cd_change=None,
                    linearization_error=None,merit=None,controlled=False))
                if reset:self.state.clear()
                records[-1].update(restarted=reset,moment_step=int(self.state[next(layer.parameters())].get('step',0)))
                return value
            original=[p.detach().clone() for p in layer.parameters()]
            value=super().step(closure)
            actual=[p.detach().double()-a.double() for p,a in zip(layer.parameters(),original)]
            info=controlled_step(layer,state[level],state['target'],original,actual,kind,91000+level*50+step)
            if reset:self.state.clear()
            records.append(dict(level=level,step=step,**info,controlled=True,restarted=reset,
                                moment_step=int(self.state[next(layer.parameters())].get('step',0))))
            return value
    env.update(Deformation_Pyramid=constructor,optimize_deformation_pyramid=optimizer,
               optim=SimpleNamespace(Adam=ControlledAdam))
    return env
