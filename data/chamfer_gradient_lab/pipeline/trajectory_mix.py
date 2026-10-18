"""Nine controlled replacements in a fixed50-update-per-level native trajectory."""
from types import SimpleNamespace
from .model import torch
from .optim import fixed_official
from .native_mix import directions, native_trials


def trajectory_factory(kind,records,pair,pose=None,levels=None):
    if kind not in ('adam','adam_trust','cd_ridge','fgr_gdmix','fgr_ridgemix'):
        raise ValueError(kind)
    env=fixed_official();build=env['Deformation_Pyramid'];optimize=env['optimize_deformation_pyramid']
    state={}
    def constructor(**kwargs):
        net=build(**kwargs);state['net']=net
        instance=state['instance'];x,y=instance.src_pcd,instance.tgt_pcd
        generator=torch.Generator().set_state(torch.get_rng_state())
        ids=torch.randperm(len(x),generator=generator)[:instance.config.samples].to(x.device)
        if kind.startswith('fgr'):
            transform=torch.as_tensor(pose,device=x.device,dtype=torch.float64)
            state['anchor']=x[ids].double()@transform[:3,:3].T+transform[:3,3]-y.mean(0).double()
        for level,layer in enumerate(net.pyramid):
            layer.register_forward_pre_hook(lambda module,inputs,i=level:state.update({i:inputs[0].detach()}))
        return net
    def run(instance):
        if instance.config.budget!=[50]*instance.config.m:
            raise ValueError('The frozen intervention schedule requires50 updates per level')
        if levels is not None and any(i not in range(instance.config.m) for i in levels):
            raise ValueError('Intervention levels must belong to the pyramid')
        state['instance']=instance;original=env['compute_truncated_chamfer_distance']
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
            if kind=='adam' or step!=49 or (levels is not None and level not in levels):
                value=super().step(closure)
                records.append(dict(level=level,step=step,controlled=False,
                    moment_step=int(self.state[next(layer.parameters())]['step'])))
                return value
            original=[p.detach().clone() for p in layer.parameters()]
            value=super().step(closure)
            actual=[p.detach().double()-v.double() for p,v in zip(layer.parameters(),original)]
            with torch.no_grad():
                for p,v in zip(layer.parameters(),original):p.copy_(v)
            record=dict(level=level,step=step,controlled=True,
                moment_step=int(self.state[next(layer.parameters())]['step']))
            method='adam' if kind=='adam_trust' else kind
            try:
                result=directions(layer,state[level],state['target'],state.get('anchor'),
                    93000+100*pair+level,method=method,adam_delta=actual)
                item=result['directions'][method]
                trials=native_trials(layer,state[level],state['target'],result['native'],result['initial'],
                    item['delta'],item['velocity'],result['gradient'],caps=(1.,))
                next(trials);trial,expected=next(trials);trials.close()
                with torch.no_grad():
                    for p,v,d in zip(layer.parameters(),original,item['delta']):p.copy_(v.double()+trial['fraction']*d)
                    applied=layer(state[level])[0].detach().double()
                assert torch.equal(applied,expected),'Applied native move differs from accepted trial'
                record.update(**trial,beta=item.get('beta'),solvers=result['solvers'],
                    bound=result['bound'],initial_cd=result['initial'],applied_exact=True,valid=True)
            except (ValueError,RuntimeError) as error:
                with torch.no_grad():
                    for p,v in zip(layer.parameters(),original):p.copy_(v)
                record.update(valid=False,fraction=0.,accepted=False,error=str(error))
                records.append(record)
                raise  # Preserve the failed record and stop; never substitute another update.
            records.append(record)
            return value
    env.update(Deformation_Pyramid=constructor,optimize_deformation_pyramid=run,
               optim=SimpleNamespace(Adam=ControlledAdam))
    return env
