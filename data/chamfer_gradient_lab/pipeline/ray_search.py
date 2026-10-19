"""First native-accepted ray on a fixed grid, without any task-label inputs."""
from .continuation import tick
from .field_steps import scale_direction
from .native_mix import directions,native_trials


def first_accepted(layer,inputs,target,anchor,seed):
    begin=tick(inputs)
    result=directions(layer,inputs,target,anchor,seed,method='fgr_search')
    forward=result.pop('forward');a=result['directions']['fgr_pcg'];c=result['directions']['cd_gd']
    attempts=[];selected=None
    for j in range(9):
        beta=j/8;raw=[(1-beta)*v+beta*w for v,w in zip(a['delta'],c['delta'])]
        delta,velocity=scale_direction(raw,forward,result['bound'])
        for trial,output in native_trials(layer,inputs,target,result['native'],result['initial'],delta,velocity,result['gradient'],caps=(1.,)):
            if trial['policy']=='trust':
                attempts.append(dict(beta=beta,**trial))
                if trial['accepted']:selected=dict(beta=beta,delta=delta,velocity=velocity)
        if selected is not None:break
    if selected is None:
        selected=dict(beta=None,delta=[v*0 for v in c['delta']],velocity=c['velocity']*0)
        trial=dict(trial,cd_slope=0.);output=result['native']
    result.update(item=selected,trial=trial,output=output,attempted=attempts,
                  production_seconds=tick(inputs)-begin)
    return result
