"""One fixed FGR/CD mixing rule at original endpoints; common CD acceptance."""
import ndp_field_probe as base
import ndp_fgr_probe as anchor
import sys
from pipeline.direction_mix import mix_fraction
from pipeline.mse_reference import mse_reference

PRECISE = '--precise-metrics' in sys.argv
OUT = base.ROOT/'paper_revision/mix_20261018'/('precise' if PRECISE else 'finite')


def compare(pair,level,layer,opt,inputs,target,truth):
    torch=base.torch
    before=[v.detach().double().clone() for v in layer.parameters()]
    path=base.continuation(layer,opt,inputs,target,base.cd,steps=1)
    _,native_start,_,_=next(path);_,native_after,_,native_params=next(path)
    adam=[v.detach().double()-p for v,p in zip(native_params,before)]
    native_start,native_after=native_start.detach(),native_after.detach();path.close()
    model=base.copy.deepcopy(layer).double()
    inputs,q,gt=inputs.double(),target.double(),truth.double()
    p=model(inputs)[0];params=list(model.parameters());loss=base.cd(p,q)
    gradient=torch.autograd.grad(loss,p,retain_graph=True)[0].detach()
    forward,transpose=base.jacobian_products(p,params)
    start=p.detach();task=2*(start-gt)/len(start);risk=float((start-gt).square().sum()/len(start))
    seed=93000+100*pair+level; methods={}; begin=base.tick(p)
    for name,force,objective,info in base.build_fields(start,q,seed):
        delta,solver=base.ridge_lift(forward,transpose,-force,seed,damping_ratio=.0001,maxiter=2048)
        base.append('solvers',dict(pair=pair,level=level,method=name,**solver))
        assert solver['converged'] and solver['relative_residual']<=1e-8
        methods[name+'_gd']=(transpose(force),force,objective)
        methods[name+'_ridge']=(delta,force,objective)
        base.append('fields',dict(pair=pair,level=level,method=name,**info,
            point_derivative=float((task*force).sum()),force_norm=float(force.norm()),
            point_task_cosine=float(-(task*force).sum()/(task.norm()*force.norm()))))
    bound=len(start)**.5*info['spacing']
    cd_delta,cd_velocity=base.scale_direction(methods['cd_gd'][0],forward,bound)
    sc=float((gradient*cd_velocity).sum())
    for transfer in ('gd','ridge'):
        delta,velocity=base.scale_direction(methods['fgr_'+transfer][0],forward,bound)
        sa=float((gradient*velocity).sum()); beta=mix_fraction(sa,sc)
        mixed=[(1-beta)*a+beta*c for a,c in zip(delta,cd_delta)]
        mixed_velocity=forward(mixed)
        linear=(1-beta)*velocity+beta*cd_velocity
        error=float((mixed_velocity-linear).norm())/bound
        assert error<1e-10
        if transfer=='ridge':
            expected=CHOICES[pair,level]
            assert abs(beta-expected['beta'])<1e-10
        methods['fgr_'+transfer+'mix']=(mixed,-gradient,lambda z:base.cd(z,q))
        base.append('mixes',dict(pair=pair,level=level,transfer=transfer,beta=beta,cd_a=sa,cd_c=sc,
            jvp_combination_error=error,pre_normalization_ratio=float(mixed_velocity.norm())/bound))
    methods['adam']=(adam,-gradient,lambda z:base.cd(z,q))
    seconds=base.tick(p)-begin; maximum=0.
    for name,(raw,force,objective) in methods.items():
        delta,velocity=base.scale_direction(raw,forward,bound)
        record=dict(pair=pair,level=level,method=name,risk_derivative=float((task*velocity).sum()),
            cd_derivative=float((gradient*velocity).sum()),objective_derivative=float((-force*velocity).sum()),
            output_norm=float(velocity.norm()),motion_bound=bound,
            raw_parameter_norm=float(sum(v.square().sum() for v in raw).sqrt()),
            transfer=base.transfer_error(force,forward(raw)))
        base.append('directions',record)
        if name=='fgr_ridgemix':
            low,high=INTERVALS[pair,level]
            assert low-1e-12<=record['risk_derivative']<=high+1e-12
            assert record['cd_derivative']<=.5*sc+1e-12*abs(sc)
        for row,output in base.finite_trials(model,inputs,start,q,delta,velocity,objective,-force,gradient):
            displacement=output-start
            change=float((output-gt).square().sum()/len(start))-risk
            identity=float((task*displacement).sum()+displacement.square().sum()/len(start))
            error=abs(change-identity)
            float64_change,float64_error=change,error
            reference={}
            if PRECISE or error>1e-12:
                reference=mse_reference(*[v.detach().cpu().numpy() for v in (start,output,gt)])
                reference['float64_change_error']=change-reference['reference_change']
            if PRECISE:
                change=reference['reference_change'];error=reference['reference_identity_error']
            maximum=max(maximum,error)
            base.append('finite',dict(pair=pair,level=level,method=name,**row,risk_change=change,
                mse_identity_error=error,precision_reference=reference,
                float64_risk_change=float64_change,float64_identity_error=float64_error,
                epe_change=float((output-gt).norm(dim=1).mean()-(start-gt).norm(dim=1).mean())))
    base.append('states',dict(pair=pair,level=level,cd=float(loss.detach()),mse=risk,
        mse_identity_error=maximum,field_seconds=seconds,spacing=info['spacing'],
        native_adam_cd_change=float(base.cd(native_after,target)-base.cd(native_start,target)),
        native_adam_mse_change=float((native_after-truth).square().sum(1).mean()-(native_start-truth).square().sum(1).mean()),
        precision_output_difference=float((start-native_start.double()).norm()),peak_cuda_bytes=torch.cuda.max_memory_allocated()))
    (OUT/'progress.json').write_text(base.json.dumps(dict(pair=pair,level=level))+'\n')


if __name__=='__main__':
    observed=base.json.loads((OUT.parents[1]/'fgr_20261018/observed.json').read_text())
    anchor.POSES={r['pair']:r['transform'] for r in observed['rows']}
    CHOICES={(r['pair'],r['level']):r for r in base.json.loads((OUT.parent/'choices.json').read_text())['choices']}
    INTERVALS={(r['pair'],r['level']):r['normalized_task_interval'] for r in base.json.loads((OUT.parent/'states.json').read_text())}
    base.OUT=OUT; base.environment=anchor.environment; base.compare=compare; base.main()
