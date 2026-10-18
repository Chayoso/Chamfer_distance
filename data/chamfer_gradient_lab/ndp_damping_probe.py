"""Fixed-field ridge path on exact baseline clones; preserve every failed solve."""
import copy
import csv
import json
import sys
import time
import open3d as o3d
from pipeline.model import ROOT,np,torch,config
from pipeline.datasets import transfer
from pipeline.runner import fit
from pipeline.lifting import jacobian_products,weighted_lift
from pipeline.continuation import transfer_error
from pipeline.capacity import plan_field,frozen_change
from pipeline.features import feature_penalty
from pipeline.controlled import cd
from pipeline.trust import select_step
from ndp_direction_audit import instrumented

OUT=ROOT/'paper_revision/damping_20261018'
KINDS=('feature_soft','descriptor_soft')
RATIOS={'r01':.01,'r001':.001,'r0001':.0001,'r00001':.00001}
MAXITER=512


def observe(before,after,inputs,q,gt,history,group,scale,dirs,key,records):
    before._forward_pre_hooks.clear();before=before.double()
    inputs,q,gt=inputs.double(),q.double(),gt.double()
    params=list(before.parameters());saved=[a.detach().clone() for a in params]
    actual=[b.detach().double()-a for a,b in zip(saved,after.parameters())]
    p=before(inputs)[0];loss=cd(p,q)
    gp,=torch.autograd.grad(loss,p,retain_graph=True);gp=gp.detach()
    task=2*(p.detach()-gt)/len(p);risk=(p.detach()-gt).square().sum()/len(p)
    forward,transpose=jacobian_products(p,params);speed=forward(actual).norm()
    proposals=[('actual_adam',actual,None,None),('parameter_gd',[-g for g in transpose(gp)],None,None)]
    penalty,_=feature_penalty(p.detach(),q)
    for name in KINDS:
        force,mass,info=plan_field(p.detach(),q,'soft',penalty=penalty,feature_only=name.startswith('descriptor_'))
        assert info['converged'] and info['residual']<=1e-8
        cosine=float((force*(-task)).sum()/(force.norm()*task.norm()).clamp_min(1e-30))
        records['states'].append(dict(pair=key[0],level=key[1],step=key[2],method=name,task_cosine=cosine,**info))
        root=mass.sqrt()[:,None];target=force/root.clamp_min(torch.finfo(force.dtype).tiny)
        for suffix,ratio in RATIOS.items():
            method=name+'_'+suffix;torch.cuda.synchronize();start=time.perf_counter()
            delta,solver=weighted_lift(forward,transpose,force,mass,81000+key[0]*100+key[1]*10+key[2],ratio,maxiter=MAXITER)
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            record=dict(pair=key[0],level=key[1],step=key[2],method=method,seconds=seconds,**solver)
            records['solver'].append(record)
            (OUT/'progress.json').write_text(json.dumps(record,indent=2)+'\n')
            if not solver['converged']:
                (OUT/'numerical_failure.json').write_text(json.dumps(record,indent=2)+'\n')
                raise RuntimeError('Unconverged path solve; do not interpret as transfer obstruction')
            velocity=forward(delta);metrics=transfer_error(target,root*velocity)
            records['transfer'].append(dict(pair=key[0],level=key[1],step=key[2],method=method,
                barycentric_ray_error=metrics['ray_error'],weighted_relative_error=metrics['relative_error'],
                parameter_norm=float(sum(a.square().sum() for a in delta).sqrt()),output_norm=float(velocity.norm()),
                damping=solver['damping']))
            proposals.append((method,delta,force,mass))
    trial=copy.deepcopy(before);live=list(trial.parameters());mse_error=0.
    for method,raw,force,mass in proposals:
        norm=forward(raw).norm()
        delta=[d*speed/norm for d in raw] if norm>1e-20 else [torch.zeros_like(d) for d in raw]
        velocity=forward(delta);gradient=gp if force is None else -force
        objective=lambda z:cd(z,q) if force is None else frozen_change(z,p.detach(),force,mass)
        initial=float(objective(p.detach()));slope=float((gradient*velocity).sum());cd_slope=float((gp*velocity).sum())
        records['directions'].append(dict(pair=key[0],level=key[1],step=key[2],method=method,
            risk_derivative=float((task*velocity).sum()),objective_derivative=slope,cd_derivative=cd_slope,output_norm=float(velocity.norm())))
        cache={}
        def evaluate(h,common=False):
            if h not in cache:
                with torch.no_grad():
                    for t,a,d in zip(live,saved,delta):t.copy_(a+h*d)
                    output=trial(inputs)[0];value=float(objective(output));cd_value=float(cd(output,q))
                cache[h]=(output.detach(),value,cd_value)
            output,value,cd_value=cache[h]
            return output,cd_value if common else value
        for cap in (1/16,.25,1.):
            selected,attempts=select_step(evaluate,p.detach(),velocity,initial,slope,cap)
            selected_cd,attempts_cd=select_step(lambda h:evaluate(h,True),p.detach(),velocity,float(loss.detach()),cd_slope,cap)
            for policy,h,tries in [('raw',cap,1),('trust',selected,attempts),('cd_trust',selected_cd,attempts_cd)]:
                output,value=evaluate(h) if h else (p.detach(),initial)
                displacement=output-p.detach();change=(output-gt).square().sum()/len(p)-risk
                identity=(task*displacement).sum()+displacement.square().sum()/len(p)
                mse_error=max(mse_error,float(abs(change-identity)))
                with torch.no_grad():cd_change=float(cd(output,q)-loss.detach())
                error=float((displacement-h*velocity).norm()/(h*speed)) if h else 0.
                records['finite'].append(dict(pair=key[0],level=key[1],step=key[2],method=method,policy=policy,cap=cap,
                    fraction=h,accepted=h>0,risk_change=float(change),objective_change=value-initial,cd_change=cd_change,
                    output_norm=float(displacement.norm()),linearization_error=error,attempts=tries))
    assert mse_error<1e-12
    records['identities'].append(dict(pair=key[0],level=key[1],step=key[2],mse_identity_error=mse_error))


def save(records,replays,folder):
    for name,rows in records.items():
        if not rows:continue
        with (folder/(name+'.txt')).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (folder/'integrity.json').write_text(json.dumps(dict(exact_replays=replays,rows={k:len(v) for k,v in records.items()}),indent=2)+'\n')


def main():
    OUT.mkdir(exist_ok=True);folder=OUT/'probe';folder.mkdir(exist_ok=False)
    cfg=config();cfg.budget=[50]*9;torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    records={k:[] for k in ('directions','finite','states','solver','transfer','identities')};replays=[]
    try:
        for pair in (100,101,124,125,200,201):
            assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<800_000_000
            x,y,truth,_=transfer(pair)
            factory=lambda:instrumented(pair,x,y,truth,None,None,records,observer=observe)
            got=fit(pair,0,'official',0,folder,loader=transfer,cfg=cfg,factory=factory)
            stem=f'p{pair}_s0_official_w0';ref=ROOT/'paper_revision'/('replication_4d' if pair<200 else 'replication_hf')
            old=json.loads((ref/(stem+'.json')).read_text())
            for k in old:
                if k in got and k!='seconds':assert old[k]==got[k],(pair,k)
            assert np.array_equal(np.loadtxt(ref/(stem+'_trace.txt'),delimiter=',',skiprows=1),np.loadtxt(folder/(stem+'_trace.txt'),delimiter=',',skiprows=1))
            replays.append(pair);save(records,replays,folder)
    finally:save(records,replays,folder)
    assert {k:len(v) for k,v in records.items()}==dict(directions=540,finite=4860,states=108,solver=432,transfer=432,identities=54)


if __name__=='__main__':
    if '--accurate' in sys.argv:OUT=OUT/'precision';MAXITER=2048
    main()
