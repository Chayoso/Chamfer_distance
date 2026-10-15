"""Full-sample marginal coupling audit with weighted-plan and common-CD controls."""
import copy
import csv
import json
import time
import sys
from functools import partial
from pipeline.model import ROOT,np,torch,config
from pipeline.datasets import transfer
from pipeline.runner import fit
from pipeline.lifting import jacobian_products,ridge_lift,weighted_lift
from pipeline.continuation import transfer_error
from pipeline.capacity import plan_field,frozen_change
from pipeline.trust import select_step
from pipeline.controlled import cd
from ndp_direction_audit import instrumented

OUT=ROOT/'paper_revision/capacity_20261013'
MAXITER=4096
NEWTON=False


def build_fields(p,q,key):
    for name in ('soft','semi','unbalanced','balanced'):
        torch.cuda.synchronize();start=time.perf_counter()
        force,mass,info=plan_field(p,q,name,maxiter=MAXITER,newton=NEWTON)
        torch.cuda.synchronize()
        yield name,force,mass,info,time.perf_counter()-start


def observe(before,after,inputs,q,gt,history,group,scale,dirs,key,records,field_builder=None,out=None,weighted=False):
    before._forward_pre_hooks.clear();before=before.double()
    inputs,q,gt=inputs.double(),q.double(),gt.double()
    params=list(before.parameters());saved=[p.detach().clone() for p in params]
    actual=[p.detach().double()-a for a,p in zip(saved,after.parameters())]
    p=before(inputs)[0];loss=cd(p,q)
    gp,=torch.autograd.grad(loss,p,retain_graph=True);gp=gp.detach()
    task=2*(p.detach()-gt)/len(p);risk=(p.detach()-gt).square().sum()/len(p)
    forward,transpose=jacobian_products(p,params);speed=forward(actual).norm()
    def normalize(delta):
        norm=forward(delta).norm()
        return [d*speed/norm for d in delta] if norm>1e-20 else [torch.zeros_like(d) for d in delta]
    proposals=[('actual_adam',actual,None,None),('parameter_gd',[-a for a in transpose(gp)],None,None)]
    fields=[]
    destination=OUT if out is None else out
    for name,force,mass,info,seconds in (field_builder or build_fields)(p.detach(),q,key):
        progress=dict(pair=key[0],level=key[1],step=key[2],method=name,seconds=seconds,**info)
        (destination/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
        if not info['converged']:
            (destination/'numerical_failure.json').write_text(json.dumps(progress,indent=2)+'\n')
            raise RuntimeError('Transport numerical gate failed; no approximate envelope direction adopted')
        fields.append((name,force,info,seconds))
        gd=transpose(force);proposals.append((name+'_gd',gd,force,mass))
        measured=[('gd',gd,0.)]
        seed=81000+key[0]*100+key[1]*10+key[2]
        for kind in (('lift','weighted') if weighted else ('lift',)):
            torch.cuda.synchronize();start=time.perf_counter()
            delta,solver=(weighted_lift(forward,transpose,force,mass,seed) if kind=='weighted'
                          else ridge_lift(forward,transpose,-force,seed))
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            if not solver['converged']:raise RuntimeError(f'Unconverged {name}_{kind}: {solver}')
            proposals.append((name+'_'+kind,delta,force,mass));measured.append((kind,delta,solver['damping']))
            records['solver'].append([*key,name+'_'+kind,solver['converged'],solver['relative_residual'],solver['iterations'],elapsed])
        if weighted:
            root=mass.sqrt()[:,None];target=force/root.clamp_min(torch.finfo(force.dtype).tiny)
            for kind,delta,damping in measured:
                velocity=forward(delta);force_fit=transfer_error(force,velocity)
                bary_fit=transfer_error(target,root*velocity)
                records['transfer'].append([*key,name+'_'+kind,force_fit['ray_error'],bary_fit['ray_error'],bary_fit['relative_error'],damping])
    trial=copy.deepcopy(before);live=list(trial.parameters());identity_error=0.
    for method,raw,force,mass in proposals:
        delta=normalize(raw);velocity=forward(delta)
        gradient=gp if force is None else -force
        objective=lambda z:cd(z,q) if force is None else frozen_change(z,p.detach(),force,mass)
        initial=float(objective(p.detach()));slope=float((gradient*velocity).sum());cd_slope=float((gp*velocity).sum())
        records['directions'].append([*key,method,float((task*velocity).sum()),slope,cd_slope,float(velocity.norm())])
        cache={}
        def evaluate(h,common=False):
            if h not in cache:
                with torch.no_grad():
                    for t,a,d in zip(live,saved,delta):t.copy_(a+h*d)
                    output=trial(inputs)[0];value=float(objective(output));cd_value=float(cd(output,q))
                cache[h]=(output.detach(),value,cd_value)
            out,value,cd_value=cache[h]
            return out,cd_value if common else value
        for cap in (1/16,.25,1.):
            selected,attempts=select_step(evaluate,p.detach(),velocity,initial,slope,cap)
            cd_selected,cd_attempts=select_step(lambda h:evaluate(h,True),p.detach(),velocity,float(loss.detach()),cd_slope,cap)
            for policy,h,tries in [('raw',cap,1),('trust',selected,attempts),('cd_trust',cd_selected,cd_attempts)]:
                output,value=evaluate(h) if h else (p.detach(),initial)
                displacement=output-p.detach();change=(output-gt).square().sum()/len(p)-risk
                identity=(task*displacement).sum()+displacement.square().sum()/len(p)
                identity_error=max(identity_error,float(abs(change-identity)))
                with torch.no_grad():cd_change=float(cd(output,q)-loss.detach())
                error=float((displacement-h*velocity).norm()/(h*speed)) if h else 0.
                records['finite'].append([*key,method,policy,cap,h,h>0,float(change),value-initial,cd_change,float(displacement.norm()),error,tries])
    for name,force,info,seconds in fields:
        cosine=float((force*(-task)).sum()/(force.norm()*task.norm()).clamp_min(1e-30))
        records['states'].append([*key,name,cosine,float(force.norm()),info['epsilon'],info['total_mass'],info['max_source_relative_mass'],
            info['source_relative_std'],info['target_relative_std'],info['residual'],info['iterations'],info['converged'],seconds,
            identity_error,torch.cuda.max_memory_allocated(),info['scaling_residual'],info['newton_iterations']])


def run(out,field_builder=None,weighted=False):
    out.mkdir(exist_ok=True)
    folder=out/'probe';folder.mkdir(exist_ok=False)
    cfg=config();cfg.budget=[50]*9;torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    records={k:[] for k in ('directions','finite','states','solver')}
    headers={
        'directions':'pair level step method risk_derivative objective_derivative cd_derivative output_norm',
        'finite':'pair level step method policy cap fraction accepted risk_change objective_change cd_change output_norm linearization_error attempts',
        'states':'pair level step method task_cosine field_norm epsilon total_mass max_source_relative_mass source_relative_std target_relative_std residual iterations converged seconds mse_identity_error cuda_peak_bytes scaling_residual newton_iterations',
        'solver':'pair level step method converged relative_residual iterations seconds'}
    if weighted:
        records['transfer']=[]
        headers['transfer']='pair level step method force_ray_error barycentric_ray_error weighted_relative_error damping'
    replays=[]
    for pair in (100,101,124,125,200,201):
        assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<800_000_000
        torch.cuda.reset_peak_memory_stats();x,y,truth,_=transfer(pair)
        observer=partial(observe,field_builder=field_builder,out=out,weighted=weighted)
        factory=lambda:instrumented(pair,x,y,truth,None,None,records,observer=observer)
        result=fit(pair,0,'official',0,folder,loader=transfer,cfg=cfg,factory=factory)
        stem=f'p{pair}_s0_official_w0';ref=ROOT/'paper_revision'/('replication_4d' if pair<200 else 'replication_hf')
        old=json.loads((ref/(stem+'.json')).read_text())
        for key in old:
            if key in result and key!='seconds':assert old[key]==result[key],(pair,key)
        assert np.array_equal(np.loadtxt(ref/(stem+'_trace.txt'),delimiter=',',skiprows=1),np.loadtxt(folder/(stem+'_trace.txt'),delimiter=',',skiprows=1)),pair
        replays.append(pair)
        for name,rows in records.items():
            with (folder/(name+'.txt')).open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(headers[name].split());writer.writerows(rows)
        (folder/'integrity.json').write_text(json.dumps(dict(exact_replays=replays,rows={k:len(v) for k,v in records.items()}),indent=2)+'\n')
    expected=dict(directions=756,finite=6804,states=216,solver=432,transfer=648) if weighted else dict(directions=540,finite=4860,states=216,solver=216)
    assert {k:len(v) for k,v in records.items()}==expected


def main():
    global OUT,MAXITER,NEWTON
    if '--accurate' in sys.argv:OUT=OUT/'precision';MAXITER=65536
    if '--newton' in sys.argv:
        assert '--accurate' not in sys.argv
        OUT=OUT/'newton_512';NEWTON=True
    run(OUT)


if __name__=='__main__':main()
