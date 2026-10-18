"""Full-sample transport gates; separate weighted-plan and common-CD controllers."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from trust_report import read
from lift_report import key,table

ROOT=Path(__file__).resolve().parent
PHASE=ROOT/'paper_revision/capacity_20261013'
OUT=PHASE/'newton_512'
KINDS=('soft','semi','unbalanced','balanced')
METHODS=tuple(k+'_'+m for k in KINDS for m in ('gd','lift'))
BASES=('actual_adam','parameter_gd')


def verify_coverage(dr,fr,sr,solvers,integrity,kinds=KINDS,methods=METHODS,solver_methods=None):
    pairs=(100,101,124,125,200,201)
    keys={(p,l,t) for p in pairs for l in (0,4,8) for t in (0,25,49)}
    expected=[{(k,m) for k in keys for m in BASES+methods},
              {(k,m,p,c) for k in keys for m in BASES+methods for p in ('raw','trust','cd_trust') for c in (1/16,.25,1.)},
              {(k,m) for k in keys for m in kinds},
              {(k,m) for k in keys for m in (solver_methods if solver_methods is not None else tuple(v+'_lift' for v in kinds))}]
    for name,rows,want in zip(('directions','finite','states','solver'),(dr,fr,sr,solvers),expected):
        labels=[(key(r),r['method'],r['policy'],r['cap']) if name=='finite' else (key(r),r['method']) for r in rows]
        assert len(labels)==len(want) and set(labels)==want, 'Incomplete or duplicated '+name+' records'
        assert integrity['rows'][name]==len(labels), 'Integrity count mismatch: '+name
    assert sorted(integrity['exact_replays'])==list(pairs), 'Incomplete or duplicated baseline replay list'


def assess(d,f,keys,methods=METHODS,comparison=None):
    counts=[];gates=[]
    for periodic in (False,True):
        for group,pick in [('4DMatch',lambda p:p<200),('HF',lambda p:p>=200)]:
            ks=[k for k in keys if pick(k[0])]
            for candidate in methods:
                soft=comparison[candidate] if comparison is not None else 'soft_'+candidate.rsplit('_',1)[1]
                for base in list(BASES)+([] if candidate==soft else [soft]):
                    for base_policy in ('raw','trust'):
                        for cap in (0,1/16,.25,1.):
                            if cap==0 and base_policy=='trust':continue
                            for controller in ('trust','cd_trust'):
                                def values(method,policy):
                                    vals=[]
                                    for k in ks:
                                        skip=periodic and k[2]==0
                                        m='actual_adam' if skip and method not in BASES else method
                                        v=d[k,m]['risk_derivative'] if cap==0 else f[k,m,'raw' if skip else policy,cap]['risk_change']
                                        vals.append(v)
                                    return np.array(vals)
                                a,b=values(base,base_policy if base in BASES or base_policy=='raw' else controller),values(candidate,controller)
                                for eps in (1e-12,1e-10,1e-8):
                                    rescue=int(((a>eps)&(b < -eps)).sum());damage=int(((a < -eps)&(b>eps)).sum())
                                    uncertain=int(((a < -eps)&(abs(b)<=eps)).sum())
                                    counts.append(dict(group=group,candidate=candidate,baseline=base,base_policy=base_policy,cap=cap,
                                        controller=controller,periodic=periodic,tolerance=eps,states=len(ks),rescued=rescue,damaged=damage,
                                        uncertain_damage=uncertain,conservative_net=rescue-damage-uncertain,mean_base=float(a.mean()),mean_candidate=float(b.mean())))
        for candidate in methods:
            rows=[r for r in counts if r['candidate']==candidate and r['periodic']==periodic and r['controller']=='trust' and r['tolerance']==1e-12 and r['cap'] in (0,.25,1.)]
            primary=[r for r in rows if r['baseline'] in BASES and r['base_policy']=='raw']
            controlled=[r for r in rows if r['baseline'] in BASES and r['base_policy']=='trust']
            attribution=[r for r in rows if r['baseline'] not in BASES and (r['cap']==0 or r['base_policy']=='trust')]
            gates.append(dict(candidate=candidate,periodic=periodic,primary_gate=all(r['conservative_net']>0 for r in primary),
                controlled_baseline_gate=all(r['conservative_net']>0 for r in controlled),
                attribution_gate=bool(attribution) and all(r['conservative_net']>0 for r in attribution)))
    return counts,gates


def main():
    dr,fr,sr,solvers=[read(OUT/'probe'/n) for n in ('directions.txt','finite.txt','states.txt','solver.txt')]
    integrity=json.loads((OUT/'probe/integrity.json').read_text())
    verify_coverage(dr,fr,sr,solvers,integrity)
    d={(key(r),r['method']):r for r in dr};f={(key(r),r['method'],r['policy'],r['cap']):r for r in fr}
    s={(key(r),r['method']):r for r in sr};keys=sorted({key(r) for r in sr})
    assert len(keys)==54 and len(dr)==len(d)==540 and len(fr)==len(f)==4860 and len(sr)==len(s)==216 and len(solvers)==216
    assert all(np.isfinite(v) for r in dr+fr+sr+solvers for v in r.values() if isinstance(v,float))
    old=ROOT/'paper_revision/trust_20261013/adaptive/probe'
    od={(key(r),r['method']):r for r in read(old/'directions.txt')}
    of={(key(r),r['method'],r['policy'],r['cap']):r for r in read(old/'finite.txt')}
    os={(key(r),r['method']):r for r in read(ROOT/'paper_revision/soft_20261013/precision/probe/directions.txt')}
    errors=[];soft_errors=[]
    for k in keys:
        for m in BASES:
            errors.extend(abs(d[k,m][n]-od[k,m][n]) for n in ('risk_derivative','cd_derivative','output_norm'))
            for cap in (1/16,.25,1.):
                for policy in ('raw','trust','cd_trust'):
                    old_policy='trust' if policy=='cd_trust' else policy
                    errors.extend(abs(f[k,m,policy,cap][n]-of[k,m,old_policy,cap][n]) for n in ('risk_change','cd_change','fraction'))
        for m in ('soft_gd','soft_lift'):
            soft_errors.extend(abs(d[k,m][n]-os[k,m][n]) for n in ('risk_derivative','cd_derivative','output_norm'))
        for m in METHODS:
            assert d[k,m]['objective_derivative']<=1e-14
            assert d[k,m]['output_norm']==0 or abs(d[k,m]['output_norm']-d[k,'actual_adam']['output_norm'])<1e-10
    assert max(errors)<1e-8 and max(soft_errors)<1e-8
    for r in fr:
        if r['policy']!='raw' and r['accepted']:
            field='cd' if r['policy']=='cd_trust' else 'objective'
            assert r[field+'_change']<=.1*r['fraction']*d[key(r),r['method']][field+'_derivative']+1e-14
            assert r['linearization_error']<=.1+1e-12
    counts,gates=assess(d,f,keys);table(OUT/'counts.txt',counts)
    numerical=all(r['converged'] and r['residual']<=1e-8 for r in sr) and all(r['converged'] for r in solvers)
    mse=max(r['mse_identity_error'] for r in sr);numerical=numerical and mse<1e-12 and len(integrity['exact_replays'])==6
    summary=dict(numerical_gate=numerical,exact_replays=integrity['exact_replays'],existing_development_states=54,new_independent_scenes=0,
        initial_partial_numerical_failures=3,max_control_difference=max(errors),max_soft_direction_difference=max(soft_errors),
        max_transport_residual=max(r['residual'] for r in sr),max_ridge_residual=max(r['relative_residual'] for r in solvers),max_mse_identity_error=mse,
        plan_statistics={m:dict(median_seconds=float(np.median([r['seconds'] for r in sr if r['method']==m])),
            max_iterations=max(r['iterations'] for r in sr if r['method']==m),
            newton_refined_states=sum(r['newton_iterations']>0 for r in sr if r['method']==m),
            max_newton_iterations=max(r['newton_iterations'] for r in sr if r['method']==m),
            total_mass_range=[min(r['total_mass'] for r in sr if r['method']==m),max(r['total_mass'] for r in sr if r['method']==m)],
            improved_point_cosine=sum(s[k,m]['task_cosine']>s[k,'soft']['task_cosine'] for k in keys)) for m in KINDS},
        gates=gates,advance_to_periodic_native_fit=[m for m in METHODS if not m.startswith('soft') and numerical and all(r['primary_gate'] and r['controlled_baseline_gate'] and r['attribution_gate'] for r in gates if r['candidate']==m)])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    paths=list((OUT/'probe').glob('*.txt'))+[OUT/'probe/integrity.json',PHASE/'PROTOCOL.txt',PHASE/'NUMERICAL_AMENDMENT.txt',ROOT/'pipeline/capacity.py',ROOT/'ndp_capacity_probe.py',Path(__file__)]
    paths=[p for p in paths if p.name not in ('PROTOCOL.txt','PROOF.txt','NUMERICAL_AMENDMENT.txt') or p.is_file()]
    (OUT/'input_hashes.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
    figure(counts,sr);print(json.dumps(summary,indent=2))
    for r in counts:
        if r['cap'] in (0,1) and r['tolerance']==1e-12 and not r['periodic'] and r['controller']=='trust' and (r['cap']==0 or r['base_policy']=='trust'):
            print(r['group'],r['candidate'],r['baseline'],r['cap'],r['rescued'],r['damaged'],r['conservative_net'])


def figure(counts,states):
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for j,group in enumerate(('4DMatch','HF')):
        ax=axes[0,j];x=np.arange(len(METHODS))
        for shift,base,c,label in [(-.18,'actual_adam','#27886d','vs Adam'),(.18,'parameter_gd','#c65353','vs GD')]:
            rows=[next(r for r in counts if r['group']==group and r['candidate']==m and r['baseline']==base and r['cap']==1 and r['base_policy']=='trust' and r['controller']=='trust' and r['tolerance']==1e-12 and not r['periodic']) for m in METHODS]
            ax.bar(x+shift,[r['conservative_net'] for r in rows],.35,color=c,label=label)
        ax.axhline(0,c='black',lw=.7);ax.set(xticks=x,xticklabels=[m.replace('_','\n') for m in METHODS],title=group+': finite cap1, controlled baselines',ylabel='Conservative net rescued states');ax.legend(fontsize=8);ax.tick_params(axis='x',labelsize=8)
    for m in KINDS:
        rows=[r for r in states if r['method']==m]
        axes[1,0].plot([r['max_source_relative_mass']/r['total_mass'] for r in rows],'.',label=m)
        axes[1,1].semilogy([max(r['seconds'],1e-6) for r in rows],'.',label=m)
    axes[1,0].set(xlabel='Development-state index',ylabel='Largest source share / uniform share',title='Conditional source concentration');axes[1,0].legend(fontsize=8)
    axes[1,1].set(xlabel='Development-state index',ylabel='Plan and force seconds',title='Measured diagnostic cost');axes[1,1].legend(fontsize=8)
    fig.suptitle('Global marginal coupling: same full2000-point samples and bandwidth')
    fig.text(.5,.01,'54 correlated states / six development pairs. Standard OT controls, not full RobOT. Weighted-plan controller; common-CD controls retained in TXT.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.04,1,.96]);fig.savefig(OUT/'capacity_results.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
