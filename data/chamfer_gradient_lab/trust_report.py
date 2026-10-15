"""Frozen development gates, paired controls and observable-step diagnostics."""
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lift_report import key,table

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'paper_revision/trust_20261013'


def read(path):
    with path.open() as f:
        return [{k:None if v=='' else v if k in ('method','policy','group','baseline') else (v=='True' if v in ('True','False') else float(v))
                 for k,v in r.items()} for r in csv.DictReader(f)]


def main():
    global OUT
    if '--adaptive' in sys.argv:OUT=OUT/'adaptive'
    head='--head' in sys.argv
    if head:OUT=ROOT/'paper_revision/head_20261013'
    full='--full-target' in sys.argv
    if full:OUT=ROOT/'paper_revision/density_ridge_20261013'
    periodic='--periodic' in sys.argv
    if periodic:OUT=OUT/'periodic'
    OUT.mkdir(exist_ok=True)
    source=(OUT.parent if head or full else OUT.parent/'adaptive') if periodic else OUT
    dr,fr,sr=[read(source/'probe'/name) for name in ('directions.txt','finite.txt','solver.txt')]
    if periodic:
        od={(key(r),r['method']):r for r in dr}
        of={(key(r),r['method'],r['policy'],r['cap']):r for r in fr}
        dr=[dict(od[key(r),'actual_adam'],method='blend',adam_weight=1.)
            if r['method']=='blend' and r['step']==0 else r for r in dr]
        fr=[dict(of[key(r),'actual_adam','raw',r['cap']],method='blend',policy=r['policy'],controller_applied=False)
            if r['method']=='blend' and r['step']==0 else r for r in fr]
    d={(key(r),r['method']):r for r in dr}
    f={(key(r),r['method'],r['policy'],r['cap']):r for r in fr}
    keys=sorted({key(r) for r in sr})
    assert len(keys)==54 and len(sr)==54 and len(d)==len(dr)==162 and len(f)==len(fr)==972
    assert all(np.isfinite(v) for r in dr+fr+sr for v in r.values() if isinstance(v,float))
    old=ROOT/'paper_revision/lift_20261013/probe'
    od={(key(r),r['method']):r for r in read(old/'directions.txt')}
    of={(key(r),r['method'],r['fraction']):r for r in read(old/'finite.txt')}
    errors=[]
    for k in keys:
        for method in ('actual_adam','parameter_gd'):
            for name in ('risk_derivative','cd_derivative'):
                errors.append(abs(d[k,method][name]-od[k,method][name]))
            for h in (1/16,.25,1.):
                for name in ('risk_change','cd_change'):
                    a,b=f[k,method,'raw',h][name],of[k,method,h][name]
                    assert np.isclose(a,b,rtol=1e-8,atol=1e-10),(k,method,h,name)
                    errors.append(abs(a-b))
        alpha=d[k,'blend'].get('adam_weight',.4)
        lift_slope=d[k,'blend'].get('lift_risk_derivative',od[k,'ridge_lift']['risk_derivative'])
        expected=alpha*od[k,'actual_adam']['risk_derivative']+(1-alpha)*lift_slope
        assert expected*d[k,'blend']['risk_derivative']>0
    counts=[]
    groups=[('4DMatch',lambda p:p<200),('HF',lambda p:p>=200)]
    groups += [(str(p),lambda q,p=p:p==q) for p in sorted({k[0] for k in keys})]
    for group,pick in groups:
        ks=[k for k in keys if pick(k[0])]
        for method in ('actual_adam','parameter_gd'):
            for policy in ('raw','trust'):
                for cap in (0,1/16,.25,1.):
                    if cap==0 and policy=='trust': continue
                    def values(m,pol):
                        return np.array([d[k,m]['risk_derivative'] if cap==0 else
                                         f[k,m,pol,cap]['risk_change'] for k in ks])
                    base,candidate=values(method,policy),values('blend','trust')
                    for eps in (1e-12,1e-10,1e-8):
                        rescue=int(((base>eps)&(candidate < -eps)).sum())
                        damage=int(((base < -eps)&(candidate>eps)).sum())
                        uncertain=int(((base < -eps)&(abs(candidate)<=eps)).sum())
                        counts.append(dict(group=group,baseline=method,policy=policy,cap=cap,tolerance=eps,
                            states=len(ks),rescued=rescue,damaged=damage,uncertain_damage=uncertain,
                            conservative_net=rescue-damage-uncertain,mean_base=float(base.mean()),
                            mean_candidate=float(candidate.mean()),median_paired_change=float(np.median(candidate-base))))
    table(OUT/'counts.txt',counts)
    gate=[r for r in counts if r['group'] in ('4DMatch','HF') and r['cap'] in (0,.25,1.) and r['tolerance']==1e-12]
    diagnostics=[]
    for group,pick in groups[:2]:
        ks=[k for k in keys if pick(k[0])]
        for method in ('actual_adam','parameter_gd','blend'):
            for policy in ('raw','trust'):
                for cap in (1/16,.25,1.):
                    rows=[f[k,method,policy,cap] for k in ks]
                    accepted=[r for r in rows if r['accepted']]
                    if policy=='trust':
                        assert all(r['cd_change']<=.1*r['fraction']*d[key(r),method]['cd_derivative']+1e-14
                                   and r['linearization_error']<=.1+1e-12 for r in accepted if r.get('controller_applied',True))
                    diagnostics.append(dict(group=group,method=method,policy=policy,cap=cap,states=len(ks),
                        rejected=sum(not r['accepted'] for r in rows),
                        task_harm=sum(r['risk_change']>1e-12 for r in rows),
                        local_descent_but_harm=sum(d[key(r),method]['risk_derivative'] < -1e-12 and r['risk_change']>1e-12 for r in rows),
                        cd_increases=sum(r['cd_change']>1e-12 for r in rows),
                        median_fraction=float(np.median([r['fraction'] for r in rows])),
                        median_attempts=float(np.median([r['attempts'] for r in rows])),
                        max_linearization_error=max(r['linearization_error'] for r in rows)))
    table(OUT/'diagnostics.txt',diagnostics)
    integrity=json.loads((source/'probe/integrity.json').read_text())
    numerical=all(r['converged'] for r in sr) and len(integrity['exact_replays'])==6 and max(errors)<1e-8
    primary=all(r['conservative_net']>0 for r in gate if r['policy']=='raw')
    controlled=all(r['conservative_net']>0 for r in gate if r['policy']=='trust')
    summary=dict(states=54,existing_development_pairs=6,alpha_selected_using_development_labels=.4,
        adaptive_merit_threshold=.8 if head or full or OUT.name in ('adaptive','periodic') else None,
        new_replays=0 if periodic else 6,reused_states=periodic,
        retained_adam_states=sum(d[k,'blend'].get('adam_weight',.4)==1. for k in keys),
        primary_gate=primary,controlled_baseline_gate=controlled,numerical_gate=numerical,
        advance_to_native_fit=primary and controlled and numerical,
        max_control_difference=max(errors),max_cg_residual=max(r['relative_residual'] for r in sr),
        lift_type='translation_head' if head else 'full_parameters',
        candidate_target='full_observed' if full else 'sample2000',
        max_head_velocity_error=max(r.get('head_velocity_error',0.) for r in sr),
        max_mse_identity_error=max(r['mse_identity_error'] for r in sr),gate_rows=gate,
        interpretation='Development only; no independent generalization or optimizer novelty claim.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    paths=list((source/'probe').glob('*.txt'))+[source/'probe/integrity.json',old/'directions.txt',old/'finite.txt']
    (OUT/'input_hashes.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                                                  for p in paths},indent=2)+'\n')
    figure(d,f,keys,gate)
    print(json.dumps({k:v for k,v in summary.items() if k!='gate_rows'},indent=2))
    for r in gate:print(r['group'],r['baseline'],r['policy'],r['cap'],r['rescued'],r['damaged'],r['conservative_net'])


def figure(d,f,keys,gate):
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,group in zip(axes[0],('4DMatch','HF')):
        rows=[r for r in gate if r['group']==group and r['cap']==1.]
        x=np.arange(len(rows))
        ax.bar(x-.18,[r['rescued'] for r in rows],.35,label='Rescued',color='#27886d')
        ax.bar(x+.18,[r['damaged'] for r in rows],.35,label='Damaged',color='#c65353')
        for i,r in enumerate(rows):
            for offset,name in [(-.18,'rescued'),(.18,'damaged')]:
                ax.text(i+offset,r[name]+.1,str(r[name]),ha='center')
        ax.set(xticks=x,xticklabels=[('Adam' if r['baseline']=='actual_adam' else 'GD')+'\n'+r['policy'] for r in rows],
               title=group+': controlled blend versus each baseline',ylabel='State count',ylim=(0,10))
        ax.legend(fontsize=8)
    methods=['actual_adam','parameter_gd','blend']
    fractions=[[f[k,m,'trust',1.]['fraction'] for k in keys] for m in methods]
    axes[1,0].boxplot(fractions)
    axes[1,0].set(xticks=[1,2,3],xticklabels=['Adam','GD','Blend'],ylabel='Accepted fraction',title='Identical controller; cap = 1')
    for m,color in zip(methods,('#357cba','#777777','#c16b24')):
        axes[1,1].scatter([d[k,m]['risk_derivative'] for k in keys],
                          [f[k,m,'trust',1.]['risk_change'] for k in keys],s=17,alpha=.7,label=m,c=color)
    axes[1,1].axhline(0,c='gray',lw=.8);axes[1,1].axvline(0,c='gray',lw=.8)
    axes[1,1].set_xscale('symlog',linthresh=1e-5);axes[1,1].set_yscale('symlog',linthresh=1e-6)
    axes[1,1].set(xlabel='Local task-risk slope',ylabel='Finite task-risk change',title='Observed-loss acceptance does not guarantee task descent')
    axes[1,1].legend(fontsize=8)
    name={'adaptive':'Adaptive','periodic':'Periodic adaptive','head_20261013':'Translation-head',
          'density_ridge_20261013':'Full-target'}.get(OUT.name,'Fixed')
    fig.suptitle(name+' Adam/ridge blend with measured output-linearization control')
    fig.text(.5,.015,'54 correlated states / 6 existing development pairs. Blend coefficient selected using development GT.\n'
             'Runtime selection uses observed CD and predicted/actual output change only.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,.96]);fig.savefig(OUT/'trust_results.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
