"""Complete feature-control audit using the unchanged conservative adoption gates."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from trust_report import read
from lift_report import key,table
from capacity_report import BASES,verify_coverage,assess

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'paper_revision/feature_20261015'
KINDS=('geo_soft','geo_unbalanced','feature_soft','feature_unbalanced')
METHODS=tuple(k+'_'+m for k in KINDS for m in ('gd','lift'))
COMPARISON={m:m.replace('feature_','geo_',1) for m in METHODS}


def main():
    dr,fr,sr,solvers=[read(OUT/'probe'/n) for n in ('directions.txt','finite.txt','states.txt','solver.txt')]
    integrity=json.loads((OUT/'probe/integrity.json').read_text())
    verify_coverage(dr,fr,sr,solvers,integrity,KINDS,METHODS)
    assert all(np.isfinite(v) for r in dr+fr+sr+solvers for v in r.values() if isinstance(v,float))
    d={(key(r),r['method']):r for r in dr};f={(key(r),r['method'],r['policy'],r['cap']):r for r in fr}
    s={(key(r),r['method']):r for r in sr};keys=sorted({key(r) for r in sr})
    features=[json.loads(line) for line in (OUT/'features.txt').read_text().splitlines()]
    assert len(features)==54 and {key(r) for r in features}==set(keys)
    assert all(np.isfinite(r['spacing']) and r['spacing']>0 for r in features)
    old=ROOT/'paper_revision/capacity_20261013/newton_512/probe'
    od={(key(r),r['method']):r for r in read(old/'directions.txt')}
    of={(key(r),r['method'],r['policy'],r['cap']):r for r in read(old/'finite.txt')}
    errors=[]
    for k in keys:
        for m in BASES+tuple(m for m in METHODS if m.startswith('geo_')):
            prior=m.removeprefix('geo_')
            errors.extend(abs(d[k,m][n]-od[k,prior][n]) for n in ('risk_derivative','objective_derivative','cd_derivative','output_norm'))
            for cap in (1/16,.25,1.):
                for policy in ('raw','trust','cd_trust'):
                    errors.extend(abs(f[k,m,policy,cap][n]-of[k,prior,policy,cap][n]) for n in ('risk_change','objective_change','cd_change','fraction'))
        for m in METHODS:
            assert d[k,m]['objective_derivative']<=1e-14
            assert d[k,m]['output_norm']==0 or abs(d[k,m]['output_norm']-d[k,'actual_adam']['output_norm'])<1e-10
    assert max(errors)<1e-8
    for r in fr:
        if r['policy']!='raw' and r['accepted']:
            field='cd' if r['policy']=='cd_trust' else 'objective'
            assert r[field+'_change']<=.1*r['fraction']*d[key(r),r['method']][field+'_derivative']+1e-14
            assert r['linearization_error']<=.1+1e-12
    counts,gates=assess(d,f,keys,METHODS,COMPARISON);table(OUT/'counts.txt',counts)
    mse=max(r['mse_identity_error'] for r in sr)
    numerical=all(r['converged'] and r['residual']<=1e-8 for r in sr) and all(r['converged'] for r in solvers) and mse<1e-12
    summary=dict(numerical_gate=numerical,exact_replays=integrity['exact_replays'],existing_development_states=54,new_independent_scenes=0,
        max_control_difference=max(errors),max_transport_residual=max(r['residual'] for r in sr),
        max_ridge_residual=max(r['relative_residual'] for r in solvers),max_mse_identity_error=mse,
        feature_extraction=dict(open3d=sorted({r['open3d'] for r in features}),median_seconds=float(np.median([r['seconds'] for r in features])),
            max_source_zero_rows=max(r['source_zero_features'] for r in features),max_target_zero_rows=max(r['target_zero_features'] for r in features)),
        feature_point_cosine_improvements={m:sum(s[k,m]['task_cosine']>s[k,m.replace('feature_','geo_',1)]['task_cosine'] for k in keys) for m in KINDS if m.startswith('feature_')},
        gates=gates,advance_to_periodic_native_fit=[m for m in METHODS if m.startswith('feature_') and numerical and all(r['primary_gate'] and r['controlled_baseline_gate'] and r['attribution_gate'] for r in gates if r['candidate']==m)])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'failure_decomposition.json').write_text(json.dumps(decomposition(d,sr),indent=2)+'\n')
    paths=list((OUT/'probe').glob('*.txt'))+[OUT/'probe/integrity.json',OUT/'features.txt',OUT/'PROTOCOL.txt',ROOT/'pipeline/features.py',ROOT/'pipeline/capacity.py',ROOT/'ndp_capacity_probe.py',ROOT/'ndp_feature_probe.py',ROOT/'capacity_report.py',Path(__file__)]
    paths=[p for p in paths if p.name not in ('PROTOCOL.txt','PROOF.txt','NUMERICAL_AMENDMENT.txt') or p.is_file()]
    (OUT/'input_hashes.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
    figure(counts,sr);print(json.dumps(summary,indent=2))
    for r in counts:
        if r['candidate'].startswith('feature_') and r['cap'] in (0,1) and r['tolerance']==1e-12 and not r['periodic'] and r['controller']=='trust' and (r['cap']==0 or r['base_policy']=='trust'):
            print(r['group'],r['candidate'],r['baseline'],r['cap'],r['rescued'],r['damaged'],r['uncertain_damage'],r['conservative_net'])


def decomposition(d,states):
    rows=[]
    for group in ('4DMatch','HF'):
        for method in KINDS:
            chosen=[r for r in states if r['method']==method and (r['pair']<200)==(group=='4DMatch')]
            row=dict(group=group,method=method,states=len(chosen),point_descent=sum(r['task_cosine']>1e-12 for r in chosen))
            for lift in ('gd','lift'):
                row[lift+'_descent']=sum(d[key(r),method+'_'+lift]['risk_derivative'] < -1e-12 for r in chosen)
                row[lift+'_positive_point_to_bad_parameter']=sum(r['task_cosine']>1e-12 and d[key(r),method+'_'+lift]['risk_derivative']>1e-12 for r in chosen)
            rows.append(row)
    return rows


def figure(counts,states):
    fig,axes=plt.subplots(2,2,figsize=(13,8));candidates=[m for m in METHODS if m.startswith('feature_')]
    for j,group in enumerate(('4DMatch','HF')):
        ax=axes[0,j];x=np.arange(len(candidates))
        for shift,base,color,label in [(-.24,'actual_adam','#27886d','vs Adam'),(0,'parameter_gd','#c65353','vs GD'),(.24,None,'#537eac','vs same geo matching')]:
            rows=[next(r for r in counts if r['group']==group and r['candidate']==m and r['baseline']==(base or COMPARISON[m]) and r['cap']==1 and r['base_policy']=='trust' and r['controller']=='trust' and r['tolerance']==1e-12 and not r['periodic']) for m in candidates]
            ax.bar(x+shift,[r['conservative_net'] for r in rows],.23,color=color,label=label)
        ax.axhline(0,c='black',lw=.7);ax.set(xticks=x,xticklabels=[m.removeprefix('feature_').replace('_','\n') for m in candidates],title=group+': finite cap1, controlled baselines',ylabel='Conservative net rescued states');ax.legend(fontsize=8)
    for m in KINDS:
        rows=[r for r in states if r['method']==m]
        axes[1,0].plot([r['max_source_relative_mass']/r['total_mass'] for r in rows],'.',label=m)
        axes[1,1].semilogy([r['seconds'] for r in rows],'.',label=m)
    axes[1,0].set(xlabel='Development-state index',ylabel='Largest source share / uniform share',title='Conditional source concentration');axes[1,0].legend(fontsize=8)
    axes[1,1].set(xlabel='Development-state index',ylabel='Field seconds (feature extraction included)',title='Measured diagnostic cost');axes[1,1].legend(fontsize=8)
    fig.suptitle('Observable FPFH likelihood: fixed coordinate/marginal/gradient controls')
    fig.text(.5,.01,'54 correlated states / six development pairs. No learned features or GT in updates. Not a full RobOT or PointTruss reproduction.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.04,1,.96]);fig.savefig(OUT/'feature_results.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
