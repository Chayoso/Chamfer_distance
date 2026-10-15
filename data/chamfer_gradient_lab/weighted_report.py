"""Unchanged adoption gates for mass-weighted versus force-space projection."""
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
OUT=ROOT/'paper_revision/weighted_20261015'
KINDS=('geo_soft','geo_unbalanced','feature_soft','feature_unbalanced')
METHODS=tuple(k+'_'+m for k in KINDS for m in ('gd','lift','weighted'))
CANDIDATES=tuple(k+'_weighted' for k in KINDS)
COMPARISONS={control:{m:m.removesuffix('_weighted')+'_'+control for m in CANDIDATES} for control in ('lift','gd')}


def main():
    dr,fr,sr,solvers,tr=[read(OUT/'probe'/n) for n in ('directions.txt','finite.txt','states.txt','solver.txt','transfer.txt')]
    integrity=json.loads((OUT/'probe/integrity.json').read_text())
    solver_methods=tuple(k+'_'+m for k in KINDS for m in ('lift','weighted'))
    verify_coverage(dr,fr,sr,solvers,integrity,KINDS,METHODS,solver_methods)
    expected={(p,l,s) for p in (100,101,124,125,200,201) for l in (0,4,8) for s in (0,25,49)}
    labels=[(key(r),r['method']) for r in tr]
    assert len(labels)==len(set(labels))==648 and set(labels)=={(k,m) for k in expected for m in METHODS}
    assert integrity['rows']['transfer']==648
    assert all(np.isfinite(v) for r in dr+fr+sr+solvers+tr for v in r.values() if isinstance(v,float))
    assert all(r['converged'] and r['relative_residual']<=1e-8 for r in solvers)
    assert all(r['converged'] and r['residual']<=1e-8 for r in sr)
    d={(key(r),r['method']):r for r in dr}
    f={(key(r),r['method'],r['policy'],r['cap']):r for r in fr}
    t={(key(r),r['method']):r for r in tr};keys=sorted(expected)
    old=ROOT/'paper_revision/feature_20261015/probe';errors=[]
    for r in read(old/'directions.txt'):
        got=d[key(r),r['method']]
        errors.extend(abs(got[n]-r[n]) for n in ('risk_derivative','objective_derivative','cd_derivative','output_norm'))
    for r in read(old/'finite.txt'):
        got=f[key(r),r['method'],r['policy'],r['cap']]
        errors.extend(abs(got[n]-r[n]) for n in ('risk_change','objective_change','cd_change','fraction','output_norm','linearization_error'))
    assert max(errors)<1e-8
    old_transfer=[json.loads(line) for line in (ROOT/'paper_revision/budget_20261015/transfer.txt').read_text().splitlines()]
    transfer_errors=[abs(t[key(r),r['method']]['force_ray_error']-r['raw_ray_error']) for r in old_transfer if r['method'] in METHODS]
    assert len(transfer_errors)==432 and max(transfer_errors)<1e-10
    for k in keys:
        for method in METHODS:
            assert d[k,method]['objective_derivative']<=1e-14
            assert d[k,method]['output_norm']==0 or abs(d[k,method]['output_norm']-d[k,'actual_adam']['output_norm'])<1e-10
    for r in fr:
        if r['policy']!='raw' and r['accepted']:
            field='cd' if r['policy']=='cd_trust' else 'objective'
            assert r[field+'_change']<=.1*r['fraction']*d[key(r),r['method']][field+'_derivative']+1e-14
            assert r['linearization_error']<=.1+1e-12
    mse=max(r['mse_identity_error'] for r in sr);assert mse<1e-12
    counts=[];gates=[]
    for control,comparison in COMPARISONS.items():
        rows,decisions=assess(d,f,keys,CANDIDATES,comparison)
        counts.extend(dict(r,attribution_control=control) for r in rows)
        gates.extend(dict(r,attribution_control=control) for r in decisions)
    table(OUT/'counts.txt',counts)
    transfer=[]
    for group in ('4DMatch','HF'):
        for field in KINDS:
            ks=[k for k in keys if (k[0]<200)==(group=='4DMatch')]
            for method in ('gd','lift','weighted'):
                rows=[t[k,field+'_'+method] for k in ks]
                transfer.append(dict(group=group,field=field,method=method,states=len(ks),
                    force_ray_median=float(np.median([r['force_ray_error'] for r in rows])),
                    barycentric_ray_median=float(np.median([r['barycentric_ray_error'] for r in rows])),
                    raw_weighted_error_median=float(np.median([r['weighted_relative_error'] for r in rows]))))
    table(OUT/'transfer_summary.txt',transfer)
    summary=dict(numerical_gate=True,exact_replays=integrity['exact_replays'],unique_development_states=54,new_scenes=0,
        direction_rows=len(dr),finite_rows=len(fr),solver_rows=len(solvers),transfer_rows=len(tr),
        max_old_control_difference=max(errors),max_old_transfer_difference=max(transfer_errors),max_mse_identity_error=mse,
        max_transport_residual=max(r['residual'] for r in sr),max_solver_residual=max(r['relative_residual'] for r in solvers),
        weighted_solver_seconds=np.percentile([r['seconds'] for r in solvers if r['method'].endswith('_weighted')],[0,50,100]).tolist(),
        gates=gates,advance_to_native=[m for m in CANDIDATES if all(r['primary_gate'] and r['controlled_baseline_gate'] and r['attribution_gate'] for r in gates if r['candidate']==m)],
        primary_cap1=[r for r in counts if r['attribution_control']=='lift' and not r['periodic'] and r['controller']=='trust' and r['cap']==1 and r['base_policy']=='trust' and r['tolerance']==1e-12],
        transfer_summary=transfer)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    figure(counts,transfer)
    paths=list((OUT/'probe').glob('*.txt'))+[OUT/'PROTOCOL.txt',OUT/'PROOF.txt',OUT/'probe/integrity.json',ROOT/'pipeline/lifting.py',ROOT/'ndp_weighted_probe.py',ROOT/'ndp_capacity_probe.py',ROOT/'ndp_feature_probe.py',ROOT/'capacity_report.py',Path(__file__)]
    (OUT/'report_hashes.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
    print(json.dumps(summary,indent=2))


def figure(counts,transfer):
    fig,axes=plt.subplots(2,2,figsize=(13,9));x=np.arange(4)
    for j,group in enumerate(('4DMatch','HF')):
        for shift,base,color,label in [(-.24,'actual_adam','#27886d','vs Adam'),(0,'parameter_gd','#c65353','vs GD'),(.24,None,'#537eac','vs same force ridge')]:
            rows=[next(r for r in counts if r['group']==group and r['candidate']==m and r['baseline']==(base or COMPARISONS['lift'][m]) and
                       r['attribution_control']=='lift' and not r['periodic'] and r['controller']=='trust' and r['cap']==1 and r['base_policy']=='trust' and r['tolerance']==1e-12) for m in CANDIDATES]
            axes[0,j].bar(x+shift,[r['conservative_net'] for r in rows],.23,color=color,label=label)
        axes[0,j].axhline(0,c='black',lw=.6);axes[0,j].set(xticks=x,xticklabels=[m.replace('_','\n') for m in KINDS],ylabel='Conservative net rescued states',title=group+': finite cap 1, controlled baselines');axes[0,j].legend(fontsize=8)
        for shift,method,color in [(-.18,'lift','#c89544'),(.18,'weighted','#416fa0')]:
            rows=[next(r for r in transfer if r['group']==group and r['field']==field and r['method']==method) for field in KINDS]
            axes[1,j].bar(x+shift,[r['barycentric_ray_median'] for r in rows],.35,color=color,label=method)
        axes[1,j].set(xticks=x,xticklabels=[m.replace('_','\n') for m in KINDS],ylabel='Median mass-weighted ray error',title=group+': displacement alignment');axes[1,j].legend(fontsize=8)
    fig.suptitle('Same correspondence; force ridge versus weighted displacement fitting')
    fig.text(.5,.01,'54 correlated development states / 6 pairs. Standard weighted least squares. GT is evaluation-only; original adoption gates retained.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.96]);fig.savefig(OUT/'weighted_results.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
