"""Separate ridge suppression from correspondence and finite-transfer outcomes."""
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
PHASE=ROOT/'paper_revision/damping_20261018'
OUT=PHASE/'precision'
KINDS=('feature_soft','descriptor_soft')
SUFFIXES=('r01','r001','r0001','r00001')
METHODS=tuple(k+'_'+s for k in KINDS for s in SUFFIXES)
CANDIDATES=tuple(m for m in METHODS if not m.endswith('_r01'))
COMPARISON={m:m.rsplit('_',1)[0]+'_r01' for m in CANDIDATES}


def main():
    files=('directions','finite','states','solver','transfer','identities')
    dr,fr,sr,solvers,tr,identities=[read(OUT/'probe'/(n+'.txt')) for n in files]
    integrity=json.loads((OUT/'probe/integrity.json').read_text())
    verify_coverage(dr,fr,sr,solvers,integrity,KINDS,METHODS,METHODS)
    keys=sorted({key(r) for r in sr})
    for rows,expected in [(tr,{(k,m) for k in keys for m in METHODS}),
                          (identities,set(keys))]:
        labels=[(key(r),r['method']) if 'method' in r else key(r) for r in rows]
        assert len(labels)==len(set(labels))==len(expected) and set(labels)==expected
    assert integrity['rows']['transfer']==432 and integrity['rows']['identities']==54
    assert all(np.isfinite(v) for r in dr+fr+sr+solvers+tr+identities for v in r.values() if isinstance(v,float))
    assert all(r['converged'] and r['relative_residual']<=1e-8 for r in solvers)
    assert all(r['converged'] and r['residual']<=1e-8 for r in sr)
    assert max(r['mse_identity_error'] for r in identities)<1e-12
    d={(key(r),r['method']):r for r in dr};s={(key(r),r['method']):r for r in sr}
    f={(key(r),r['method'],r['policy'],r['cap']):r for r in fr}
    t={(key(r),r['method']):r for r in tr};sol={(key(r),r['method']):r for r in solvers}
    errors=[];old=ROOT/'paper_revision/descriptor_20261015/probe'
    for filename,lookup,names in [
        ('directions',d,('risk_derivative','objective_derivative','cd_derivative','output_norm')),
        ('finite',f,('risk_change','objective_change','cd_change','fraction','linearization_error','output_norm')),
        ('transfer',t,('barycentric_ray_error','weighted_relative_error','damping')),
        ('states',s,('task_cosine','epsilon','total_mass'))]:
        for row in read(old/(filename+'.txt')):
            m=row['method']
            if m not in BASES+KINDS+tuple(k+'_weighted' for k in KINDS):continue
            m=m.removesuffix('_weighted')+'_r01' if m.endswith('_weighted') else m
            label=(key(row),m,row['policy'],row['cap']) if filename=='finite' else (key(row),m)
            errors.extend(abs(row[n]-lookup[label][n]) for n in names)
    assert max(errors)<1e-8,'Original .01 control must reproduce before interpreting damping'
    path=[]
    for group in ('4DMatch','HF'):
        for field in KINDS:
            for suffix in SUFFIXES:
                method=field+'_'+suffix;base=field+'_r01'
                ks=[k for k in keys if (k[0]<200)==(group=='4DMatch')]
                drops=[t[k,base]['weighted_relative_error']-t[k,method]['weighted_relative_error'] for k in ks]
                bounds=[sol[k,base]['relative_residual']+sol[k,method]['relative_residual'] for k in ks]
                path.append(dict(group=group,field=field,suffix=suffix,states=len(ks),
                    raw_error_median=float(np.median([t[k,method]['weighted_relative_error'] for k in ks])),
                    ray_error_median=float(np.median([t[k,method]['barycentric_ray_error'] for k in ks])),
                    raw_error_drop_median=float(np.median(drops)),certified_decreases=sum(a>b+1e-12 for a,b in zip(drops,bounds)),
                    parameter_growth_median=float(np.median([t[k,method]['parameter_norm']/t[k,base]['parameter_norm'] for k in ks])),
                    point_descent=sum(s[k,field]['task_cosine']>0 for k in ks),
                    parameter_descent=sum(d[k,method]['risk_derivative'] < -1e-12 for k in ks),
                    finite_descent=sum(f[k,method,'trust',1.]['risk_change'] < -1e-12 for k in ks),
                    accepted_cap1=sum(f[k,method,'trust',1.]['accepted'] for k in ks),
                    fraction_median=float(np.median([f[k,method,'trust',1.]['fraction'] for k in ks]))))
    violations=[]
    for k in keys:
        for field in KINDS:
            for hi,lo in zip(SUFFIXES,SUFFIXES[1:]):
                a,b=field+'_'+hi,field+'_'+lo
                error=t[k,b]['weighted_relative_error']-t[k,a]['weighted_relative_error']
                bound=sol[k,a]['relative_residual']+sol[k,b]['relative_residual']
                if error>bound+1e-12:violations.append(dict(state=k,field=field,hi=hi,lo=lo,excess=error-bound))
    assert not violations,'Raw residual monotonicity outside numerical error bound'
    for r in fr:
        if r['policy']!='raw' and r['accepted']:
            objective='cd' if r['policy']=='cd_trust' else 'objective'
            assert r[objective+'_change']<=.1*r['fraction']*d[key(r),r['method']][objective+'_derivative']+1e-14
            assert r['linearization_error']<=.1+1e-12
    counts,gates=assess(d,f,keys,CANDIDATES,COMPARISON)
    table(OUT/'counts.txt',counts);table(OUT/'path_summary.txt',path)
    trajectory=[]
    for group in ('4DMatch','HF'):
        for level in (0,4,8):
            ks=[k for k in keys if (k[0]<200)==(group=='4DMatch') and k[1]==level]
            for m in METHODS:
                field=m.rsplit('_',1)[0]
                trajectory.append(dict(group=group,level=level,method=m,states=len(ks),
                    point_descent=sum(s[k,field]['task_cosine']>0 for k in ks),
                    parameter_descent=sum(d[k,m]['risk_derivative'] < -1e-12 for k in ks),
                    finite_descent=sum(f[k,m,'trust',1.]['risk_change'] < -1e-12 for k in ks),
                    weighted_ray_median=float(np.median([t[k,m]['barycentric_ray_error'] for k in ks]))))
    table(OUT/'trajectory_alignment.txt',trajectory)
    summary=dict(numerical_gate=True,initial_partial_failed_solves=1,exact_replays=integrity['exact_replays'],unique_development_states=54,
        new_independent_scenes=0,candidate_full_training_runs=0,rows=integrity['rows'],max_old_control_difference=max(errors),
        max_solver_residual=max(r['relative_residual'] for r in solvers),max_solver_iterations=max(r['iterations'] for r in solvers),
        monotonicity_violations=violations,gates=gates,
        advance_to_native=[m for m in CANDIDATES if all(g['primary_gate'] and g['controlled_baseline_gate'] and g['attribution_gate'] for g in gates if g['candidate']==m)],
        primary_cap1=[r for r in counts if not r['periodic'] and r['controller']=='trust' and r['cap']==1 and r['base_policy']=='trust' and r['tolerance']==1e-12])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');figure(path)
    paths=[OUT/'probe'/(n+'.txt') for n in files]+[PHASE/'PROTOCOL.txt',PHASE/'PROOF.txt',PHASE/'NUMERICAL_AMENDMENT.txt',ROOT/'ndp_damping_probe.py',ROOT/'pipeline/lifting.py',Path(__file__)]
    paths=[p for p in paths if p.name not in ('PROTOCOL.txt','PROOF.txt','NUMERICAL_AMENDMENT.txt') or p.is_file()]
    (OUT/'report_hashes.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
    print(json.dumps(summary,indent=2))


def figure(path):
    fig,axes=plt.subplots(2,3,figsize=(13,8));x=np.arange(4)
    for i,group in enumerate(('4DMatch','HF')):
        for field,color in [('feature_soft','#537eac'),('descriptor_soft','#c89544')]:
            rows=[r for r in path if r['group']==group and r['field']==field]
            axes[i,0].plot(x,[r['raw_error_median'] for r in rows],'-o',color=color,label=field)
            axes[i,1].semilogy(x,[r['parameter_growth_median'] for r in rows],'-o',color=color,label=field)
            axes[i,2].plot(x,[r['parameter_descent']/r['states'] for r in rows],'--o',color=color,label=field+' local')
            axes[i,2].plot(x,[r['finite_descent']/r['states'] for r in rows],'-o',color=color,label=field+' finite')
        for j,title in enumerate(('Weighted fit residual','Raw parameter norm / original','Fraction of GT-descent states')):
            axes[i,j].set(xticks=x,xticklabels=['1e-2','1e-3','1e-4','1e-5'],xlabel='Damping / power estimate',title=group+': '+title)
            axes[i,j].legend(fontsize=7)
        axes[i,2].set_ylim(0,1)
    fig.suptitle('Same proposal and Jacobian; vary ridge regularization only')
    fig.text(.5,.01,'54 correlated development states. Finite cap 1 uses the original Q controller and Adam speed. Residuals do not prove a range obstruction.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.04,1,.96]);fig.savefig(OUT/'damping_results.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
