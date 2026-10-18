"""Position-cost ablation with unchanged gates and matched transfer controls."""
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
OUT=ROOT/'paper_revision/descriptor_20261015'
KINDS=('feature_soft','feature_unbalanced','descriptor_soft','descriptor_unbalanced')
METHODS=tuple(k+'_'+m for k in KINDS for m in ('gd','lift','weighted'))
CANDIDATES=tuple(m for m in METHODS if m.startswith('descriptor_'))
COMPARISON={m:m.replace('descriptor_','feature_',1) for m in CANDIDATES}


def main():
    dr,fr,sr,solvers,tr=[read(OUT/'probe'/n) for n in ('directions.txt','finite.txt','states.txt','solver.txt','transfer.txt')]
    integrity=json.loads((OUT/'probe/integrity.json').read_text())
    verify_coverage(dr,fr,sr,solvers,integrity,KINDS,METHODS,tuple(k+'_'+m for k in KINDS for m in ('lift','weighted')))
    keys=sorted({key(r) for r in sr});labels=[(key(r),r['method']) for r in tr]
    assert len(labels)==len(set(labels))==integrity['rows']['transfer']==648
    assert set(labels)=={(k,m) for k in keys for m in METHODS}
    assert all(np.isfinite(v) for r in dr+fr+sr+solvers+tr for v in r.values() if isinstance(v,float))
    assert all(r['converged'] and r['relative_residual']<=1e-8 for r in solvers)
    assert all(r['converged'] and r['residual']<=1e-8 for r in sr)
    d={(key(r),r['method']):r for r in dr};s={(key(r),r['method']):r for r in sr}
    f={(key(r),r['method'],r['policy'],r['cap']):r for r in fr};t={(key(r),r['method']):r for r in tr}
    old=ROOT/'paper_revision/weighted_20261015/probe';errors=[]
    for filename,lookup,names in [
        ('directions.txt',d,('risk_derivative','objective_derivative','cd_derivative','output_norm')),
        ('finite.txt',f,('risk_change','objective_change','cd_change','fraction','output_norm','linearization_error')),
        ('transfer.txt',t,('force_ray_error','barycentric_ray_error','weighted_relative_error','damping')),
        ('states.txt',s,('task_cosine','field_norm','epsilon','total_mass'))]:
        for r in read(old/filename):
            if not r['method'].startswith('feature_') and r['method'] not in BASES:continue
            label=(key(r),r['method'],r['policy'],r['cap']) if filename=='finite.txt' else (key(r),r['method'])
            errors.extend(abs(lookup[label][n]-r[n]) for n in names)
    assert max(errors)<1e-8
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
    counts,gates=assess(d,f,keys,CANDIDATES,COMPARISON);table(OUT/'counts.txt',counts)
    alignment=[]
    for group in ('4DMatch','HF'):
        for level in (0,4,8):
            ks=[k for k in keys if (k[0]<200)==(group=='4DMatch') and k[1]==level]
            for field in KINDS:
                for transfer in ('gd','lift','weighted'):
                    method=field+'_'+transfer
                    a=np.array([s[k,field]['task_cosine']>0 for k in ks])
                    b=np.array([d[k,method]['risk_derivative'] < -1e-12 for k in ks])
                    alignment.append(dict(group=group,level=level,field=field,transfer=transfer,states=len(ks),
                        point_descent=int(a.sum()),parameter_descent=int(b.sum()),point_descent_lost=int((a&~b).sum()),
                        point_ascent_rescued=int((~a&b).sum()),
                        finite_gt_descent=sum(f[k,method,'trust',1.]['risk_change'] < -1e-12 for k in ks),
                        barycentric_ray_median=float(np.median([t[k,method]['barycentric_ray_error'] for k in ks]))))
    table(OUT/'trajectory_alignment.txt',alignment)
    summary=dict(numerical_gate=True,exact_replays=integrity['exact_replays'],unique_development_states=54,new_scenes=0,
        direction_rows=len(dr),finite_rows=len(fr),solver_rows=len(solvers),transfer_rows=len(tr),
        max_old_control_difference=max(errors),max_mse_identity_error=mse,
        max_transport_residual=max(r['residual'] for r in sr),max_solver_residual=max(r['relative_residual'] for r in solvers),
        gates=gates,advance_to_native=[m for m in CANDIDATES if all(r['primary_gate'] and r['controlled_baseline_gate'] and r['attribution_gate'] for r in gates if r['candidate']==m)],
        primary_cap1=[r for r in counts if not r['periodic'] and r['controller']=='trust' and r['cap']==1 and r['base_policy']=='trust' and r['tolerance']==1e-12])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    figure(counts,alignment)
    paths=list((OUT/'probe').glob('*.txt'))+[OUT/'PROTOCOL.txt',OUT/'PROOF.txt',OUT/'probe/integrity.json',ROOT/'pipeline/capacity.py',ROOT/'ndp_descriptor_probe.py',Path(__file__)]
    paths=[p for p in paths if p.name not in ('PROTOCOL.txt','PROOF.txt','NUMERICAL_AMENDMENT.txt') or p.is_file()]
    (OUT/'report_hashes.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
    print(json.dumps(summary,indent=2))


def figure(counts,alignment):
    fig,axes=plt.subplots(2,2,figsize=(13,9));x=np.arange(len(CANDIDATES))
    for j,group in enumerate(('4DMatch','HF')):
        for shift,base,color,label in [(-.24,'actual_adam','#27886d','vs Adam'),(0,'parameter_gd','#c65353','vs GD'),(.24,None,'#537eac','vs same mixed field')]:
            rows=[next(r for r in counts if r['group']==group and r['candidate']==m and r['baseline']==(base or COMPARISON[m]) and not r['periodic'] and r['controller']=='trust' and r['cap']==1 and r['base_policy']=='trust' and r['tolerance']==1e-12) for m in CANDIDATES]
            axes[0,j].bar(x+shift,[r['conservative_net'] for r in rows],.23,color=color,label=label)
        axes[0,j].axhline(0,c='black',lw=.6);axes[0,j].set(xticks=x,xticklabels=[m.removeprefix('descriptor_').replace('_','\n') for m in CANDIDATES],ylabel='Conservative net rescued states',title=group+': finite cap 1, controlled baselines');axes[0,j].legend(fontsize=8)
        for field,color,style in [('feature_soft','#c89544','--'),('descriptor_soft','#c89544','-'),('feature_unbalanced','#416fa0','--'),('descriptor_unbalanced','#416fa0','-')]:
            rows=[r for r in alignment if r['group']==group and r['field']==field and r['transfer']=='gd']
            axes[1,j].plot([r['level'] for r in rows],[r['point_descent']/r['states'] for r in rows],style+'o',color=color,label=field)
        axes[1,j].set(xticks=[0,4,8],ylim=[0,1],xlabel='Baseline deformation level',ylabel='Fraction of GT-descent point fields',title=group+': before parameter transfer');axes[1,j].legend(fontsize=8)
    fig.suptitle('Remove spatial matching cost; preserve descriptors, transfer and gates')
    fig.text(.5,.01,'54 correlated states / 6 development pairs. GT is evaluation-only. Separate spatial fitting Q; no feature-OT gradient claim.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.96]);fig.savefig(OUT/'descriptor_results.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
