"""Report the frozen ridge-lift diagnostic, including failures and solver error."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'paper_revision/lift_20261013'


def read(path):
    with path.open() as f:
        return [{k:v if k in ('method','optimizer') else (v=='True' if v in ('True','False') else float(v))
                 for k,v in row.items()} for row in csv.DictReader(f)]


def key(r): return tuple(int(r[k]) for k in ('pair','level','step'))


def table(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def numerical_checks(primary):
    report=[]
    for folder in ('numerical_probe','fine_numerical_probe'):
        rows=read(OUT/folder/'derivatives.txt')
        assert len(rows)==108
        for r in rows:
            assert r['predicted_risk']==primary[key(r),'ridge_lift']['risk_derivative']
            assert r['predicted_cd']==primary[key(r),'ridge_lift']['cd_derivative']
        for h in sorted({r['fraction'] for r in rows},reverse=True):
            chosen=[r for r in rows if r['fraction']==h]
            assert len(chosen)==54
            for name in ('risk','cd'):
                a=np.array([r['predicted_'+name] for r in chosen])
                b=np.array([r['finite_'+name] for r in chosen])
                err=abs(a-b)
                report.append(dict(folder=folder,fraction=h,quantity=name,states=len(a),
                    outside_tolerance=int((err>2e-8+1e-3*abs(a)).sum()),
                    sign_disagreements=int((a*b<0).sum()),max_absolute_error=float(err.max()),
                    max_relative_error_floor_1e8=float((err/np.maximum(abs(a),1e-8)).max())))
    table(OUT/'numerical_derivative_validation.txt',report)
    return report


def main():
    drows = read(OUT/'probe/directions.txt')
    frows = read(OUT/'probe/finite.txt')
    solver = read(OUT/'probe/solver.txt')
    d = {(key(r),r['method']):r for r in drows}
    f = {(key(r),r['method'],r['fraction']):r for r in frows}
    keys = sorted({key(r) for r in solver})
    assert len(keys)==54 and len(solver)==54 and len(d)==len(drows)==216 and len(f)==len(frows)==486
    for rows in (drows,frows,solver):
        assert all(np.isfinite(v) for r in rows for v in r.values() if isinstance(v,float))
    old = ROOT/'paper_revision/directions_20261013/probe'
    od = {(key(r),r['optimizer']):r for r in read(old/'directions.txt') if r['method']=='official'}
    of = {(key(r),r['optimizer'],r['fraction']):r for r in read(old/'finite.txt') if r['method']=='official'}
    control_errors = []
    for k in keys:
        for method,optimizer in [('actual_adam','actual_adam'),('parameter_gd','gd')]:
            for name in ('risk_derivative','cd_derivative'):
                a,b=d[k,method][name],od[k,optimizer][name]
                assert np.isclose(a,b,rtol=1e-8,atol=1e-10),(k,method,name)
                control_errors.append(abs(a-b))
            for fraction in (1/16,1/4,1.):
                for name in ('risk_change','cd_change'):
                    a,b=f[k,method,fraction][name],of[k,optimizer,fraction][name]
                    assert np.isclose(a,b,rtol=1e-8,atol=1e-10),(k,method,name,fraction)
                    control_errors.append(abs(a-b))
    groups = [('all',lambda p:True),('4DMatch',lambda p:p<200),('HF',lambda p:p>=200)]
    groups += [(str(p),lambda k,p=p:k==p) for p in sorted({k[0] for k in keys})]
    rows = []
    for group,pick in groups:
        selected = [k for k in keys if pick(k[0])]
        for method in ('actual_adam','parameter_gd'):
            for fraction in (0,1/16,1/4,1.):
                values = lambda m:np.array([d[k,m]['risk_derivative'] if fraction==0 else
                                            f[k,m,fraction]['risk_change'] for k in selected])
                base,candidate = values(method),values('ridge_lift')
                bound = np.array([d[k,'ridge_lift']['slope_error_bound'] if fraction==0 else 0 for k in selected])
                for eps in (1e-12,1e-10,1e-8):
                    good,bad = candidate < -eps-bound,candidate > eps+bound
                    uncertain = (~good)&(~bad)
                    rescue = int(((base>eps)&good).sum())
                    damage = int(((base < -eps)&bad).sum())
                    possible_damage = int(((base < -eps)&uncertain).sum())
                    rows.append(dict(group=group,baseline=method,fraction=fraction,tolerance=eps,
                        states=len(selected),baseline_ascent=int((base>eps).sum()),lift_ascent=int(bad.sum()),
                        rescued=rescue,damaged=damage,uncertain=int(uncertain.sum()),
                        conservative_net=rescue-damage-possible_damage,mean_base=float(base.mean()),
                        mean_lift=float(candidate.mean()),median_paired_change=float(np.median(candidate-base))))
    table(OUT/'counts.txt',rows)
    gate_rows = [r for r in rows if r['group'] in ('4DMatch','HF') and r['fraction'] in (0,.25,1)
                 and r['tolerance']==1e-12]
    gate = all(r['conservative_net']>0 for r in gate_rows)
    numerical = all(r['converged'] for r in solver) and all(d[k,'ridge_lift']['error_bound_valid'] for k in keys)
    descents = sum(d[k,'ridge_lift']['cd_derivative'] < 0 for k in keys)
    assert descents==54
    diagnostics = dict(solver_converged=sum(r['converged'] for r in solver),states=54,
        max_relative_residual=max(r['relative_residual'] for r in solver),
        max_control_absolute_difference=max(control_errors),
        max_mse_identity_error=max(r['mse_identity_error'] for r in solver),
        cg_iterations_range=[min(r['iterations'] for r in solver),max(r['iterations'] for r in solver)],
        median_gd_seconds=float(np.median([r['gd_seconds'] for r in solver])),
        median_lift_seconds=float(np.median([r['lift_seconds'] for r in solver])),
        median_paired_time_ratio=float(np.median([r['lift_seconds']/r['gd_seconds'] for r in solver])),
        max_cuda_allocated_bytes=max(r['cuda_peak_bytes'] for r in solver),
        max_slope_error_bound=max(d[k,'ridge_lift']['slope_error_bound'] for k in keys),
        ridge_cd_local_descents=descents,
        max_lift_parameter_norm=max(d[k,'ridge_lift']['parameter_norm'] for k in keys))
    reversals = []
    for group,pick in groups[:3]:
        ks=[k for k in keys if pick(k[0])]
        for method in ('actual_adam','parameter_gd','ridge_lift'):
            local=np.array([d[k,method]['risk_derivative'] for k in ks])
            for fraction in (1/16,.25,1.):
                finite=np.array([f[k,method,fraction]['risk_change'] for k in ks])
                reversals.append(dict(group=group,method=method,fraction=fraction,
                    descent_but_harm=int(((local < -1e-12)&(finite>1e-12)).sum()),
                    ascent_but_improved=int(((local>1e-12)&(finite < -1e-12)).sum()),
                    finite_cd_increases=sum(f[k,method,fraction]['cd_change']>1e-12 for k in ks),
                    median_linearization_error=float(np.median([f[k,method,fraction]['relative_linearization_error'] for k in ks])),
                    max_linearization_error=max(f[k,method,fraction]['relative_linearization_error'] for k in ks)))
    table(OUT/'finite_reversals.txt',reversals)
    decomposition=[]
    for k in keys:
        n=2000
        speed=d[k,'actual_adam']['raw_output_norm']
        slope=d[k,'ridge_lift']['risk_derivative']
        for fraction in (1/16,.25,1.):
            linear=fraction*slope+fraction**2*speed**2/n
            actual=f[k,'ridge_lift',fraction]['risk_change']
            decomposition.append(dict(pair=k[0],level=k[1],step=k[2],fraction=fraction,
                local_slope=slope,linearized_model_change=linear,actual_change=actual,
                nonlinear_contribution=actual-linear))
    table(OUT/'ridge_risk_decomposition.txt',decomposition)
    checks=numerical_checks(d)
    fine_ok=all(r['outside_tolerance']==0 for r in checks if r['folder']=='fine_numerical_probe')
    coarse_ok=all(r['outside_tolerance']==0 for r in checks if r['fraction']==1e-6)
    summary=dict(diagnostics=diagnostics,solver_and_replay_gate=numerical,direction_and_finite_gate=gate,
        coarse_small_step_check_pass=coarse_ok,fine_native_derivative_check_pass=fine_ok,
        advance_to_native_fit=numerical and gate and fine_ok,gate_rows=gate_rows,
        aggregate=[r for r in rows if r['group']=='all' and r['tolerance']==1e-12],
        native_derivative_checks=checks,
        interpretation='Previously used development states; no endpoint or new-scene result.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    inputs=[OUT/'probe'/name for name in ('directions.txt','finite.txt','solver.txt','integrity.json')]
    inputs += [old/'directions.txt',old/'finite.txt']
    inputs += [OUT/folder/'derivatives.txt' for folder in ('numerical_probe','fine_numerical_probe')]
    (OUT/'input_hashes.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                                                  for p in inputs},indent=2)+'\n')
    figure(d,f,keys,rows)
    print(json.dumps(summary,indent=2))


def figure(d,f,keys,rows):
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    methods=['actual_adam','parameter_gd','ridge_lift','particle_gd']
    for i,(group,pick) in enumerate([('4DMatch',lambda p:p<200),('HF',lambda p:p>=200)]):
        ks=[k for k in keys if pick(k[0])]
        counts=[sum(d[k,m]['risk_derivative']>1e-12 for k in ks) for m in methods]
        bars=axes[0,0].bar(np.arange(4)+i*.35,100*np.array(counts)/len(ks),.33,label=group)
        for bar,n in zip(bars,counts):
            axes[0,0].text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f'{n}/{len(ks)}',ha='center',fontsize=8)
    axes[0,0].set(xticks=np.arange(4)+.175,xticklabels=['Adam','Parameter GD','Ridge lift','Particle GD*'],
                  ylabel='Task-ascent directions (%)',ylim=(0,80),title='Local directions at matched output speed')
    axes[0,0].legend(fontsize=8)
    for group,pick,color in [('4DMatch',lambda p:p<200,'#357cba'),('HF',lambda p:p>=200,'#d98023')]:
        ks=[k for k in keys if pick(k[0])]
        axes[0,1].scatter([d[k,'parameter_gd']['risk_derivative'] for k in ks],
                          [d[k,'ridge_lift']['risk_derivative'] for k in ks],label=group,c=color,s=22)
    axes[0,1].axhline(0,color='gray',lw=.8);axes[0,1].axvline(0,color='gray',lw=.8)
    axes[0,1].set_xscale('symlog',linthresh=1e-4);axes[0,1].set_yscale('symlog',linthresh=1e-4)
    axes[0,1].set(xlabel='Parameter-GD risk slope',ylabel='Ridge-lift risk slope',title='Lower-right: local rescue; upper-left: damage')
    selected=[r for r in rows if r['group'] in ('4DMatch','HF') and r['fraction']==1 and r['tolerance']==1e-12]
    x=np.arange(len(selected))
    axes[1,0].bar(x-.18,[r['rescued'] for r in selected],.34,color='#27886d',label='Rescued')
    axes[1,0].bar(x+.18,[r['damaged'] for r in selected],.34,color='#c65353',label='Damaged')
    for i,r in enumerate(selected):
        for offset,keyname in [(-.18,'rescued'),(.18,'damaged')]:
            axes[1,0].text(i+offset,r[keyname]+.15,str(r[keyname]),ha='center',fontsize=9)
    axes[1,0].set(xticks=x,xticklabels=[r['group']+'\nvs '+('Adam' if r['baseline']=='actual_adam' else 'GD') for r in selected],
                  ylabel='State count',title='Full finite step: ridge lift versus each control')
    axes[1,0].set_ylim(0,max(2,max(max(r['rescued'],r['damaged']) for r in selected)+2))
    axes[1,0].legend(fontsize=8)
    values=[[max(1e-12,f[k,m,1.]['relative_linearization_error']) for k in keys] for m in methods[:3]]
    axes[1,1].boxplot(values,showfliers=True)
    axes[1,1].set_yscale('log')
    axes[1,1].set(xticks=[1,2,3],xticklabels=['Adam','Parameter GD','Ridge lift'],
                  ylabel='||actual output change - J delta|| / ||J delta||',title='Full-step nonlinearity is measured, not assumed small')
    fig.suptitle('Ridge lifting: the same Chamfer gradient through a different parameter metric',fontsize=13)
    fig.text(.5,.015,'54 correlated states / 6 existing pairs. GT is evaluation only. Fixed damping ratio 0.01.\n'
             '*Particle GD may be unrealizable by NDP. No full-fit or new-scene claim.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,.96]);fig.savefig(OUT/'ridge_lift_results.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    main()
