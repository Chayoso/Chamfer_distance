"""Audit native outcomes against conservative time-bracketed Adam continuations."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'paper_revision/budget_20261015'
METHODS = tuple(f+'_'+m for f in ('cd', 'geo_soft', 'geo_unbalanced', 'feature_soft', 'feature_unbalanced') for m in ('gd', 'lift'))
POLICIES = ('trust', 'cd_trust')
key = lambda r: (r['pair'], r['level'], r['step'])


def read(name):
    return [json.loads(line) for line in (OUT/(name+'.txt')).read_text().splitlines()]


def rescue(candidate_gain, adam_gains, threshold):
    return max(adam_gains) <= threshold and candidate_gain > threshold and candidate_gain-max(adam_gains) > 1e-8


def main():
    states, transfers, finite, comparisons, paths = [read(n) for n in ('states', 'transfer', 'finite', 'comparison', 'adam')]
    expected = {(p, l, t) for p in (100, 101, 124, 125, 200, 201) for l in (0, 4, 8) for t in (0, 25, 49)}
    for name, rows, labels, wanted in [
        ('states', states, lambda r:key(r), expected),
        ('transfer', transfers, lambda r:(key(r), r['method']), {(k,m) for k in expected for m in METHODS}),
        ('finite', finite, lambda r:(key(r), r['method'], r['policy']), {(k,m,p) for k in expected for m in METHODS for p in POLICIES}),
        ('comparison', comparisons, lambda r:(key(r), r['method'], r['policy'], r['repetition']),
         {(k,m,p,i) for k in expected for m in METHODS for p in POLICIES for i in (0,1)})]:
        identities = [labels(r) for r in rows]
        assert len(identities) == len(set(identities)) and set(identities) == wanted, name
    integrity = json.loads((OUT/'integrity.json').read_text())
    assert integrity['exact_replays'] == [100,101,124,125,200,201]
    assert len(integrity['completed_states']) == 54 and {tuple(k) for k in integrity['completed_states']} == expected
    assert all(r['exact_native_first_update'] for r in states)
    assert all(r['solver_residual'] <= 1e-8 and r['transport_residual'] <= 1e-8 for r in transfers)
    assert max(r['mse_identity_error'] for r in finite) < 1e-12
    assert all(r['linearization_error'] <= .1+1e-12 for r in finite)
    assert all(r['objective_change'] <= .1*r['fraction']*r['objective_slope']+1e-12 for r in finite)
    s = {key(r):r for r in states}; t = {(key(r),r['method']):r for r in transfers}
    f = {(key(r),r['method'],r['policy']):r for r in finite}
    c = {(key(r),r['method'],r['policy'],r['repetition']):r for r in comparisons}
    trajectories = {(k,i):[r for r in paths if key(r)==k and r['repetition']==i] for k in expected for i in (0,1)}
    repetition_error = 0.
    for k in expected:
        a,b = trajectories[k,0],trajectories[k,1]
        for path in (a,b):
            assert [r['update'] for r in path] == list(range(len(path)))
            assert path[0]['cd_change'] == 0. and path[0]['risk_change'] == 0.
            assert np.all(np.diff([r['seconds'] for r in path]) > 0)
        repetition_error = max(repetition_error, max(abs(x['cd_change']-y['cd_change']) for x,y in zip(a,b)))
    assert repetition_error == 0.
    within_best = {}; overruns = []
    for row in comparisons:
        path = trajectories[key(row),row['repetition']]
        after = next(r for r in path if r['seconds'] > row['budget'])
        prefix = path[:after['update']+1]; best = min(prefix,key=lambda r:r['cd_change'])
        assert row['after_update'] == after['update'] and row['best_update'] == best['update']
        assert row['best_cd_change'] == best['cd_change'] and row['best_risk_change'] == best['risk_change']
        within = [r for r in prefix if r['seconds'] <= row['budget']]
        within_best[key(row),row['method'],row['policy'],row['repetition']] = min(r['cd_change'] for r in within) if within else None
        overruns.append((row['after_seconds']-row['budget'])/row['budget'])
    previous = list(csv.DictReader((ROOT/'paper_revision/feature_20261015/probe/directions.txt').open()))
    errors=[]
    for r in previous:
        method = 'cd_gd' if r['method']=='parameter_gd' else r['method']
        if method not in METHODS:continue
        k = tuple(int(r[n]) for n in ('pair','level','step'))
        errors.extend(abs(t[k,method][n]-float(r[n])) for n in ('risk_derivative','cd_derivative','output_norm'))
    assert len(errors)==54*9*3 and max(errors)<1e-8
    counts=[]
    for group in ('4DMatch','HF'):
        for subset, selected_steps in [('all',(0,25,49)),('periodic',(25,49)),('last',(49,))]:
            keys = sorted(k for k in expected if (k[0]<200)==(group=='4DMatch') and k[2] in selected_steps)
            for method in METHODS:
                for policy in POLICIES:
                    gains=[-f[k,method,policy]['cd_change']/s[k]['initial_cd'] for k in keys]
                    adam=[[-c[k,method,policy,i]['best_cd_change']/s[k]['initial_cd'] for i in (0,1)] for k in keys]
                    for threshold in (1e-6,1e-4,1e-3):
                        rescued=[rescue(g,a,threshold) for g,a in zip(gains,adam)]
                        rows=[f[k,method,policy] for k in keys]
                        counts.append(dict(group=group,subset=subset,method=method,policy=policy,threshold=threshold,states=len(keys),
                            adam_plateau=sum(max(a)<=threshold for a in adam),candidate_improves=sum(g>threshold for g in gains),
                            both_improve=sum(g>threshold and min(a)>threshold for g,a in zip(gains,adam)),cd_rescues=sum(rescued),
                            cd_rescues_with_gt=sum(flag and r['risk_change'] < -1e-10 for flag,r in zip(rescued,rows)),
                            beats_adam_cd=sum(g-max(a)>1e-8 for g,a in zip(gains,adam)),
                            gt_descent=sum(r['risk_change'] < -1e-10 for r in rows),
                            gt_descent_beats_adam=sum(r['fraction']>0 and r['risk_change'] < -1e-10 and
                                r['risk_change'] < min(c[k,method,policy,i]['best_risk_change'] for i in (0,1))-1e-10 for r,k in zip(rows,keys)),
                            finite_nonnegative_slope_cd_gain=sum(g>threshold and t[k,method]['cd_derivative']>=0 for g,k in zip(gains,keys))))
    alignment=[]
    for group in ('4DMatch','HF'):
        for method in METHODS:
            keys=[k for k in expected if (k[0]<200)==(group=='4DMatch')]
            for threshold in (.1,.25,.5):
                rows=[f[k,method,'trust'] for k in keys]
                aligned=[r['fraction']>0 and r['finite_ray_error'] is not None and r['finite_ray_error']<=threshold for r in rows]
                alignment.append(dict(group=group,method=method,threshold=threshold,states=len(keys),
                    finite_aligned=sum(aligned),aligned_without_gt_gain=sum(a and r['risk_change']>=-1e-10 for a,r in zip(aligned,rows)),
                    median_raw_error=float(np.median([t[k,method]['raw_relative_error'] for k in keys])),
                    median_ray_error=float(np.median([t[k,method]['raw_ray_error'] for k in keys]))))
    for name,rows in [('counts',counts),('alignment',alignment)]:
        with (OUT/(name+'.csv')).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(exact_replays=integrity['exact_replays'],unique_development_states=54,new_scenes=0,
                 native_trial_rows=len(finite),time_bracket_rows=len(comparisons),adam_evaluation_rows=len(paths),
                 max_previous_direction_difference=max(errors),timing_repetition_cd_difference=repetition_error,
                 max_mse_identity_error=max(r['mse_identity_error'] for r in finite),
                 max_native_double_discrepancy=max(r['native_double_discrepancy'] for r in states),
                 budget_seconds=np.percentile([r['budget'] for r in finite],[0,50,100]).tolist(),
                 adam_updates=np.percentile([r['after_update'] for r in comparisons],[0,50,100]).tolist(),
                 minimum_strict_within_budget_adam_gain=min(-v/s[k]['initial_cd'] for (k,m,p,i),v in within_best.items() if v is not None),
                 missing_within_budget_evaluations=sum(v is None for v in within_best.values()),
                 overrun_relative_median_max=np.percentile(overruns,[50,100]).tolist(),
                 ot_ridge_beats_strict_within_budget_adam={policy:sum(r['method']!='cd_lift' and r['method'].endswith('_lift') and r['policy']==policy and
                     all(within_best[key(r),r['method'],policy,i] is not None and r['cd_change'] < within_best[key(r),r['method'],policy,i]-1e-8*s[key(r)]['initial_cd'] for i in (0,1)) for r in finite) for policy in POLICIES},
                 median_raw_transfer={m:float(np.median([r['raw_relative_error'] for r in transfers if r['method']==m])) for m in METHODS},
                 median_ray_transfer={m:float(np.median([r['raw_ray_error'] for r in transfers if r['method']==m])) for m in METHODS},
                 main_counts=[r for r in counts if r['subset']=='all' and r['policy']=='trust' and r['threshold']==1e-4],
                 adoption='unchanged: prior gates failed; no native full-training or new-scene expansion')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    figure(s,t,f,c)
    sources=[OUT/(n+'.txt') for n in ('states','transfer','finite','comparison','adam','PROTOCOL','PROOF')]
    sources += [ROOT/'pipeline/continuation.py',ROOT/'ndp_budget_probe.py',Path(__file__)]
    sources=[p for p in sources if p.name not in ('PROTOCOL.txt','PROOF.txt','NUMERICAL_AMENDMENT.txt') or p.is_file()]
    (OUT/'report_hashes.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2)+'\n')
    print(json.dumps(summary,indent=2))


def figure(states,transfer,finite,comparison):
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    colors={'cd_lift':'#5274aa','geo_soft_lift':'#bc7939','feature_unbalanced_lift':'#31856e'}
    for ax,group in zip(axes[0],('4DMatch','HF')):
        for method,color in colors.items():
            keys=[k for k in states if (k[0]<200)==(group=='4DMatch')]
            x=[max(-comparison[k,method,'trust',i]['best_cd_change']/states[k]['initial_cd'] for i in (0,1)) for k in keys]
            y=[-finite[k,method,'trust']['cd_change']/states[k]['initial_cd'] for k in keys]
            ax.scatter(x,y,s=24,alpha=.65,c=color,label=method)
        ax.plot([0,1],[0,1],'k--',lw=.7);ax.axhline(0,c='gray',lw=.5)
        ax.set_xscale('symlog',linthresh=1e-4);ax.set_yscale('symlog',linthresh=1e-4)
        ax.set(xlabel='Best bracketed Adam relative CD gain',
               ylabel='One native candidate relative CD gain',title=group+' (same-state measured compute)')
        ax.legend(fontsize=8)
    methods=list(colors)+['geo_unbalanced_lift','feature_soft_lift'];x=np.arange(len(methods))
    for offset,field,label in [(-.18,'raw_relative_error','Raw force residual'),(.18,'raw_ray_error','Best positive-ray residual')]:
        values=[np.median([r[field] for (k,m),r in transfer.items() if m==method]) for method in methods]
        axes[1,0].bar(x+offset,values,.36,label=label)
    axes[1,0].set(xticks=x,xticklabels=[m.replace('_','\n') for m in methods],ylabel='Median relative error',title='Ridge transfer: amplitude and direction');axes[1,0].legend(fontsize=8)
    for group,marker,color in [('4DMatch','o','#31856e'),('HF','x','#b84a4a')]:
        keys=[k for k in states if (k[0]<200)==(group=='4DMatch')]
        rows=[finite[k,'feature_unbalanced_lift','trust'] for k in keys]
        axes[1,1].scatter([-r['cd_change']/states[k]['initial_cd'] for r,k in zip(rows,keys)],
            [-r['risk_change']/max(states[k]['initial_risk'],1e-30) for r,k in zip(rows,keys)],label=group,marker=marker,c=color,alpha=.7)
    axes[1,1].axhline(0,c='gray',lw=.6);axes[1,1].axvline(0,c='gray',lw=.6)
    axes[1,1].set(xlabel='Relative observed CD gain',ylabel='Relative GT MSE gain',title='Feature-UOT ridge: observed loss vs registration');axes[1,1].legend()
    fig.suptitle('54 correlated development states / 6 pairs; GT is evaluation-only',fontsize=12)
    fig.tight_layout();fig.savefig(OUT/'budget_results.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    main()
