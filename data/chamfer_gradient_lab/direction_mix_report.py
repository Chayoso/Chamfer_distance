"""Predeclared observable mixture; certify tangent signs before any new replay."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fields_report import PAIRS, coverage, net_counts
from pipeline.direction_mix import mix_fraction, normalized_slope_interval, interval_sign, feasible_task_range

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'paper_revision/mix_20261018'
SOURCE = ROOT/'paper_revision/fgr_20261018/neural'
TOLERANCES = (1e-12,1e-10,1e-8)


def read(name):
    return [json.loads(s) for s in (SOURCE/(name+'.txt')).read_text().splitlines()]


def choices():
    # Deliberately restrict the selection table to observed quantities.
    observed = {(r['pair'],r['level'],r['method']):{k:r[k] for k in
        ('cd_derivative','motion_bound','output_norm')} for r in read('directions')}
    norms = {(r['pair'],r['level']):r['force_norm'] for r in read('fields') if r['method']=='cd'}
    selected = []
    for pair in PAIRS:
        for level in range(9):
            a,c = [observed[pair,level,m] for m in ('fgr_ridge','cd_gd')]
            assert a['motion_bound']==c['motion_bound']
            for r in (a,c): assert abs(r['output_norm']/r['motion_bound']-1)<1e-10
            sa,sc = a['cd_derivative'],c['cd_derivative']
            beta = mix_fraction(sa,sc)
            selected.append(dict(pair=pair,level=level,beta=beta,retain=.5,cd_a=sa,cd_c=sc,
                mixed_cd=(1-beta)*sa+beta*sc,bound=a['motion_bound'],cd_norm=norms[pair,level]))
    document = dict(choices=selected,source_hashes={n:hashlib.sha256((SOURCE/(n+'.txt')).read_bytes()).hexdigest()
        for n in ('directions','fields')},uses_task_for_selection=False)
    serialized=json.dumps(document,indent=2,allow_nan=False)+'\n'
    path=OUT/'choices.json'
    if path.exists(): assert path.read_text()==serialized, 'Frozen choices changed'
    else: path.write_text(serialized)
    return selected


def main():
    selected = choices()  # All choices persist before task slopes enter evaluation.
    states={(p,l) for p in PAIRS for l in range(9)}
    coverage(selected,('pair','level'),states)
    d={(r['pair'],r['level'],r['method']):r for r in read('directions')}
    results=[]
    for r in selected:
        p,l=r['pair'],r['level']
        ta,tc=[d[p,l,m]['risk_derivative'] for m in ('fgr_ridge','cd_gd')]
        interval=normalized_slope_interval(ta,tc,r['beta'],r['cd_a'],r['cd_c'],r['bound'],r['cd_norm'])
        assert r['mixed_cd'] <= .5*r['cd_c']+1e-12*abs(r['cd_c'])
        result=dict(**r,task_a=ta,task_c=tc,normalized_task_interval=interval,
            classifications={str(e):interval_sign(interval,e) for e in TOLERANCES},
            existential=[dict(retain=eta,**feasible_task_range(ta,tc,r['cd_a'],r['cd_c'],eta)) for eta in (0.,.5)])
        results.append(result)
    counts,groups,oracle=[],[],[]
    for group,pairs in (('4DMatch',PAIRS[:4]),('HF',PAIRS[4:])):
        rows=[r for r in results if r['pair'] in pairs]
        for eps in TOLERANCES:
            signs=[r['classifications'][str(eps)] for r in rows]
            for baseline in ('adam','cd_gd','cd_ridge','fgr_ridge'):
                a=[d[r['pair'],r['level'],baseline]['risk_derivative'] for r in rows]
                count=net_counts(a,signs,eps) if None not in signs else None
                counts.append(dict(group=group,baseline=baseline,tolerance=eps,counts=count))
        signs=[r['classifications'][str(1e-12)] for r in rows]
        groups.append(dict(group=group,states=len(rows),mixed_descents=signs.count(-1),ambiguous=signs.count(None),
            baseline_descents={m:sum(d[r['pair'],r['level'],m]['risk_derivative'] < -1e-12 for r in rows)
                               for m in ('adam','cd_gd','cd_ridge','fgr_ridge')},
            active_corrections=sum(r['beta']>0 for r in rows),original_cd_ascents=sum(r['cd_a']>=0 for r in rows),
            median_beta=float(np.median([r['beta'] for r in rows])),
            helpful_lost=sum(r['task_a'] < -1e-12 and s in (0,1) for r,s in zip(rows,signs)),
            harmful_rescued=sum(r['task_a'] > 1e-12 and s==-1 for r,s in zip(rows,signs))))
        for eta in (0.,.5):
            minima=[next(v['minimum'] for v in r['existential'] if v['retain']==eta) for r in rows]
            optimistic=[1 if v>1e-12 else -1 for v in minima]
            oracle.append(dict(group=group,retain=eta,robust_existence=sum(v < -1e-12 for v in minima),
                robust_impossibility=sum(v > 1e-12 for v in minima),uncertain=sum(abs(v)<=1e-12 for v in minima),
                optimistic_net_upper={m:net_counts([d[r['pair'],r['level'],m]['risk_derivative'] for r in rows],
                    optimistic,1e-12)['net'] for m in ('adam','cd_gd')},evaluation_only=True))
    primary=[r for r in counts if r['baseline'] in ('adam','cd_gd') and r['tolerance']==1e-12]
    assert len(primary)==4
    ambiguous=any(r['counts'] is None for r in primary)
    passed=not ambiguous and all(r['counts']['net']>0 for r in primary)
    summary=dict(states=54,new_replays=0,new_scenes=0,groups=groups,necessary_tangent_gate=passed,
        ambiguous_primary=ambiguous,primary_cells=primary,existence_diagnostics=oracle,adopted=False,
        prior_fgr_numerical_gate=False,finite_stage_required=passed or ambiguous,
        prior_failed_numerical_gate_overridden=False)
    for name,data in (('summary.json',summary),('counts.json',counts),('states.json',results)):
        (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    lines=['One fixed observable direction mix;54 correlated states from six reused pairs.',
        'No new neural replay. Existing failed numerical and finite gates are not overridden.']
    lines += [str(r) for r in groups]+['Original GD/Adam tangent cells:']+[str(r) for r in primary]
    lines += ['Necessary tangent gate: '+str(passed),'Evaluation-only existence diagnostics:']+[str(r) for r in oracle]
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines));figure(summary,results)


def figure(summary,results):
    fig,axes=plt.subplots(1,3,figsize=(13,4.5))
    for ax,group in zip(axes[:2],summary['groups']):
        names=('cd_gd','adam','fgr_ridge','mixed')
        values=[group['baseline_descents'][m] for m in names[:-1]]+[group['mixed_descents']]
        ax.bar(names,values,color=['gray','gray','tab:blue','tab:orange'])
        ax.set(title=group['group']+': task descent tangents',ylabel='Endpoint count',ylim=(0,group['states']+3))
        ax.tick_params(axis='x',labelrotation=15)
    for pair in PAIRS:
        rows=[r for r in results if r['pair']==pair]
        axes[2].plot([r['level'] for r in rows],[r['beta'] for r in rows],'.-',label=str(pair))
    axes[2].set(title='Observed-only reference mixing',xlabel='NDP level',ylabel='CD-GD fraction',ylim=(-.03,1.03))
    axes[2].legend(ncol=2,fontsize=8)
    fig.text(.5,.01,'Mix signs certified for matched speed by norm bounds. GT used only after choices freeze. No finite-step success claim.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(OUT/'direction_mix.png',dpi=160);plt.close(fig)


if __name__=='__main__': main()
