"""Separate force, neural transfer, CD mixing and actual realization signs."""
from collections import Counter
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ndp_global_event_chain import ROOT,OUT,OLD,PAIRS,np,exact_reference

RAYS=('force','pcg','cd_reference','mixed','actual')
CHAIN=('force','pcg','mixed','actual_linear','finite')


def values(row,basis):
    return [row['products'][k]['task_'+basis] for k in CHAIN[:3]]+[row['task']['linear'],row['task']['change']]


def collect():
    assert json.loads((OUT/'integrity.json').read_text())['completed_replays']==6
    rows=[]
    for pair in PAIRS:
        folder=OUT/'runs';stem=f'p{pair}_s0_official_w0';read=lambda suffix:json.loads((folder/(stem+suffix+'.json')).read_text())
        result,updates,prefix,events=[read(s) for s in ('','_updates','_prefix','_events')]
        exact_reference(folder,OLD/'runs',stem,result,updates)
        assert prefix==json.loads((OLD/'runs'/(stem+'_prefix.json')).read_text()) and result['calls_by_level']==[51]*9
        previous={(r['level'],r['step']):r for r in json.loads((OLD/'runs'/(stem+'_movements.json')).read_text())}
        assert [(r['level'],r['step']) for r in events]==[(l,s) for l in range(9) for s in (7,15)]
        for row in events:
            prior=previous[row['level'],row['step']];u=updates[row['level']*50+row['step']]
            for key in ('task','anchor','fraction','beta','accepted','cd_change','linearization_error'):assert row[key]==prior[key],key
            assert u['valid'] and u['applied_exact'] and u['solvers']==row['solvers'] and u['bound']==row['bound']
            assert all(s['converged'] and s['relative_residual']<=1e-8 for s in row['solvers'].values())
            assert set(row['products'])==set(RAYS) and max(row['reconstruction_errors'].values())<1e-12
            assert row['actual_task_dot_error']<1e-12 and row['points']==2000 and 0<=row['beta']<1
            assert row['force_zero']==(row['force_norm']==0) and row['native_double_start_gap']>=0
            for ray in RAYS:
                assert np.isfinite(list(row['products'][ray].values())).all()
                if ray not in ('actual','force') or (ray=='force' and not row['force_zero']):
                    assert abs(row['products'][ray]['norm']-row['bound'])<1e-12
            assert abs(row['products']['actual']['norm']**2/row['points']-row['task']['quadratic'])<1e-12
            p=row['products'];scale=row['bound']/row['mixed_raw_norm'];beta=row['beta']
            for key in ('task_double','task_native','cd_double'):
                predicted=scale*((1-beta)*p['pcg'][key]+beta*p['cd_reference'][key])
                assert abs(predicted-p['mixed'][key])<1e-12
            assert p['mixed']['cd_double']<0 and p['mixed']['cd_double']<=.5*p['cd_reference']['cd_double']+1e-12
            if row['accepted']:
                assert row['cd_change']<=.1*row['fraction']*p['mixed']['cd_double']+1e-14 and row['linearization_error']<=.1+1e-12
            else:assert row['fraction']==row['task']['change']==row['products']['actual']['norm']==0
            rows.append(dict(row,pair=pair,group='4DMatch' if pair<200 else 'HF'))
    return rows


def summarize(rows,basis):
    sequences=[values(r,basis) for r in rows];harmful=[r for r in rows if r['accepted'] and r['task']['change']>0]
    sign=lambda x:'D' if x<0 else ('U' if x>0 else 'Z')
    patterns=Counter(''.join(sign(x) for x in seq) for seq in sequences)
    first=Counter(CHAIN[next(i for i,v in enumerate(values(r,basis)) if v>=0)] for r in harmful)
    return dict(basis=basis,events=len(rows),accepted=sum(r['accepted'] for r in rows),harmful_accepted=len(harmful),
        descents={k:sum(r['products'][k]['task_'+basis]<0 for r in rows) for k in RAYS},
        finite_descents=sum(r['task']['change']<0 for r in rows),first_nondescent_in_harmful=dict(sorted(first.items())),
        patterns=[dict(pattern=k,count=v) for k,v in sorted(patterns.items())],
        force_good_pcg_bad=sum(seq[0]<0<=seq[1] for seq in sequences),force_bad_pcg_good=sum(seq[0]>=0>seq[1] for seq in sequences),
        pcg_good_mix_bad=sum(seq[1]<0<=seq[2] for seq in sequences),pcg_bad_mix_good=sum(seq[1]>=0>seq[2] for seq in sequences),
        mix_good_actual_bad=sum(r['accepted'] and values(r,basis)[2]<0<=r['task']['linear'] for r in rows),
        near_zero_products={k:sum(abs(r['products'][k]['task_'+basis])<=1e-12 for r in rows) for k in RAYS},
        mixed_events=sum(r['beta']>0 for r in rows),zero_force_events=sum(r['force_zero'] for r in rows),
        mean_task_change=float(np.mean([r['task']['change'] for r in rows])))


def figure(rows,groups):
    fig,axes=plt.subplots(1,3,figsize=(14,4.6));labels=['Force','PCG','CD ref.','Mixed','Actual','Finite']
    for ax,group in zip(axes[:2],('4DMatch','HF')):
        r=next(r for r in groups if r['group']==group and r['basis']=='native')
        counts=[r['descents'][k] for k in RAYS]+[r['finite_descents']]
        ax.bar(range(6),np.asarray(counts)/r['events'],color=['C0','C1','gray','C2','C3','C4'])
        ax.set(title=f"{group}: {r['events']} events",xticks=range(6),xticklabels=labels,ylim=(0,1),ylabel='Fraction with task descent')
        for i,c in enumerate(counts):ax.text(i,c/r['events']+.015,str(c),ha='center',fontsize=9)
    axis=axes[2];points=axis.scatter([r['products']['pcg']['task_native'] for r in rows],
        [r['products']['mixed']['task_native'] for r in rows],c=[r['beta'] for r in rows],s=22,cmap='viridis',vmin=0,vmax=1)
    axis.axhline(0,color='black',lw=.8);axis.axvline(0,color='black',lw=.8)
    axis.set_xscale('symlog',linthresh=1e-6);axis.set_yscale('symlog',linthresh=1e-6)
    axis.set(title='PCG to CD-mixed direction',xlabel='PCG task directional product',ylabel='Mixed task directional product')
    fig.colorbar(points,ax=axis,label='CD mixing fraction beta')
    fig.text(.5,.01,'Native task-gradient basis shown; double-basis counts also reported. D/U/Z denote negative/positive/zero; no new optimizer.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(OUT/'event_chain.png',dpi=160);plt.close(fig)


def main():
    rows=collect();groups=[];pairs=[]
    for group in ('4DMatch','HF'):
        for basis in ('double','native'):groups.append(dict(group=group,**summarize([r for r in rows if r['group']==group],basis)))
    for pair in PAIRS:
        for basis in ('double','native'):pairs.append(dict(pair=pair,**summarize([r for r in rows if r['pair']==pair],basis)))
    sign=lambda x:(x>0)-(x<0)
    result=dict(exact_replays=6,exact_update_records=2700,exact_prior_finite_events=108,events=108,numerical_gate=True,
        max_reconstruction_error=max(max(r['reconstruction_errors'].values()) for r in rows),
        max_actual_task_dot_error=max(r['actual_task_dot_error'] for r in rows),
        max_native_double_start_gap=max(r['native_double_start_gap'] for r in rows),
        native_double_task_sign_changes={k:sum(sign(r['products'][k]['task_native'])!=sign(r['products'][k]['task_double']) for r in rows) for k in RAYS},
        groups=groups,posthoc_diagnostic=True,new_candidates=0,adopted=False,new_scenes=0)
    for name,value in (('summary.json',result),('events.json',rows),('pair_counts.json',pairs)):
        (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    lines=['Six exact paths; sign patterns are force,PCG,mixed,actual linear,finite task change. D/U/Z:negative/positive/zero.',
        'First non-descent is an accounting label, not unique causal attribution. No new candidate or adoption.']
    lines += [str({k:v for k,v in result.items() if k!='groups'})]+[str(r) for r in groups]
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines));figure(rows,groups)


if __name__=='__main__':main()
