"""Original full-task gates plus short-Adam controls for the cost allocation."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ndp_dense_trajectory import ROOT,OUT,OLD,PAIRS,ARMS,PRIMARY,np,allocation,settings,deterministic,exact_reference

BASES=ARMS[:4]


def collect():
    n=allocation();rows=[];saved={}
    assert json.loads((OUT/'integrity.json').read_text())['completed_fits']==96
    poses={r['pair']:r['transform'] for r in json.loads((ROOT/'paper_revision/fgr_20261018/observed.json').read_text())['rows']}
    for repeat in (0,1):
        for pair in PAIRS:
            for arm in ARMS:
                kind,steps,events=settings(arm,n);folder=OUT/'runs'/f'r{repeat}'/arm;stem=f'p{pair}_s0_official_w0'
                read=lambda suffix:json.loads((folder/(stem+suffix+'.json')).read_text())
                result,ledger,updates=read(''),read('_ledger'),read('_updates')
                trace=np.loadtxt(folder/(stem+'_trace.txt'),delimiter=',',skiprows=1)
                assert trace.shape==(27,8) and np.isfinite(trace).all()
                assert np.array_equal(trace[:,:2],np.array([(l,s) for l in range(9) for s in (0,steps//2,steps)]))
                assert result['calls_by_level']==[steps+1]*9 and ledger['updates']==9*steps and ledger['loss_calls']==9*(steps+1)
                assert ledger['kind']==kind and ledger['updates_per_level']==steps and ledger['event_steps']==list(events)
                assert ledger['total_seconds']==ledger['fit_seconds']+ledger['preparation_seconds']
                assert ledger['total_seconds']>0 and result['seconds']==ledger['fit_seconds']
                assert all(v is None or not isinstance(v,float) or np.isfinite(v) for v in result.values())
                controlled=[r for r in updates if r['controlled']]
                if kind=='adam':assert not updates
                else:
                    assert [(r['level'],r['step']) for r in updates]==[(l,s) for l in range(9) for s in range(steps)]
                    assert [(r['level'],r['step']) for r in controlled]==[(l,s) for l in range(9) for s in events]
                    assert all(r['moment_step']==r['step']+1 for r in updates)
                for r in controlled:
                    assert r['valid'] and r['applied_exact'] and r['cap']==1 and r['policy']=='trust'
                    assert all(s['converged'] and s['relative_residual']<=1e-8 for s in r['solvers'].values())
                    if r['accepted']:
                        assert r['cd_change']<=.1*r['fraction']*r['cd_slope']+1e-14 and r['linearization_error']<=.1+1e-12
                    else:assert r['fraction']==r['cd_change']==r['output_norm']==0
                if kind.startswith('fgr'):
                    assert ledger['pose']['exact'] and ledger['preparation_seconds']>0
                    assert np.array_equal(np.asarray(ledger['pose']['transform']),np.asarray(poses[pair]))
                else:assert ledger['pose'] is None and ledger['preparation_seconds']==0
                if arm.startswith('full'):
                    assert ledger['exact_full_replay'];exact_reference(folder,OLD/'r0'/kind,stem,result,updates)
                if repeat:
                    old=saved[pair,arm]
                    assert deterministic(result)==deterministic(old[0]) and deterministic(updates)==deterministic(old[1])
                    assert np.array_equal(trace,old[2]) and ledger['repeat_exact']
                else:saved[pair,arm]=result,updates,trace
                auc=float(np.mean([(t[0,5]+2*t[1,5]+t[2,5])/4 for t in trace.reshape(9,3,8)]))
                rows.append(dict(pair=pair,group='4DMatch' if pair<200 else 'HF',arm=arm,repetition=repeat,
                    held_epe=result['held_epe'],full_epe=result['full_epe'],train_mse_auc=auc,final_cd=float(trace[-1,3]),
                    total_seconds=ledger['total_seconds'],fit_seconds=ledger['fit_seconds'],preparation_seconds=ledger['preparation_seconds'],
                    rejected=sum(not r['accepted'] for r in controlled),accepted=sum(r['accepted'] for r in controlled),
                    fractions=[r['fraction'] for r in controlled],betas=[r['beta'] for r in controlled],
                    solvers=sum(len(r['solvers']) for r in controlled),attempted_updates=9*steps))
    return rows


def assess(rows):
    index={(r['repetition'],r['pair'],r['arm']):r for r in rows};gates=[];comparisons=[];attribution=[]
    mean=lambda seq,key:float(np.mean([r[key] for r in seq]))
    for repeat in (0,1):
        for group,pairs in (('4DMatch',PAIRS[:4]),('HF',PAIRS[4:])):
            for arm in ARMS[4:]:
                candidate=[index[repeat,p,arm] for p in pairs]
                for baseline in BASES:
                    base=[index[repeat,p,baseline] for p in pairs]
                    record=dict(repetition=repeat,arm=arm,group=group,baseline=baseline,
                        held_epe=mean(candidate,'held_epe'),baseline_held_epe=mean(base,'held_epe'),
                        held_epe_improvement=1-mean(candidate,'held_epe')/mean(base,'held_epe'),
                        train_mse_auc_ratio=mean(candidate,'train_mse_auc')/mean(base,'train_mse_auc'),
                        median_total_time_ratio=float(np.median([c['total_seconds']/b['total_seconds'] for c,b in zip(candidate,base)])),
                        worst_pair_epe_ratio=max(c['held_epe']/b['held_epe'] for c,b in zip(candidate,base)))
                    comparisons.append(record)
                    if arm==PRIMARY:
                        gates.extend([dict(repetition=repeat,criterion='held_epe',group=group,baseline=baseline,
                            value=record['held_epe_improvement'],required='>=0.02',passed=record['held_epe_improvement']>=.02),
                            dict(repetition=repeat,criterion='train_auc',group=group,baseline=baseline,
                            value=record['train_mse_auc_ratio'],required='<=1',passed=record['train_mse_auc_ratio']<=1)])
            candidate=[index[repeat,p,PRIMARY] for p in pairs]
            for baseline in ARMS[4:7]:
                base=[index[repeat,p,baseline] for p in pairs]
                attribution.append(dict(repetition=repeat,group=group,baseline=baseline,
                    held_epe_improvement=1-mean(candidate,'held_epe')/mean(base,'held_epe'),
                    train_mse_auc_ratio=mean(candidate,'train_mse_auc')/mean(base,'train_mse_auc')))
        worst=max(index[repeat,p,PRIMARY]['held_epe']/index[repeat,p,'full_adam']['held_epe'] for p in PAIRS)
        timing=float(np.median([index[repeat,p,PRIMARY]['total_seconds']/index[repeat,p,'full_adam']['total_seconds'] for p in PAIRS]))
        gates.extend([dict(repetition=repeat,criterion='worst_pair',value=worst,required='<=1.05',passed=worst<=1.05),
            dict(repetition=repeat,criterion='total_time',value=timing,required='<=2',passed=timing<=2)])
    assert len(gates)==36
    return comparisons,gates,attribution


def figure(rows):
    index={(r['pair'],r['arm']):r for r in rows if r['repetition']==0}
    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    for ax,key,title,limit in zip(axes,('held_epe','train_mse_auc','total_seconds'),
            ('Held-point EPE','Normalized training-MSE AUC','Per-pair time (gate uses median)'),(1.,1.,2.)):
        for arm in ARMS[1:]:
            ax.plot(range(6),[index[p,arm][key]/index[p,'full_adam'][key] for p in PAIRS],
                'o-',label=arm.replace('_',' '),alpha=1. if arm==PRIMARY else .6,lw=2 if arm==PRIMARY else 1)
        ax.axhline(limit,color='black',ls='--',lw=1)
        ax.set(xticks=range(6),xticklabels=PAIRS,title=title,ylabel='Ratio to full plain Adam',xlabel='Reused development pair')
        ax.axvline(3.5,color='gray',ls=':',lw=1)
    axes[0].legend(fontsize=7)
    fig.text(.5,.01,'Execution 1 shown; task records must match execution 2. All gates apply separately per execution; no new scenes.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(OUT/'dense_trajectory.png',dpi=160);plt.close(fig)


def main():
    rows=collect();comparisons,gates,attribution=assess(rows)
    result=dict(completed_fits=96,development_pairs=6,repetitions=2,updates_per_short_level=16,
        numerical_gate=True,passed_cells=sum(g['passed'] for g in gates),required_cells=36,
        advance_to_new_scenes=all(g['passed'] for g in gates),adopted=False,new_scenes=0,
        failed_cells=[g for g in gates if not g['passed']],comparisons=comparisons,attribution=attribution)
    for name,value in (('summary.json',result),('pairs.json',rows),('gates.json',gates)):
        (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    lines=['One budget-allocation study: six reused pairs, eight arms, two executions; not96 independent samples.']
    lines += [str(r) for r in comparisons if r['arm']==PRIMARY]+['Attribution:']+[str(r) for r in attribution]
    lines += ['Gates:']+[str(g) for g in gates]+[f"Advance to new scenes: {result['advance_to_new_scenes']}. Adopted: False."]
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines));figure(rows)


if __name__=='__main__':main()
