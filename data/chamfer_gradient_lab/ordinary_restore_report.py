"""Fixed-event restoration against all unchanged full/short task criteria."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ndp_ordinary_restore import ROOT,OUT,OLD,PAIRS,ARMS,np,deterministic,exact_reference
from dense_trajectory_report import collect as dense_collect,assess

PRIMARY=ARMS[-1]


def collect():
    assert json.loads((OUT/'integrity.json').read_text())['completed_fits']==36
    old=dense_collect();rows=[];saved={};branches={}
    poses={r['pair']:r['transform'] for r in json.loads((ROOT/'paper_revision/fgr_20261018/observed.json').read_text())['rows']}
    for rep in (0,1):
        for pair in PAIRS:
            for arm in ARMS:
                folder=OUT/'runs'/f'r{rep}'/arm;stem=f'p{pair}_s0_official_w0'
                read=lambda suffix:json.loads((folder/(stem+suffix+'.json')).read_text())
                result,ledger,updates,prefix=read(''),read('_ledger'),read('_updates'),read('_prefix')
                steps=16 if arm==ARMS[1] else 50
                trace=np.loadtxt(folder/(stem+'_trace.txt'),delimiter=',',skiprows=1)
                assert trace.shape==(27,8) and np.isfinite(trace).all()
                assert np.array_equal(trace[:,:2],np.array([(l,s) for l in range(9) for s in (0,steps//2,steps)]))
                assert result['calls_by_level']==[steps+1]*9 and ledger['updates']==9*steps and ledger['loss_calls']==9*(steps+1)
                assert ledger['updates_per_level']==steps and ledger['total_seconds']==result['seconds']+ledger['preparation_seconds']>0
                assert ledger['fit_seconds']==result['seconds'] and len(prefix['outputs'])==17
                controlled=[r for r in updates if r['controlled']]
                if arm==ARMS[0]:
                    assert not updates and ledger['pose'] is None and ledger['preparation_seconds']==0 and not ledger['event_steps']
                else:
                    assert ledger['event_steps']==[7,15] and ledger['pose']['exact'] and ledger['preparation_seconds']>0
                    assert np.array_equal(np.asarray(ledger['pose']['transform']),np.asarray(poses[pair]))
                    assert [(r['level'],r['step']) for r in updates]==[(l,s) for l in range(9) for s in range(steps)]
                    assert [(r['level'],r['step']) for r in controlled]==[(l,s) for l in range(9) for s in (7,15)]
                    assert all(r['moment_step']==r['step']+1 for r in updates)
                    for r in controlled:
                        assert r['valid'] and r['applied_exact'] and r['cap']==1 and r['policy']=='trust'
                        assert all(s['converged'] and s['relative_residual']<=1e-8 for s in r['solvers'].values())
                        if r['accepted']:
                            assert r['cd_change']<=.1*r['fraction']*r['cd_slope']+1e-14 and r['linearization_error']<=.1+1e-12
                        else:assert r['fraction']==r['cd_change']==r['output_norm']==0
                if arm!=PRIMARY:
                    assert ledger['exact_saved_replay'];exact_reference(folder,OLD/'runs/r0'/arm,stem,result,updates)
                if rep:
                    expected=saved[pair,arm]
                    assert deterministic(result)==deterministic(expected[0]) and deterministic(updates)==deterministic(expected[1])
                    assert prefix==expected[2] and np.array_equal(trace,expected[3]) and ledger['repeat_exact']
                else:saved[pair,arm]=result,updates,prefix,trace
                branches[rep,pair,arm]=(prefix,updates)
                auc=float(np.mean([(t[0,5]+2*t[1,5]+t[2,5])/4 for t in trace.reshape(9,3,8)]))
                rows.append(dict(pair=pair,group='4DMatch' if pair<200 else 'HF',arm=arm,repetition=rep,
                    held_epe=result['held_epe'],full_epe=result['full_epe'],train_mse_auc=auc,final_cd=float(trace[-1,3]),
                    total_seconds=ledger['total_seconds'],fit_seconds=ledger['fit_seconds'],preparation_seconds=ledger['preparation_seconds'],
                    accepted=sum(r['accepted'] for r in controlled),rejected=sum(not r['accepted'] for r in controlled),
                    fractions=[r['fraction'] for r in controlled],betas=[r['beta'] for r in controlled],attempted_updates=9*steps))
            a,b=[branches[rep,pair,k] for k in ARMS[1:]]
            assert a[0]==b[0] and deterministic(a[1][:16])==deterministic(b[1][:16])
    # Only replace controls actually rerun. Others retain their exact saved task records.
    comparison=rows+[r for r in old if r['arm'] not in ARMS]
    return rows,comparison


def figure(rows):
    index={(r['pair'],r['arm']):r for r in rows if r['repetition']==0}
    fig,axes=plt.subplots(1,3,figsize=(13,4.3))
    for ax,key,title,limit in zip(axes,('held_epe','train_mse_auc','total_seconds'),
            ('Held-point EPE','Normalized training-MSE AUC','Total time (gate uses median)'),(1.,1.,2.)):
        for arm,label in ((ARMS[1],'16 updates; events7,15'),(PRIMARY,'50 updates; events7,15')):
            ax.plot(range(6),[index[p,arm][key]/index[p,ARMS[0]][key] for p in PAIRS],'o-',label=label)
        ax.axhline(limit,color='black',ls='--',lw=1);ax.axvline(3.5,color='gray',ls=':',lw=1)
        ax.set(title=title,xticks=range(6),xticklabels=PAIRS,ylabel='Ratio to fresh full Adam',xlabel='Reused development pair')
    axes[0].legend(fontsize=8)
    fig.text(.5,.01,'First execution shown; repeated task scores match exactly. Both schedules retain18 global corrections.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(OUT/'ordinary_restore.png',dpi=160);plt.close(fig)


def main():
    rows,comparison=collect()
    comparisons,gates,restoration=assess(comparison,primary=PRIMARY,candidates=(PRIMARY,),attribution_bases=(ARMS[1],))
    for r in comparisons:r['time_reference']='fresh' if r['baseline']=='full_adam' else 'historical; task reference only'
    summary=dict(completed_fits=36,development_pairs=6,repetitions=2,numerical_gate=True,exact_shared_prefixes=12,
        passed_cells=sum(g['passed'] for g in gates),required_cells=36,failed_cells=[g for g in gates if not g['passed']],
        advance_to_new_scenes=all(g['passed'] for g in gates),adopted=False,new_scenes=0,comparisons=comparisons,restoration=restoration)
    for name,value in (('summary.json',summary),('pairs.json',rows),('gates.json',gates)):
        (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    lines=['Fixed events7,15; restore34 ordinary updates per stage. Six reused pairs, not36 independent samples.']
    lines += [str(r) for r in comparisons]+['Restoration contrasts:']+[str(r) for r in restoration]
    lines += ['Gates:']+[str(g) for g in gates]+[f"Advance to new scenes: {summary['advance_to_new_scenes']}. Adopted: False."]
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines));figure(rows)


if __name__=='__main__':main()
