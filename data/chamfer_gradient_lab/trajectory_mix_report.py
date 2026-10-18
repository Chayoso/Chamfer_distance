"""Unchanged held-task, trajectory-AUC and total-time gates, by repetition."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'paper_revision/trajectory_mix_20261018'
PAIRS=(100,101,124,125,200,201)
ARMS=('adam','adam_trust','cd_ridge','fgr_gdmix','fgr_ridgemix')


def deterministic(value):
    if isinstance(value,dict):return {k:deterministic(v) for k,v in value.items() if 'seconds' not in k}
    if isinstance(value,list):return [deterministic(v) for v in value]
    return value


def collect():
    assert json.loads((OUT/'integrity.json').read_text())['completed_fits']==60
    rows=[];saved={}
    for repeat in (0,1):
        for pair in PAIRS:
            for arm in ARMS:
                folder=OUT/'runs'/f'r{repeat}'/arm;stem=f'p{pair}_s0_official_w0'
                read=lambda suffix:json.loads((folder/(stem+suffix+'.json')).read_text())
                result,ledger,updates=read(''),read('_ledger'),read('_updates')
                trace=np.loadtxt(folder/(stem+'_trace.txt'),delimiter=',',skiprows=1)
                assert trace.shape==(27,8) and np.isfinite(trace).all()
                assert np.array_equal(trace[:,:2],np.array([(i,s) for i in range(9) for s in (0,25,50)]))
                assert result['calls_by_level']==[51]*9 and ledger['updates']==450 and ledger['loss_calls']==459
                assert ledger['total_seconds']==ledger['fit_seconds']+ledger['preparation_seconds']
                assert ledger['total_seconds']>0 and result['seconds']==ledger['fit_seconds']
                assert all(v is None or not isinstance(v,float) or np.isfinite(v) for v in result.values())
                controlled=[r for r in updates if r['controlled']]
                if arm!='adam':
                    assert [(r['level'],r['step']) for r in updates]==[(i,s) for i in range(9) for s in range(50)]
                    assert [(r['level'],r['step']) for r in controlled]==[(i,49) for i in range(9)]
                    assert all(r['moment_step']==r['step']+1 for r in updates)
                else:
                    assert not updates and ledger['exact_plain_replay']
                    ref=ROOT/'paper_revision'/('replication_4d' if pair<200 else 'replication_hf')
                    expected=json.loads((ref/(stem+'.json')).read_text())
                    values=dict(result)
                    if pair>=200:
                        radius=ledger['source_radius_m']
                        values.update(source_radius_m=radius,**{g+'_epe_m':result[g+'_epe']*radius for g in ('full','held','vis','occ')})
                    assert all(values[k]==v for k,v in expected.items() if k not in ('seconds','updates_per_level'))
                    assert np.array_equal(trace,np.loadtxt(ref/(stem+'_trace.txt'),delimiter=',',skiprows=1))
                for r in controlled:
                    assert r['valid'] and r['applied_exact'] and r['cap']==1 and r['policy']=='trust'
                    assert all(s['converged'] and s['relative_residual']<=1e-8 for s in r['solvers'].values())
                    if r['accepted']:
                        assert r['cd_change']<=.1*r['fraction']*r['cd_slope']+1e-14 and r['linearization_error']<=.1+1e-12
                    else:assert r['fraction']==r['cd_change']==r['output_norm']==0
                if arm.startswith('fgr'):assert ledger['pose']['exact'] and ledger['preparation_seconds']>0
                if repeat:
                    old=saved[pair,arm]
                    assert deterministic(result)==deterministic(old[0]) and deterministic(updates)==deterministic(old[1])
                    assert np.array_equal(trace,old[2]) and ledger['repeat_exact']
                else:saved[pair,arm]=result,updates,trace
                # Each 50-update level has the same trapezoid spacing and weight.
                auc=float(np.mean([(t[0,5]+2*t[1,5]+t[2,5])/4 for t in trace.reshape(9,3,8)]))
                rows.append(dict(pair=pair,group='4DMatch' if pair<200 else 'HF',arm=arm,repetition=repeat,
                    held_epe=result['held_epe'],full_epe=result['full_epe'],train_mse_auc=auc,final_cd=float(trace[-1,3]),
                    total_seconds=ledger['total_seconds'],fit_seconds=ledger['fit_seconds'],preparation_seconds=ledger['preparation_seconds'],
                    rejected=sum(not r['accepted'] for r in controlled),accepted=sum(r['accepted'] for r in controlled),
                    fractions=[r['fraction'] for r in controlled],betas=[r['beta'] for r in controlled],
                    solvers=sum(len(r['solvers']) for r in controlled)))
    return rows


def assess(rows):
    index={(r['repetition'],r['pair'],r['arm']):r for r in rows}
    gates=[];comparisons=[]
    for repeat in (0,1):
        for arm in ARMS[1:]:
            for group,pairs in (('4DMatch',PAIRS[:4]),('HF',PAIRS[4:])):
                candidate=[index[repeat,p,arm] for p in pairs]
                for baseline in ('adam','adam_trust'):
                    base=[index[repeat,p,baseline] for p in pairs]
                    mean=lambda seq,key:float(np.mean([r[key] for r in seq]))
                    record=dict(repetition=repeat,arm=arm,group=group,baseline=baseline,
                        held_epe=mean(candidate,'held_epe'),baseline_held_epe=mean(base,'held_epe'),
                        held_epe_improvement=1-mean(candidate,'held_epe')/mean(base,'held_epe'),
                        train_mse_auc_ratio=mean(candidate,'train_mse_auc')/mean(base,'train_mse_auc'),
                        median_total_time_ratio=float(np.median([c['total_seconds']/b['total_seconds'] for c,b in zip(candidate,base)])),
                        worst_pair_epe_ratio=max(c['held_epe']/b['held_epe'] for c,b in zip(candidate,base)))
                    comparisons.append(record)
                    if arm=='fgr_ridgemix':
                        gates.append(dict(repetition=repeat,criterion='held_epe',group=group,baseline=baseline,
                            value=record['held_epe_improvement'],required='>=0.02',passed=record['held_epe_improvement']>=.02))
                        gates.append(dict(repetition=repeat,criterion='train_auc',group=group,baseline=baseline,
                            value=record['train_mse_auc_ratio'],required='<=1',passed=record['train_mse_auc_ratio']<=1))
            if arm=='fgr_ridgemix':
                worst=max(index[repeat,p,arm]['held_epe']/index[repeat,p,'adam']['held_epe'] for p in PAIRS)
                timing=float(np.median([index[repeat,p,arm]['total_seconds']/index[repeat,p,'adam']['total_seconds'] for p in PAIRS]))
                gates.extend([dict(repetition=repeat,criterion='worst_pair',value=worst,required='<=1.05',passed=worst<=1.05),
                    dict(repetition=repeat,criterion='total_time',value=timing,required='<=2',passed=timing<=2)])
    return comparisons,gates


def figure(rows):
    index={(r['pair'],r['arm']):r for r in rows if r['repetition']==0}
    fig,axes=plt.subplots(1,3,figsize=(14,4.7))
    for ax,key,title,limit in zip(axes,('held_epe','train_mse_auc','total_seconds'),
            ('Held-point EPE','Training MSE AUC','Total wall time'),(1.,1.,2.)):
        for arm in ARMS[1:]:
            values=[index[p,arm][key]/index[p,'adam'][key] for p in PAIRS]
            ax.plot(range(6),values,'o-',label=arm.replace('_',' '),alpha=.85)
        ax.axhline(limit,color='black',ls='--',lw=1)
        ax.set(xticks=range(6),xticklabels=PAIRS,title=title,ylabel='Ratio to plain Adam (lower is better)',xlabel='Reused development pair')
        ax.axvline(3.5,color='gray',ls=':',lw=1)
    axes[0].legend(fontsize=8)
    fig.text(.5,.01,'Repeat 1 shown; task metrics/traces must match repeat 2 exactly. All gates are assessed in each repeat. No new scenes.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(OUT/'trajectory_mix.png',dpi=160);plt.close(fig)


def main():
    rows=collect();comparisons,gates=assess(rows)
    result=dict(completed_fits=60,development_pairs=6,repetitions=2,numerical_gate=True,
        passed_cells=sum(g['passed'] for g in gates),required_cells=len(gates),
        advance_to_new_scenes=all(g['passed'] for g in gates),adopted=False,new_scenes=0,
        failed_cells=[g for g in gates if not g['passed']],comparisons=comparisons)
    for name,value in (('summary.json',result),('pairs.json',rows),('gates.json',gates)):
        (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    lines=['Own trajectories: six reused pairs, five arms, two timing/reproducibility repeats; not60 independent samples.']
    lines += [str(r) for r in comparisons if r['arm']=='fgr_ridgemix']
    lines += ['Gates:']+[str(g) for g in gates]
    lines += [f"Advance to new scenes: {result['advance_to_new_scenes']}. Adopted: False."]
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines));figure(rows)


if __name__=='__main__':main()
