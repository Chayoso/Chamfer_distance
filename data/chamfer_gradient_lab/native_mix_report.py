"""Native precision and matched-time gates for the frozen corrected FGR rule."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fields_report import PAIRS, coverage, net_counts

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'paper_revision/native_mix_20261018'
METHODS=('cd_gd','cd_ridge','adam','fgr_ridge','fgr_gdmix','fgr_ridgemix')


def read(name):return [json.loads(s) for s in (OUT/(name+'.txt')).read_text().splitlines()]


def main():
    rows={n:read(n) for n in ('directions','finite','solvers','states','budget','adam')}
    states={(p,l,r) for p in PAIRS for l in range(9) for r in (0,1)}
    coverage(rows['states'],('pair','level','repetition'),states)
    coverage(rows['directions'],('pair','level','repetition','method'),{(*s,m) for s in states for m in METHODS})
    coverage(rows['finite'],('pair','level','repetition','method','policy','cap'),
        {(*s,m,p,c) for s in states for m in METHODS for p in ('raw','trust') for c in (1/16,.25,1.)})
    coverage(rows['solvers'],('pair','level','repetition','field'),{(*s,m) for s in states for m in ('cd','fgr')})
    coverage(rows['budget'],('pair','level','repetition','account'),{(*s,a) for s in states for a in ('amortized','full')})
    assert json.loads((OUT/'integrity.json').read_text())['exact_replays']==list(PAIRS)
    poses=json.loads((OUT/'poses.json').read_text())
    assert len(poses)==6 and all(r['exact_pose_replay'] and r['seconds']>0 for r in poses)
    d={(r['pair'],r['level'],r['repetition'],r['method']):r for r in rows['directions']}
    f={(r['pair'],r['level'],r['repetition'],r['method'],r['policy'],r['cap']):r for r in rows['finite']}
    b={(r['pair'],r['level'],r['repetition'],r['account']):r for r in rows['budget']}
    for r in rows['finite']:
        assert r['controller_seconds']>0
        if r['policy']=='trust' and r['accepted']:
            assert r['cd_change']<=.1*r['fraction']*r['cd_slope']+1e-14 and r['linearization_error']<=.1+1e-12
        if r['policy']=='trust' and not r['accepted']:
            assert r['risk_change']==r['cd_change']==r['output_norm']==0
        if r['repetition']==1:
            prior=f[r['pair'],r['level'],0,r['method'],r['policy'],r['cap']]
            assert {k:v for k,v in r.items() if k not in ('controller_seconds','repetition')}=={
                k:v for k,v in prior.items() if k not in ('controller_seconds','repetition')}
    assert all(r['converged'] and r['relative_residual']<=1e-8 for r in rows['solvers'])
    histories={s:[] for s in states}
    for r in rows['adam']:histories[r['pair'],r['level'],r['repetition']].append(r)
    for key,history in histories.items():
        assert [r['update'] for r in history]==list(range(len(history)))
        assert all(x['seconds']<y['seconds'] for x,y in zip(history,history[1:]))
        for account in ('amortized','full'):
            r=b[(*key,account)];deadline=r['budget']
            after=next(v for v in history if v['seconds']>deadline)
            eligible=history[:after['update']+1]
            best=min(eligible,key=lambda v:v['cd_change'])
            assert r['after']['update']==after['update'] and r['best']['update']==best['update']
            if r['last'] is not None:assert r['last']['seconds']<=deadline and r['last']['update']==after['update']-1
            assert r['best']['cd_change']==best['cd_change']
    counts,groups,time_counts,time_groups=[],[],[],[]
    for group,pairs in (('4DMatch',PAIRS[:4]),('HF',PAIRS[4:])):
        for repeat in (0,1):
            keys=[(p,l,repeat) for p in pairs for l in range(9)]
            for baseline in ('adam','cd_gd','cd_ridge','fgr_gdmix'):
                for cap in (0.,1/16,.25,1.):
                    for policy in (('raw',) if cap==0 else ('raw','trust')):
                        a=[d[(*k,baseline)]['risk_derivative'] if cap==0 else f[(*k,baseline,policy,cap)]['risk_change'] for k in keys]
                        c=[d[(*k,'fgr_ridgemix')]['risk_derivative'] if cap==0 else f[(*k,'fgr_ridgemix','trust',cap)]['risk_change'] for k in keys]
                        for eps in (1e-12,1e-10,1e-8):counts.append(dict(group=group,repetition=repeat,baseline=baseline,cap=cap,
                            base_policy=policy,tolerance=eps,**net_counts(a,c,eps)))
            for method in METHODS:
                chosen=[f[(*k,method,'trust',1.)] for k in keys]
                groups.append(dict(group=group,repetition=repeat,method=method,states=len(keys),
                    tangent_descents=sum(d[(*k,method)]['risk_derivative'] < -1e-12 for k in keys),
                    finite_descents=sum(r['risk_change'] < -1e-12 for r in chosen),rejected=sum(not r['accepted'] for r in chosen),
                    cd_descents=sum(r['cd_change'] < -1e-12 for r in chosen),
                    mean_mse_change=float(np.mean([r['risk_change'] for r in chosen])),
                    median_fraction=float(np.median([r['fraction'] for r in chosen]))))
            for account in ('amortized','full'):
                selected=[b[(*k,account)] for k in keys]
                for label in ('last','after','best'):
                    for eps in (1e-12,1e-10,1e-8):
                        a=[r[label]['risk_change'] if r[label] is not None else 0. for r in selected]
                        c=[r['candidate']['risk_change'] for r in selected]
                        time_counts.append(dict(group=group,repetition=repeat,account=account,baseline=label,
                            tolerance=eps,**net_counts(a,c,eps)))
                time_groups.append(dict(group=group,repetition=repeat,account=account,
                    median_budget=float(np.median([r['budget'] for r in selected])),
                    median_after_updates=float(np.median([r['after']['update'] for r in selected])),
                    best_adam_descents=sum(r['best']['risk_change'] < -1e-12 for r in selected),
                    candidate_descents=sum(r['candidate']['risk_change'] < -1e-12 for r in selected),
                    candidate_beats_best_cd=sum(r['candidate']['cd_change'] < r['best']['cd_change']-1e-8 for r in selected),
                    candidate_mean_mse_change=float(np.mean([r['candidate']['risk_change'] for r in selected])),
                    best_adam_mean_mse_change=float(np.mean([r['best']['risk_change'] for r in selected]))))
    primary=[r for r in counts if r['cap'] in (0,.25,1.) and r['tolerance']==1e-12]
    timed=[r for r in time_counts if r['baseline'] in ('best','after') and r['tolerance']==1e-12]
    assert len(primary)==80 and len(timed)==16
    identities=[r['mse_identity_error'] for r in rows['finite']]
    identities += [r[label]['mse_identity_error'] for r in rows['budget'] for label in ('last','after','best') if r[label] is not None]
    numerical=max(identities)<1e-12
    native=all(r['net']>0 for r in primary);budget=all(r['net']>0 for r in timed)
    result=dict(states=54,repetitions=2,exact_original_replays=6,exact_pose_replays=6,solves=len(rows['solvers']),
        finite_rows=len(rows['finite']),budget_comparisons=len(rows['budget']),adam_evaluation_rows=len(rows['adam']),
        adam_updates=sum(r['adam_updates'] for r in rows['states']),numerical_gate=numerical,max_mse_identity_error=max(identities),
        native_gate=native,native_positive_cells=sum(r['net']>0 for r in primary),native_required_cells=80,
        time_gate=budget,time_positive_cells=sum(r['net']>0 for r in timed),time_required_cells=16,
        advance_to_full_trajectory=numerical and native and budget,adopted=False,new_scenes=0,native_candidate_full_fits=0,
        groups=groups,time_groups=time_groups,failed_native_cells=[r for r in primary if r['net']<=0],
        failed_time_cells=[r for r in timed if r['net']<=0])
    for name,value in (('summary.json',result),('counts.json',counts),('time_counts.json',time_counts)):
        (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    lines=['Native float32 and two timed repetitions;54 correlated original states, not108 independent states.']
    lines += [str(r) for r in groups if r['method'] in ('adam','fgr_ridgemix')]+[str(r) for r in time_groups]
    lines += ['Gates: '+str({k:v for k,v in result.items() if k not in ('groups','time_groups','failed_native_cells','failed_time_cells')})]
    lines += ['Failed native cells:']+[str(r) for r in result['failed_native_cells']]
    lines += ['Failed time cells:']+[str(r) for r in result['failed_time_cells']]
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines));figure(result)


def figure(result):
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for ax,group in zip(axes,('4DMatch','HF')):
        names=['CD-GD','Adam','GD mix','Ridge mix','Budget Adam\namortized','Budget Adam\nfull']
        for repeat,offset in ((0,-.18),(1,.18)):
            values=[next(r['finite_descents'] for r in result['groups'] if r['group']==group and r['repetition']==repeat and r['method']==m)
                    for m in ('cd_gd','adam','fgr_gdmix','fgr_ridgemix')]
            values += [next(r['best_adam_descents'] for r in result['time_groups'] if r['group']==group and r['repetition']==repeat and r['account']==a) for a in ('amortized','full')]
            ax.bar(np.arange(6)+offset,values,.36,label=f'Timing repetition {repeat+1}')
        ax.set(xticks=range(6),xticklabels=names,title=group,ylabel='GT task-descent states',ylim=(0,39 if group=='4DMatch' else 20))
        ax.tick_params(axis='x',labelrotation=20);ax.legend(fontsize=8)
    fig.text(.5,.01,'Native CD acceptance at cap1; budget Adam selects lowest CD including first over-budget evaluation. No GT selection.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(OUT/'native_mix.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
