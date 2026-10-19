"""Require each old native/time cell and paired deployment-cost improvement."""
import json
import ndp_pcg_probe as probe
base=probe.base;screen=probe.screen
OUT=base.ROOT/'paper_revision/pcg_timing_20261018'


def read(name):return screen.read(OUT,name)


def main(out=OUT,probe_module=probe,primary='fgr_pcgmix',cost_key='pcg_seconds'):
    OUT=out;probe=probe_module;read=lambda name:screen.read(OUT,name)
    summaries=[];finite={}
    for repeat in (0,1):
        probe.OUT=OUT/f'r{repeat}';probe.report()
        summaries.append(json.loads((probe.OUT/'summary.json').read_text()))
        finite.update({(r['pair'],r['level'],repeat):r for r in screen.read(probe.OUT,'finite')
            if r['method']==primary and r['policy']=='trust' and r['cap']==1})
        for name in ('finite','reference_finite','directions','reference_directions','solvers','states'):
            current=screen.read(probe.OUT,name)
            old=screen.read(OUT/'r0',name)
            clean=lambda rows:[{k:v for k,v in r.items() if not k.endswith('_seconds')} for r in rows]
            assert clean(current)==clean(old),(repeat,name)
    keys={(p,l,r) for p in base.PAIRS for l in range(9) for r in (0,1)}
    budget=read('budget');speed=read('speed');histories={k:[] for k in keys}
    screen.coverage(speed,('pair','level','repetition'),keys)
    screen.coverage(budget,('pair','level','repetition','account'),{(*k,a) for k in keys for a in ('amortized','full')})
    indexed={(r['pair'],r['level'],r['repetition'],r['account']):r for r in budget}
    costs={(r['pair'],r['level'],r['repetition']):r for r in speed}
    for r in budget:
        key=(r['pair'],r['level'],r['repetition']);cost=costs[key]
        assert r['budget']==cost[cost_key]+cost['fgr_seconds']/(9 if r['account']=='amortized' else 1)
        assert all(v==finite[key][k] for k,v in r['candidate'].items())
    for r in read('adam'):histories[r['pair'],r['level'],r['repetition']].append(r)
    for key,history in histories.items():
        assert [r['update'] for r in history]==list(range(len(history)))
        assert all(a['seconds']<b['seconds'] for a,b in zip(history,history[1:]))
        for account in ('amortized','full'):
            row=indexed[(*key,account)];after=next(r for r in history if r['seconds']>row['budget'])
            best=min(history[:after['update']+1],key=lambda r:r['cd_change'])
            assert row['after']['update']==after['update'] and row['best']['update']==best['update']
            assert row['best']['cd_change']==best['cd_change']
            if row['last'] is None:assert after['update']==0
            else:assert row['last']['update']==after['update']-1 and row['last']['seconds']<=row['budget']
            for label in ('last','after','best'):
                chosen=row[label]
                if chosen is not None:assert all(chosen[k]==history[chosen['update']][k] for k in ('seconds','cd_change'))
    counts=[];groups=[];speeds=[]
    for group,pairs in (('4DMatch',base.PAIRS[:4]),('HF',base.PAIRS[4:])):
        for repeat in (0,1):
            for account in ('amortized','full'):
                rows=[r for r in budget if r['pair'] in pairs and r['repetition']==repeat and r['account']==account]
                for label in ('last','after','best'):
                    a=[r[label]['risk_change'] if r[label] is not None else 0. for r in rows]
                    c=[r['candidate']['risk_change'] for r in rows]
                    for eps in (1e-12,1e-10,1e-8):counts.append(dict(group=group,repetition=repeat,account=account,baseline=label,tolerance=eps,**screen.net_counts(a,c,eps)))
                groups.append(dict(group=group,repetition=repeat,account=account,states=len(rows),
                    candidate_descents=sum(r['candidate']['risk_change'] < -1e-12 for r in rows),
                    best_adam_descents=sum(r['best']['risk_change'] < -1e-12 for r in rows),
                    candidate_mean_mse_change=float(base.np.mean([r['candidate']['risk_change'] for r in rows])),
                    best_adam_mean_mse_change=float(base.np.mean([r['best']['risk_change'] for r in rows])),
                    candidate_mean_epe_change=float(base.np.mean([r['candidate']['epe_change'] for r in rows])),
                    best_adam_mean_epe_change=float(base.np.mean([r['best']['epe_change'] for r in rows])),
                    median_budget=float(base.np.median([r['budget'] for r in rows])),
                    median_after_updates=float(base.np.median([r['after']['update'] for r in rows]))))
            selected=[r for r in speed if r['pair'] in pairs and r['repetition']==repeat]
            record=dict(group=group,repetition=repeat)
            for label,weight in (('operations',0),('amortized',1/9),('full',1)):
                record[label]=float(base.np.median([(r['default_seconds']+weight*r['fgr_seconds'])/(r[cost_key]+weight*r['fgr_seconds']) for r in selected]))
            speeds.append(record)
    required=[r for r in counts if r['tolerance']==1e-12 and r['baseline'] in ('after','best')]
    assert len(required)==16
    errors=[r[label]['mse_identity_error'] for r in budget for label in ('last','after','best','candidate') if r[label] is not None]
    native=all(r['advance_to_timing'] for r in summaries);timed=all(r['net']>0 for r in required)
    numerical=all(r['numerical_gate'] for r in summaries) and max(errors)<1e-12
    accelerated=all(r['operations']>1 for r in speeds)
    result=dict(states=54,repetitions=2,exact_original_replays=6,exact_common_finite_rows=2592,
        native_positive_cells=sum(r['native_positive_cells'] for r in summaries),native_required_cells=80,
        time_positive_cells=sum(r['net']>0 for r in required),time_required_cells=16,
        native_gate=native,time_gate=timed,numerical_gate=numerical,acceleration_gate=accelerated,
        max_mse_identity_error=max(errors),advance_to_cost_based_schedule=native and timed and numerical and accelerated,
        adopted=False,new_scenes=0,candidate_full_fits=0,groups=groups,speeds=speeds,
        failed_time_cells=[r for r in required if r['net']<=0],failed_speed_cells=[r for r in speeds if r['operations']<=1])
    for name,value in (('summary.json',result),('time_counts.json',counts)):
        (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    lines=['Two executions per original state; summed synchronized deployment operations, not full-fit wall time.']
    lines += [str(r) for r in groups]+[str(r) for r in speeds]
    lines += ['Gates: '+str({k:v for k,v in result.items() if k not in ('groups','speeds','failed_time_cells','failed_speed_cells')})]
    lines += ['Failed time cells: '+str(result['failed_time_cells']),'Failed speed cells: '+str(result['failed_speed_cells'])]
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines));figure(result,OUT,'PCG mix' if primary=='fgr_pcgmix' else 'Norm. mix')


def figure(result,out=OUT,label='PCG mix'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for col,group in enumerate(('4DMatch','HF')):
        for repeat,offset in ((0,-.18),(1,.18)):
            rows=[r for r in result['groups'] if r['group']==group and r['repetition']==repeat]
            values=[rows[0]['candidate_descents']]+[next(r['best_adam_descents'] for r in rows if r['account']==a) for a in ('amortized','full')]
            axes[0,col].bar(base.np.arange(3)+offset,values,.36,label=f'Repetition {repeat+1}')
            row=next(r for r in result['speeds'] if r['group']==group and r['repetition']==repeat)
            axes[1,col].bar(base.np.arange(3)+offset,[row[k] for k in ('operations','amortized','full')],.36)
        axes[0,col].set(title=group,xticks=range(3),xticklabels=(label,'Best Adam\namortized','Best Adam\nfull'),ylabel='Task-MSE descent states')
        axes[0,col].legend(fontsize=8)
        axes[1,col].axhline(1,color='black',ls=':',lw=1)
        ylabel='Median paired old / PCG time' if label=='PCG mix' else 'Median full-ridge / normalized-PCG time'
        axes[1,col].set(xticks=range(3),xticklabels=('Operations','+ FGR / 9','+ full FGR'),ylabel=ylabel)
    fig.tight_layout();fig.savefig(out/'timing.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
