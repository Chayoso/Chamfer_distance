"""Retain all native/time gates while timing only deployed normalized-mix work."""
import json
import shutil
import sys
import ndp_normalized_mix_probe as probe
from pipeline.continuation import adam_path,budget_bracket

base=probe.base;native=probe.native;screen=probe.screen
OUT=base.ROOT/'paper_revision/normalized_timing_20261018'
PREVIOUS=base.ROOT/'paper_revision/normalized_mix_20261018'


def append(name,row):
    with (OUT/(name+'.txt')).open('a') as handle:handle.write(json.dumps(row,allow_nan=False)+'\n')


def legacy_cost(pair,level,layer,inputs,target,truth,anchor,anchor_seconds,repetition):
    result=native.directions(layer,inputs,target,anchor,93000+100*pair+level,method='fgr_ridgemix')
    item=result['directions']['fgr_ridgemix']
    for trial,output in native.native_trials(layer,inputs,target,result['native'],result['initial'],
            item['delta'],item['velocity'],result['gradient'],caps=(1.,)):
        row=dict(pair=pair,level=level,repetition=repetition,method='fgr_ridgemix',**trial,
            **native.task_metrics(result['native'],output,truth.double()),
            builder_seconds=result['production_seconds'],anchor_seconds=anchor_seconds)
        old=probe.F[pair,level,'fgr_ridgemix',trial['policy'],1.]
        assert all(v==old[k] for k,v in row.items() if k!='repetition' and not k.endswith('_seconds'))
        append('legacy',row)
    return result['production_seconds']+anchor_seconds+trial['controller_seconds']


def compare(pair,level,layer,opt,inputs,target,truth,anchor,anchor_seconds):
    history=[]
    for parameter in layer.parameters():
        state=opt.state[parameter]
        history.append((state['exp_avg'].detach().clone(),state['exp_avg_sq'].detach().clone(),int(state['step'])))
    group=dict(opt.param_groups[0])
    for repetition in (0,1):
        first=(base.PAIRS.index(pair)+level+repetition)%2==0
        args=(pair,level,layer,inputs,target,truth,anchor,anchor_seconds,repetition)
        if first:legacy=legacy_cost(*args)
        base.OUT=probe.OUT=OUT/f'r{repetition}'
        try:result=probe.compare(pair,level,layer,opt,inputs,target,truth,anchor,anchor_seconds,repetition)
        finally:base.OUT=OUT
        if not first:legacy=legacy_cost(*args)
        costs=result['costs'];fgr=POSES[pair]['seconds'];cost=costs[probe.PRIMARY]
        budgets={'amortized':cost+fgr/9,'full':cost+fgr}
        path=adam_path(layer,inputs,target,history,group,max(budgets.values()))
        if len(path)>1:assert base.torch.equal(path[1]['output'],result['adam_after'])
        selections={account:budget_bracket(path,budget) for account,budget in budgets.items()}
        chosen={id(v):v for selection in selections.values() for v in selection.values() if v is not None}
        for item in chosen.values():item['task']=native.task_metrics(result['native'],item['output'].double(),truth.double())
        for item in path:append('adam',dict(pair=pair,level=level,repetition=repetition,update=item['step'],
            seconds=item['seconds'],cd_change=item['cd']-result['initial']))
        for account,selection in selections.items():
            row=dict(pair=pair,level=level,repetition=repetition,account=account,budget=budgets[account],candidate=result['candidate'])
            for label,item in selection.items():
                row[label]=None if item is None else dict(update=item['step'],seconds=item['seconds'],
                    cd_change=item['cd']-result['initial'],**item['task'])
            append('budget',row)
        append('speed',dict(pair=pair,level=level,repetition=repetition,default_seconds=legacy,
            candidate_seconds=cost,old_pcg_seconds=costs['fgr_pcgmix'],fgr_seconds=fgr,adam_updates=len(path)-1,legacy_first=first,
            candidate_builder_seconds=result['builder_seconds'][probe.PRIMARY],old_pcg_builder_seconds=result['builder_seconds']['fgr_pcgmix'],anchor_seconds=anchor_seconds))
        (OUT/'progress.json').write_text(json.dumps(dict(pair=pair,level=level,repetition=repetition))+'\n')


def report():
    strip=lambda rows:[{k:v for k,v in r.items() if not k.endswith('seconds')} for r in rows]
    names=('directions','reference_directions','finite','reference_finite','states','solvers')
    for repetition in (0,1):
        for name in names:
            assert strip(screen.read(OUT/f'r{repetition}',name))==strip(screen.read(PREVIOUS,name)),(repetition,name)
    legacy=screen.read(OUT,'legacy');_,old,_=screen.references()
    screen.coverage(legacy,('pair','level','repetition','policy'),
        {(p,l,r,c) for p in base.PAIRS for l in range(9) for r in (0,1) for c in ('raw','trust')})
    for row in legacy:
        saved=old[row['pair'],row['level'],'fgr_ridgemix',row['policy'],1.]
        assert all(v==saved[k] for k,v in row.items() if k!='repetition' and not k.endswith('_seconds'))
    speeds=screen.read(OUT,'speed');legacy_index={(r['pair'],r['level'],r['repetition']):r for r in legacy if r['policy']=='trust'}
    for repetition in (0,1):
        trials={ (r['pair'],r['level'],r['method']):r for name in ('finite','reference_finite')
            for r in screen.read(OUT/f'r{repetition}',name) if r['policy']=='trust' and r['cap']==1}
        for row in (r for r in speeds if r['repetition']==repetition):
            p,l=row['pair'],row['level'];old=legacy_index[p,l,repetition]
            assert row['default_seconds']==old['builder_seconds']+old['anchor_seconds']+old['controller_seconds']
            for method,prefix in ((probe.PRIMARY,'candidate'),('fgr_pcgmix','old_pcg')):
                assert row[prefix+'_seconds']==row[prefix+'_builder_seconds']+row['anchor_seconds']+trials[p,l,method]['controller_seconds']
    import pcg_timing_report
    pcg_timing_report.main(OUT,probe,probe.PRIMARY,'candidate_seconds')
    budget=screen.read(OUT,'budget');speed=screen.read(OUT,'speed');pairs=[]
    for pair in base.PAIRS:
        for repetition in (0,1):
            costs=[r for r in speed if r['pair']==pair and r['repetition']==repetition]
            for account in ('amortized','full'):
                rows=[r for r in budget if r['pair']==pair and r['repetition']==repetition and r['account']==account]
                pairs.append(dict(pair=pair,repetition=repetition,account=account,states=len(rows),
                    candidate_descents=sum(r['candidate']['risk_change'] < -1e-12 for r in rows),
                    best_adam_descents=sum(r['best']['risk_change'] < -1e-12 for r in rows),
                    candidate_mean_mse=float(base.np.mean([r['candidate']['risk_change'] for r in rows])),
                    best_adam_mean_mse=float(base.np.mean([r['best']['risk_change'] for r in rows])),
                    candidate_mean_epe=float(base.np.mean([r['candidate']['epe_change'] for r in rows])),
                    best_adam_mean_epe=float(base.np.mean([r['best']['epe_change'] for r in rows])),
                    median_old_pcg_over_candidate=float(base.np.median([r['old_pcg_seconds']/r['candidate_seconds'] for r in costs]))))
    (OUT/'pair_counts.json').write_text(json.dumps(pairs,indent=2)+'\n')


if __name__=='__main__':
    if '--report-only' not in sys.argv:
        assert json.loads((PREVIOUS/'summary.json').read_text())['advance_to_timing']
        assert json.loads((PREVIOUS/'VALIDATION.json').read_text())['source_files']==176
        for r in (0,1):(OUT/f'r{r}').mkdir(exist_ok=False)
        probe.D,probe.F,probe.STATES=screen.references()
        for row in screen.read(probe.OLD,'directions'):probe.D[row['pair'],row['level'],row['method']]=row
        for row in screen.read(probe.OLD,'finite'):probe.F[row['pair'],row['level'],row['method'],row['policy'],row['cap']]=row
        native.OUT=OUT;POSES=native.pose_controls();base.OUT=OUT
        base.environment=lambda pair,truth:screen.environment(pair,truth,callback=compare,poses=POSES)
        base.main()
        for r in (0,1):
            for name in ('poses.json','integrity.json'):shutil.copy2(OUT/name,OUT/f'r{r}'/name)
    report()
