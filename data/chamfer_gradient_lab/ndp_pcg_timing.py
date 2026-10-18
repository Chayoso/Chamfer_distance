"""Two native executions and original-history Adam at both declared budgets."""
import json
import shutil
import ndp_pcg_probe as probe
from pipeline.continuation import adam_path,budget_bracket

base=probe.base;native=probe.native
OUT=base.ROOT/'paper_revision/pcg_timing_20261018'


def append(name,row):
    with (OUT/(name+'.txt')).open('a') as handle:handle.write(json.dumps(row,allow_nan=False)+'\n')


def compare(pair,level,layer,opt,inputs,target,truth,anchor,anchor_seconds):
    history=[]
    for p in layer.parameters():
        state=opt.state[p]
        history.append((state['exp_avg'].detach().clone(),state['exp_avg_sq'].detach().clone(),int(state['step'])))
    group=dict(opt.param_groups[0])
    for repetition in (0,1):
        folder=OUT/f'r{repetition}';base.OUT=probe.OUT=folder
        try:result=probe.compare(pair,level,layer,opt,inputs,target,truth,anchor,anchor_seconds,repetition)
        finally:base.OUT=OUT
        costs=result['costs'];fgr=POSES[pair]['seconds']
        budgets={'amortized':costs['fgr_pcgmix']+fgr/9,'full':costs['fgr_pcgmix']+fgr}
        path=adam_path(layer,inputs,target,history,group,max(budgets.values()))
        if len(path)>1:assert base.torch.equal(path[1]['output'],result['adam_after'])
        selections={account:budget_bracket(path,budget) for account,budget in budgets.items()}
        chosen={id(v):v for selection in selections.values() for v in selection.values() if v is not None}
        for item in chosen.values():item['task']=native.task_metrics(result['native'],item['output'].double(),truth.double())
        for item in path:append('adam',dict(pair=pair,level=level,repetition=repetition,update=item['step'],
            seconds=item['seconds'],cd_change=item['cd']-result['initial']))
        for account,selection in selections.items():
            record=dict(pair=pair,level=level,repetition=repetition,account=account,budget=budgets[account],candidate=result['candidate'])
            for label,item in selection.items():
                record[label]=None if item is None else dict(update=item['step'],seconds=item['seconds'],
                    cd_change=item['cd']-result['initial'],**item['task'])
            append('budget',record)
        append('speed',dict(pair=pair,level=level,repetition=repetition,default_seconds=costs['fgr_ridgemix'],
            pcg_seconds=costs['fgr_pcgmix'],fgr_seconds=fgr,adam_updates=len(path)-1))
        (OUT/'progress.json').write_text(json.dumps(dict(pair=pair,level=level,repetition=repetition))+'\n')


if __name__=='__main__':
    previous=base.ROOT/'paper_revision/head_preconditioner_20261018'
    assert json.loads((previous/'summary.json').read_text())['advance_to_timing']
    assert json.loads((previous/'VALIDATION.json').read_text())['source_files']==157
    for r in (0,1):(OUT/f'r{r}').mkdir(exist_ok=False)
    probe.D,probe.F,probe.STATES=probe.screen.references();native.OUT=OUT;POSES=native.pose_controls()
    base.OUT=OUT;base.environment=lambda pair,truth:probe.screen.environment(pair,truth,callback=compare,poses=POSES)
    base.main()
    for r in (0,1):
        for name in ('poses.json','integrity.json'):shutil.copy2(OUT/name,OUT/f'r{r}'/name)
    from pcg_timing_report import main
    main()
