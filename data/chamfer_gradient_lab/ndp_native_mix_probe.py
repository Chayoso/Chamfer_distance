"""Native parameter rounding and two-repetition, time-bracketed Adam control."""
import time
import ndp_field_probe as base
from global_registration_control import load, prepare
from pipeline.native_mix import directions, native_trials
from pipeline.continuation import adam_path, budget_bracket
from pipeline.mse_reference import mse_reference

OUT=base.ROOT/'paper_revision/native_mix_20261018'
METHODS=('cd_gd','cd_ridge','adam','fgr_ridge','fgr_gdmix','fgr_ridgemix')


def task_metrics(start,after,gt):
    reference=mse_reference(*[v.detach().cpu().numpy() for v in (start,after,gt)])
    return dict(risk_change=reference['reference_change'],mse_identity_error=reference['reference_identity_error'],
        epe_change=float((after-gt).norm(dim=1).mean()-(start-gt).norm(dim=1).mean()))


def pose_controls():
    import open3d as o3d
    assert o3d.__version__=='0.19.0'
    old=base.json.loads((base.ROOT/'paper_revision/fgr_20261018/observed.json').read_text())['rows']
    rows=[];reg=o3d.pipelines.registration
    for row in old:
        x,y,_,_=load(row['pair']);begin=time.perf_counter()
        voxel,clouds,features=prepare(x,y)
        o3d.utility.random.seed(row['seed'])
        result=reg.registration_fgr_based_on_feature_matching(*clouds,*features,reg.FastGlobalRegistrationOption(**row['options']))
        seconds=time.perf_counter()-begin
        assert voxel==row['voxel'] and base.np.array_equal(result.transformation,base.np.asarray(row['transform']))
        rows.append(dict(pair=row['pair'],transform=result.transformation.tolist(),seconds=seconds,exact_pose_replay=True))
    (OUT/'poses.json').write_text(base.json.dumps(rows,indent=2)+'\n')
    return {r['pair']:r for r in rows}


def compare(pair,level,layer,opt,inputs,target,truth,anchor,anchor_seconds):
    torch=base.torch;gt=truth.double()
    history=[]
    for p in layer.parameters():
        state=opt.state[p]
        history.append((state['exp_avg'].detach().clone(),state['exp_avg_sq'].detach().clone(),int(state['step'])))
    group=dict(opt.param_groups[0])
    for repetition in (0,1):
        begin=base.tick(inputs)
        before=[v.detach().double().clone() for v in layer.parameters()]
        path=base.continuation(layer,opt,inputs,target,base.cd,steps=1)
        _,native_start,_,_=next(path);_,native_after,_,params=next(path)
        adam=[v.detach().double()-p for v,p in zip(params,before)]
        native_start,native_after=native_start.detach(),native_after.detach();path.close()
        adam_seconds=base.tick(inputs)-begin
        result=directions(layer,inputs,target,anchor,93000+100*pair+level)
        p=result['start'];task=2*(p-gt)/len(p);native=result['native'];gradient=result['gradient']
        assert torch.equal(native,native_start.double())
        # Adam's historical direction uses the same continuous Jacobian as before.
        begin=base.tick(inputs);model=base.copy.deepcopy(layer).double()
        forward,_=base.jacobian_products(model(inputs.double())[0],list(model.parameters()))
        delta,velocity=base.scale_direction(adam,forward,result['bound'])
        result['directions']['adam']=dict(delta=delta,velocity=velocity,seconds=adam_seconds+base.tick(inputs)-begin)
        del forward,model
        for field,solver in result['solvers'].items():
            base.append('solvers',dict(pair=pair,level=level,repetition=repetition,field=field,**solver))
        budgets={};candidate=None
        for method in METHODS:
            item=result['directions'][method];velocity=item['velocity']
            row=dict(pair=pair,level=level,repetition=repetition,method=method,
                risk_derivative=float((task*velocity).sum()),cd_derivative=float((gradient*velocity).sum()),
                native_start_task_predictor=float((2*(native-gt)*velocity).sum()/len(native)),
                motion_bound=result['bound'],output_norm=float(velocity.norm()),
                setup_seconds=result['setup_seconds'],direction_seconds=item['seconds'],
                anchor_seconds=anchor_seconds if method.startswith('fgr') else 0.,beta=item.get('beta'))
            expected=EXPECTED[pair,level,method]
            for key in ('risk_derivative','cd_derivative','motion_bound','output_norm'):
                base.np.testing.assert_allclose(row[key],expected[key],rtol=1e-10,atol=1e-12)
            base.append('directions',row)
            for trial,output in native_trials(layer,inputs,target,native,result['initial'],item['delta'],velocity,gradient):
                base.append('finite',dict(pair=pair,level=level,repetition=repetition,method=method,**trial,
                    **task_metrics(native,output,gt)))
                if method=='fgr_ridgemix' and trial['cap']==1 and trial['policy']=='trust':
                    shared=result['setup_seconds']+anchor_seconds+item['seconds']+trial['controller_seconds']
                    budgets={'amortized':shared+POSES[pair]['seconds']/9,'full':shared+POSES[pair]['seconds']}
                    candidate=task_metrics(native,output,gt)
                    candidate.update(cd_change=trial['cd_change'],fraction=trial['fraction'])
        # No task metric is passed to continuation or the bracket selector.
        path=adam_path(layer,inputs,target,history,group,max(budgets.values()))
        if len(path)>1:assert torch.equal(path[1]['output'],native_after)
        choices={account:budget_bracket(path,budget) for account,budget in budgets.items()}
        selected={id(r):r for selection in choices.values() for r in selection.values() if r is not None}
        for item in selected.values():item['task']=task_metrics(native,item['output'].double(),gt)
        for item in path:
            base.append('adam',dict(pair=pair,level=level,repetition=repetition,update=item['step'],
                seconds=item['seconds'],cd_change=item['cd']-result['initial']))
        for account,selection in choices.items():
            record=dict(pair=pair,level=level,repetition=repetition,account=account,budget=budgets[account],
                candidate=candidate,within_evaluated=selection['last'] is not None)
            for label,item in selection.items():
                record[label]=None if item is None else dict(update=item['step'],seconds=item['seconds'],
                    cd_change=item['cd']-result['initial'],**item['task'])
            base.append('budget',record)
        base.append('states',dict(pair=pair,level=level,repetition=repetition,native_cd=result['initial'],
            native_double_difference=float((native-p).norm()),fgr_seconds=POSES[pair]['seconds'],
            adam_updates=len(path)-1,max_budget=max(budgets.values())))
        (OUT/'progress.json').write_text(base.json.dumps(dict(pair=pair,level=level,repetition=repetition))+'\n')


def environment(pair,truth):
    x=base.torch.as_tensor(load(pair)[0],device=truth.device,dtype=base.torch.float32)
    transform=base.torch.as_tensor(POSES[pair]['transform'],device=x.device,dtype=base.torch.float64)
    def end(level,iteration,count,previous,value,opt,layer,inputs,q,cached,ids,mean):
        begin=base.tick(inputs)
        anchor=x[ids].double()@transform[:3,:3].T+transform[:3,3]-mean.double()
        elapsed=base.tick(inputs)-begin
        compare(pair,level,layer,opt,inputs,q,truth[ids]-mean,anchor,elapsed)
    return base.stopping_environment(lambda *args:None,end)[0]


if __name__=='__main__':
    POSES=pose_controls()
    old=base.ROOT/'paper_revision/mix_20261018/precise/directions.txt'
    EXPECTED={(r['pair'],r['level'],r['method']):r for r in [base.json.loads(s) for s in old.read_text().splitlines()]}
    base.OUT=OUT;base.environment=environment;base.main()
