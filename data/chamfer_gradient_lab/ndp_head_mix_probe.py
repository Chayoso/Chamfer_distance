"""Frozen affine-head screen at the same 54 original endpoints; no full fits."""
import sys
import ndp_native_mix_probe as native
from fields_report import coverage,net_counts

base=native.base
OUT=base.ROOT/'paper_revision/fgr_head_20261018'
OLD=base.ROOT/'paper_revision/native_mix_20261018'
METHODS=('cd_gd','adam','fgr_head','fgr_headmix')


def read(folder,name):return [base.json.loads(s) for s in (folder/(name+'.txt')).read_text().splitlines()]


def compare(pair,level,layer,opt,inputs,target,truth,anchor,anchor_seconds):
    result=native.directions(layer,inputs,target,anchor,93000+100*pair+level,method='fgr_headmix')
    before=[p.detach().double().clone() for p in layer.parameters()]
    path=base.continuation(layer,opt,inputs,target,base.cd,steps=1)
    _,start,_,_=next(path);_,_,_,params=next(path)
    adam=[v.detach().double()-p for v,p in zip(params,before)];path.close()
    check=native.directions(layer,inputs,target,None,93000+100*pair+level,method='adam',adam_delta=adam)
    assert base.torch.equal(start.double(),result['native']) and result['initial']==STATES[pair,level]['native_cd']
    result['directions']['adam']=check['directions']['adam']
    gt=truth.double();task=2*(result['start']-gt)/len(gt)
    for field,solver in result['solvers'].items():base.append('solvers',dict(pair=pair,level=level,field=field,**solver))
    for method in METHODS:
        item=result['directions'][method];velocity=item['velocity']
        row=dict(pair=pair,level=level,method=method,risk_derivative=float((task*velocity).sum()),
            cd_derivative=float((result['gradient']*velocity).sum()),motion_bound=result['bound'],
            output_norm=float(velocity.norm()),setup_seconds=result['setup_seconds'],direction_seconds=item['seconds'],
            anchor_seconds=anchor_seconds,beta=item.get('beta'))
        if method in ('cd_gd','adam'):
            for key in ('risk_derivative','cd_derivative','motion_bound','output_norm'):
                base.np.testing.assert_allclose(row[key],D[pair,level,method][key],rtol=1e-10,atol=1e-12)
        base.append('directions',row)
        for trial,output in native.native_trials(layer,inputs,target,result['native'],result['initial'],
                item['delta'],velocity,result['gradient']):
            value=dict(pair=pair,level=level,method=method,**trial,**native.task_metrics(result['native'],output,gt))
            if method in ('cd_gd','adam'):
                old=F[pair,level,method,trial['policy'],trial['cap']]
                assert all(v==old[k] for k,v in value.items() if k!='controller_seconds'),(pair,level,method)
            base.append('finite',value)
    base.append('states',dict(pair=pair,level=level,native_cd=result['initial'],exact_common_controls=True))
    (OUT/'progress.json').write_text(base.json.dumps(dict(pair=pair,level=level))+'\n')


def environment(pair,truth,callback=None,poses=None):
    callback=compare if callback is None else callback;poses=POSES if poses is None else poses
    x=base.torch.as_tensor(native.load(pair)[0],device=truth.device,dtype=base.torch.float32)
    pose=base.torch.as_tensor(poses[pair]['transform'],device=x.device,dtype=base.torch.float64)
    def end(level,iteration,count,previous,value,opt,layer,inputs,q,cached,ids,mean):
        begin=base.tick(inputs);anchor=x[ids].double()@pose[:3,:3].T+pose[:3,3]-mean.double()
        callback(pair,level,layer,opt,inputs,q,truth[ids]-mean,anchor,base.tick(inputs)-begin)
    return base.stopping_environment(lambda *args:None,end)[0]


def references():
    d={(r['pair'],r['level'],r['method']):r for r in read(OLD,'directions') if r['repetition']==0}
    f={(r['pair'],r['level'],r['method'],r['policy'],r['cap']):r for r in read(OLD,'finite') if r['repetition']==0}
    states={(r['pair'],r['level']):r for r in read(OLD,'states') if r['repetition']==0}
    return d,f,states


def report(out=OUT,primary='fgr_headmix',raw='fgr_head'):
    OUT=out;METHODS=('cd_gd','adam',raw,primary)
    d,f,_=references();states={(p,l) for p in base.PAIRS for l in range(9)}
    rows={name:read(OUT,name) for name in ('directions','finite','states','solvers')}
    assert base.json.loads((OUT/'integrity.json').read_text())['exact_replays']==list(base.PAIRS)
    poses=base.json.loads((OUT/'poses.json').read_text())
    assert len(poses)==6 and all(p['exact_pose_replay'] for p in poses)
    coverage(rows['states'],('pair','level'),states)
    coverage(rows['solvers'],('pair','level','field'),{(*s,'fgr') for s in states})
    coverage(rows['directions'],('pair','level','method'),{(*s,m) for s in states for m in METHODS})
    coverage(rows['finite'],('pair','level','method','policy','cap'),
        {(*s,m,p,c) for s in states for m in METHODS for p in ('raw','trust') for c in (1/16,.25,1.)})
    assert all(r['converged'] and r['relative_residual']<=1e-8 and r.get('jvp_relative_error',0)<=1e-10 for r in rows['solvers'])
    for r in rows['directions']:
        key=(r['pair'],r['level'],r['method'])
        if r['method'] in ('cd_gd','adam'):
            for name in ('risk_derivative','cd_derivative','motion_bound','output_norm'):
                base.np.testing.assert_allclose(r[name],d[key][name],rtol=1e-10,atol=1e-12)
        assert r['output_norm']==0 or abs(r['output_norm']/r['motion_bound']-1)<1e-10
        d[key]=r
    for r in rows['finite']:
        key=(r['pair'],r['level'],r['method'],r['policy'],r['cap'])
        if r['method'] in ('cd_gd','adam'):assert all(v==f[key][k] for k,v in r.items() if k!='controller_seconds')
        if r['policy']=='trust' and r['accepted']:
            assert r['cd_change']<=.1*r['fraction']*r['cd_slope']+1e-14 and r['linearization_error']<=.1+1e-12
        if r['policy']=='trust' and not r['accepted']:assert r['risk_change']==r['cd_change']==r['output_norm']==0
        f[key]=r
    counts=[];groups=[]
    for group,pairs in (('4DMatch',base.PAIRS[:4]),('HF',base.PAIRS[4:])):
        keys=[(p,l) for p in pairs for l in range(9)]
        for baseline in ('adam','cd_gd','cd_ridge','fgr_gdmix'):
            for cap in (0.,1/16,.25,1.):
                for policy in (('raw',) if cap==0 else ('raw','trust')):
                    a=[d[(*k,baseline)]['risk_derivative'] if cap==0 else f[(*k,baseline,policy,cap)]['risk_change'] for k in keys]
                    c=[d[(*k,primary)]['risk_derivative'] if cap==0 else f[(*k,primary,'trust',cap)]['risk_change'] for k in keys]
                    for eps in (1e-12,1e-10,1e-8):counts.append(dict(group=group,baseline=baseline,cap=cap,base_policy=policy,tolerance=eps,**net_counts(a,c,eps)))
        for method in ('cd_gd','adam','cd_ridge','fgr_gdmix','fgr_ridgemix',raw,primary):
            chosen=[f[(*k,method,'trust',1.)] for k in keys]
            groups.append(dict(group=group,method=method,states=len(keys),
                tangent_descents=sum(d[(*k,method)]['risk_derivative'] < -1e-12 for k in keys),
                finite_descents=sum(r['risk_change'] < -1e-12 for r in chosen),rejected=sum(not r['accepted'] for r in chosen),
                epe_descents=sum(r['epe_change'] < -1e-12 for r in chosen),
                mean_mse_change=float(base.np.mean([r['risk_change'] for r in chosen])),
                mean_epe_change=float(base.np.mean([r['epe_change'] for r in chosen])),
                median_fraction=float(base.np.median([r['fraction'] for r in chosen]))))
    required=[r for r in counts if r['cap'] in (0,.25,1.) and r['tolerance']==1e-12]
    assert len(required)==40
    error=max(r['mse_identity_error'] for r in rows['finite']);passed=all(r['net']>0 for r in required) and error<1e-12
    result=dict(states=54,exact_original_replays=6,exact_pose_replays=6,exact_common_finite_rows=648,
        solver_max_relative_residual=max(r['relative_residual'] for r in rows['solvers']),
        head_max_jvp_relative_error=max((r['jvp_relative_error'] for r in rows['solvers'] if 'jvp_relative_error' in r),default=None),
        native_positive_cells=sum(r['net']>0 for r in required),native_required_cells=40,
        numerical_gate=error<1e-12,max_mse_identity_error=error,advance_to_timing=passed,
        adopted=False,new_scenes=0,candidate_full_fits=0,groups=groups,failed_cells=[r for r in required if r['net']<=0])
    for name,value in (('summary.json',result),('counts.json',counts)):
        (OUT/name).write_text(base.json.dumps(value,indent=2,allow_nan=False)+'\n')
    lines=['Same 54 correlated endpoints. Saved controls reused only after exact common native trials.']
    lines += [str(r) for r in groups]+['Gates: '+str({k:v for k,v in result.items() if k not in ('groups','failed_cells')})]
    lines += ['Failed cells:']+[str(r) for r in result['failed_cells']]
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines));figure(groups,out,primary,raw)


def figure(groups,out=OUT,primary='fgr_headmix',raw='fgr_head'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    names=('cd_gd','adam','fgr_ridgemix',raw,primary)
    for ax,group in zip(axes,('4DMatch','HF')):
        selected=[next(r for r in groups if r['group']==group and r['method']==m) for m in names]
        ax.bar(range(5),[r['finite_descents'] for r in selected])
        labels=('Raw head','Head mix') if primary=='fgr_headmix' else ('Raw PCG','PCG mix')
        ax.set(xticks=range(5),xticklabels=('CD-GD','Adam','Full ridge mix',*labels),title=group,
            ylabel='Task-MSE descent states at cap 1',ylim=(0,39 if group=='4DMatch' else 20))
        ax.tick_params(axis='x',labelrotation=18)
    fig.tight_layout();fig.savefig(out/'head_mix.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    if '--report-only' not in sys.argv:
        D,F,STATES=references();native.OUT=OUT;POSES=native.pose_controls()
        base.OUT=OUT;base.environment=environment;base.main()
    report()
