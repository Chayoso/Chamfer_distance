"""Compare preconditioned/full ridge at the same native original endpoints."""
import sys
import ndp_head_mix_probe as screen
native=screen.native;base=screen.base
OUT=base.ROOT/'paper_revision/head_preconditioner_20261018'
METHODS=('cd_gd','adam','fgr_pcg','fgr_pcgmix')


def compare(pair,level,layer,opt,inputs,target,truth,anchor,anchor_seconds,repetition=0):
    results={};costs={};candidate=None
    # The repeated audit reverses both proposal and controller order.
    order=('fgr_ridgemix','fgr_pcgmix') if (base.PAIRS.index(pair)+level+repetition)%2==0 else ('fgr_pcgmix','fgr_ridgemix')
    for name in order:results[name]=native.directions(layer,inputs,target,anchor,93000+100*pair+level,method=name)
    result,reference=results['fgr_pcgmix'],results['fgr_ridgemix']
    a,b=reference['solvers']['fgr'],result['solvers']['fgr']
    assert a['damping']==b['damping'] and a['spectral_estimate']==b['spectral_estimate']
    assert base.torch.equal(reference['start'],result['start'])
    base.append('paired',dict(pair=pair,level=level,first=order[0],default_solver=a,pcg_solver=b,
        default_proposal_seconds=reference['setup_seconds']+reference['directions']['fgr_ridgemix']['seconds'],
        pcg_proposal_seconds=result['setup_seconds']+result['directions']['fgr_pcgmix']['seconds'],
        raw_velocity_relative_difference=float((result['directions']['fgr_pcg']['velocity']-reference['directions']['fgr_ridge']['velocity']).norm())/result['bound'],
        mixed_velocity_relative_difference=float((result['directions']['fgr_pcgmix']['velocity']-reference['directions']['fgr_ridgemix']['velocity']).norm())/result['bound']))
    before=[p.detach().double().clone() for p in layer.parameters()]
    path=base.continuation(layer,opt,inputs,target,base.cd,steps=1)
    _,start,_,_=next(path);_,native_after,_,params=next(path)
    native_after=native_after.detach()
    adam=[v.detach().double()-p for v,p in zip(params,before)];path.close()
    check=native.directions(layer,inputs,target,None,93000+100*pair+level,method='adam',adam_delta=adam)
    assert base.torch.equal(start.double(),result['native']) and result['initial']==STATES[pair,level]['native_cd']
    result['directions']['adam']=check['directions']['adam']
    gt=truth.double();task=2*(result['start']-gt)/len(gt)
    base.append('solvers',dict(pair=pair,level=level,field='fgr',**b))
    for method in ('cd_gd','adam',*[m for name in order for m in (name.removesuffix('mix'),name)]):
        item=(reference if method in ('fgr_ridge','fgr_ridgemix') else result)['directions'][method]
        velocity=item['velocity'];old_control=method not in ('fgr_pcg','fgr_pcgmix')
        row=dict(pair=pair,level=level,method=method,risk_derivative=float((task*velocity).sum()),
            cd_derivative=float((result['gradient']*velocity).sum()),motion_bound=result['bound'],
            output_norm=float(velocity.norm()),setup_seconds=result['setup_seconds'],direction_seconds=item['seconds'],
            anchor_seconds=anchor_seconds,beta=item.get('beta'))
        if old_control:
            for key in ('risk_derivative','cd_derivative','motion_bound','output_norm'):
                base.np.testing.assert_allclose(row[key],D[pair,level,method][key],rtol=1e-10,atol=1e-12)
        base.append('reference_directions' if method not in METHODS else 'directions',row)
        for trial,output in native.native_trials(layer,inputs,target,result['native'],result['initial'],item['delta'],velocity,result['gradient']):
            value=dict(pair=pair,level=level,method=method,**trial,**native.task_metrics(result['native'],output,gt))
            if old_control:
                old=F[pair,level,method,trial['policy'],trial['cap']]
                assert all(v==old[k] for k,v in value.items() if k!='controller_seconds'),(pair,level,method)
            base.append('reference_finite' if method not in METHODS else 'finite',value)
            if method in ('fgr_pcgmix','fgr_ridgemix') and trial['cap']==1 and trial['policy']=='trust':
                source=reference if method=='fgr_ridgemix' else result
                costs[method]=source['setup_seconds']+item['seconds']+anchor_seconds+trial['controller_seconds']
                if method=='fgr_pcgmix':candidate={k:value[k] for k in ('risk_change','epe_change','mse_identity_error','cd_change','fraction')}
    base.append('states',dict(pair=pair,level=level,native_cd=result['initial'],exact_common_controls=True))
    (OUT/'progress.json').write_text(base.json.dumps(dict(pair=pair,level=level))+'\n')
    return dict(native=result['native'],initial=result['initial'],adam_after=native_after,candidate=candidate,costs=costs)


def report():
    screen.report(OUT,'fgr_pcgmix','fgr_pcg')
    rows=screen.read(OUT,'paired');states={(p,l) for p in base.PAIRS for l in range(9)}
    screen.coverage(rows,('pair','level'),states)
    d,f,_=screen.references()
    refs=screen.read(OUT,'reference_finite')
    screen.coverage(refs,('pair','level','method','policy','cap'),
        {(*s,m,p,c) for s in states for m in ('fgr_ridge','fgr_ridgemix') for p in ('raw','trust') for c in (1/16,.25,1.)})
    for r in refs:
        old=f[r['pair'],r['level'],r['method'],r['policy'],r['cap']]
        assert all(v==old[k] for k,v in r.items() if k!='controller_seconds')
    for r in rows:
        assert r['default_solver']['damping']==r['pcg_solver']['damping']
        assert all(r[name]['converged'] and r[name]['relative_residual']<=1e-8 for name in ('default_solver','pcg_solver'))
    comparisons=[]
    actual=screen.read(OUT,'finite')
    for r in actual:
        if r['method'] not in ('fgr_pcg','fgr_pcgmix'):continue
        name={'fgr_pcg':'fgr_ridge','fgr_pcgmix':'fgr_ridgemix'}[r['method']]
        old=f[r['pair'],r['level'],name,r['policy'],r['cap']]
        comparisons.append(dict(pair=r['pair'],level=r['level'],method=r['method'],policy=r['policy'],cap=r['cap'],
            same_fraction=r['fraction']==old['fraction'],same_acceptance=r['accepted']==old['accepted'],
            risk_change_difference=r['risk_change']-old['risk_change'],epe_change_difference=r['epe_change']-old['epe_change']))
    summary=dict(exact_reference_finite_rows=len(refs),same_fractions=sum(r['same_fraction'] for r in comparisons),
        same_acceptance=sum(r['same_acceptance'] for r in comparisons),compared_finite_rows=len(comparisons),
        max_raw_velocity_relative_difference=max(r['raw_velocity_relative_difference'] for r in rows),
        max_mixed_velocity_relative_difference=max(r['mixed_velocity_relative_difference'] for r in rows),
        groups=[dict(group=group,default_median_iterations=float(base.np.median([r['default_solver']['iterations'] for r in rows if r['pair'] in pairs])),
            pcg_median_iterations=float(base.np.median([r['pcg_solver']['iterations'] for r in rows if r['pair'] in pairs])),
            median_paired_proposal_speedup=float(base.np.median([r['default_proposal_seconds']/r['pcg_proposal_seconds'] for r in rows if r['pair'] in pairs])))
            for group,pairs in (('4DMatch',base.PAIRS[:4]),('HF',base.PAIRS[4:]))],
        timing_is_preliminary=True,matched_time_audit_completed=False,adopted=False)
    for name,value in (('solver_comparison.json',summary),('finite_comparison.json',comparisons)):
        (OUT/name).write_text(base.json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(base.json.dumps(summary),flush=True);figure(rows)


def figure(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for group,pairs in (('4DMatch',base.PAIRS[:4]),('HF',base.PAIRS[4:])):
        selected=[r for r in rows if r['pair'] in pairs]
        axes[0].scatter([r['default_solver']['iterations'] for r in selected],[r['pcg_solver']['iterations'] for r in selected],label=group,alpha=.7)
        axes[1].scatter([r['default_proposal_seconds'] for r in selected],[r['pcg_proposal_seconds'] for r in selected],label=group,alpha=.7)
    for ax,unit in zip(axes,('CG iterations','proposal seconds')):
        limit=max(ax.get_xlim()[1],ax.get_ylim()[1]);ax.plot([0,limit],[0,limit],color='black',ls=':',lw=1)
        ax.set(xlabel='Unpreconditioned '+unit,ylabel='Head-preconditioned '+unit,xlim=(0,limit),ylim=(0,limit));ax.legend()
    fig.text(.5,.01,'Same 54 correlated endpoints; below diagonal is smaller. Proposal timing only; matched-time audit pending.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(OUT/'solver_comparison.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    if '--report-only' not in sys.argv:
        D,F,STATES=screen.references();native.OUT=OUT;POSES=native.pose_controls()
        base.OUT=OUT;base.environment=lambda pair,truth:screen.environment(pair,truth,callback=compare,poses=POSES)
        base.main()
    report()
