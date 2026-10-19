"""One scalar-rule change on all54 original states, with exact native controls."""
import sys
import ndp_head_mix_probe as screen

native=screen.native;base=screen.base
OUT=base.ROOT/'paper_revision/normalized_mix_20261018'
OLD=base.ROOT/'paper_revision/head_preconditioner_20261018'
PRIMARY='fgr_normmix'


def compare(pair,level,layer,opt,inputs,target,truth,anchor,anchor_seconds):
    order=('fgr_pcgmix',PRIMARY) if (base.PAIRS.index(pair)+level)%2==0 else (PRIMARY,'fgr_pcgmix')
    results={name:native.directions(layer,inputs,target,anchor,93000+100*pair+level,method=name) for name in order}
    result,old=results[PRIMARY],results['fgr_pcgmix'];rays=result['directions']
    for name in ('cd_gd','fgr_pcg'):
        assert base.torch.equal(rays[name]['velocity'],old['directions'][name]['velocity'])
    assert rays[PRIMARY]['beta']<=old['directions']['fgr_pcgmix']['beta']
    before=[p.detach().double().clone() for p in layer.parameters()]
    path=base.continuation(layer,opt,inputs,target,base.cd,steps=1)
    _,start,_,_=next(path);_,_,_,params=next(path)
    adam=[v.detach().double()-p for v,p in zip(params,before)];path.close()
    check=native.directions(layer,inputs,target,None,93000+100*pair+level,method='adam',adam_delta=adam)
    assert base.torch.equal(start.double(),result['native']) and result['initial']==STATES[pair,level]['native_cd']
    rays['adam']=check['directions']['adam'];gt=truth.double();task=2*(result['start']-gt)/len(gt)
    base.append('solvers',dict(pair=pair,level=level,field='fgr',**result['solvers']['fgr']))
    for method in ('cd_gd','adam','fgr_pcg',*order):
        item=old['directions'][method] if method=='fgr_pcgmix' else rays[method];velocity=item['velocity']
        row=dict(pair=pair,level=level,method=method,risk_derivative=float((task*velocity).sum()),
            cd_derivative=float((result['gradient']*velocity).sum()),motion_bound=result['bound'],output_norm=float(velocity.norm()),
            setup_seconds=result['setup_seconds'],direction_seconds=item['seconds'],anchor_seconds=anchor_seconds,beta=item.get('beta'))
        if method!=PRIMARY:
            for key in ('risk_derivative','cd_derivative','motion_bound','output_norm'):
                base.np.testing.assert_allclose(row[key],D[pair,level,method][key],rtol=1e-10,atol=1e-12)
        base.append('reference_directions' if method=='fgr_pcgmix' else 'directions',row)
        for trial,output in native.native_trials(layer,inputs,target,result['native'],result['initial'],item['delta'],velocity,result['gradient']):
            value=dict(pair=pair,level=level,method=method,**trial,**native.task_metrics(result['native'],output,gt))
            if method!=PRIMARY:
                saved=F[pair,level,method,trial['policy'],trial['cap']]
                assert all(v==saved[k] for k,v in value.items() if k!='controller_seconds'),(pair,level,method)
            base.append('reference_finite' if method=='fgr_pcgmix' else 'finite',value)
    base.append('states',dict(pair=pair,level=level,native_cd=result['initial'],exact_common_controls=True))
    (OUT/'progress.json').write_text(base.json.dumps(dict(pair=pair,level=level))+'\n')


def report():
    screen.report(OUT,PRIMARY,'fgr_pcg')
    old={ (r['pair'],r['level']):r for r in screen.read(OUT,'reference_finite') if r['cap']==1 and r['policy']=='trust'}
    finite=[r for r in screen.read(OUT,'finite') if r['method']==PRIMARY and r['cap']==1 and r['policy']=='trust']
    directions={(r['pair'],r['level']):r for r in screen.read(OUT,'directions') if r['method']==PRIMARY}
    refs={(r['pair'],r['level']):r for r in screen.read(OUT,'reference_directions')}
    pairs=[]
    for pair in base.PAIRS:
        chosen=[r for r in finite if r['pair']==pair];prior=[old[pair,r['level']] for r in chosen]
        pairs.append(dict(pair=pair,events=len(chosen),
            old_descents=sum(r['risk_change'] < -1e-12 for r in prior),new_descents=sum(r['risk_change'] < -1e-12 for r in chosen),
            old_mean_mse=float(base.np.mean([r['risk_change'] for r in prior])),new_mean_mse=float(base.np.mean([r['risk_change'] for r in chosen])),
            old_mean_epe=float(base.np.mean([r['epe_change'] for r in prior])),new_mean_epe=float(base.np.mean([r['epe_change'] for r in chosen])),
            changed_beta=sum(directions[pair,r['level']]['beta']<refs[pair,r['level']]['beta'] for r in chosen)))
    (OUT/'pair_counts.json').write_text(base.json.dumps(pairs,indent=2)+'\n')


if __name__=='__main__':
    if '--report-only' not in sys.argv:
        D,F,STATES=screen.references()
        for r in screen.read(OLD,'directions'):D[r['pair'],r['level'],r['method']]=r
        for r in screen.read(OLD,'finite'):F[r['pair'],r['level'],r['method'],r['policy'],r['cap']]=r
        native.OUT=OUT;POSES=native.pose_controls();base.OUT=OUT
        base.environment=lambda pair,truth:screen.environment(pair,truth,callback=compare,poses=POSES)
        base.main()
    report()
