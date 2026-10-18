"""Replay existing early interventions; observe deformation/gradient/NN changes."""
import json
import sys
import open3d  # Server ABI import order.
from pipeline.model import ROOT,np,torch,config
from pipeline.runner import fit
from pipeline.datasets import transfer
from pipeline.losses import objective
from pipeline.horizon import observer_factory,digest
from pipeline.projection import point_snapshot,projection_metrics
from ndp_mix_trajectory import deterministic

OUT=ROOT/'paper_revision/trajectory_projection_20261018'
PAIRS=(100,101,124,125,200,201)


def measure():
    runs=OUT/'replays';runs.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    cfg=config();cfg.budget=[50]*9;rows=[];completed=[];peak_ram=0
    for pair in PAIRS:
        stem=f'p{pair}_s0_official_w0';history=[];target_hash=None
        _,_,truth,_=transfer(pair)
        pose=json.loads((ROOT/'paper_revision/trajectory_mix_20261018/runs/r0/fgr_ridgemix'/(stem+'_ledger.json')).read_text())['pose']['transform']
        for arm in ('adam','fgr_ridgemix'):
            assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<800_000_000
            updates=[];observations=[];metadata={};index=0;initial=None
            def loss(p,q,method,weight,scale,directions):
                nonlocal index,initial,target_hash
                value=objective(p,q,method,weight,scale,directions)
                snapshot=point_snapshot(p,q,value);current_hash=digest([q])
                if target_hash is None:target_hash=current_hash
                assert current_hash==target_hash
                if arm=='adam':history.append(snapshot)
                else:
                    previous=history[index]
                    if index<50:assert all(torch.equal(snapshot[k],previous[k]) for k in snapshot)
                    else:
                        if initial is None:initial=snapshot['points'].double()-previous['points'].double()
                        level,step=divmod(index,51)
                        rows.append(dict(pair=pair,level=level,step=step,updates_since_intervention=level*50+step-50,
                            **projection_metrics(snapshot,previous,initial)))
                index+=1
                return value
            factory=lambda:observer_factory(arm,pair,0,pose,truth,updates,observations,metadata)
            folder=runs/arm;folder.mkdir(exist_ok=True)
            try:result=fit(pair,0,'official',0,folder,loader=transfer,loss_fn=loss,cfg=cfg,factory=factory)
            finally:
                for suffix,value in (('updates',updates),('observations',observations),('metadata',metadata)):
                    (folder/(stem+'_'+suffix+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
            prior=ROOT/'paper_revision/trajectory_horizon_20261018/runs/site0'/arm
            read=lambda suffix:json.loads((prior/(stem+suffix+'.json')).read_text())
            assert deterministic(result)==deterministic(read(''))
            assert deterministic(updates)==deterministic(read('_updates'))
            assert observations==read('_observations') and metadata==read('_metadata')
            assert index==459 and len(updates)==450 and result['calls_by_level']==[51]*9
            assert np.array_equal(np.loadtxt(folder/(stem+'_trace.txt'),delimiter=',',skiprows=1),
                                  np.loadtxt(prior/(stem+'_trace.txt'),delimiter=',',skiprows=1))
            completed.append(dict(pair=pair,arm=arm));(OUT/'progress.json').write_text(json.dumps(completed,indent=2)+'\n')
        assert len(history)==459
        peak_ram=max(peak_ram,sum(v.numel()*v.element_size() for snapshot in history for v in snapshot.values()))
        (OUT/'observed.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    assert len(completed)==12 and len(rows)==6*409
    (OUT/'integrity.json').write_text(json.dumps(dict(exact_replays=12,exact_update_records=5400,
        exact_observation_records=5508,observed_comparisons=len(rows),peak_snapshot_ram_bytes=peak_ram,
        adoption_test=False,new_scenes=0),indent=2)+'\n')


def report():
    rows=json.loads((OUT/'observed.json').read_text())
    assert json.loads((OUT/'integrity.json').read_text())['exact_replays']==12
    assert [(r['pair'],r['level'],r['step']) for r in rows]==[(p,l,s) for p in PAIRS for l in range(9) for s in range(51) if l*51+s>=50]
    risk=json.loads((ROOT/'paper_revision/trajectory_horizon_20261018/pairs.json').read_text())
    task={(r['pair'],r['level'],r['step']):r for r in json.loads((ROOT/'paper_revision/trajectory_horizon_20261018/paths.json').read_text()) if r['site']==0}
    errors=[];pairs=[];samples=config().samples
    for pair in PAIRS:
        saved=json.loads((OUT/f'replays/adam/p{pair}_s0_official_w0.json').read_text())
        assert saved['source_points']>=samples
    for row in rows:
        errors.append(abs(row['projection']**2+row['orthogonal_gain']**2-row['norm_gain']**2)/max(1,row['norm_gain']**2))
        assert all(v is None or not isinstance(v,float) or np.isfinite(v) for v in row.values())
        assert 0<=row['forward_changed']<=1 and 0<=row['reverse_changed']<=1
        for key in ('deformation_cosine','gradient_cosine'):
            assert row[key] is None or abs(row[key])<=1+1e-12
        row.update({k:v for k,v in task[row['pair'],row['level'],row['step']].items() if k.endswith('_gap')})
        # Reconstruct the exact squared-risk decomposition AFTER joining saved GT.
        row['risk_quadratic_term']=(row['norm_gain']*row['initial_deviation_norm'])**2/samples
        row['risk_linear_term']=row['train_mse_gap']-row['risk_quadratic_term']
    assert max(errors)<1e-12
    for pair in PAIRS:
        path=[r for r in rows if r['pair']==pair];start=path[0]
        assert start['level']==0 and start['step']==50 and abs(start['projection']-1)<1e-12 and abs(start['norm_gain']-1)<1e-12
        first=next(r for r in risk if r['pair']==pair and r['site']==0)['first_train_reversal']
        at=next(r for r in path if (r['level'],r['step'])==(first['level'],first['step']))
        before=[r for r in path if (r['level'],r['step'])<=(at['level'],at['step'])]
        lost=next((r for r in path if r['projection']<=0),None)
        pairs.append(dict(pair=pair,at_first_task_reversal=at,
            reconstructed_risk_type='harmful_linear_term' if at['risk_linear_term']>1e-12 else
                'quadratic_dominates_descent' if at['risk_linear_term'] < -1e-12 else 'near_zero_linear_term',
            minimum_projection_through_reversal=min(r['projection'] for r in before),
            first_nonpositive_projection=None if lost is None else {k:lost[k] for k in ('level','step','updates_since_intervention')},
            maximum_norm_gain=max(r['norm_gain'] for r in path),final=path[-1]))
    result=dict(exact_replays=12,observed_comparisons=len(rows),max_geometry_identity_error=max(errors),
        zero_baseline_gradient_rows=sum(r['baseline_gradient_norm']==0 for r in rows),
        zero_current_gradient_rows=sum(r['gradient_norm']==0 for r in rows),
        positive_projection_at_task_reversal=sum(r['at_first_task_reversal']['projection']>0 for r in pairs),
        projection_at_least_half_at_task_reversal=sum(r['at_first_task_reversal']['projection']>=.5 for r in pairs),
        positive_projection_through_task_reversal=sum(r['minimum_projection_through_reversal']>0 for r in pairs),
        projection_at_least_half_through_task_reversal=sum(r['minimum_projection_through_reversal']>=.5 for r in pairs),
        nonamplified_at_task_reversal=sum(r['at_first_task_reversal']['norm_gain']<=1 for r in pairs),
        point_gradient_cosine_over_098_at_task_reversal=sum(r['at_first_task_reversal']['gradient_cosine'] is not None and r['at_first_task_reversal']['gradient_cosine']>.98 for r in pairs),
        harmful_reconstructed_linear_terms=sum(r['reconstructed_risk_type']=='harmful_linear_term' for r in pairs),
        sample_points=samples,risk_terms_reconstructed=True,
        adopted=False,new_scenes=0,adoption_test=False,pairs=pairs)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (OUT/'joined.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    lines=['Observed path geometry joined with saved task gaps after all replay choices; six reused pairs, no adoption claim.']
    lines += [str(r) for r in pairs]+['Geometry identity maximum relative error: '+str(max(errors))]
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines));figure(rows)


def figure(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,7))
    for ax,pair in zip(axes.flat,PAIRS):
        path=[r for r in rows if r['pair']==pair]
        ax.plot([r['updates_since_intervention'] for r in path],[r['projection'] for r in path],label='Initial direction projection')
        ax.plot([r['updates_since_intervention'] for r in path],[r['orthogonal_gain'] for r in path],label='Orthogonal norm / initial norm',alpha=.8)
        first=next(r for r in path if r['train_mse_gap']>1e-12)
        ax.axvline(first['updates_since_intervention'],color='red',ls=':',label='First task reversal')
        ax.axhline(0,color='black',lw=.7);ax.set(title=f'Pair {pair}',xlabel='Remaining Adam updates',yscale='symlog')
    axes[0,0].legend(fontsize=7)
    fig.text(.5,.01,'Paired deviation from the ordinary-Adam branch. Signed projection and orthogonal growth are different; symmetric-log scale.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(OUT/'projection.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    if '--report-only' not in sys.argv:measure()
    report()
