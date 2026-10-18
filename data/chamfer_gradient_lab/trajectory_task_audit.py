"""Post-hoc task diagnosis of every mixed-trajectory intervention; no tuning."""
import copy
import json
import open3d  # Preserve server import order.
from pipeline.model import ROOT,torch,np,config
from pipeline.runner import fit
from pipeline.datasets import transfer
from pipeline.controlled import cd
from pipeline.mse_reference import mse_reference
import pipeline.trajectory_mix as trajectory
from ndp_mix_trajectory import deterministic

OUT=ROOT/'paper_revision/trajectory_mix_20261018';AUDIT=OUT/'task_audit'


def change(start,after,truth):
    result=mse_reference(*[v.detach().cpu().numpy() for v in (start,after,truth)])
    return dict(mse_change=result['reference_change'],identity_error=result['reference_identity_error'])


def main():
    AUDIT.mkdir(exist_ok=False);folder=AUDIT/'replays';folder.mkdir()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    original_direction,original_trials=trajectory.directions,trajectory.native_trials
    cfg=config();cfg.budget=[50]*9;rows=[];completed=[]
    try:
        for pair in (100,101,124,125,200,201):
            assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<800_000_000
            _,y,truth,_=transfer(pair);state={};updates=[];stem=f'p{pair}_s0_official_w0'
            reference=OUT/'runs/r0/fgr_ridgemix'
            pose=json.loads((reference/(stem+'_ledger.json')).read_text())['pose']['transform']
            def factory():
                env=trajectory.trajectory_factory('fgr_ridgemix',updates,pair,pose);build=env['Deformation_Pyramid']
                def constructor(**kwargs):
                    net=build(**kwargs)
                    generator=torch.Generator().set_state(torch.get_rng_state())
                    ids=torch.randperm(len(truth),generator=generator)[:cfg.samples].to(truth.device)
                    state['truth']=(truth[ids]-y.mean(0)).double()
                    return net
                env['Deformation_Pyramid']=constructor
                return env
            def direction(layer,inputs,target,anchor,seed,**kwargs):
                result=original_direction(layer,inputs,target,anchor,seed,**kwargs)
                gt=state['truth'];task=2*(result['start']-gt)/len(gt)
                row=dict(pair=pair,level=len([r for r in rows if r['pair']==pair]),
                    point_anchor_derivative=float((task*(anchor-result['start'])).sum()),
                    task_slopes={m:float((task*d['velocity']).sum()) for m,d in result['directions'].items()},
                    beta=result['directions']['fgr_ridgemix']['beta'])
                # Replay the exact actual Adam delta already computed from this own state.
                model=copy.deepcopy(layer);model._forward_pre_hooks.clear()
                with torch.no_grad():
                    for p,v,d in zip(model.parameters(),layer.parameters(),kwargs['adam_delta']):p.copy_(v.double()+d)
                    after=model(inputs)[0].detach()
                row['actual_adam']=dict(**change(result['native'],after.double(),gt),
                    cd_change=float(cd(after,target))-result['initial'])
                state.update(row=row,truth=gt)
                return result
            def trials(*args,**kwargs):
                for selected,output in original_trials(*args,**kwargs):
                    if selected['policy']=='trust':
                        row=state['row']
                        row['selected']=dict(**change(args[3],output,state['truth']),
                            accepted=selected['accepted'],fraction=selected['fraction'],cd_change=selected['cd_change'])
                        rows.append(row)
                    yield selected,output
            trajectory.directions=direction;trajectory.native_trials=trials
            result=fit(pair,0,'official',0,folder,loader=transfer,cfg=cfg,factory=factory)
            assert deterministic(result)==deterministic(json.loads((reference/(stem+'.json')).read_text()))
            assert deterministic(updates)==deterministic(json.loads((reference/(stem+'_updates.json')).read_text()))
            assert np.array_equal(np.loadtxt(folder/(stem+'_trace.txt'),delimiter=',',skiprows=1),
                                  np.loadtxt(reference/(stem+'_trace.txt'),delimiter=',',skiprows=1))
            completed.append(pair)
            (AUDIT/'interventions.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    finally:
        trajectory.directions=original_direction;trajectory.native_trials=original_trials
    assert [(r['pair'],r['level']) for r in rows]==[(p,l) for p in completed for l in range(9)] and len(rows)==54
    summary=[]
    for group,pairs in (('4DMatch',(100,101,124,125)),('HF',(200,201))):
        chosen=[r for r in rows if r['pair'] in pairs]
        summary.append(dict(group=group,states=len(chosen),accepted=sum(r['selected']['accepted'] for r in chosen),
            point_anchor_descents=sum(r['point_anchor_derivative']<-1e-12 for r in chosen),
            tangent_descents={m:sum(r['task_slopes'][m]<-1e-12 for r in chosen) for m in ('cd_gd','fgr_ridge','fgr_ridgemix')},
            actual_adam_descents=sum(r['actual_adam']['mse_change']<-1e-12 for r in chosen),
            selected_descents=sum(r['selected']['mse_change']<-1e-12 for r in chosen),
            selected_worse_than_actual_adam=sum(r['selected']['mse_change']>r['actual_adam']['mse_change']+1e-12 for r in chosen),
            mean_selected_mse_change=float(np.mean([r['selected']['mse_change'] for r in chosen])),
            mean_actual_adam_mse_change=float(np.mean([r['actual_adam']['mse_change'] for r in chosen]))))
    identity=max(r[k]['identity_error'] for r in rows for k in ('selected','actual_adam'));assert identity<1e-12
    report=dict(exact_replays=completed,exact_update_records=2700,states=54,max_identity_error=identity,groups=summary,
        post_hoc=True,changes_to_method=0,new_scenes=0)
    (AUDIT/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
