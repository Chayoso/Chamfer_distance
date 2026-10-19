"""Restore ordinary updates while preserving both early global event indices."""
import json
from ndp_dense_trajectory import ROOT,np,torch,config,fit,transfer,pose_control,deterministic,exact_reference,PAIRS
from pipeline.optim import fixed_official
from pipeline.trajectory_mix import trajectory_factory
from pipeline.prefix import prefix_observer

OUT=ROOT/'paper_revision/ordinary_restore_20261018'
OLD=ROOT/'paper_revision/dense_pcg_20261018'
ARMS=('full_adam','short_fgr_pcgmix','restored_fgr_pcgmix')


def main():
    assert json.loads((ROOT/'paper_revision/pcg_timing_20261018/summary.json').read_text())['advance_to_cost_based_schedule']
    assert json.loads((OLD/'VALIDATION.json').read_text())['verified_full_fit_records']==96
    root=OUT/'runs';root.mkdir(exist_ok=False);completed=[]
    for repeat in (0,1):
        for i,pair in enumerate(PAIRS):
            offset=(i+repeat)%3;order=ARMS[offset:]+ARMS[:offset];branches={}
            for arm in order:
                assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<800_000_000
                steps=16 if arm=='short_fgr_pcgmix' else 50;cfg=config();cfg.budget=[steps]*9
                folder=root/f'r{repeat}'/arm;folder.mkdir(parents=True,exist_ok=True)
                stem=f'p{pair}_s0_official_w0';updates=[];metadata={}
                pose=pose_control(pair) if arm!='full_adam' else None
                base=fixed_official if arm=='full_adam' else lambda:trajectory_factory('fgr_pcgmix',updates,pair,
                    pose['transform'],updates_per_level=steps,event_steps=(7,15))
                factory,loss=prefix_observer(base,metadata)
                try:result=fit(pair,0,'official',0,folder,loader=transfer,cfg=cfg,factory=factory,loss_fn=loss,
                    trace_steps=None if steps==50 else (0,8,16))
                finally:
                    for suffix,value in (('updates',updates),('prefix',metadata)):
                        (folder/(stem+'_'+suffix+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
                assert result['calls_by_level']==[steps+1]*9 and len(metadata['outputs'])==17
                if pose is not None:
                    assert len(updates)==9*steps and all(r['moment_step']==r['step']+1 for r in updates)
                    assert [(r['level'],r['step']) for r in updates if r['controlled']]==[(l,s) for l in range(9) for s in (7,15)]
                    assert all(r['valid'] and r['applied_exact'] for r in updates if r['controlled'])
                if arm!='restored_fgr_pcgmix':exact_reference(folder,OLD/'runs/r0'/arm,stem,result,updates)
                if repeat:
                    first=root/'r0'/arm;exact_reference(folder,first,stem,result,updates)
                    assert metadata==json.loads((first/(stem+'_prefix.json')).read_text())
                prep=0. if pose is None else pose['seconds']
                ledger=dict(pair=pair,arm=arm,repetition=repeat,pose=pose,updates_per_level=steps,
                    event_steps=[] if pose is None else [7,15],fit_seconds=result['seconds'],preparation_seconds=prep,
                    total_seconds=result['seconds']+prep,updates=9*steps,loss_calls=9*(steps+1),
                    exact_saved_replay=arm!='restored_fgr_pcgmix',repeat_exact=bool(repeat))
                (folder/(stem+'_ledger.json')).write_text(json.dumps(ledger,indent=2,allow_nan=False)+'\n')
                branches[arm]=(metadata,updates)
                completed.append(dict(pair=pair,arm=arm,repetition=repeat))
                (OUT/'progress.json').write_text(json.dumps(dict(completed=completed),indent=2)+'\n')
                print(json.dumps(completed[-1]),flush=True)
            a,b=[branches[k] for k in ARMS[1:]]
            assert a[0]==b[0] and deterministic(a[1][:16])==deterministic(b[1][:16])
    assert len(completed)==36
    (OUT/'integrity.json').write_text(json.dumps(dict(completed_fits=36,exact_saved_replays=24,exact_repeat_fits=18,
        exact_shared_prefixes=12,shared_prediction_hashes=204,exact_pose_replays=24,
        attempted_updates=12528,training_loss_calls=12852),indent=2)+'\n')
    from ordinary_restore_report import main as report
    report()


if __name__=='__main__':
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;main()
