"""Frozen compute-only9x16 allocation, strong full/short controls, two repeats."""
import hashlib
import json
import sys
from ndp_mix_trajectory import ROOT,np,torch,config,fit,transfer,pose_control,deterministic
from pipeline.optim import fixed_official
from pipeline.trajectory_mix import trajectory_factory

OUT=ROOT/'paper_revision/dense_pcg_20261018'
OLD=ROOT/'paper_revision/trajectory_mix_20261018/runs'
PAIRS=(100,101,124,125,200,201)
ARMS=('full_adam','full_adam_trust','short_adam','short_adam_trust',
      'short_pcg_end','short_cd_ridge','short_fgr_gdmix','short_fgr_pcgmix')
PRIMARY='short_fgr_pcgmix'
KINDS=dict(zip(ARMS,('adam','adam_trust','adam','adam_trust','fgr_pcgmix','cd_ridge','fgr_gdmix','fgr_pcgmix')))


def allocation():
    value=json.loads((OUT/'CALIBRATION.json').read_text())
    assert not value['uses_task_metrics'] and value['selection']['updates_per_level']==16
    for name,digest in value['input_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    assert value['event_steps']==[7,15]
    return value['selection']['updates_per_level']


def settings(arm,n):
    steps=50 if arm.startswith('full') else n
    events=(steps-1,) if arm.startswith('full') or arm=='short_pcg_end' else (steps//2-1,steps-1)
    return KINDS[arm],steps,events


def exact_reference(folder,reference,stem,result,records):
    assert deterministic(result)==deterministic(json.loads((reference/(stem+'.json')).read_text()))
    assert deterministic(records)==deterministic(json.loads((reference/(stem+'_updates.json')).read_text()))
    assert (folder/(stem+'_trace.txt')).read_bytes()==(reference/(stem+'_trace.txt')).read_bytes()


def preflight():
    folder=OUT/'preflight';folder.mkdir(exist_ok=False);cfg=config();cfg.budget=[50]*9
    for pair in (100,200):
        records=[];pose=pose_control(pair);stem=f'p{pair}_s0_official_w0'
        try:result=fit(pair,0,'official',0,folder,loader=transfer,cfg=cfg,
            factory=lambda:trajectory_factory('fgr_ridgemix',records,pair,pose['transform']))
        finally:(folder/(stem+'_updates.json')).write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
        exact_reference(folder,OLD/'r0/fgr_ridgemix',stem,result,records)
    (folder/'complete.json').write_text(json.dumps(dict(exact_default_replays=2,exact_update_records=900),indent=2)+'\n')


def main():
    n=allocation()
    assert json.loads((ROOT/'paper_revision/pcg_timing_20261018/summary.json').read_text())['advance_to_cost_based_schedule']
    assert json.loads((OUT/'preflight/complete.json').read_text())['exact_default_replays']==2
    root=OUT/'runs';root.mkdir(exist_ok=False);completed=[]
    for repeat in (0,1):
        for i,pair in enumerate(PAIRS):
            offset=(i+repeat)%len(ARMS);order=ARMS[offset:]+ARMS[:offset]
            for arm in order:
                assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<800_000_000
                kind,steps,events=settings(arm,n);cfg=config();cfg.budget=[steps]*9
                folder=root/f'r{repeat}'/arm;folder.mkdir(parents=True,exist_ok=True)
                stem=f'p{pair}_s0_official_w0';records=[];pose=pose_control(pair) if kind.startswith('fgr') else None
                factory=fixed_official if kind=='adam' else lambda:trajectory_factory(kind,records,pair,
                    None if pose is None else pose['transform'],updates_per_level=steps,event_steps=events)
                try:result=fit(pair,0,'official',0,folder,loader=transfer,cfg=cfg,factory=factory,
                    trace_steps=None if steps==50 else (0,steps//2,steps))
                finally:(folder/(stem+'_updates.json')).write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
                assert result['calls_by_level']==[steps+1]*9
                if kind!='adam':
                    assert len(records)==9*steps and all(r['moment_step']==r['step']+1 for r in records)
                    assert [(r['level'],r['step']) for r in records if r['controlled']]==[(l,s) for l in range(9) for s in events]
                    assert all(r['valid'] and r['applied_exact'] for r in records if r['controlled'])
                trace=np.loadtxt(folder/(stem+'_trace.txt'),delimiter=',',skiprows=1)
                assert trace.shape==(27,8) and np.isfinite(trace).all()
                assert np.array_equal(trace[:,:2],np.asarray([(l,s) for l in range(9) for s in (0,steps//2,steps)]))
                if arm.startswith('full'):exact_reference(folder,OLD/'r0'/kind,stem,result,records)
                if repeat:exact_reference(folder,root/'r0'/arm,stem,result,records)
                prep=0. if pose is None else pose['seconds']
                ledger=dict(pair=pair,arm=arm,kind=kind,repetition=repeat,pose=pose,updates_per_level=steps,event_steps=list(events),
                    fit_seconds=result['seconds'],preparation_seconds=prep,total_seconds=result['seconds']+prep,
                    updates=9*steps,loss_calls=9*(steps+1),exact_full_replay=arm.startswith('full'),repeat_exact=bool(repeat))
                (folder/(stem+'_ledger.json')).write_text(json.dumps(ledger,indent=2,allow_nan=False)+'\n')
                completed.append(dict(pair=pair,arm=arm,repetition=repeat));print(json.dumps(completed[-1]),flush=True)
                (OUT/'progress.json').write_text(json.dumps(dict(completed=completed),indent=2)+'\n')
    assert len(completed)==96
    (OUT/'integrity.json').write_text(json.dumps(dict(completed_fits=96,exact_full_replays=24,exact_repeat_fits=48,
        attempted_updates=21168,training_loss_calls=22032,exact_pose_replays=36),indent=2)+'\n')
    from dense_trajectory_report import main as report
    report()


if __name__=='__main__':
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    preflight() if '--preflight' in sys.argv else main()
