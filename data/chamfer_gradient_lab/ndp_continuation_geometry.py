"""Replay six verified paths and measure actual update geometry without changes."""
import json
from ndp_ordinary_restore import ROOT,np,torch,config,fit,transfer,pose_control,exact_reference,PAIRS
from pipeline.trajectory_mix import trajectory_factory
from pipeline.prefix import prefix_observer
from pipeline.update_geometry import geometry_observer

OUT=ROOT/'paper_revision/continuation_geometry_20261018'
OLD=ROOT/'paper_revision/ordinary_restore_20261018'


def replay(pair,folder,reference):
    cfg=config();cfg.budget=[50]*9;updates=[];snapshots=[];movements=[];metadata={}
    _,_,truth,_=transfer(pair);pose=pose_control(pair)
    base=lambda:trajectory_factory('fgr_pcgmix',updates,pair,pose['transform'],updates_per_level=50,event_steps=(7,15))
    factory,loss=geometry_observer(base,truth,pose['transform'],updates,snapshots,movements)
    factory,loss=prefix_observer(factory,metadata,loss_fn=loss)
    stem=f'p{pair}_s0_official_w0'
    try:result=fit(pair,0,'official',0,folder,loader=transfer,cfg=cfg,factory=factory,loss_fn=loss)
    finally:
        for suffix,value in (('updates',updates),('prefix',metadata),('snapshots',snapshots),('movements',movements)):
            (folder/(stem+'_'+suffix+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    exact_reference(folder,reference,stem,result,updates)
    assert metadata==json.loads((reference/(stem+'_prefix.json')).read_text())
    assert [(r['level'],r['step']) for r in snapshots]==[(l,s) for l in range(9) for s in range(51)]
    assert [(r['level'],r['step']) for r in movements]==[(l,s) for l in range(9) for s in range(50)]
    assert result['calls_by_level']==[51]*9 and len(updates)==450
    return dict(pair=pair,exact_native_replay=True,exact_prefix=True,snapshots=len(snapshots),movements=len(movements),pose_exact=pose['exact'])


def main():
    assert json.loads((OLD/'VALIDATION.json').read_text())['verified_full_fit_records']==36
    folder=OUT/'runs';folder.mkdir(exist_ok=False);completed=[]
    for pair in PAIRS:
        assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<800_000_000
        completed.append(replay(pair,folder,OLD/'runs/r0/restored_fgr_pcgmix'))
        (OUT/'progress.json').write_text(json.dumps(completed,indent=2)+'\n');print(json.dumps(completed[-1]),flush=True)
    (OUT/'integrity.json').write_text(json.dumps(dict(completed_replays=6,exact_update_records=2700,
        observations=2754,actual_movements=2700,held_observations=108,new_candidates=0,new_scenes=0),indent=2)+'\n')
    from continuation_geometry_report import main as report
    report()


if __name__=='__main__':
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;main()
