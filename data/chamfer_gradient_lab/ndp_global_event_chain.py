"""Observe all stages of108 already committed global events on six exact paths."""
import json
from ndp_ordinary_restore import ROOT,np,torch,config,fit,transfer,pose_control,exact_reference,PAIRS
from pipeline.prefix import prefix_observer
from pipeline.event_chain import event_chain_factory

OUT=ROOT/'paper_revision/global_event_chain_20261018'
OLD=ROOT/'paper_revision/continuation_geometry_20261018'


def replay(pair,folder):
    cfg=config();cfg.budget=[50]*9;updates=[];events=[];metadata={};pose=pose_control(pair)
    _,_,truth,_=transfer(pair)
    factory,loss=prefix_observer(lambda:event_chain_factory(updates,events,pair,pose['transform'],truth),metadata)
    stem=f'p{pair}_s0_official_w0'
    try:result=fit(pair,0,'official',0,folder,loader=transfer,cfg=cfg,factory=factory,loss_fn=loss)
    finally:
        for suffix,value in (('updates',updates),('prefix',metadata),('events',events)):
            (folder/(stem+'_'+suffix+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    exact_reference(folder,OLD/'runs',stem,result,updates)
    assert metadata==json.loads((OLD/'runs'/(stem+'_prefix.json')).read_text())
    assert [(r['level'],r['step']) for r in events]==[(l,s) for l in range(9) for s in (7,15)]
    previous={(r['level'],r['step']):r for r in json.loads((OLD/'runs'/(stem+'_movements.json')).read_text())}
    for row in events:
        prior=previous[row['level'],row['step']]
        for key in ('task','anchor','fraction','beta','accepted','cd_change','linearization_error'):assert row[key]==prior[key],key
    return dict(pair=pair,exact_native_replay=True,exact_prefix=True,exact_prior_finite_events=18,pose_exact=pose['exact'])


def main():
    assert json.loads((OLD/'VALIDATION.json').read_text())['verified_native_replays']==6
    folder=OUT/'runs';folder.mkdir(exist_ok=False);completed=[]
    for pair in PAIRS:
        assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<800_000_000
        completed.append(replay(pair,folder))
        (OUT/'progress.json').write_text(json.dumps(completed,indent=2)+'\n');print(json.dumps(completed[-1]),flush=True)
    (OUT/'integrity.json').write_text(json.dumps(dict(completed_replays=6,exact_update_records=2700,
        exact_prior_finite_events=108,global_events=108,new_candidates=0,new_scenes=0),indent=2)+'\n')
    from global_event_chain_report import main as report
    report()


if __name__=='__main__':
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;main()
