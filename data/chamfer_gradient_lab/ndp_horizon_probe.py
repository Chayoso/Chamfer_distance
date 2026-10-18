"""Single early/late intervention with an identical prefix and unchanged future."""
import json
import open3d  # Server ABI import order.
from pipeline.model import ROOT,np,torch,config
from pipeline.runner import fit
from pipeline.datasets import transfer
from pipeline.horizon import observer_factory
from ndp_mix_trajectory import deterministic

OUT=ROOT/'paper_revision/trajectory_horizon_20261018'
PAIRS=(100,101,124,125,200,201)


def main():
    root=OUT/'runs';root.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    cfg=config();cfg.budget=[50]*9;completed=[]
    for pair in PAIRS:
        _,_,truth,_=transfer(pair)
        stem=f'p{pair}_s0_official_w0';prior=ROOT/'paper_revision/trajectory_mix_20261018/runs/r0'
        pose=json.loads((prior/'fgr_ridgemix'/(stem+'_ledger.json')).read_text())['pose']['transform']
        for site in (0,8):
            branches={}
            for arm in ('adam','fgr_ridgemix'):
                assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<800_000_000
                folder=root/f'site{site}'/arm;folder.mkdir(parents=True,exist_ok=True)
                updates=[];observations=[];metadata={}
                factory=lambda:observer_factory(arm,pair,site,pose,truth,updates,observations,metadata)
                try:
                    result=fit(pair,0,'official',0,folder,loader=transfer,cfg=cfg,factory=factory)
                finally:
                    for suffix,value in (('updates',updates),('observations',observations),('metadata',metadata)):
                        (folder/(stem+'_'+suffix+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
                assert result['calls_by_level']==[51]*9 and len(updates)==450
                assert [(r['level'],r['step']) for r in observations]==[(l,s) for l in range(9) for s in range(51)]
                controls=[r for r in updates if r['controlled']]
                assert [(r['level'],r['step']) for r in controls]==([] if arm=='adam' else [(site,49)])
                assert all(r['moment_step']==r['step']+1 for r in updates)
                assert all(r['valid'] and r['applied_exact'] for r in controls)
                trace=np.loadtxt(folder/(stem+'_trace.txt'),delimiter=',',skiprows=1)
                assert trace.shape==(27,8) and np.isfinite(trace).all()
                if arm=='adam':
                    expected=json.loads((prior/arm/(stem+'.json')).read_text())
                    assert deterministic(result)==deterministic(expected)
                    assert np.array_equal(trace,np.loadtxt(prior/arm/(stem+'_trace.txt'),delimiter=',',skiprows=1))
                branches[arm]=dict(metadata=metadata,observations=observations,updates=updates)
                completed.append(dict(pair=pair,site=site,arm=arm))
                (OUT/'progress.json').write_text(json.dumps(completed,indent=2)+'\n')
            a,b=branches['adam'],branches['fgr_ridgemix']
            assert a['metadata']==b['metadata']
            before=site*51+50
            assert a['observations'][:before]==b['observations'][:before]
            assert deterministic(a['updates'][:site*50+49])==deterministic(b['updates'][:site*50+49])
            print(json.dumps(dict(pair=pair,site=site,exact_prefix=True)),flush=True)
    assert len(completed)==24
    (OUT/'integrity.json').write_text(json.dumps(dict(completed_fits=24,exact_plain_replays=12,
        exact_shared_prefixes=12,controlled_updates=12,attempted_updates=10800,training_loss_calls=11016,
        adoption_test=False,new_scenes=0),indent=2)+'\n')


if __name__=='__main__':main()
