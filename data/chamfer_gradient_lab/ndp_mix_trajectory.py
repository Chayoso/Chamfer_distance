"""Frozen five-arm, two-repeat own-trajectory study; no GT update inputs."""
import time
import open3d as o3d  # Server ABI import order.
from pipeline.model import ROOT, np, torch, config
from pipeline.optim import fixed_official
from pipeline.runner import fit
from pipeline.datasets import transfer
from pipeline.trajectory_mix import trajectory_factory
from global_registration_control import load, prepare
import json
import sys

OUT=ROOT/'paper_revision/trajectory_mix_20261018'
PAIRS=(100,101,124,125,200,201)
ARMS=('adam','adam_trust','cd_ridge','fgr_gdmix','fgr_ridgemix')


def pose_control(pair):
    assert o3d.__version__=='0.19.0'
    row=next(r for r in json.loads((ROOT/'paper_revision/fgr_20261018/observed.json').read_text())['rows'] if r['pair']==pair)
    x,y,_,_=load(pair);begin=time.perf_counter()
    voxel,clouds,features=prepare(x,y);reg=o3d.pipelines.registration
    o3d.utility.random.seed(row['seed'])
    result=reg.registration_fgr_based_on_feature_matching(*clouds,*features,reg.FastGlobalRegistrationOption(**row['options']))
    seconds=time.perf_counter()-begin
    assert voxel==row['voxel'] and np.array_equal(result.transformation,np.asarray(row['transform']))
    return dict(transform=result.transformation.tolist(),seconds=seconds,exact=True)


def deterministic(value):
    if isinstance(value,dict):return {k:deterministic(v) for k,v in value.items() if 'seconds' not in k}
    if isinstance(value,list):return [deterministic(v) for v in value]
    return value


def main():
    assert json.loads((OUT/'preflight/complete.json').read_text())['exact_methods']==24
    resume='--resume' in sys.argv
    root=OUT/'runs';root.mkdir(exist_ok=resume)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    cfg=config();cfg.budget=[50]*9;completed=[]
    for repetition in (0,1):
        for i,pair in enumerate(PAIRS):
            offset=(i+repetition)%len(ARMS);order=ARMS[offset:]+ARMS[:offset]
            for arm in order:
                assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<800_000_000
                folder=root/f'r{repetition}'/arm;folder.mkdir(parents=True,exist_ok=True)
                stem=f'p{pair}_s0_official_w0';records=[];pose=None
                if resume and (folder/(stem+'_ledger.json')).exists():
                    completed.append(dict(pair=pair,arm=arm,repetition=repetition));continue
                if resume and (folder/(stem+'.json')).exists():
                    # Only a completed plain fit can be recovered without a pose ledger.
                    assert arm=='adam'
                    result=json.loads((folder/(stem+'.json')).read_text())
                    records=json.loads((folder/(stem+'_updates.json')).read_text())
                else:
                    if arm.startswith('fgr'):pose=pose_control(pair)
                    factory=fixed_official if arm=='adam' else lambda:trajectory_factory(arm,records,pair,None if pose is None else pose['transform'])
                    try:
                        result=fit(pair,0,'official',0,folder,loader=transfer,cfg=cfg,factory=factory)
                    finally:
                        (folder/(stem+'_updates.json')).write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
                assert result['calls_by_level']==[51]*9
                if arm!='adam':
                    assert len(records)==450 and [(r['level'],r['step']) for r in records if r['controlled']]==[(j,49) for j in range(9)]
                    assert all(r['moment_step']==r['step']+1 for r in records)
                    assert all(r['valid'] and r['applied_exact'] for r in records if r['controlled'])
                trace=np.loadtxt(folder/(stem+'_trace.txt'),delimiter=',',skiprows=1)
                assert trace.shape==(27,8) and np.isfinite(trace).all()
                radius=None
                if arm=='adam':
                    reference=ROOT/'paper_revision'/('replication_4d' if pair<200 else 'replication_hf')
                    expected=json.loads((reference/(stem+'.json')).read_text())
                    values=dict(result)
                    if pair>=200:
                        with np.load(ROOT/f'assets/hf_3dmatch/{pair}.npz') as asset:radius=float(asset['scale'])
                        values.update(source_radius_m=radius,**{g+'_epe_m':result[g+'_epe']*radius for g in ('full','held','vis','occ')})
                    assert all(values[k]==v for k,v in expected.items() if k not in ('seconds','updates_per_level'))
                    assert expected['updates_per_level']==[50]*9
                    assert np.array_equal(trace,np.loadtxt(reference/(stem+'_trace.txt'),delimiter=',',skiprows=1))
                if repetition:
                    first=root/'r0'/arm
                    assert deterministic(result)==deterministic(json.loads((first/(stem+'.json')).read_text()))
                    assert deterministic(records)==deterministic(json.loads((first/(stem+'_updates.json')).read_text()))
                    assert np.array_equal(trace,np.loadtxt(first/(stem+'_trace.txt'),delimiter=',',skiprows=1))
                ledger=dict(pair=pair,arm=arm,repetition=repetition,pose=pose,fit_seconds=result['seconds'],
                    source_radius_m=radius,
                    preparation_seconds=0. if pose is None else pose['seconds'],
                    total_seconds=result['seconds']+(0. if pose is None else pose['seconds']),updates=450,loss_calls=459,
                    exact_plain_replay=arm=='adam',repeat_exact=bool(repetition))
                (folder/(stem+'_ledger.json')).write_text(json.dumps(ledger,indent=2,allow_nan=False)+'\n')
                completed.append(dict(pair=pair,arm=arm,repetition=repetition))
                (OUT/'progress.json').write_text(json.dumps(dict(completed=completed),indent=2)+'\n')
                print(json.dumps(completed[-1]),flush=True)
    assert len(completed)==60
    (OUT/'integrity.json').write_text(json.dumps(dict(completed_fits=60,exact_plain_replays=12,exact_repeat_fits=30,
        attempted_updates=27000,training_loss_calls=27540,exact_pose_replays=24),indent=2)+'\n')


if __name__=='__main__':main()
