"""Standard point-to-point ICP as an explicit rigid-task capacity control."""
import json
import argparse
import time
from scipy.spatial import cKDTree
from ndp_pipeline import ROOT,np,torch,official


def align(p,q,all_points):
    a,b=p.mean(0),q.mean(0)
    u,_,vt=np.linalg.svd(np.einsum('ni,nj->ij',p-a,q-b))
    sign=np.linalg.det(vt.T@u.T);r=vt.T@np.diag([1,1,sign])@u.T
    return np.einsum('ij,nj->ni',r,all_points-a)+b


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--initialization',choices=['centroid','identity'],default='centroid');args=parser.parse_args()
    torch.set_num_threads(2);env=official();rows=[]
    for dataset,pairs,seeds in [('stanford_raw',[4,5,6,7,16,17,18,19],[0,1]),('hf_3dmatch',range(200,212),[0,1,2])]:
        for pair in pairs:
            with np.load(ROOT/f'assets/{dataset}/{pair:02d}.npz',allow_pickle=False) as z:
                scale=float(z['scale']);x,y,g=z['source']/scale,z['target']/scale,z['truth']/scale
            for seed in seeds:
                torch.manual_seed(17000+seed)
                net=env['Deformation_Pyramid'](depth=3,width=128,device='cpu',k0=-8,m=9,rotation_format='axis_angle',nonrigidity_est=False,motion='SE3')
                ids=torch.randperm(len(x))[:2000].numpy();idy=torch.randperm(len(y))[:2000].numpy()
                held=np.ones(len(x),bool);held[ids]=False;target=y[idy];tree=cKDTree(target)
                pred=x-x.mean(0)+y.mean(0) if args.initialization=='centroid' else x.copy();previous=np.inf;start=time.perf_counter()
                for step in range(100):
                    distance,nearest=tree.query(pred[ids]);err=np.mean(distance**2)
                    pred=align(pred[ids],target[nearest],pred)
                    if abs(previous-err)<1e-6*max(previous,1e-15):break
                    previous=err
                epe=np.linalg.norm(pred-g,axis=1)*scale
                rows.append(dict(dataset=dataset,pair=pair,seed=seed,full_epe_m=float(epe.mean()),held_epe_m=float(epe[held].mean()),steps=step+1,seconds=time.perf_counter()-start,
                    identity_epe_m=float(np.linalg.norm(x-g,axis=1).mean()*scale),centroid_epe_m=float(np.linalg.norm(x-x.mean(0)+y.mean(0)-g,axis=1).mean()*scale)))
    assert len(rows)==52 and all(np.isfinite(r['full_epe_m']) for r in rows)
    filename='rigid_control.json' if args.initialization=='centroid' else 'rigid_control_identity.json'
    (ROOT/'paper_revision'/filename).write_text(json.dumps(rows,indent=2)+'\n')
    print('52 rigid ICP controls, no label-based fitting or tuning.')
