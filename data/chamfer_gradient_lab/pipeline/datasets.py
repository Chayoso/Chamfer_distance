"""Existing dataset loaders; preserve original normalization and correspondence checks."""
from .model import ROOT, np, torch


def four_d_match(pair):
    with np.load(ROOT/f'assets/4dmatch_subset/{pair:02d}.npz', allow_pickle=False) as z:
        x, y = z['s_pc'].astype(float), z['t_pc'].astype(float)
        moved = x+z['s2t_flow']; rot = z['rot']
        truth = np.stack([sum(moved[:,j]*rot[i,j] for j in range(3)) for i in range(3)], 1)+z['trans'].reshape(1,3)
        overlap = np.zeros(len(x), dtype=bool); overlap[z['correspondences'][:,0]] = True
        corr = z['correspondences']
        assert np.linalg.norm(truth[corr[:,0]]-y[corr[:,1]], axis=1).max() < .015001
    cast = lambda a: torch.as_tensor(a, device='cuda:0', dtype=torch.float32)
    return cast(x), cast(y), cast(truth), torch.as_tensor(overlap, device='cuda:0')



def scans(pair):
    with np.load(ROOT/f'assets/stanford_raw/{pair:02d}.npz',allow_pickle=False) as z:
        cast=lambda key:torch.as_tensor(z[key]/z['scale'],device='cuda:0',dtype=torch.float32)
        return cast('source'),cast('target'),cast('truth'),torch.as_tensor(z['overlap'],device='cuda:0')



def transfer(pair):
    path=ROOT/f'assets/4dmatch_replication/{pair}.npz' if pair<200 else ROOT/f'assets/hf_3dmatch/{pair}.npz'
    with np.load(path,allow_pickle=False) as z:
        if pair<200:
            x,y=z['s_pc'].astype(float),z['t_pc'].astype(float);m=x+z['s2t_flow'];r=z['rot']
            g=np.stack([sum(m[:,j]*r[i,j] for j in range(3)) for i in range(3)],1)+z['trans'].reshape(1,3)
            c=z['correspondences'];assert np.linalg.norm(g[c[:,0]]-y[c[:,1]],axis=1).max()<.015001
            visible=np.zeros(len(x),bool);visible[c[:,0]]=True
        else:x,y,g,visible=z['source']/z['scale'],z['target']/z['scale'],z['truth']/z['scale'],z['overlap']
    cast=lambda a:torch.as_tensor(a,device='cuda:0',dtype=torch.float32)
    return cast(x),cast(y),cast(g),torch.as_tensor(visible,device='cuda:0')
