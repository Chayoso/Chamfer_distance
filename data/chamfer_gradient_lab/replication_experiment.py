"""Fixed-budget, six-method replication; two exactly decomposed SW controls."""
import argparse
import json
from ndp_pipeline import ROOT, np, torch, config, components, objective
from ndp_prior import prior
from ndp_matched_budget import fixed_official
import ndp_experiment as run


def loss(p,q,method,weight,scale,directions):
    if method=='dcd':return prior(p,q,method,weight,scale,directions)
    if method in ['centered_sw','moment_sw']:
        a,b=components(p,q)
        delta=(p.mean(0)-q.mean(0))@directions
        if method=='moment_sw':coupling=delta.square().mean()
        else:coupling=(((p-p.mean(0))@directions).sort(0).values-((q-q.mean(0))@directions).sort(0).values).square().mean()
        return a.mean()+b.mean()+weight*coupling/scale
    return objective(p,q,method,weight,scale,directions)


from pipeline.datasets import transfer as data


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('dataset',choices=['4d','hf']);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--start',type=int);parser.add_argument('--end',type=int);args=parser.parse_args()
    folder=ROOT/'paper_revision'/('replication_'+args.dataset);folder.mkdir(exist_ok=args.resume)
    cfg=config();cfg.budget=[50]*9
    run.data=data;run.objective=loss;run.official=fixed_official;run.config=lambda:cfg
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    for pair in (range(100,148) if args.dataset=='4d' else range(200,212)):
        if (args.start is not None and pair<args.start) or (args.end is not None and pair>=args.end):continue
        for seed in [0,1,2]:
            for method,weight in [('official',0),('dcd',10),('sw',.1),('uot',.1),('centered_sw',.1),('moment_sw',.1)]:
                path=folder/f'p{pair:02d}_s{seed}_{method}_w{weight:g}.json'
                if args.resume and path.exists():
                    row=json.loads(path.read_text());assert row['calls_by_level']==[51]*9 and row['updates_per_level']==[50]*9
                    if args.dataset=='hf':assert 'full_epe_m' in row
                    continue
                assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<800_000_000
                run.fit(pair,seed,method,weight,folder)
                row=json.loads(path.read_text());assert row['calls_by_level']==[51]*9
                if args.dataset=='hf':
                    with np.load(ROOT/f'assets/hf_3dmatch/{pair}.npz') as z:radius=float(z['scale'])
                    row.update(source_radius_m=radius,**{s+'_epe_m':row[s+'_epe']*radius for s in ['full','held','vis','occ']})
                row['updates_per_level']=[50]*9;path.write_text(json.dumps(row,indent=2)+'\n')
