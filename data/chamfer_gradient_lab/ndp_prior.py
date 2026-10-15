"""Published InfoCD/DCD loss interventions in exactly the same NDP pipeline."""
import ast
import json
import sys
from ndp_pipeline import ROOT, torch, np, components
import ndp_experiment as run


def chamfer(p,q):
    d=torch.cdist(p,q,compute_mode='donot_use_mm_for_euclid_dist')
    a,i=d.min(2);b,j=d.min(1)
    return a.square(),b.square(),i,j


def published():
    env=dict(torch=torch,chamfer_dist=chamfer,cd=lambda:chamfer)
    for path,names in [('infocd_official/loss_utils.py',['calc_cd_like_InfoV2']),
                       ('dcd_official/model_utils.py',['calc_dcd','calc_cd'])]:
        tree=ast.parse((ROOT/'assets'/path).read_text())
        tree.body=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
        exec(compile(tree,path,'exec'),env)
    return env


ENV=published()


def prior(p,q,method,weight,scale,directions):
    if method=='infocd_code': return 4/len(p)*ENV['calc_cd_like_InfoV2'](p[None],q[None])
    if method=='dcd': return ENV['calc_dcd']((p/scale)[None],(q/scale)[None],alpha=weight)[0].mean()
    a,b=components(p,q)
    return a.mean()+b.mean()+2*weight*(torch.logsumexp(-.5*a,0)-np.log(len(a))+torch.logsumexp(-.5*b,0)-np.log(len(b)))


if __name__=='__main__':
    phase=sys.argv[1];assert phase in ['validation','test']
    folder=ROOT/'paper_revision'/('ndp_prior_'+phase);folder.mkdir(exist_ok=False)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    run.objective=prior
    methods=[('infocd_code',1e-7),('infocd_paper',.1),('infocd_paper',1),('dcd',10),('dcd',100),('dcd',1000)]
    if phase=='test':
        weights=json.loads((ROOT/'paper_revision/ndp_prior_weights.json').read_text())
        methods=[('infocd_code',1e-7),('infocd_paper',weights['infocd_paper']),('dcd',weights['dcd'])]
    for pair in (range(4) if phase=='validation' else range(4,28)):
        for seed in ([0] if phase=='validation' else [0,1]):
            for method,weight in methods:
                if sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())>800_000_000:raise RuntimeError('Storage guard')
                run.fit(pair,seed,method,weight,folder)
