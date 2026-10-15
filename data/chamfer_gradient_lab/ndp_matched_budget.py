"""Historical matched-budget CLI; optimization lives in pipeline.optim."""
import json
from ndp_pipeline import ROOT, torch, official, config
from pipeline.optim import fixed_official
from ndp_prior import prior
import ndp_experiment as run


if __name__=='__main__':
    folder=ROOT/'paper_revision/ndp_matched_budget';folder.mkdir(exist_ok=False)
    weights=json.loads((ROOT/'paper_revision/ndp_weights.json').read_text());pw=json.loads((ROOT/'paper_revision/ndp_prior_weights.json').read_text())
    methods=[('official',0),('squared',0),('trimmed',0),('sw',weights['sw']),('uot',weights['uot']),
             ('infocd_code',1e-7),('infocd_paper',pw['infocd_paper']),('dcd',pw['dcd'])]
    original=run.objective;run.official=fixed_official
    run.objective=lambda p,q,m,w,s,d:prior(p,q,m,w,s,d) if m in pw or m=='infocd_code' else original(p,q,m,w,s,d)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    for pair in [4,5,6,16,17,18]:
        for seed in [0,1]:
            ref=json.loads((ROOT/f'paper_revision/ndp_test/p{pair:02d}_s{seed}_official_w0.json').read_text())
            assert max(ref['calls_by_level'])<500, 'Max-iteration endpoint needs explicit update count; do not infer it'
            cfg=config();cfg.budget=[c-1 for c in ref['calls_by_level']];run.config=lambda:cfg
            for method,weight in methods:
                if sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())>800_000_000:raise RuntimeError('Storage guard')
                run.fit(pair,seed,method,weight,folder)
                result=json.loads((folder/f'p{pair:02d}_s{seed}_{method}_w{weight:g}.json').read_text())
                assert result['calls_by_level']==ref['calls_by_level']
                if method=='official':assert abs(result['full_epe']-ref['full_epe'])<1e-9
