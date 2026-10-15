"""Historical CLI and dependency overrides; implementation lives in pipeline.runner."""
import argparse
import json
from ndp_pipeline import ROOT, np, torch, official, config, components, objective, data
from pipeline.runner import fit as _fit


def fit(pair, seed, method, weight, folder, *, loader=None, loss_fn=None, cfg=None, factory=None):
    return _fit(pair, seed, method, weight, folder,
                loader=data if loader is None else loader,
                loss_fn=objective if loss_fn is None else loss_fn,
                cfg=config() if cfg is None else cfg,
                factory=official if factory is None else factory)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['pilot','validation','test'],required=True)
    args=parser.parse_args();folder=ROOT/'paper_revision'/('ndp_'+args.phase);folder.mkdir(exist_ok=False)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    methods=[('official',0),('squared',0),('trimmed',0),('sw',.1),('sw',1),('uot',.1),('uot',1)]
    if args.phase=='test':
        rates=json.loads((ROOT/'paper_revision/ndp_weights.json').read_text())
        methods=[('official',0),('squared',0),('trimmed',0),('sw',rates['sw']),('uot',rates['uot'])]
    for pair in ([0] if args.phase=='pilot' else range(4) if args.phase=='validation' else range(4,28)):
        for seed in ([0,1] if args.phase=='test' else [0]):
            for method,weight in methods:
                if sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())>800_000_000: raise RuntimeError('Storage guard')
                fit(pair,seed,method,weight,folder)
