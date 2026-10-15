"""Single fixed-budget optimizer control; see the frozen protocol before running."""
import argparse
import json
from pipeline.model import ROOT, torch, config
from pipeline.optim import fixed_official
from pipeline.runner import fit
from pipeline.datasets import scans
from pipeline.datasets import transfer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['scan', 'transfer'], default='scan')
    parser.add_argument('--output', type=str, required=True, help='New directory beneath this lab')
    args = parser.parse_args()
    folder = (ROOT/args.output).resolve()
    if not folder.is_relative_to(ROOT.resolve()) or folder == ROOT.resolve():
        raise ValueError('Output must stay inside this lab')
    folder.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = config()
    cfg.budget = [50]*9
    pairs = [4,5,6,7,16,17,18,19] if args.phase == 'scan' else [100,101,102,103,200,201,202,203]
    # Both methods rerun on the same machine; alternate order without changing seeds.
    for index, pair in enumerate(pairs):
        for seed in ([0,1,2] if args.phase == 'scan' else [0]):
            methods = ['official','cosine'] if (index+seed)%2 == 0 else ['cosine','official']
            for method in methods:
                if sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) > 800_000_000:
                    raise RuntimeError('Storage guard: keep room below 1 GB')
                cfg.schedule = 'constant' if method == 'official' else 'cosine'
                row = fit(pair, seed, method, 0, folder, cfg=cfg, factory=fixed_official,
                          loader=scans if args.phase == 'scan' else transfer)
                assert row['calls_by_level'] == [51]*9
                row.update(optimizer_updates=450, schedule=cfg.schedule)
                (folder/f'p{pair:02d}_s{seed}_{method}_w0.json').write_text(json.dumps(row, indent=2)+'\n')


if __name__ == '__main__':
    main()
