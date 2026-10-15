"""Generate six fixed-budget reference fits from already prepared local datasets."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.model import ROOT, torch, config
from pipeline.datasets import transfer
from pipeline.optim import fixed_official
from pipeline.runner import fit


def main():
    cfg = config()
    cfg.budget = [50]*9
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    for family, pairs in [('4d', (100, 101, 124, 125)), ('hf', (200, 201))]:
        folder = ROOT/'paper_revision'/f'replication_{family}'
        folder.mkdir(parents=True, exist_ok=False)
        for pair in pairs:
            if sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) >= 800_000_000:
                raise RuntimeError('Storage guard: lab must remain below 800 MB before a fit')
            fit(pair, 0, 'official', 0, folder, loader=transfer, cfg=cfg,
                factory=fixed_official)


if __name__ == '__main__':
    main()
