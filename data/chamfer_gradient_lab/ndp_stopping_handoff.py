"""Budget-only 500-update control using the existing fresh-endpoint loop."""
import open3d
from pipeline.model import ROOT, torch, config
from pipeline.datasets import transfer
from pipeline.optim import fixed_official
from pipeline.runner import fit


def main():
    folder = ROOT/'paper_revision/stopping_20261018/fresh500'
    folder.mkdir(exist_ok=False)
    cfg = config(); cfg.budget = [500]*9
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    for pair in (100, 101, 124, 125, 200, 201):
        assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) < 800_000_000
        fit(pair, 0, 'official', 0, folder, loader=transfer, cfg=cfg, factory=fixed_official)


if __name__ == '__main__':
    main()
