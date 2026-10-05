"""DCO two-sided CD / fwd / rev vs. point count under the paper's stated DCO
settings (AMSGrad, lr 0.01, 40x30 steps, grid-unit scale 16). Collapse signature:
fwd << rev. Usage: python scripts/dco_scaling_sweep.py 1000 4000 20000 89000"""
import sys, time
import torch
from chamfer_collapse.chamfer import chamfer_components
from chamfer_collapse.dco import DCOConfig, run_dco
from chamfer_collapse.shapes import load_shape_points
from chamfer_collapse.experiments.common import write_csv

SCALE = 16.0
rows = []
for n in map(int, sys.argv[1:] or [1000, 4000, 20000]):
    s, _ = load_shape_points("sphere", n, seed=0)
    t, _ = load_shape_points("bunny", int(n * 0.83), seed=1)
    s, t = s * SCALE, t * SCALE
    for label, cfg in (("DCO", DCOConfig()), ("DCD", DCOConfig(use_dcd=True))):
        t0 = time.time()
        pts, _ = run_dco(s, t, cfg)
        r = chamfer_components(pts, torch.as_tensor(t, dtype=torch.float32))
        row = {"n_source": n, "method": label, "fwd": r.fwd_rms.item(), "rev": r.rev_rms.item(),
               "two_sided": r.two_sided.item(), "seconds": round(time.time() - t0)}
        rows.append(row); print(row, flush=True)
        write_csv(rows, "dco_scaling_sweep.csv")
