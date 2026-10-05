"""Reproduces Table 1: effect of adding local regularizers and DCD to DCO,
plus the physics-only baseline, on isosphere -> Stanford Bunny (Sec. 3.2-3.3).

Usage: python -m chamfer_collapse.experiments.table1_regularization [--preset quick|default|paper]
"""

from __future__ import annotations

from ..progress import log

import argparse
import time

import torch

from ..chamfer import chamfer_components
from ..dco import DCOConfig, run_dco
from ..mpm.presets import PRESETS
from ..mpm.run_pair import run_pair
from .common import load_pair_points, paper_equivalent_lr, print_table, write_csv

CONDITIONS = [
    ("DCO (no reg.)", DCOConfig()),
    ("+ Repulsion (lambda=0.01)", DCOConfig(repulsion_weight=0.01)),
    ("+ Repulsion (lambda=0.1)", DCOConfig(repulsion_weight=0.1)),
    ("+ Rep. + Smooth", DCOConfig(repulsion_weight=0.1, smoothness_weight=0.01)),
    ("+ Rep. + Smooth + Vol.", DCOConfig(repulsion_weight=0.1, smoothness_weight=0.01, volume_weight=0.01)),
    ("DCD", DCOConfig(use_dcd=True)),
]


def run(preset: str = "default", n_frames: int | None = None, steps_per_frame: int | None = None):
    src_pts, tgt_pts = load_pair_points("sphere", "bunny", preset)
    target_t = torch.as_tensor(tgt_pts, dtype=torch.float32)

    rows = []
    dco_baseline_two_sided = None
    for label, cfg in CONDITIONS:
        cfg = DCOConfig(**{**cfg.__dict__, "lr": paper_equivalent_lr(preset), **({"n_frames": n_frames} if n_frames else {}),
                            **({"steps_per_frame": steps_per_frame} if steps_per_frame else {})})
        t0 = time.time()
        points, _ = run_dco(src_pts, tgt_pts, cfg)
        res = chamfer_components(points, target_t)
        two_sided = res.two_sided.item()
        if label == "DCO (no reg.)":
            dco_baseline_two_sided = two_sided
        vs_dco = "--" if label == "DCO (no reg.)" else (
            "worse" if two_sided > dco_baseline_two_sided else "better"
        )
        rows.append({
            "condition": label,
            "s->t": res.fwd_rms.item(),
            "t->s": res.rev_rms.item(),
            "two_sided": two_sided,
            "vs_DCO": vs_dco,
            "seconds": round(time.time() - t0, 2),
        })

    mpm_preset = PRESETS[preset if preset in PRESETS else "default"]
    t0 = time.time()
    phys = run_pair("sphere", "bunny", mpm_preset, use_chamfer=False)
    rows.append({
        "condition": "Physics-only",
        "s->t": phys.fwd_rms,
        "t->s": phys.rev_rms,
        "two_sided": phys.two_sided,
        "vs_DCO": (f"physics/DCO CD ratio = {phys.two_sided / dco_baseline_two_sided:.2f}" if dco_baseline_two_sided else "--"),
        "seconds": round(time.time() - t0, 2),
    })

    print_table(rows, title="Table 1: Regularization ablation (Sphere -> Bunny)")
    path = write_csv(rows, "table1_regularization.csv")
    log(f"Saved: {path}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["quick", "default", "paper"], default="default")
    args = parser.parse_args()
    run(preset=args.preset)
