"""2D circle -> star collapse experiment (Sec. 4 / Supp. 2D table + Fig. 3).

python -m chamfer_collapse.experiments.exp_2d
"""
from __future__ import annotations

from ..progress import log

import numpy as np

from ..experiment_2d import (make_circle, make_star, run_dco_2d, run_dco_repulsion_2d,
                             run_fourier_2d, summarize)
from .common import RESULTS_DIR, print_table, write_csv

PAPER = {"DCO": (0.352, 38, 97), "DCO + repulsion": (0.299, 50, 97), "Shared-basis (Fourier)": (0.123, 62, 81)}


def run(make_figure: bool = True):
    circle, star = make_circle(), make_star()
    trajs = {"DCO": run_dco_2d(circle, star), "DCO + repulsion": run_dco_repulsion_2d(circle, star),
             "Shared-basis (Fourier)": run_fourier_2d(star)}
    rows = []
    for name, tr in trajs.items():
        s = summarize(tr, star)
        p = PAPER[name]
        rows.append({"method": name, **s, "paper_cd": p[0], "paper_cov": p[1], "paper_cluster": p[2]})
    print_table(rows, "2D experiment: circle -> star")
    log(f"Saved: {write_csv(rows, "experiment_2d.csv")}")
    if make_figure:
        _figure(trajs, star)
    return rows


def _figure(trajs, star):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), constrained_layout=True)
    for ax, (name, tr) in zip(axes, trajs.items()):
        ax.scatter(*star.T, s=6, color="#9aa0a6", label="target")
        ax.scatter(*tr.positions[-1].T, s=4, color="#1f6feb", label="optimized")
        ax.set_title(name, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "fig_2d_experiment.png"
    fig.savefig(out, dpi=160)
    log(f"Saved: {out}")


if __name__ == "__main__":
    run()
