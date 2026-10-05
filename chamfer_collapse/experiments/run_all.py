"""Run every table and the 2D experiment: python -m chamfer_collapse.experiments.run_all --preset quick"""
from __future__ import annotations

from ..progress import log

import argparse
import time

from . import (table1_regularization, table2_shape_generalization, table3_coupled_results,
               table4_pairwise, table5_dragon)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["quick", "default", "paper"], default="default")
    ap.add_argument("--skip-2d", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    if not a.skip_2d:
        from .exp_2d import run as run_2d
        run_2d()
    for mod in (table1_regularization, table2_shape_generalization, table3_coupled_results,
                table4_pairwise, table5_dragon):
        mod.run(preset=a.preset)
    log(f"All done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
