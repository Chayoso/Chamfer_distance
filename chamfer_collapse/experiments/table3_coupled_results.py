"""Table 3 (+ Hausdorff/F1 table): Physics-only vs. joint optimization,
Sphere source -> four targets. Shares cached runs with Table 4."""
from __future__ import annotations

from ..progress import log

import argparse

from .common import print_table, run_pair_cached, write_csv
from .table2_shape_generalization import TARGETS


def run(preset: str = "default"):
    rows = []
    for tgt in TARGETS:
        for use in (False, True):
            r = run_pair_cached("sphere", tgt, preset, use_chamfer=use)
            rows.append({"target": tgt, "method": r["method"], "s->t": r["s->t"], "t->s": r["t->s"],
                         "two_sided": r["two_sided"], "hausdorff": r["hausdorff"], "f1": r["f1"]})
    print_table(rows, "Table 3: Physics-only vs. Ours (Sphere source)")
    log(f"Saved: {write_csv(rows, "table3_coupled_results.csv")}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["quick", "default", "paper"], default="default")
    run(ap.parse_args().preset)
