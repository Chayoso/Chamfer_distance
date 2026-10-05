"""Table 2: DCO / DCD / physics-only across four targets (Sphere source)."""
from __future__ import annotations

from ..progress import log

import argparse

from ..dco import DCOConfig
from .common import load_pair_points, print_table, run_dco_row, run_pair_cached, write_csv

TARGETS = ["bunny", "duck", "cow", "teapot"]


def run(preset: str = "default"):
    rows = []
    for tgt in TARGETS:
        s, t = load_pair_points("sphere", tgt, preset)
        dco = run_dco_row("DCO", DCOConfig(), s, t, preset)
        dcd = run_dco_row("DCD", DCOConfig(use_dcd=True), s, t, preset)
        phys = run_pair_cached("sphere", tgt, preset, use_chamfer=False)
        rows.append({"target": tgt, "DCO": dco["two_sided"], "DCD": dcd["two_sided"],
                     "Physics": phys["two_sided"],
                     "DCO/Phys": dco["two_sided"] / phys["two_sided"],
                     "DCD/Phys": dcd["two_sided"] / phys["two_sided"]})
    print_table(rows, "Table 2: Shape generalization (Sphere source), two-sided CD")
    log(f"Saved: {write_csv(rows, "table2_shape_generalization.csv")}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["quick", "default", "paper"], default="default")
    run(ap.parse_args().preset)
