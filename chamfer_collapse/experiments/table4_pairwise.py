"""Table 4: all 20 directed pairs among five shapes, Physics vs. Ours."""
from __future__ import annotations

from ..progress import log

import argparse
import itertools

from .common import print_table, run_pair_cached, write_csv

SHAPES = ["sphere", "bunny", "duck", "cow", "teapot"]


def run(preset: str = "default"):
    rows = []
    for s, t in itertools.permutations(SHAPES, 2):
        ph = run_pair_cached(s, t, preset, use_chamfer=False)
        ou = run_pair_cached(s, t, preset, use_chamfer=True)
        rows.append({"source": s, "target": t, "physics": ph["two_sided"], "ours": ou["two_sided"],
                     "phys/ours": ph["two_sided"] / ou["two_sided"],
                     "ours_wins": ou["two_sided"] < ph["two_sided"]})
    print_table(rows, "Table 4: 20 directed pairs, two-sided CD")
    wins = sum(r["ours_wins"] for r in rows)
    log(f"Ours better on {wins}/{len(rows)} pairs")
    log(f"Saved: {write_csv(rows, "table4_pairwise.csv")}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["quick", "default", "paper"], default="default")
    run(ap.parse_args().preset)
