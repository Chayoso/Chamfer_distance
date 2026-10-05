"""Table 5: Sphere -> Dragon. DCO, DCD, Physics, Ours at 3 and 4 particles
per cell (PPC^(1/3) in the paper's notation, i.e. 27 / 64 per cell)."""
from __future__ import annotations

from ..progress import log

import argparse

from ..dco import DCOConfig
from ..mpm.particles import estimate_particle_count
from ..mpm.presets import PRESETS
from ..shapes import load_shape_mesh
from .common import load_pair_points, print_table, run_dco_row, run_pair_cached, write_csv


def _n_for(preset: str, per_axis: int) -> int:
    p = PRESETS[preset]
    mesh, _ = load_shape_mesh("dragon")
    scaled = mesh.copy()
    scaled.apply_scale(p.object_scale)
    return estimate_particle_count(scaled, p.mpm_params.dx, per_axis ** 3)


def run(preset: str = "default"):
    s, t = load_pair_points("sphere", "dragon", preset)
    rows = [run_dco_row("DCO", DCOConfig(), s, t, preset), run_dco_row("DCD", DCOConfig(use_dcd=True), s, t, preset)]
    base = PRESETS[preset].n_particles
    def row(label, r):
        return {"method": label, **{k: r[k] for k in ("s->t", "t->s", "two_sided", "hausdorff", "f1", "seconds")}}
    rows.append(row("Physics-only", run_pair_cached("sphere", "dragon", preset, False)))
    rows.append(row("Ours (base N)", run_pair_cached("sphere", "dragon", preset, True)))
    n4 = max(int(base * (4 / 3) ** 3), base + 1)
    rows.append(row(f"Ours (4/3)^3 x N = {n4}", run_pair_cached("sphere", "dragon", preset, True, n_particles=n4)))
    print_table(rows, "Table 5: Sphere -> Dragon")
    log(f"(paper-scale ppc counts: 3ppc={_n_for('paper', 3)}, 4ppc={_n_for('paper', 4)})")
    log(f"Saved: {write_csv(rows, "table5_dragon.csv")}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["quick", "default", "paper"], default="default")
    run(ap.parse_args().preset)
