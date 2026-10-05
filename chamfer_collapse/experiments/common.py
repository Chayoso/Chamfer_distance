"""Shared plumbing for the table-reproduction scripts: point-cloud scale
presets (separate from the MPM particle-count presets in
``chamfer_collapse.mpm.presets``, since DCO has no simulation cost and can
run closer to paper scale by default), CSV writing, and pretty-printing.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ..shapes import load_shape_points

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"

#: (n_source, n_target) point counts for the DCO-family experiments.
#: "paper" matches Sec. 3.1: isosphere ~89K source, surface-sampled Bunny
#: ~74K target (other targets are sampled at the same count as their
#: source for a fair comparison, since the paper does not vary target
#: density by shape).
DCO_POINT_PRESETS = {
    "quick": (400, 400),
    "default": (4000, 3000),
    "paper": (89000, 74000),
}


def load_pair_points(source: str, target: str, preset: str, seed: int = 0):
    """Surface-sampled source/target clouds for the DCO family, scaled by the
    MPM preset's ``object_scale`` so DCO and MPM metrics (CD, Hausdorff, F1
    tau) live in the same units -- the paper's Hausdorff values (~3) imply
    grid-cell units rather than unit-normalized ones."""
    from ..mpm.presets import PRESETS
    n_src, n_tgt = DCO_POINT_PRESETS[preset]
    scale = PRESETS[preset].object_scale
    src_pts, _ = load_shape_points(source, n_src, seed=seed)
    tgt_pts, _ = load_shape_points(target, n_tgt, seed=seed + 1)
    return src_pts * scale, tgt_pts * scale


def paper_equivalent_lr(preset: str, base_lr: float = 0.01) -> float:
    """The paper's DCO lr (0.01) is in grid units on a [-16,16]^3 domain
    (object_scale 16). Running at a smaller object_scale with the same
    absolute lr would take proportionally larger steps and hide collapse
    (verified: scripts/dco_scaling_sweep.py), so scale lr with the object."""
    from ..mpm.presets import PRESETS
    return base_lr * PRESETS[preset].object_scale / 16.0


def write_csv(rows: list[dict], name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    if not rows:
        path.write_text("")
        return path
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def print_table(rows: list[dict], title: str | None = None) -> None:
    from ..progress import table_out as print_  # tables go to the stderr log
    if title:
        print_(f"\n=== {title} ===")
    if not rows:
        print_("(no rows)")
        return
    cols = list(rows[0].keys())
    widths = [max(len(c), *(len(f'{r[c]:.4f}' if isinstance(r[c], float) else str(r[c])) for r in rows)) for c in cols]

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    print_(header)
    print_("-" * len(header))
    for r in rows:
        print_("  ".join(fmt(r[c]).ljust(w) for c, w in zip(cols, widths)))


# ---------------------------------------------------------------------------
# Shared experiment helpers (DCO rows, cached MPM pair runs)
# ---------------------------------------------------------------------------
import json  # noqa: E402
import time  # noqa: E402


def run_dco_row(label: str, cfg, src_pts, tgt_pts, preset: str = "default") -> dict:
    """Run one DCO-family condition and return a table row."""
    import dataclasses
    import torch
    cfg = dataclasses.replace(cfg, lr=paper_equivalent_lr(preset))
    from ..chamfer import chamfer_components
    from ..dco import run_dco
    from ..metrics import f1_score, hausdorff_distance
    t0 = time.time()
    points, _ = run_dco(src_pts, tgt_pts, cfg)
    res = chamfer_components(points, torch.as_tensor(tgt_pts, dtype=torch.float32, device=points.device))
    pn = points.cpu().numpy()
    return {"method": label, "s->t": res.fwd_rms.item(), "t->s": res.rev_rms.item(),
            "two_sided": res.two_sided.item(), "hausdorff": hausdorff_distance(pn, tgt_pts),
            "f1": f1_score(pn, tgt_pts, tau=0.2), "seconds": round(time.time() - t0, 2)}


def run_pair_cached(source: str, target: str, preset: str, use_chamfer: bool,
                    n_particles: int | None = None, tag: str = "") -> dict:
    """MPM pair run with a JSON cache in results/cache/ so Tables 3 and 4
    (and re-runs of run_all) share work. Returns a metrics dict."""
    import dataclasses
    from ..mpm.presets import PRESETS
    from ..mpm.run_pair import run_pair
    p = PRESETS[preset]
    if n_particles:
        p = dataclasses.replace(p, n_particles=n_particles)
    from ..shapes import shape_provenance
    prov = shape_provenance([source, target])
    real = "".join("R" if v.startswith("real") else "P" for v in prov.values())   # R=real mesh, P=stand-in
    key = f"{preset}_{source}_{target}_{real}_{'ours' if use_chamfer else 'phys'}_{p.n_particles}{tag}.json"
    path = RESULTS_DIR / "cache" / key
    if path.exists():
        return json.loads(path.read_text())
    t0 = time.time()
    r = run_pair(source, target, p, use_chamfer=use_chamfer)
    out = {"source": source, "target": target, "method": "Ours" if use_chamfer else "Physics",
           "s->t": r.fwd_rms, "t->s": r.rev_rms, "two_sided": r.two_sided,
           "hausdorff": r.hausdorff, "f1": r.f1, "seconds": round(time.time() - t0, 2)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out
