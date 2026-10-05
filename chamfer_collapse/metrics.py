"""Evaluation metrics used throughout the paper: two-sided Chamfer distance
(``chamfer.py``), Hausdorff distance, F1-score, coverage, and cluster
fraction (Table 3/4/5 and the 2D experiment in the supplementary material).
"""

from __future__ import annotations

import numpy as np

from .geometry import nn_indices_and_dists


def hausdorff_distance(source: np.ndarray, target: np.ndarray) -> float:
    """Symmetric Hausdorff distance: max over both directions of the worst
    per-point nearest-neighbor distance."""
    d_fwd, _ = nn_indices_and_dists(source, target)
    d_rev, _ = nn_indices_and_dists(target, source)
    return float(max(d_fwd.max(), d_rev.max()))


def f1_score(source: np.ndarray, target: np.ndarray, tau: float = 0.2) -> float:
    """F1-score at distance threshold ``tau``: precision is the fraction of
    source points within ``tau`` of some target point, recall is the
    fraction of target points within ``tau`` of some source point."""
    d_fwd, _ = nn_indices_and_dists(source, target)
    d_rev, _ = nn_indices_and_dists(target, source)
    precision = float((d_fwd < tau).mean())
    recall = float((d_rev < tau).mean())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def coverage(source: np.ndarray, target: np.ndarray, eps: float = 0.05) -> float:
    """Fraction of target points with at least one source point within
    ``eps`` (2D experiment, Supplementary Sec. 2.2)."""
    d_rev, _ = nn_indices_and_dists(target, source)
    return 100.0 * float((d_rev < eps).mean())


def cluster_fraction(source: np.ndarray, eps: float = 0.02) -> float:
    """Fraction of source points that have another source point within
    ``eps`` -- a direct measure of many-to-one collapse (2D experiment,
    Supplementary Sec. 2.2).

    Caveat for methods that keep points at *fixed angular positions* on a
    parameterized curve (e.g. the shared-basis/Fourier condition): if the
    curve has N points and local radius r, adjacent-point spacing is
    ~2*pi*r/N regardless of collapse. For N=600 and eps=0.02 this floor is
    r > ~1.9, comfortably above the star's outer radius of 1.0, so this
    metric saturates near 100% for that condition independent of how well
    it suppresses true collapse and should be read as non-discriminative
    there; it remains discriminative for DCO / DCO+Repulsion, where points
    move freely and only genuine coincidence triggers it.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(source)
    # k=2: the first match is always the point itself (distance 0).
    d2, _ = tree.query(source, k=2)
    nearest_other = d2[:, 1]
    return 100.0 * float((nearest_other < eps).mean())


def two_sided_chamfer_np(source: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    """NumPy convenience version of the two-sided Chamfer distance, returning
    (s->t RMS, t->s RMS, two-sided)."""
    d_fwd, _ = nn_indices_and_dists(source, target)
    d_rev, _ = nn_indices_and_dists(target, source)
    fwd_rms = float(np.sqrt((d_fwd ** 2).mean()))
    rev_rms = float(np.sqrt((d_rev ** 2).mean()))
    two_sided = float(np.sqrt((d_fwd ** 2).mean() + (d_rev ** 2).mean()))
    return fwd_rms, rev_rms, two_sided
