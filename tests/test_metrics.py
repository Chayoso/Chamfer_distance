import numpy as np

from chamfer_collapse.metrics import cluster_fraction, coverage, f1_score, hausdorff_distance, two_sided_chamfer_np


def test_hausdorff_zero_for_identical_clouds():
    pts = np.random.default_rng(0).normal(size=(40, 3))
    assert hausdorff_distance(pts, pts) < 1e-8


def test_f1_score_perfect_for_identical_clouds():
    pts = np.random.default_rng(1).normal(size=(40, 3))
    assert f1_score(pts, pts, tau=0.2) == 1.0


def test_coverage_full_when_source_equals_target():
    pts = np.random.default_rng(2).normal(size=(40, 2))
    assert coverage(pts, pts, eps=0.05) == 100.0


def test_cluster_fraction_zero_for_well_separated_points():
    pts = np.arange(10, dtype=float).reshape(-1, 1) * 10.0
    pts = np.concatenate([pts, np.zeros_like(pts)], axis=-1)
    assert cluster_fraction(pts, eps=0.02) == 0.0


def test_cluster_fraction_full_for_exact_duplicates():
    base = np.array([[0.0, 0.0]])
    pts = np.repeat(base, 5, axis=0)
    assert cluster_fraction(pts, eps=0.02) == 100.0


def test_two_sided_chamfer_np_matches_component_identity():
    rng = np.random.default_rng(3)
    source = rng.normal(size=(25, 3))
    target = rng.normal(size=(18, 3))
    fwd_rms, rev_rms, two_sided = two_sided_chamfer_np(source, target)
    assert abs(two_sided - (fwd_rms ** 2 + rev_rms ** 2) ** 0.5) < 1e-8
