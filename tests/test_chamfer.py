import numpy as np
import torch

from chamfer_collapse.chamfer import chamfer_components


def test_two_sided_cd_matches_component_identity():
    """Table 1's reported two-sided CD is sqrt(fwd_rms^2 + rev_rms^2), e.g.
    DCO: sqrt(0.220^2 + 0.182^2) = 0.2855 =~ 0.286."""
    fwd_rms, rev_rms = 0.220, 0.182
    expected_two_sided = (fwd_rms ** 2 + rev_rms ** 2) ** 0.5
    assert abs(expected_two_sided - 0.286) < 1e-2


def test_chamfer_zero_for_identical_clouds():
    pts = torch.randn(50, 3)
    res = chamfer_components(pts, pts.clone())
    assert res.two_sided.item() < 1e-5


def test_chamfer_components_internal_consistency():
    torch.manual_seed(0)
    source = torch.randn(30, 3)
    target = torch.randn(20, 3)
    res = chamfer_components(source, target)
    two_sided = res.two_sided.item()
    recomputed = (res.fwd_rms.item() ** 2 + res.rev_rms.item() ** 2) ** 0.5
    assert abs(two_sided - recomputed) < 1e-5
