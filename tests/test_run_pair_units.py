"""DCO and MPM metrics must share units (see experiments/common.load_pair_points)."""
import numpy as np

from chamfer_collapse.experiments.common import load_pair_points
from chamfer_collapse.mpm.presets import PRESETS


def test_dco_points_use_mpm_object_scale():
    s, t = load_pair_points("sphere", "bunny", "quick")
    scale = PRESETS["quick"].object_scale
    assert np.abs(s).max() <= scale * 1.01 and np.abs(s).max() > 0.3 * scale
