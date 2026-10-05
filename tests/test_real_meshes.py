"""Real-asset handling (skipped when the meshes are not present)."""
import numpy as np
import pytest

from chamfer_collapse.mpm.particles import get_sampler, sample_volume_points
from chamfer_collapse.shapes import ASSETS_DIR, load_shape_mesh, shape_provenance

have = lambda f: (ASSETS_DIR / f).exists()


@pytest.mark.skipif(not have("isosphere.obj"), reason="no isosphere.obj")
def test_aliases_resolve_to_real_files():
    prov = shape_provenance(["sphere"])
    assert prov["sphere"] == "real:isosphere.obj"
    if have("spot.obj"):
        assert shape_provenance(["cow"])["cow"] == "real:spot.obj"


@pytest.mark.parametrize("name,f", [("sphere", "isosphere.obj"), ("bunny", "bunny.obj"), ("cow", "spot.obj"), ("teapot", "teapot.obj")])
def test_repaired_real_mesh_is_watertight_and_voxel_volume_close(name, f):
    if not have(f):
        pytest.skip(f"no {f}")
    s = get_sampler(load_shape_mesh(name)[0])
    assert s.watertight
    assert s.voxel_volume == pytest.approx(s.exact_volume, rel=0.07)   # voxel-fill bias is +2-5%
    assert s.volume == s.exact_volume


@pytest.mark.skipif(not have("bunny.obj"), reason="no bunny.obj")
def test_volume_samples_lie_inside_repaired_bunny():
    mesh, real = load_shape_mesh("bunny")
    assert real
    pts = sample_volume_points(mesh, 2000, seed=0)
    s = get_sampler(mesh)
    assert s.contains(pts).all()
    # not a hollow shell: interior points exist well away from the surface
    d = np.abs(s.repaired.nearest.signed_distance(pts[:300]))
    assert np.median(d) > 0.02
