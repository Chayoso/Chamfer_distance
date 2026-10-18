import numpy as np
import torch
from rigid_witness import geometry, experiment


def test_exact_assignment_gaps_and_unique_rigid_bijection():
    result = experiment(); p, q = geometry()
    gaps = np.array(result['costs'])-result['costs'][0]
    expected = [0., 10*np.sqrt(3), np.sqrt(3), 10*np.sqrt(3)+3, 10*np.sqrt(3)-3, 9*np.sqrt(3)]
    np.testing.assert_allclose(gaps, expected, atol=1e-13)
    distances = lambda z: ((z[:, None]-z[None, :])**2).sum(2)
    valid = [perm for perm in result['assignments'] if np.allclose(distances(p), distances(q[list(perm)]))]
    assert valid == [(0, 1, 2)]


def test_native_chamfer_autograd_reciprocity_and_finite_reversal():
    p, q = geometry()
    source = torch.tensor(p, requires_grad=True); target = torch.tensor(q)
    d = torch.cdist(source, target)
    grad = torch.autograd.grad(d.min(0).values.mean()+d.min(1).values.mean(), source)[0].numpy()
    result = experiment()
    for row in result['rows']:
        motion = np.array(row['motion'])
        assert abs((grad*motion).sum()-row['cd_slope']) < 1e-14
    cd, ot = result['rows'][:2]
    assert cd['cd_change'] < 0 < cd['mse_change']
    assert ot['mse_change'] < 0 < ot['cd_change']
