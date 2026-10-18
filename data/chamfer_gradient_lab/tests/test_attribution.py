import torch
from slope_attribution import measure
from pipeline.lifting import weighted_lift


def test_exact_projection_can_reverse_a_helpful_proposal():
    jacobian = torch.tensor([[1.], [1.]], dtype=torch.float64)
    forward = lambda ds: (jacobian@ds[0]).reshape(1, 2)
    transpose = lambda v: [jacobian.T@v.reshape(2, 1)]
    force = torch.tensor([[1., 0.]], dtype=torch.float64)
    mass = torch.ones(1, dtype=force.dtype)
    delta, _ = weighted_lift(forward, transpose, force, mass, 4, .5)
    motion = forward(delta)
    torch.testing.assert_close(motion, torch.full_like(force, 1/3))
    result = measure(torch.zeros_like(force), torch.tensor([[.5, -1.]], dtype=force.dtype), force, mass, motion)
    assert result['proposal_slope'] == -1.
    assert abs(result['projected_slope']-1/3) < 1e-12
    assert result['slope_identity_error'] < 1e-12
    assert not result['directional_descent_certificate']


def test_mass_weighting_and_unsupported_motion_are_separated():
    p = torch.zeros(3, 1, dtype=torch.float64)
    task = torch.tensor([[-1.], [2.], [3.]], dtype=p.dtype)
    mass = torch.tensor([10., 1., 0.], dtype=p.dtype)
    force = torch.tensor([[10.], [1.], [0.]], dtype=p.dtype)
    motion = torch.tensor([[1.], [1.], [1.]], dtype=p.dtype)
    result = measure(p, -1.5*task, force, mass, motion)
    assert result['force_slope'] == -8. and result['proposal_slope'] == 1.
    assert result['projected_slope'] == 4. and result['zero_mass_rows'] == 1
    assert result['supported_correction_slope'] == 0.
    assert result['unsupported_motion_slope'] == 3.
    assert result['weighted_transfer_error'] == 0.
