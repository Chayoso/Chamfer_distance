import numpy as np
import torch
from transfer_mechanism import P, Q, FORCE, TASK, witness
from pipeline.lifting import jacobian_products, ridge_lift


def test_witness_is_actual_unique_nn_unsquared_chamfer_gradient():
    p = torch.tensor(P[:, None], requires_grad=True)
    q = torch.tensor(Q[:, None])
    distance = torch.cdist(p, q)
    loss = distance.min(0).values.mean()+distance.min(1).values.mean()
    gradient = torch.autograd.grad(loss, p)[0].numpy().ravel()
    np.testing.assert_allclose(-gradient, FORCE)
    np.testing.assert_allclose(P-Q, TASK)
    assert TASK@FORCE < 0


def test_neural_lift_and_closed_form_agree_in_both_rank_cases():
    for mu in (.001, 0.):
        parameter = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        j = torch.diag(torch.tensor([1., mu**.5], dtype=torch.float64))
        output = torch.tensor(P)+j@parameter
        forward, transpose = jacobian_products(output, [parameter])
        for ratio in (.01, .0001):
            delta, info = ridge_lift(forward, transpose, -torch.ones_like(output), 72, ratio)
            assert info['converged']
            actual = forward(delta).detach().numpy()
            damping = info['damping']
            expected = np.array([1/(1+damping), mu/(mu+damping)])
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
            expected_row = witness(mu, damping)
            assert abs(TASK@actual-expected_row['raw_slope']) < 1e-10


def test_sign_threshold_and_finite_counterexample():
    threshold = .001/(1-2*.001)
    assert abs(witness(.001, threshold)['raw_slope']) < 1e-14
    assert witness(.001, .01)['mse_change'] > 0
    assert witness(.001, .0001)['mse_change'] < 0
    assert witness(0., .0001)['mse_change'] > 0
