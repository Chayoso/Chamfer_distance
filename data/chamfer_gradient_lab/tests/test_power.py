"""Checks of power-loss extensions and a limitation of the sufficient bound."""
import numpy as np
from scipy.integrate import solve_ivp


def loss(x, q, power, weight=1.):
    d = np.linalg.norm(x[:, None]-q, axis=-1)**power
    return d.min(1).mean()+weight*d.min(0).mean()


def test_power_minimum_formula_and_open_neighborhood():
    rng = np.random.default_rng(711)
    q = np.array([[-1., .1, -.1], [.5, .02, .03], [1.5, -.05, .1]])
    distance = np.linalg.norm(q[2]-q[1])
    for power in [1., 1.1, 1.5, 2., 3., 4.]:
        for weight in [.3, 1., 2.]:
            t = 0. if power == 1 else 1/(1+((1+weight)/weight)**(1/(power-1)))
            x = np.array([q[0], q[0], q[1]+t*(q[2]-q[1])])
            value = weight*distance/3 if power == 1 else distance**power/3*((1+weight)*t**power+weight*(1-t)**power)
            assert abs(loss(x, q, power, weight)-value) < 1e-14
            for _ in range(30):
                assert loss(x+rng.normal(size=x.shape)*.001, q, power, weight) > value
            if power > 1:
                assert abs((1+weight)*t**(power-1)-weight*(1-t)**(power-1)) < 1e-13


def test_one_sided_flow_and_right_derivative():
    q = np.array([[-1.], [0.], [1.]])
    for power in [1.1, 1.5, 2., 3., 4.]:
        flow = solve_ivp(lambda t, z: power/3*((1-z)**(power-1)-z**(power-1)),
                         [0., 4.], [.31], rtol=1e-11, atol=1e-13, max_step=.02)
        assert np.all(np.diff(flow.y[0]) > 0) and flow.y[0, -1] < .5
        x = np.array([[-1.], [0.], [.5]])
        moved = x.copy(); moved[2, 0] += 1e-7
        numeric = (loss(moved, q, power)-loss(x, q, power))/1e-7
        assert abs(numeric+2*power*.5**(power-1)/3) < 2e-7
    for s in [.1, .2, .3, .4, .49]:
        assert abs(loss(np.array([[-1.], [0.], [s]]), q, 1.)-1/3) < 1e-14


def test_zero_bound_can_still_have_strict_actual_improvement():
    import torch
    from shared_branch import branch_system, branch_value
    x = torch.tensor([[-1.], [-1.], [5/6]], dtype=torch.float64)
    q = torch.tensor([[-1.], [.5], [1.5]], dtype=torch.float64)
    u, coef = torch.eye(3, dtype=x.dtype), torch.zeros((3, 1), dtype=x.dtype)
    center, matrix, a, b = branch_system(x, q, u, coef, 0.)
    labels = a.clone(); labels[1] = 1
    candidate = center+torch.linalg.solve(matrix, u[1])[:, None]*(q[1]-q[a[1]])/3
    old = branch_value(x, q, u, coef, a, b, 0.)
    assert abs(float(old-branch_value(x, q, u, candidate, labels, b, 0.))) < 1e-14
    assert abs(loss((x+candidate).numpy(), q.numpy(), 2.)-5/27) < 1e-14
    assert abs(float(old)-2/9) < 1e-14
