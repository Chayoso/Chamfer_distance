"""Independent dense/primal references, including nonzero refresh coordinates."""
import numpy as np
import torch
from scipy.optimize import root
from scipy.special import logsumexp
from pipeline.joint import correspondence, fixed_lift, joint_path


def primal_plan(p, q, epsilon):
    """Solve the primal KKT equations in log(pi), without production Sinkhorn."""
    cost = .5*((p[:, None]-q[None, :])**2).sum(2)
    n, m = cost.shape
    def equation(flat):
        log = flat.reshape(n, m)
        return (cost/epsilon+log+np.log(n*m)
                +2*(logsumexp(log, axis=1)+np.log(n))[:, None]
                +2*(logsumexp(log, axis=0)+np.log(m))[None, :]).ravel()
    result = root(equation, -cost.ravel()/epsilon-np.log(n*m), tol=1e-11)
    assert np.max(np.abs(equation(result.x))) < 1e-10
    plan = np.exp(result.x.reshape(n, m))
    def kl(a, b):
        return (a*np.log(a/b)-a+b).sum()
    value = (plan*cost).sum()+epsilon*kl(plan, 1/(n*m))
    value += 2*epsilon*(kl(plan.sum(1), 1/n)+kl(plan.sum(0), 1/m))
    return plan, value


def reference(p, q, z, kind, epsilon):
    n, m = len(p), len(q)
    if kind == 'uot':
        weights, value = primal_plan(z, q, epsilon)
    else:
        distance = np.linalg.norm(z[:, None]-q[None, :], axis=2)
        ai, bi = distance.argmin(1), distance.argmin(0)
        weights = np.zeros((n, m))
        for i in range(n): weights[i, ai[i]] += 1/(n*distance[i, ai[i]])
        for j in range(m): weights[bi[j], j] += 1/(m*distance[bi[j], j])
        value = distance.min(1).mean()+distance.min(0).mean()
    mass = weights.sum(1)
    return weights@q-mass[:, None]*p, mass, value


def fixture():
    p = np.array([[-.3, .2], [.4, -.1], [.2, .5]])
    q = np.array([[.8, .7], [1.2, .1], [.7, 1.3], [1.6, .9]])
    j = np.array([[1., 0., .2], [0., 1., -.1], [1., 0., .4],
                  [0., 1., .3], [1., 0., -.2], [0., 1., .1]])
    epsilon = (((p[:, None]-q[None, :])**2).sum(2).min(0).mean()/2)
    return p, q, j, epsilon


def test_refreshed_force_coordinates_gradient_and_independent_primal_value():
    p, q, j, epsilon = fixture()
    z = p+(j@np.array([.13, -.09, .2])).reshape(p.shape)
    for kind in ('cd', 'uot'):
        force, mass, value, _ = correspondence(torch.tensor(p), torch.tensor(q), kind)(torch.tensor(z))
        expected, weight, objective = reference(p, q, z, kind, epsilon)
        np.testing.assert_allclose(force, expected, rtol=2e-8, atol=1e-10)
        np.testing.assert_allclose(mass, weight, rtol=2e-8, atol=1e-10)
        assert abs(value-objective) < 1e-10
        # Force is relative to ORIGINAL p; the current negative gradient differs.
        current = force.numpy()-mass.numpy()[:, None]*(z-p)
        direction = np.array([[.3, -.2], [.1, .4], [-.2, .3]])
        h = 1e-5
        plus = reference(p, q, z+h*direction, kind, epsilon)[2]
        minus = reference(p, q, z-h*direction, kind, epsilon)[2]
        assert abs((plus-minus)/(2*h)+(current*direction).sum()) < 2e-8
        assert np.linalg.norm(force.numpy()-current) > .01


def test_all_four_total_solves_match_independent_dense_reference():
    p, q, j, epsilon = fixture(); matrix = torch.tensor(j)
    forward = lambda delta: (matrix@delta[0]).reshape(p.shape)
    transpose = lambda v: [matrix.T@v.flatten()]
    for kind in ('cd', 'uot'):
        rows = list(joint_path(torch.tensor(p), torch.tensor(q), forward, transpose, 71, kind))
        gamma = rows[0][-1][0]['solve']['damping']
        z = p.copy(); expected = None
        for iteration in range(1, 5):
            force, mass, value = reference(p, q, z, kind, epsilon)
            a = j.T@(np.repeat(mass, 2)[:, None]*j)+gamma*np.eye(j.shape[1])
            expected = np.linalg.solve(a, j.T@force.ravel())
            delta, info = fixed_lift(forward, transpose, torch.tensor(force), torch.tensor(mass), gamma)
            assert info['converged']
            np.testing.assert_allclose(delta[0], expected, rtol=1e-7, atol=1e-8)
            record = rows[-1][-1][iteration-1]
            assert abs(value-record['objective']) < 2e-7
            z = p+(j@expected).reshape(p.shape)
            if iteration in (1, 4):
                actual = rows[0 if iteration == 1 else 1][1][0].numpy()
                np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)


def test_incremental_and_total_normal_equations_require_regularizer_shift():
    p, q, j, epsilon = fixture(); previous = np.array([.13, -.09, .2])
    z = p+(j@previous).reshape(p.shape); gamma = .2
    for kind in ('cd', 'uot'):
        force, mass, _ = reference(p, q, z, kind, epsilon)
        a = j.T@(np.repeat(mass, 2)[:, None]*j)+gamma*np.eye(j.shape[1])
        total = np.linalg.solve(a, j.T@force.ravel())
        incremental_force = force-mass[:, None]*(z-p)
        increment = np.linalg.solve(a, j.T@incremental_force.ravel()-gamma*previous)
        np.testing.assert_allclose(previous+increment, total, atol=1e-12)
        # Omitting the shift changes the optimization problem, even with correct NN.
        wrong = previous+np.linalg.solve(a, j.T@incremental_force.ravel())
        assert np.linalg.norm(wrong-total) > .01
