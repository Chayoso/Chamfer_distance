"""Independent checks of the trajectory derivations and measurement formulas."""
import numpy as np
import torch
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from experiment import chamfer, distances
from trajectory_audit import components, first_boundary


def test_closed_form_against_matrix_exponential():
    rng = np.random.default_rng(871)
    for _ in range(100):
        u, v = rng.uniform(-.02, .02, 2); delta = rng.uniform(.001, .015)
        rho = rng.uniform(.05, .95); t = rng.uniform(0, 10)
        k = (1-rho)*np.eye(3)+rho*np.ones((3, 3))
        z = expm(-4*t*k/3)@np.array([u, v, -delta])
        formula = np.exp(-4*(1-rho)*t/3)*((-2*delta-u-v)+(u+v-delta)*np.exp(-4*rho*t))/3
        assert abs(z[2]-formula) < 1e-15
        assert np.linalg.norm(z) <= np.linalg.norm([u, v, delta])+1e-15


def test_continuous_event_time_and_assignment_self_consistency():
    q = np.arange(-1., 2.); center = np.array([-1., 0., .5])
    for rho in [.2, .5, .8]:
        k = (1-rho)*np.eye(3)+rho*np.ones((3, 3))
        for v in [-.035, -.03, -.021]:
            expected = np.log((.01-v)/(-.02-v))/(4*rho)
            event = lambda t, z: z[2]
            event.terminal = True; event.direction = 1
            sol = solve_ivp(lambda t, z: -4*k@z/3, (0, 2*expected), [0, v, -.01],
                            events=event, rtol=1e-11, atol=1e-13, max_step=.02)
            assert abs(sol.t_events[0][0]-expected) < 1e-8
            for z in sol.y[:, :-1].T:
                d = (center[:, None]+z[:, None]-q)**2
                assert np.array_equal(d.argmin(1), [0, 1, 1])
                assert np.array_equal(d.argmin(0), [0, 1, 2])
            incoming = (-4*k@sol.y[:, -1]/3)[2]
            assert incoming > 0 and incoming+2/3 > 0


def test_discrete_crossing_condition_including_nonrescue():
    rho, alpha, delta = .5, .1, .01
    k = (1-rho)*np.eye(3)+rho*np.ones((3, 3))
    a, b = 1-alpha*4*(1-rho)/3, 1-alpha*4*(1+2*rho)/3
    for u in [-.01, 0., .01]:
        for v in [-.035, -.03, -.02, -.01, .01]:
            z = np.array([u, v, -delta]); crossed = False
            for step in range(1, 401):
                z = z-alpha*4*k@z/3
                expected = a**step*((-2*delta-u-v)+(u+v-delta)*(b/a)**step)/3
                assert abs(z[2]-expected) < 1e-15
                if z[2] > 1e-13: crossed = True; break
            assert crossed == (u+v < -2*delta-1e-12)
    # A tie-rule-sensitive zero is not counted as a strict crossing.


def test_forward_and_reverse_polynomials_and_first_forward_event():
    rng = np.random.default_rng(199)
    for _ in range(25):
        p, q, v = [rng.normal(size=(n, 3)) for n in [7, 9, 7]]
        v /= np.linalg.norm(v)
        d = ((p[:, None]-q)**2).sum(-1); a, b = d.argmin(1), d.argmin(0)
        for s in [.01, .3, 2.]:
            direct = ((p[:, None]+s*v[:, None]-q)**2).sum(-1)
            f = d-d[np.arange(7), a, None]+2*s*(v[:, None]*(q[a, None]-q)).sum(-1)
            assert np.allclose(f, direct-direct[np.arange(7), a, None], atol=1e-13)
            reverse = d-d[b, np.arange(9)]+2*s*((p[:, None]-q)*v[:, None]).sum(-1)
            reverse -= 2*s*((p[b]-q)*v[b]).sum(-1)
            reverse += s*s*((v*v).sum(-1)[:, None]-(v[b]*v[b]).sum(-1))
            assert np.allclose(reverse, direct-direct[b, np.arange(9)], atol=1e-13)
        p, q, v = map(torch.tensor, [p, q, v]); a = torch.tensor(a)
        radius = first_boundary(p, q, a, v)
        if np.isfinite(radius) and radius > 1e-10:
            assert torch.equal(distances(p+.999*radius*v, q).argmin(1), a)
            assert not torch.equal(distances(p+1.001*radius*v, q).argmin(1), a)


def test_weighted_pairs_rotate_total_gradient():
    p = torch.zeros((1, 2), dtype=torch.float64, requires_grad=True)
    q = torch.tensor([[1., 0.], [0., 2.]], dtype=torch.float64)
    d = distances(p, q)
    g = torch.autograd.grad(d.min(1).values.mean()+d.min(0).values.mean(), p, retain_graph=True)[0]
    weighted = torch.autograd.grad(d.min(1).values.square().mean()+d.min(0).values.square().mean(), p)[0]
    assert torch.equal(g, torch.tensor([[-3., -2.]]))
    assert torch.equal(weighted, torch.tensor([[-6., -16.]]))


def test_gradient_decomposition_matches_autograd():
    torch.manual_seed(871)
    p = torch.randn(13, 3, dtype=torch.float64, requires_grad=True)
    q = torch.randn(19, 3, dtype=torch.float64)
    gf, gr, _, _ = components(p, q)
    assert torch.allclose(gf+gr, torch.autograd.grad(chamfer(p, q), p)[0], atol=1e-14, rtol=1e-14)
