"""Independent numerical checks; these do not replace the exact proofs."""
import numpy as np
import torch
from experiment import chamfer
from shared_branch import branch_system, branch_value, switch_proposal


def test_branch_system_gradient_and_center():
    torch.manual_seed(601)
    for _ in range(20):
        x, y, u, coef = [torch.randn(*s, dtype=torch.float64) for s in [(7, 3), (9, 3), (7, 4), (4, 3)]]
        coef.requires_grad_(); ridge = .03
        center, matrix, a, b = branch_system(x, y, u, coef, ridge)
        actual = chamfer(x+u@coef, y)+ridge*coef.square().sum()
        branch = branch_value(x, y, u, coef, a, b, ridge)
        assert torch.allclose(actual, branch, atol=1e-13)
        grad, = torch.autograd.grad(branch, coef)
        assert torch.allclose(grad, 2*matrix@(coef-center), atol=1e-12)
        fixed = center.detach().requires_grad_()
        g, = torch.autograd.grad(branch_value(x, y, u, fixed, a, b, ridge), fixed)
        assert g.norm() < 1e-12


def test_all_label_gain_formula_against_direct_resolves():
    torch.manual_seed(602)
    for _ in range(15):
        x, y, u, coef = [torch.randn(*s, dtype=torch.float64) for s in [(5, 2), (6, 2), (5, 3), (3, 2)]]
        ridge = .04
        center, matrix, a, b = branch_system(x, y, u, coef, ridge)
        old = branch_value(x, y, u, coef, a, b, ridge)
        gains, candidates = [], []
        for i in range(len(x)):
            for j in range(len(y)):
                if j == a[i]: continue
                labels = a.clone(); labels[i] = j
                zero = torch.zeros_like(coef, requires_grad=True)
                gradient, = torch.autograd.grad(branch_value(x, y, u, zero, labels, b, ridge), zero)
                candidate = torch.linalg.solve(2*matrix, -gradient)
                gains.append(float(old-branch_value(x, y, u, candidate, labels, b, ridge)))
                candidates.append(candidate)
        proposal, gain, i, j, leverage = switch_proposal(x, y, u, coef, ridge)
        assert abs(gain-max(gains)) < 1e-11
        assert torch.allclose(proposal, candidates[np.argmax(gains)], atol=1e-11)
        new = chamfer(x+u@proposal, y)+ridge*proposal.square().sum()
        assert float(old-new) >= gain-1e-11
        assert 0 <= leverage <= 1+1e-12


def test_shared_stationarity_does_not_imply_particle_stationarity():
    x = torch.tensor([[-1.], [1.]], dtype=torch.float64)
    p = (x+.5).requires_grad_(); y = torch.zeros((1, 1), dtype=x.dtype)
    loss = chamfer(p, y); g, = torch.autograd.grad(loss, p)
    assert abs(float(loss.detach())-1.5) < 1e-14
    assert abs(float(g.sum())) < 1e-14 and g.norm() > 2
    for delta in [-.001, .001]: assert chamfer(p.detach()+delta, y) > loss


def test_nonlinear_stationary_maximum_and_active_tie_rescue():
    theta = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    loss = 2*(theta.square()-1).square(); g, = torch.autograd.grad(loss, theta, create_graph=True)
    h, = torch.autograd.grad(g, theta)
    assert g == 0 and h == -8
    x = torch.tensor([[-1.], [0.], [.5]], dtype=torch.float64)
    y = torch.tensor([[-1.], [0.], [1.]], dtype=torch.float64)
    u, coef = torch.eye(3, dtype=x.dtype), torch.zeros((3, 1), dtype=x.dtype)
    proposal, bound, *_ = switch_proposal(x, y, u, coef, 0.)
    assert bound > 0 and chamfer(x+u@proposal, y) < chamfer(x, y)


def test_ridge_branch_center_response_to_source_positions():
    torch.manual_seed(612)
    for ridge in [.001, .03, 1.]:
        u, x, q = [torch.randn(*shape, dtype=torch.float64) for shape in [(5, 3), (5, 2), (4, 2)]]
        a, b = torch.tensor([0, 1, 2, 3, 1]), torch.tensor([0, 2, 3, 4])
        weight = 1/5+torch.bincount(b, minlength=5).double()/4
        matrix = u.T@(weight[:, None]*u)+ridge*torch.eye(3, dtype=u.dtype)
        response = torch.eye(5, dtype=u.dtype)-u@torch.linalg.solve(matrix, u.T*weight)
        inverse = torch.linalg.inv(torch.eye(5, dtype=u.dtype)+(u@u.T)*weight/ridge)
        assert torch.allclose(response, inverse, atol=2e-12, rtol=0)
        def center_at(source):
            rhs = (q[a]-source)/5
            rhs.index_add_(0, b, (q-source[b])/4)
            return source+u@torch.linalg.solve(matrix, u.T@rhs)
        delta = torch.randn_like(x)*.01
        assert torch.allclose(center_at(x+delta)-center_at(x), response@delta, atol=1e-13, rtol=0)
