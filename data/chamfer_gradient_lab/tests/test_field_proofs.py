import torch
from pipeline.fields import gaussian_force


def test_squared_cell_gradient_and_geometric_median_gradient_are_distinct():
    p = torch.tensor([[.2], [2.7]], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([[0.], [.5], [3.]], dtype=torch.float64)
    distances = (p[:, None]-q[None, :]).square().sum(2)
    a, b = distances.argmin(1), distances.argmin(0)
    squared = distances.min(1).values.mean()+distances.min(0).values.mean()
    euclidean = distances.sqrt().min(1).values.mean()+distances.sqrt().min(0).values.mean()
    g2 = torch.autograd.grad(squared, p, retain_graph=True)[0]
    g1 = torch.autograd.grad(euclidean, p)[0]
    expected2, expected1 = [], []
    for i in range(len(p)):
        residuals = p[i]-q[b == i]
        forward = p[i]-q[a[i]]
        expected2.append(2*forward/len(p)+2*residuals.sum(0)/len(q))
        expected1.append(forward/forward.norm()/len(p)+(residuals/residuals.norm(dim=1, keepdim=True)).sum(0)/len(q))
    torch.testing.assert_close(g2, torch.stack(expected2))
    torch.testing.assert_close(g1, torch.stack(expected1))
    assert not torch.allclose(g1, g2)


def test_positive_ridge_transfer_can_reverse_task_descent():
    kernel = torch.diag(torch.tensor([1., .01], dtype=torch.float64))
    force = torch.ones(2, dtype=torch.float64)
    task = torch.tensor([1., -2.], dtype=torch.float64)
    velocity = kernel@torch.linalg.solve(kernel+torch.eye(2, dtype=torch.float64), force)
    assert task@force < 0 and task@velocity > 0 and force@velocity > 0


def test_gaussian_centroid_remainder_bound():
    generator = torch.Generator().manual_seed(91)
    p, q = [torch.randn(n, 2, generator=generator, dtype=torch.float64) for n in (5, 8)]
    distance = max(float(torch.cdist(p, p).max()), float(torch.cdist(p, q).max()))
    for scale in (.5, 2., 10., 30.):
        force = gaussian_force(p, q, scale)
        centroid = 2*(q.mean(0)-p.mean(0)).expand_as(p)/(len(p)*scale**2)
        bound = 2*distance**3/(len(p)**.5*scale**4)
        assert float((force-centroid).norm()) <= bound+1e-14


def test_same_unlabeled_target_opposite_truths_cannot_both_improve():
    angle = torch.arange(3, dtype=torch.float64)*2*torch.pi/3
    a = torch.stack((angle.cos(), angle.sin()), 1)
    b = a.roll(1, 0)
    p = (a+b)/2
    assert float(torch.pdist(p).min()) > 0
    generator = torch.Generator().manual_seed(55)
    for _ in range(30):
        motion = torch.randn(p.shape, generator=generator, dtype=p.dtype)*.3
        changes = [((p+motion-g).square().sum()-(p-g).square().sum())/len(p) for g in (a, b)]
        torch.testing.assert_close(sum(changes), 2*motion.square().sum()/len(p), atol=1e-14, rtol=1e-12)
        assert max(changes) > 0


def test_centroid_task_guarantee_needs_correct_target_mean():
    truth = torch.tensor([[0.], [1.], [2.]], dtype=torch.float64)
    p = truth+2
    for step in (.1, 1., 1.9):
        motion = step*(truth.mean(0)-p.mean(0))
        change = ((p+motion-truth).square().sum()-(p-truth).square().sum())/len(p)
        torch.testing.assert_close(change, -step*(2-step)*(truth.mean(0)-p.mean(0)).square().sum())
    # A correctly aligned source is harmed by the mean of a visible target subset.
    partial_target = truth[1:]
    motion = partial_target.mean(0)-truth.mean(0)
    assert (truth+motion-truth).square().sum() > 0
