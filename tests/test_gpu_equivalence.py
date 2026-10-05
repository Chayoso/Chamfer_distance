"""GPU code paths must agree with the CPU reference (skipped when CUDA is absent)."""
import numpy as np
import pytest
import torch

from chamfer_collapse.chamfer import chamfer_components
from chamfer_collapse.geometry import knn_indices, nn_indices

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")


@cuda
def test_gpu_nn_matches_kdtree_distances():
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(3000, 3, generator=g), torch.randn(2500, 3, generator=g)
    i_cpu, i_gpu = nn_indices(a, b), nn_indices(a.cuda(), b.cuda()).cpu()
    d_cpu = (a - b[i_cpu]).norm(dim=1); d_gpu = (a - b[i_gpu]).norm(dim=1)
    assert torch.allclose(d_cpu, d_gpu, atol=1e-5)          # same nearest distance
    assert (i_cpu == i_gpu).float().mean() > 0.999            # indices equal except float ties


@cuda
def test_gpu_knn_matches_kdtree_distances():
    p = torch.randn(1500, 3, generator=torch.Generator().manual_seed(1))
    c, g = knn_indices(p, 6), knn_indices(p.cuda(), 6).cpu()
    assert torch.allclose((p[:, None] - p[c]).norm(dim=-1), (p[:, None] - p[g]).norm(dim=-1), atol=1e-5)


@cuda
def test_gpu_chamfer_and_gradient_match_cpu():
    g = torch.Generator().manual_seed(2)
    s, t = torch.randn(800, 3, generator=g), torch.randn(900, 3, generator=g)
    s_cpu = s.clone().requires_grad_(True); s_gpu = s.cuda().requires_grad_(True)
    r1, r2 = chamfer_components(s_cpu, t), chamfer_components(s_gpu, t.cuda())
    (r1.fwd_sq.mean() + r1.rev_sq.mean()).backward(); (r2.fwd_sq.mean() + r2.rev_sq.mean()).backward()
    assert r1.two_sided.item() == pytest.approx(r2.two_sided.item(), rel=1e-5)
    assert torch.allclose(s_cpu.grad, s_gpu.grad.cpu(), atol=1e-6)
