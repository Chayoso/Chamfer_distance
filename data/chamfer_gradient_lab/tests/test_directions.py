"""Independent optimizer and exact risk identities used in the direction audit."""
import numpy as np
import torch
from ndp_direction_audit import adam_direction


def test_adam_formula_with_and_without_history():
    gen = torch.Generator().manual_seed(81)
    for eps in [1e-8,1e-3]:
        p = torch.randn(7,generator=gen,dtype=torch.float64,requires_grad=True)
        opt = torch.optim.Adam([p],lr=.023,betas=(.83,.97),eps=eps)
        for _ in range(4):
            g = torch.randn(7,generator=gen,dtype=p.dtype)
            h = opt.state[p]
            history = [(h.get('exp_avg',torch.zeros_like(p)).clone(),
                        h.get('exp_avg_sq',torch.zeros_like(p)).clone(),int(h.get('step',0)))]
            proposed, = adam_direction([g],history,opt.param_groups[0])
            before = p.detach().clone()
            p.grad = g
            opt.step()
            torch.testing.assert_close(p.detach()-before,proposed,rtol=1e-12,atol=3e-16)


def test_coupled_linear_risk_expectation():
    """Enumerate a finite noise law; compare exact task risk with mean/covariance."""
    rng = np.random.default_rng(18)
    j = rng.normal(size=(6,4))
    k = j@j.T
    p, truth = rng.normal(size=(2,6))
    gstar = 2*(p-truth)/2  # Two 3D points, mean squared point error.
    samples = rng.normal(size=(5,6))
    probability = np.array([.1,.15,.2,.25,.3])
    mu = probability@samples
    centered = samples-mu
    covariance = (centered.T*probability)@centered
    eta = .013
    actual = sum(w*((p-eta*k@g-truth)**2).sum()/2 for w,g in zip(probability,samples))
    formula = ((p-truth)**2).sum()/2-eta*gstar@k@mu
    formula += eta**2/2*((k@mu)@(k@mu)+np.trace(k@covariance@k.T))
    np.testing.assert_allclose(actual,formula,rtol=1e-13,atol=1e-13)


def test_positive_coupling_can_reverse_task_alignment():
    k = np.array([[100.,5.],[5.,1.]])
    task, observed = np.ones(2), np.array([-.1,1.])
    assert np.linalg.eigvalsh(k).min() > 0
    assert task@observed > 0 and task@k@observed < 0


def test_infocd_code_small_coefficient_identity_away_from_clamp():
    from ndp_prior import prior
    from pipeline.losses import components
    gen = torch.Generator().manual_seed(149)
    p = torch.randn(13,3,generator=gen,dtype=torch.float64,requires_grad=True)
    q = torch.randn(13,3,generator=gen,dtype=torch.float64)+3
    a,b = components(p,q)
    assert min(float(a.min()),float(b.min())) > np.sqrt(1e-9)
    base = a.mean()+b.mean()
    explicit = base+2e-7*(torch.log((torch.exp(-.5*a)+1e-7).sum())+
                          torch.log((torch.exp(-.5*b)+1e-7).sum()))
    original = prior(p,q,'infocd_code',.1,1,torch.eye(3))
    torch.testing.assert_close(original,explicit,rtol=1e-13,atol=1e-13)
    gi, = torch.autograd.grad(original,p,retain_graph=True)
    ge, = torch.autograd.grad(explicit,p,retain_graph=True)
    gc, = torch.autograd.grad(base,p)
    torch.testing.assert_close(gi,ge,rtol=1e-12,atol=1e-13)
    assert float((gi-gc).norm()) <= 2e-7+1e-13
