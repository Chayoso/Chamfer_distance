"""Damped least-squares lifting of a point gradient; no ground truth or full J."""
from .model import torch


def jacobian_products(output, params):
    dummy = torch.zeros_like(output,requires_grad=True)
    graph = torch.autograd.grad(output,params,grad_outputs=dummy,
                                create_graph=True,retain_graph=True)

    def transpose(vector):
        return [g.detach() for g in torch.autograd.grad(output,params,
                grad_outputs=vector.detach(),retain_graph=True)]

    def forward(direction):
        scalar = sum((a*b.detach()).sum() for a,b in zip(graph,direction))
        return torch.autograd.grad(scalar,dummy,retain_graph=True)[0].detach()

    return forward,transpose


def cg(operator, rhs, rtol=1e-8, maxiter=128):
    x = torch.zeros_like(rhs)
    r = rhs.detach().clone()
    direction = r.clone()
    rr, initial = (r*r).sum(), float(r.norm())
    if initial == 0:
        return x,dict(iterations=0,residual_norm=0.,relative_residual=0.,converged=True)
    iterations = 0
    for iterations in range(1,maxiter+1):
        product = operator(direction)
        curvature = (direction*product).sum()
        if not torch.isfinite(curvature) or curvature <= 0:
            raise RuntimeError('CG encountered nonpositive/nonfinite curvature')
        step = rr/curvature
        x, r = x+step*direction, r-step*product
        next_rr = (r*r).sum()
        if float(next_rr.sqrt()) <= rtol*initial: break
        direction = r+(next_rr/rr)*direction
        rr = next_rr
    residual = float((rhs-operator(x)).norm())
    return x,dict(iterations=iterations,residual_norm=residual,
                  relative_residual=residual/initial,converged=residual <= rtol*initial)


def ridge_lift(forward, transpose, gradient, seed, damping_ratio=.01, maxiter=128):
    """Return -J^T(K+lambda I)^-1 g and auditable numerical diagnostics."""
    calls = 0

    def kernel(vector):
        nonlocal calls
        calls += 1
        return forward(transpose(vector))

    generator = torch.Generator(device=gradient.device).manual_seed(seed)
    vector = torch.randn(gradient.shape,generator=generator,device=gradient.device,dtype=gradient.dtype)
    vector /= vector.norm()
    for _ in range(8):
        product = kernel(vector)
        if product.norm() == 0: raise RuntimeError('Zero sampled Jacobian direction')
        vector = product/product.norm()
    spectral = float((vector*kernel(vector)).sum())
    damping = damping_ratio*spectral
    if damping <= 0 or not torch.isfinite(torch.tensor(damping)):
        raise RuntimeError('Invalid damping')
    dual, info = cg(lambda z:kernel(z)+damping*z,gradient.detach(),maxiter=maxiter)
    delta = [-g for g in transpose(dual)]
    info.update(spectral_estimate=spectral,damping=damping,kernel_calls=calls)
    return delta,info


def weighted_lift(forward, transpose, force, mass, seed, damping_ratio=.01, maxiter=128):
    """Regularized mass-weighted displacement fit, including zero-mass rows."""
    if mass.shape != force.shape[:-1] or not torch.isfinite(mass).all() or not torch.isfinite(force).all():
        raise ValueError('Finite force and one mass per point are required')
    if (mass < 0).any() or mass.max() <= 0 or (force[mass == 0] != 0).any():
        raise ValueError('Nonnegative mass with positive total and zero force at zero mass required')
    root = mass.sqrt()[:, None]
    target = force/root.clamp_min(torch.finfo(force.dtype).tiny)
    weighted_forward = lambda delta: root*forward(delta)
    weighted_transpose = lambda vector: transpose(root*vector)
    return ridge_lift(weighted_forward, weighted_transpose, -target, seed, damping_ratio, maxiter)
