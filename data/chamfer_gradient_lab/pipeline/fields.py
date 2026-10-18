"""Observed-geometry direction controls; frozen scales and local random features."""
from .model import torch
from .controlled import cd
from .losses import components
from .capacity import plan_field, frozen_change


def geometry_scales(p, q):
    spacings = []
    for cloud in (p, q):
        distance = torch.cdist(cloud, cloud, compute_mode='donot_use_mm_for_euclid_dist')
        distance.fill_diagonal_(float('inf'))
        spacings.append(distance.min(1).values)
    h = float(torch.cat(spacings).mean())
    radius = float(sum((z-z.mean(0)).square().sum(1).mean().sqrt() for z in (p, q))/2)
    if not 0 < h <= radius*2 or not torch.isfinite(p).all() or not torch.isfinite(q).all():
        raise ValueError('Finite nondegenerate clouds required')
    return h, radius


def robust_nn(p, q, scale):
    """Continuous saturating Euclidean NN loss; still nonsmooth at NN ties."""
    a, b = components(p, q)
    return -scale*(torch.expm1(-a/scale).mean()+torch.expm1(-b/scale).mean())


def rff_objective(q, bandwidth, seed, features=512):
    generator = torch.Generator(device=q.device).manual_seed(seed)
    frequencies = torch.randn(q.shape[1], features, generator=generator,
                              dtype=q.dtype, device=q.device)/bandwidth
    def embedding(z):
        phase = z@frequencies
        return torch.cat((phase.sin().mean(0), phase.cos().mean(0)))/features**.5
    target = embedding(q).detach()
    return lambda z: (embedding(z)-target).square().sum()


def gaussian_force(p, q, bandwidth):
    """Negative exact biased Gaussian MMD gradient, including source repulsion."""
    pp = torch.cdist(p, p, compute_mode='donot_use_mm_for_euclid_dist').square()
    pq = torch.cdist(p, q, compute_mode='donot_use_mm_for_euclid_dist').square()
    a, b = (-pp/(2*bandwidth**2)).exp(), (-pq/(2*bandwidth**2)).exp()
    return 2/bandwidth**2*((b@q-b.sum(1)[:, None]*p)/(len(p)*len(q))
                          +(a.sum(1)[:, None]*p-a@p)/len(p)**2)


def gaussian_objective(q, bandwidth):
    def kernel(a, b):
        distance = torch.cdist(a, b, compute_mode='donot_use_mm_for_euclid_dist')
        return (-distance.square()/(2*bandwidth**2)).exp().mean()
    constant = kernel(q, q).detach()
    return lambda p: kernel(p, p)+constant-2*kernel(p, q)


def build_fields(p, q, seed):
    """Every choice depends only on current observed clouds and a fixed seed."""
    p, q = p.detach(), q.detach()
    h, radius = geometry_scales(p, q)
    objectives = [('cd', lambda z: cd(z, q)),
                  ('robust', lambda z: robust_nn(z, q, 4*h)),
                  ('centroid', lambda z: (z.mean(0)-q.mean(0)).square().sum()),
                  ('mmd_local', rff_objective(q, 4*h, seed)),
                  ('mmd_global', rff_objective(q, radius, seed))]
    for name, objective in objectives:
        variable = p.clone().requires_grad_()
        force = -torch.autograd.grad(objective(variable), variable)[0].detach()
        info = dict(spacing=h, radius=radius)
        if name.startswith('mmd'):
            bandwidth = 4*h if name == 'mmd_local' else radius
            exact = gaussian_force(p, q, bandwidth)
            info.update(bandwidth=bandwidth, features=512,
                        exact_gradient_relative_error=float((force-exact).norm()/exact.norm().clamp_min(1e-30)),
                        exact_gradient_cosine=float((force*exact).sum()/(force.norm()*exact.norm()).clamp_min(1e-30)))
        yield name, force, objective, info
    force, mass, info = plan_field(p, q, 'unbalanced')
    if not info['converged']:
        raise RuntimeError(f'Unconverged transport: {info}')
    info.update(spacing=h, radius=radius)
    yield 'uot', force, lambda z: frozen_change(z, p, force, mass), info
