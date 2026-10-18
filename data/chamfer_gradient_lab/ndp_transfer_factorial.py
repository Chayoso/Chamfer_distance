"""One frozen damping intervention; all other endpoint operations are reused."""
import ndp_field_probe as probe
from pipeline.fields import geometry_scales
from pipeline.capacity import plan_field, frozen_change
from pipeline.lifting import ridge_lift


def fields(p, q, seed):
    h, radius = geometry_scales(p, q)
    variable = p.detach().clone().requires_grad_()
    force = -probe.torch.autograd.grad(probe.cd(variable, q), variable)[0].detach()
    yield 'cd', force, lambda z: probe.cd(z, q), dict(spacing=h, radius=radius)
    force, mass, info = plan_field(p, q, 'unbalanced')
    if not info['converged']:
        raise RuntimeError(f'Unconverged transport: {info}')
    info.update(spacing=h, radius=radius)
    yield 'uot', force, lambda z: frozen_change(z, p, force, mass), info


def weak_ridge(forward, transpose, gradient, seed, maxiter=2048):
    return ridge_lift(forward, transpose, gradient, seed,
                      damping_ratio=.0001, maxiter=maxiter)


if __name__ == '__main__':
    probe.OUT = probe.ROOT/'paper_revision/mechanism_20261018/weak'
    probe.build_fields = fields
    probe.ridge_lift = weak_ridge
    probe.main()
