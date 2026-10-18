"""Transfer frozen observed FGR anchors at original NDP endpoints; no new fit rule."""
from functools import partial
import ndp_field_probe as base
from global_registration_control import load
from pipeline.fields import geometry_scales

OUT = base.ROOT/'paper_revision/fgr_20261018/neural'


def fields(p, q, seed, anchor):
    h, radius = geometry_scales(p, q)
    z = p.detach().requires_grad_(True)
    cd_force = -base.torch.autograd.grad(base.cd(z, q), z)[0].detach()
    info = dict(spacing=h, radius=radius)
    yield 'cd', cd_force, lambda z: base.cd(z, q), info
    yield 'fgr', anchor-p, lambda z: .5*(z-anchor).square().sum(), info


def environment(pair, truth):
    x = base.torch.as_tensor(load(pair)[0], device=truth.device, dtype=base.torch.float32)
    transform = base.torch.as_tensor(POSES[pair], device=x.device, dtype=base.torch.float64)
    def end(level, iteration, count, previous, value, opt, layer, inputs, q, cached, ids, mean):
        # Same original source sample and target centering as the official fit.
        anchor = x[ids].double()@transform[:3, :3].T+transform[:3, 3]-mean.double()
        base.build_fields = lambda p, target, seed: fields(p, target, seed, anchor)
        base.compare(pair, level, layer, opt, inputs, q, truth[ids]-mean)
    return base.stopping_environment(lambda *args: None, end)[0]


if __name__ == '__main__':
    observed = base.json.loads((OUT.parent/'observed.json').read_text())
    POSES = {r['pair']: r['transform'] for r in observed['rows']}
    assert set(POSES) == set(base.PAIRS)
    base.OUT = OUT
    base.environment = environment
    base.ridge_lift = partial(base.ridge_lift, damping_ratio=.0001)
    base.main()
