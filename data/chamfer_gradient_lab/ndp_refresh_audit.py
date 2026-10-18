"""Refresh only after accepted nonlinear moves, preserving all original fits."""
import ndp_field_probe as base
from pipeline.refresh import accepted_path
from pipeline.fields import geometry_scales


def jacobian_check(layer, inputs, seed):
    torch = base.torch; clone = base.copy.deepcopy(layer).double()
    p = clone(inputs)[0]; params = list(clone.parameters())
    saved = [v.detach().clone() for v in params]
    forward, transpose = base.jacobian_products(p, params)
    rng = torch.Generator(device=p.device).manual_seed(seed)
    direction = [torch.randn(v.shape, generator=rng, device=p.device, dtype=p.dtype) for v in params]
    norm = sum(v.square().sum() for v in direction).sqrt()
    direction = [v/norm for v in direction]
    u = torch.randn(p.shape, generator=rng, device=p.device, dtype=p.dtype)
    jv, jtu = forward(direction), transpose(u)
    left, right = (u*jv).sum(), sum((a*b).sum() for a, b in zip(direction, jtu))
    adjoint = float(abs(left-right)/(u.norm()*jv.norm()).clamp_min(1e-30))
    errors = []
    for exponent in range(16, 27, 2):
        epsilon = 2.**-exponent
        outputs = []
        for sign in (-1, 1):
            with torch.no_grad():
                for live, start, d in zip(params, saved, direction): live.copy_(start+sign*epsilon*d)
                outputs.append(clone(inputs)[0].detach())
        errors.append(float(((outputs[1]-outputs[0])/(2*epsilon)-jv).norm()/jv.norm()))
        if len(errors) >= 2 and min(errors) < 1e-4:
            break
    assert adjoint < 1e-8 and min(errors) < 1e-4, (adjoint, errors)
    return dict(adjoint_relative=adjoint, finite_difference_relative=errors,
                finite_difference_exponents=list(range(16, 27, 2))[:len(errors)])


def compare(pair, level, layer, opt, inputs, target, truth):
    torch = base.torch
    inputs, q, gt = inputs.double(), target.double(), truth.double()
    probe = base.copy.deepcopy(layer).double(); start = probe(inputs)[0].detach()
    h, _ = geometry_scales(start, q); bound = len(start)**.5*h
    seed = 93000+100*pair+level
    base.append('jacobians', dict(pair=pair, level=level, **jacobian_check(layer, inputs, seed)))
    for kind in ('cd', 'uot'):
        first = None
        for refresh in (False, True):
            for row, initial, p, velocity, after, old_output in accepted_path(layer, inputs, q, kind, refresh, seed, bound):
                if row['iteration'] == 1:
                    reference = OLD_D[pair, level, kind+'1']
                    expected = (reference['risk_derivative'], reference['raw_parameter_norm'])
                    actual = (float((2*(p-gt)/len(p)*velocity).sum()), row['raw_parameter_norm'])
                    trial = OLD_F[pair, level, kind+'1']
                    old_risk = float((old_output-gt).square().sum()/len(p)-(p-gt).square().sum()/len(p))
                    checks = dict(direction_exact=actual == expected,
                        slope_difference=actual[0]-expected[0], parameter_norm_difference=actual[1]-expected[1],
                        fraction_exact=row['old_first_fraction'] == trial['fraction'],
                        old_controller_mse_difference=old_risk-trial['risk_change'])
                    base.append('controls', dict(pair=pair, level=level, field=kind, refresh=refresh, **checks))
                    if kind == 'uot':
                        assert checks['direction_exact'] and checks['fraction_exact']
                        assert abs(checks['old_controller_mse_difference']) < 1e-12
                    if first is not None: torch.testing.assert_close(after, first, rtol=0, atol=0)
                    first = after.clone()
                displacement = after-p; task = 2*(p-gt)/len(p)
                change = float((after-gt).square().sum()/len(p)-(p-gt).square().sum()/len(p))
                identity = float((task*displacement).sum()+displacement.square().sum()/len(p))
                base.append('steps', dict(pair=pair, level=level, field=kind, refresh=refresh, **row,
                    task_slope=float((task*velocity).sum()), mse_change=change, mse_identity_error=abs(change-identity),
                    total_mse_change=float((after-gt).square().sum()/len(p)-(initial-gt).square().sum()/len(p)),
                    total_cd_change=float(base.cd(after, q)-base.cd(initial, q))))
    (base.OUT/'progress.json').write_text(base.json.dumps(dict(pair=pair, level=level))+'\n')


def records(name, keys):
    path = base.ROOT/'paper_revision/joint_20261018'/name
    rows = [base.json.loads(s) for s in path.read_text().splitlines()]
    if name == 'finite.txt': rows = [r for r in rows if r['policy'] == 'trust' and r['cap'] == 1]
    return {tuple(r[k] for k in keys): r for r in rows}


if __name__ == '__main__':
    OLD_D = records('directions.txt', ('pair', 'level', 'method'))
    OLD_F = records('finite.txt', ('pair', 'level', 'method'))
    base.OUT = base.ROOT/'paper_revision/refresh_audit_20261018/verified'
    base.compare = compare
    base.main()
