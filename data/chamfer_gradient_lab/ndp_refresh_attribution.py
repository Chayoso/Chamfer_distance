"""Replay unchanged accepted paths; evaluate point proposals after selection."""
import ndp_field_probe as base
from pipeline.refresh import accepted_path
from pipeline.fields import geometry_scales
from slope_attribution import measure

OUT = base.ROOT/'paper_revision/attribution_20261018'
REFERENCE = base.ROOT/'paper_revision/refresh_audit_20261018/verified'


def compare(pair, level, layer, opt, inputs, target, truth):
    torch = base.torch
    inputs, q, gt = inputs.double(), target.double(), truth.double()
    clone = base.copy.deepcopy(layer).double(); start = clone(inputs)[0].detach()
    h, _ = geometry_scales(start, q); bound = len(start)**.5*h
    seed = 93000+100*pair+level
    for kind in ('cd', 'uot'):
        for refresh in (False, True):
            diagnostic = []
            def observe(p, force, mass, raw_motion):
                diagnostic.append(measure(p, gt, force, mass, raw_motion))
            for row, initial, p, velocity, after, _ in accepted_path(layer, inputs, q, kind, refresh, seed, bound, observe=observe):
                displacement = after-p; task = 2*(p-gt)/len(p)
                change = float((after-gt).square().sum()/len(p)-(p-gt).square().sum()/len(p))
                identity = float((task*displacement).sum()+displacement.square().sum()/len(p))
                replay = dict(pair=pair, level=level, field=kind, refresh=refresh, **row,
                    task_slope=float((task*velocity).sum()), mse_change=change, mse_identity_error=abs(change-identity),
                    total_mse_change=float((after-gt).square().sum()/len(p)-(initial-gt).square().sum()/len(p)),
                    total_cd_change=float(base.cd(after, q)-base.cd(initial, q)))
                key = (pair, level, kind, refresh, row['iteration'])
                assert replay == EXPECTED[key], ('Trajectory changed', key)
                metrics = diagnostic.pop()
                assert not diagnostic
                base.append('attribution', dict(**replay, attribution=metrics))
    (base.OUT/'progress.json').write_text(base.json.dumps(dict(pair=pair, level=level))+'\n')


if __name__ == '__main__':
    EXPECTED = {tuple(r[k] for k in ('pair','level','field','refresh','iteration')): r
                for r in [base.json.loads(s) for s in (REFERENCE/'steps.txt').read_text().splitlines()]}
    assert len(EXPECTED) == 864
    base.OUT = OUT
    base.compare = compare
    base.main()
