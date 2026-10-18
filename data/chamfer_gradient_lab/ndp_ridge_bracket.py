"""Fixed initial states/plans, three damping values, unchanged true-objective trust."""
import ndp_field_probe as base
from pipeline.refresh import accepted_path
from pipeline.fields import geometry_scales
from slope_attribution import measure

OUT = base.ROOT/'paper_revision/ridge_bracket_20261018'
FACTORS = (.1, 1., 10.)


def compare(pair, level, layer, opt, inputs, target, truth):
    inputs, q, gt = inputs.double(), target.double(), truth.double()
    clone = base.copy.deepcopy(layer).double(); start = clone(inputs)[0].detach()
    h, _ = geometry_scales(start, q); bound = len(start)**.5*h
    seed = 93000+100*pair+level
    for kind in ('cd', 'uot'):
        for factor in FACTORS:
            diagnostics = []
            def observe(p, force, mass, motion):
                diagnostics.append(measure(p, gt, force, mass, motion))
            path = accepted_path(layer, inputs, q, kind, True, seed, bound, steps=1,
                                 observe=observe, damping_ratio=.0001*factor)
            for row, initial, p, velocity, after, _ in path:
                displacement = after-p; task = 2*(p-gt)/len(p)
                change = float((after-gt).square().sum()/len(p)-(p-gt).square().sum()/len(p))
                identity = float((task*displacement).sum()+displacement.square().sum()/len(p))
                record = dict(pair=pair, level=level, field=kind, refresh=True, **row,
                    task_slope=float((task*velocity).sum()), mse_change=change, mse_identity_error=abs(change-identity),
                    total_mse_change=float((after-gt).square().sum()/len(p)-(initial-gt).square().sum()/len(p)),
                    total_cd_change=float(base.cd(after, q)-base.cd(initial, q)), attribution=diagnostics.pop())
                assert not diagnostics
                if factor == 1.: assert record == EXPECTED[pair, level, kind], ('Changed central control',pair,level,kind)
                base.append('bracket', dict(**record, factor=factor))
    (base.OUT/'progress.json').write_text(base.json.dumps(dict(pair=pair, level=level))+'\n')


if __name__ == '__main__':
    reference = base.ROOT/'paper_revision/attribution_20261018/attribution.txt'
    EXPECTED = {(r['pair'],r['level'],r['field']): r for r in
                [base.json.loads(s) for s in reference.read_text().splitlines()] if r['refresh'] and r['iteration']==1}
    assert len(EXPECTED) == 108
    base.OUT = OUT; base.compare = compare; base.main()
