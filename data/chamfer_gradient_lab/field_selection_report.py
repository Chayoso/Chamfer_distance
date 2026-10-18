"""Fixed observable selectors on exact-field records; GT oracles stay diagnostic."""
import json
from pathlib import Path
from pipeline.selection import RULES, select
from fields_report import PAIRS, METHODS, net_counts

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'paper_revision/fields_20261018/exact'
OUT = ROOT/'paper_revision/fields_20261018/selection'
BASES = ('adam', 'cd_gd', 'cd_ridge')
NN = BASES+('robust_gd', 'robust_ridge')
POOLS = dict(nn=NN, centroid=NN+('centroid_gd', 'centroid_ridge'), all=METHODS)


def records(name):
    return [json.loads(line) for line in (SOURCE/(name+'.txt')).read_text().splitlines()]


def choose(d, f):
    choices = []
    for pair in PAIRS:
        for level in range(9):
            observations = []
            for method in METHODS:
                direction, trial = d[pair, level, method], f[pair, level, method, 'trust', 1.]
                observations.append(dict(method=method, cd_derivative=direction['cd_derivative'],
                    cd_change=trial['cd_change'], accepted=trial['accepted'],
                    linearization_error=trial['linearization_error'], ray_error=direction['transfer']['ray_error']))
            for pool, methods in POOLS.items():
                for rule in RULES:
                    selected = select([r for r in observations if r['method'] in methods], rule, BASES)
                    choices.append(dict(pair=pair, level=level, pool=pool, rule=rule, selected=selected))
    # No GT, scene identity, overlap labels or task outcomes enter select().
    (OUT/'choices.json').write_text(json.dumps(choices, indent=2)+'\n')
    return choices


def main():
    OUT.mkdir(exist_ok=False)
    verified = json.loads((SOURCE/'summary.json').read_text())
    assert verified['numerical_gate'] and verified['exact_replays'] == list(PAIRS)
    d = {(r['pair'], r['level'], r['method']): r for r in records('directions')}
    f = {(r['pair'], r['level'], r['method'], r['policy'], r['cap']): r for r in records('finite')}
    choices = choose(d, f)
    chosen = {(r['pair'], r['level'], r['pool'], r['rule']): r['selected'] for r in choices}
    assert len(chosen) == len(choices) == 486
    counts, gates, oracles, ideal = [], [], [], []
    for group, pairs in (('4DMatch', PAIRS[:4]), ('HF', PAIRS[4:])):
        keys = [(p, l) for p in pairs for l in range(9)]
        def outcome(key, method, policy, cap):
            if method is None:
                return 0.
            return d[key+(method,)]['risk_derivative'] if cap == 0 else f[key+(method, policy, cap)]['risk_change']
        for rule in RULES:
            for base in BASES+('nn_selector', 'centroid_selector'):
                for cap in (0, 1/16, .25, 1.):
                    for policy in (('raw',) if cap == 0 else ('raw', 'trust')):
                        a = [outcome(k, chosen[k+(base.removesuffix('_selector'), rule)] if base.endswith('_selector') else base, policy, cap) for k in keys]
                        b = [outcome(k, chosen[k+('all', rule)], 'trust', cap) for k in keys]
                        for eps in (1e-12, 1e-10, 1e-8):
                            counts.append(dict(group=group, rule=rule, baseline=base, cap=cap,
                                base_policy=policy, tolerance=eps, **net_counts(a, b, eps)))
        for pool, methods in POOLS.items():
            best = [min([0.]+[outcome(k, method, 'trust', 1.) for method in methods]) for k in keys]
            oracles.append(dict(group=group, pool=pool, uses_gt=True, descent_states=sum(v < -1e-12 for v in best),
                                states=len(keys), mean_best_mse_change=sum(best)/len(best)))
    fields = {(r['pair'], r['level'], r['method']): r for r in records('fields')}
    for group, pairs in (('4DMatch', PAIRS[:4]), ('HF', PAIRS[4:])):
        for field in ('robust', 'centroid', 'mmd_local', 'mmd_global', 'uot'):
            a, b = [], []
            for p in pairs:
                for level in range(9):
                    bound = d[p, level, 'cd_gd']['motion_bound']
                    for method, values in (('cd', a), (field, b)):
                        row = fields[p, level, method]
                        values.append(row['point_derivative']*bound/row['force_norm'] if row['force_norm'] else 0.)
            ideal.append(dict(group=group, field=field, **net_counts(a, b, 1e-12)))
    for rule in RULES:
        primary = [r for r in counts if r['rule'] == rule and r['cap'] in (0, .25, 1.) and r['tolerance'] == 1e-12]
        gates.append(dict(rule=rule, passed=all(r['net'] > 0 for r in primary),
                          passed_cells=sum(r['net'] > 0 for r in primary), required_cells=len(primary),
                          worst_net=min(r['net'] for r in primary)))
    summary = dict(gates=gates, advance_to_native=[r['rule'] for r in gates if r['passed']],
                   choices=len(choices), new_fits=0, new_scenes=0, gt_oracles=oracles, ideal_point_transfer=ideal)
    for name, value in (('counts.json', counts), ('summary.json', summary)):
        (OUT/name).write_text(json.dumps(value, indent=2)+'\n')
    lines = ['Retrospective fixed-rule analysis. All selections saved before GT scoring.',
             'No native runtime or unseen-scene claim. GT oracles are not selectors.']
    lines += [str(r) for r in gates]+['', 'GT-only upper bounds:']+[str(r) for r in oracles]
    lines += ['', 'Ideal point transfer vs equally ideal CD:']+[str(r) for r in ideal]
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
