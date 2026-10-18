"""Validate the frozen endpoint comparison and report all failed/passing screens."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'paper_revision/fields_20261018'
FIELDS = ('cd', 'robust', 'centroid', 'mmd_local', 'mmd_global', 'uot')
METHODS = tuple(f+'_'+t for f in FIELDS for t in ('gd', 'ridge'))+('adam',)
PAIRS = (100, 101, 124, 125, 200, 201)


def read(name):
    return [json.loads(line) for line in (OUT/(name+'.txt')).read_text().splitlines()]


def net_counts(baseline, candidate, tolerance):
    a, b = np.asarray(baseline), np.asarray(candidate)
    rescue = int(((a > tolerance) & (b < -tolerance)).sum())
    damage = int(((a < -tolerance) & (b > tolerance)).sum())
    lost = int(((a < -tolerance) & (abs(b) <= tolerance)).sum())
    return dict(rescued=rescue, damaged=damage, lost=lost, net=rescue-damage-lost)


def coverage(rows, labels, expected):
    observed = [tuple(row[k] for k in labels) for row in rows]
    assert len(observed) == len(expected) and set(observed) == expected, labels


def assess(d, f):
    counts, gates = [], []
    for candidate in METHODS:
        if candidate in ('cd_gd', 'cd_ridge', 'adam'):
            continue
        field, transfer = candidate.rsplit('_', 1)
        bases = {'adam', 'cd_gd', 'cd_'+transfer}
        if transfer == 'ridge':
            bases.add(field+'_gd')
        if field.startswith('mmd') or field == 'uot':
            bases.add('centroid_'+transfer)
        for group, pairs in (('4DMatch', PAIRS[:4]), ('HF', PAIRS[4:])):
            keys = [(p, l) for p in pairs for l in range(9)]
            for base in sorted(bases):
                for cap in (0., 1/16, .25, 1.):
                    for policy in (('raw',) if cap == 0 else ('raw', 'trust')):
                        for controller in (('trust',) if cap == 0 else ('trust', 'cd_trust')):
                            a = [d[k+(base,)]['risk_derivative'] if cap == 0 else f[k+(base, policy, cap)]['risk_change'] for k in keys]
                            b = [d[k+(candidate,)]['risk_derivative'] if cap == 0 else f[k+(candidate, controller, cap)]['risk_change'] for k in keys]
                            for eps in (1e-12, 1e-10, 1e-8):
                                counts.append(dict(candidate=candidate, baseline=base, group=group, cap=cap,
                                    base_policy=policy, controller=controller, tolerance=eps, **net_counts(a, b, eps)))
        primary = [r for r in counts if r['candidate'] == candidate and r['controller'] == 'trust'
                   and r['cap'] in (0, .25, 1.) and r['tolerance'] == 1e-12]
        gates.append(dict(candidate=candidate, pass_screen=all(r['net'] > 0 for r in primary),
                          passed_cells=sum(r['net'] > 0 for r in primary), required_cells=len(primary),
                          worst_net=min(r['net'] for r in primary)))
    return counts, gates


def main():
    rows = {name: read(name) for name in ('states', 'fields', 'solvers', 'directions', 'finite')}
    states = {(p, l) for p in PAIRS for l in range(9)}
    coverage(rows['states'], ('pair', 'level'), states)
    for name in ('fields', 'solvers'):
        coverage(rows[name], ('pair', 'level', 'method'), {(*s, m) for s in states for m in FIELDS})
    coverage(rows['directions'], ('pair', 'level', 'method'), {(*s, m) for s in states for m in METHODS})
    coverage(rows['finite'], ('pair', 'level', 'method', 'policy', 'cap'),
             {(*s, m, p, c) for s in states for m in METHODS for p in ('raw', 'trust', 'cd_trust') for c in (1/16, .25, 1.)})
    integrity = json.loads((OUT/'integrity.json').read_text())
    assert integrity['exact_replays'] == list(PAIRS)
    d = {(r['pair'], r['level'], r['method']): r for r in rows['directions']}
    f = {(r['pair'], r['level'], r['method'], r['policy'], r['cap']): r for r in rows['finite']}
    for row in rows['directions']:
        assert row['output_norm'] == 0 or abs(row['output_norm']/row['motion_bound']-1) < 1e-10
        if row['method'] != 'adam':
            assert row['objective_derivative'] <= 1e-12
    for row in rows['finite']:
        if row['policy'] != 'raw' and row['accepted']:
            objective = 'cd' if row['policy'] == 'cd_trust' else 'objective'
            slope = d[row['pair'], row['level'], row['method']][objective+'_derivative']
            assert row[objective+'_change'] <= .1*row['fraction']*slope+1e-14
            assert row['linearization_error'] <= .1+1e-12
    max_identity = max(r['mse_identity_error'] for r in rows['states'])
    assert max_identity < 1e-12
    assert all(r['converged'] and r['relative_residual'] <= 1e-8 for r in rows['solvers'])
    assert all(r['converged'] and r['residual'] <= 1e-8 for r in rows['fields'] if r['method'] == 'uot')
    counts, gates = assess(d, f)
    attribution = transfer_attribution(rows['fields'], d, f)
    (OUT/'transfer_attribution.json').write_text(json.dumps(attribution, indent=2)+'\n')
    (OUT/'transfer_attribution.txt').write_text('\n'.join(str(r) for r in attribution)+'\n')
    summaries = []
    for group, pairs in (('4DMatch', PAIRS[:4]), ('HF', PAIRS[4:])):
        for method in METHODS:
            ds = [d[p, l, method] for p in pairs for l in range(9)]
            fs = [f[p, l, method, 'trust', 1.] for p in pairs for l in range(9)]
            field = method.rsplit('_', 1)[0]
            ps = [r for r in rows['fields'] if r['pair'] in pairs and r['method'] == field]
            summaries.append(dict(group=group, method=method, states=len(ds),
                point_descent=sum(r['point_derivative'] < -1e-12 for r in ps) if ps else None,
                linear_descent=sum(r['risk_derivative'] < -1e-12 for r in ds),
                finite_descent=sum(r['risk_change'] < -1e-12 for r in fs),
                rejected=sum(not r['accepted'] for r in fs),
                cd_down_gt_up=sum(r['cd_change'] < -1e-12 and r['risk_change'] > 1e-12 for r in fs),
                median_ray_error=float(np.median([r['transfer']['ray_error'] for r in ds if r['transfer']['ray_error'] is not None]))))
    per_pair = [dict(pair=p, method=m, mean_mse_change=float(np.mean([f[p, l, m, 'trust', 1.]['risk_change'] for l in range(9)])),
                     descents=sum(f[p, l, m, 'trust', 1.]['risk_change'] < -1e-12 for l in range(9))) for p in PAIRS for m in METHODS]
    summary = dict(numerical_gate=True, exact_replays=integrity['exact_replays'], states=54,
        new_scenes=0, rows={k: len(v) for k, v in rows.items()}, max_mse_identity_error=max_identity,
        max_ridge_residual=max(r['relative_residual'] for r in rows['solvers']),
        gates=gates, advance_to_native=[r['candidate'] for r in gates if r['pass_screen']], groups=summaries,
        rff_diagnostics={m: dict(median_exact_cosine=float(np.median([r['exact_gradient_cosine'] for r in rows['fields'] if r['method'] == m])),
                               minimum_exact_cosine=min(r['exact_gradient_cosine'] for r in rows['fields'] if r['method'] == m)) for m in ('mmd_local', 'mmd_global')})
    for name, value in (('summary.json', summary), ('counts.json', counts), ('per_pair.json', per_pair)):
        (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    lines = ['54 correlated endpoints from six reused pairs; no new native method or scene.',
             'group method point_descent linear_descent finite_descent rejected CD_down_GT_up median_ray_error']
    lines += [' '.join(str(r[k]) for k in ('group', 'method', 'point_descent', 'linear_descent', 'finite_descent', 'rejected', 'cd_down_gt_up', 'median_ray_error')) for r in summaries]
    lines += ['', 'Strict screening (does not replace earlier failed adoption gates):']+[str(r) for r in gates]
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n')
    figure(summaries, counts, rows['fields'])
    print(json.dumps({k: v for k, v in summary.items() if k != 'groups'}, indent=2))


def transfer_attribution(fields, d, f):
    lookup = {(r['pair'], r['level'], r['method']): r for r in fields}
    result = []
    for group, pairs in (('4DMatch', PAIRS[:4]), ('HF', PAIRS[4:])):
        for method in METHODS[:-1]:
            field = method.rsplit('_', 1)[0]
            lost = gained = reversed_finite = certified = 0
            for p in pairs:
                for level in range(9):
                    point, direction = lookup[p, level, field], d[p, level, method]
                    trial = f[p, level, method, 'trust', 1.]
                    lost += point['point_derivative'] < -1e-12 and direction['risk_derivative'] > 1e-12
                    gained += point['point_derivative'] > 1e-12 and direction['risk_derivative'] < -1e-12
                    reversed_finite += direction['risk_derivative'] < -1e-12 and trial['risk_change'] > 1e-12
                    ray = direction['transfer']['ray_error']
                    certificate = ray is not None and point['point_task_cosine'] > ray+1e-12
                    if certificate:
                        assert direction['risk_derivative'] < 0
                    certified += certificate
            result.append(dict(group=group, method=method, lost_point_descents=lost,
                gained_after_transfer=gained, nonlinear_reversals=reversed_finite,
                gt_conditional_certificates=certified))
    return result


def figure(summaries, counts, fields):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    x = np.arange(len(METHODS))
    for column, group in enumerate(('4DMatch', 'HF')):
        selected = [next(r for r in summaries if r['group'] == group and r['method'] == m) for m in METHODS]
        ax = axes[0, column]
        for offset, key, label in ((-.23, 'point_descent', 'Point field'), (0., 'linear_descent', 'Transferred'), (.23, 'finite_descent', 'Finite accepted')):
            ax.bar(x+offset, [r[key] or 0 for r in selected], .23, label=label)
        ax.set(title=group+': task-descent states', ylabel='Count (point field N/A for Adam)', xticks=x,
               xticklabels=[m.replace('_', '\n') for m in METHODS], ylim=(0, 39 if group == '4DMatch' else 20))
        ax.tick_params(axis='x', labelsize=7); ax.legend(fontsize=8)
        ax = axes[1, column]
        candidates = [m for m in METHODS if m not in ('cd_gd', 'cd_ridge', 'adam')]
        for offset, base in ((-.18, 'adam'), (.18, 'cd_gd')):
            values = [next(r['net'] for r in counts if r['candidate'] == m and r['group'] == group and r['baseline'] == base
                          and r['cap'] == 1 and r['base_policy'] == 'trust' and r['controller'] == 'trust' and r['tolerance'] == 1e-12) for m in candidates]
            ax.bar(np.arange(len(candidates))+offset, values, .35, label='vs '+base)
        ax.axhline(0, color='black', lw=.7)
        ax.set(title=group+': rescue - damage - lost descents', ylabel='Conservative net count',
               xticks=np.arange(len(candidates)), xticklabels=[m.replace('_', '\n') for m in candidates])
        ax.tick_params(axis='x', labelsize=8); ax.legend(fontsize=8)
    fig.suptitle('Same NDP endpoint and observed-spacing motion bound; own-objective trust, cap 1')
    fig.text(.5, .01, '54 endpoints are correlated; 6 development pairs. Double-precision clones, not full method or unseen-scene validation.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .04, 1, .96]); fig.savefig(OUT/'field_comparison.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    main()
