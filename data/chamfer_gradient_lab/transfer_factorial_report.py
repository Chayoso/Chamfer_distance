"""Validate a paired damping intervention; report every frozen contrast."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fields_report import net_counts, coverage, PAIRS

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'paper_revision/mechanism_20261018'
OLD = ROOT/'paper_revision/fields_20261018/exact'
METHODS = ('cd_gd', 'cd_ridge', 'uot_gd', 'uot_ridge', 'adam')


def read(folder, name):
    return [json.loads(x) for x in (folder/(name+'.txt')).read_text().splitlines()]


def indexed(rows, keys):
    result = {tuple(r[k] for k in keys): r for r in rows}
    assert len(result) == len(rows)
    return result


def main():
    states = {(p, l) for p in PAIRS for l in range(9)}
    old = {n: read(OLD, n) for n in ('states', 'fields', 'solvers', 'directions', 'finite')}
    new = {n: read(OUT/'weak', n) for n in old}
    assert json.loads((OUT/'weak/integrity.json').read_text())['exact_replays'] == list(PAIRS)
    coverage(new['states'], ('pair', 'level'), states)
    for name in ('fields', 'solvers'):
        coverage(new[name], ('pair', 'level', 'method'), {(*s, m) for s in states for m in ('cd', 'uot')})
    coverage(new['directions'], ('pair', 'level', 'method'), {(*s, m) for s in states for m in METHODS})
    coverage(new['finite'], ('pair', 'level', 'method', 'policy', 'cap'),
             {(*s, m, p, c) for s in states for m in METHODS for p in ('raw', 'trust', 'cd_trust') for c in (1/16, .25, 1.)})
    keys = ('pair', 'level', 'method')
    d = {a: indexed(rows['directions'], keys) for a, rows in (('standard', old), ('weak', new))}
    f = {a: indexed(rows['finite'], keys+('policy', 'cap')) for a, rows in (('standard', old), ('weak', new))}
    fields = indexed(old['fields'], keys)
    solvers = indexed(old['solvers'], keys)
    for row in new['fields']:
        assert row == fields[tuple(row[k] for k in keys)]
    for row in new['solvers']:
        previous = solvers[tuple(row[k] for k in keys)]
        assert row['converged'] and row['relative_residual'] <= 1e-8
        assert row['spectral_estimate'] == previous['spectral_estimate']
        assert abs(row['damping']/previous['damping']-.01) < 1e-15
    controls = [r for r in new['directions'] if r['method'] in ('cd_gd', 'uot_gd', 'adam')]
    for row in controls:
        assert row == d['standard'][tuple(row[k] for k in keys)]
    for row in new['finite']:
        key = tuple(row[k] for k in keys)
        direction = d['weak'][key]
        if row['method'] in ('cd_gd', 'uot_gd', 'adam'):
            assert row == f['standard'][key+(row['policy'], row['cap'])]
        if row['policy'] != 'raw' and row['accepted']:
            objective = 'cd' if row['policy'] == 'cd_trust' else 'objective'
            assert row[objective+'_change'] <= .1*row['fraction']*direction[objective+'_derivative']+1e-14
            assert row['linearization_error'] <= .1+1e-12
    max_identity = max(r['mse_identity_error'] for r in new['states'])
    numerical = []
    for row in new['solvers']:
        key = tuple(row[k] for k in keys)
        direction = d['weak'][key[:2]+(key[2]+'_ridge',)]
        previous = d['standard'][key[:2]+(key[2]+'_ridge',)]
        point, fit = fields[key], direction['transfer']
        assert fit['relative_error'] <= previous['transfer']['relative_error']+1e-7
        raw_norm = fit['cosine']*point['force_norm']/fit['ray_scale']
        task_norm = abs(point['point_derivative']/(point['force_norm']*point['point_task_cosine']))
        bound = task_norm*row['residual_norm']*direction['output_norm']/raw_norm
        numerical.append(dict(pair=key[0], level=key[1], field=key[2], slope_error_bound=bound,
            certified=abs(direction['risk_derivative']) > bound,
            raw_parameter_norm_ratio=direction['raw_parameter_norm']/previous['raw_parameter_norm']))
    summary, per_pair = [], []
    for group, pairs in (('4DMatch', PAIRS[:4]), ('HF', PAIRS[4:])):
        for arm in ('standard', 'weak'):
            for field in ('cd', 'uot'):
                ds = [d[arm][p, l, field+'_ridge'] for p in pairs for l in range(9)]
                fs = [f[arm][p, l, field+'_ridge', 'trust', 1.] for p in pairs for l in range(9)]
                ps = [fields[p, l, field] for p in pairs for l in range(9)]
                summary.append(dict(group=group, arm=arm, field=field, n=len(ds),
                    point_good=sum(r['point_derivative'] < -1e-12 for r in ps),
                    linear_good=sum(r['risk_derivative'] < -1e-12 for r in ds),
                    finite_good=sum(r['risk_change'] < -1e-12 for r in fs),
                    lost_point_good=sum(p['point_derivative'] < -1e-12 and r['risk_derivative'] > 1e-12 for p, r in zip(ps, ds)),
                    rejected=sum(not r['accepted'] for r in fs),
                    cd_down_task_up=sum(r['cd_change'] < -1e-12 and r['risk_change'] > 1e-12 for r in fs),
                    median_fit=float(np.median([r['transfer']['relative_error'] for r in ds])),
                    median_ray=float(np.median([r['transfer']['ray_error'] for r in ds]))))
                for pair in pairs:
                    trials = [f[arm][pair, l, field+'_ridge', 'trust', 1.] for l in range(9)]
                    per_pair.append(dict(pair=pair, arm=arm, field=field,
                        finite_good=sum(r['risk_change'] < -1e-12 for r in trials),
                        mean_mse_change=float(np.mean([r['risk_change'] for r in trials]))))
    contrasts = [('damping_cd', 'standard', 'cd_ridge', 'weak', 'cd_ridge'),
                 ('damping_uot', 'standard', 'uot_ridge', 'weak', 'uot_ridge'),
                 ('global_standard', 'standard', 'cd_ridge', 'standard', 'uot_ridge'),
                 ('global_weak', 'weak', 'cd_ridge', 'weak', 'uot_ridge'),
                 ('weak_uot_vs_adam', 'standard', 'adam', 'weak', 'uot_ridge'),
                 ('weak_uot_vs_cd_gd', 'standard', 'cd_gd', 'weak', 'uot_ridge')]
    counts = []
    for name, aa, am, ba, bm in contrasts:
        for group, pairs in (('4DMatch', PAIRS[:4]), ('HF', PAIRS[4:])):
            for cap in (0., 1/16, .25, 1.):
                for policy in (('raw',) if cap == 0 else ('raw', 'trust', 'cd_trust')):
                    a = [d[aa][p, l, am]['risk_derivative'] if cap == 0 else f[aa][p, l, am, policy, cap]['risk_change'] for p in pairs for l in range(9)]
                    b = [d[ba][p, l, bm]['risk_derivative'] if cap == 0 else f[ba][p, l, bm, policy, cap]['risk_change'] for p in pairs for l in range(9)]
                    for eps in (1e-12, 1e-10, 1e-8):
                        counts.append(dict(contrast=name, group=group, cap=cap, policy=policy, tolerance=eps, **net_counts(a, b, eps)))
    gates = []
    for name, *_ in contrasts:
        selected = [r for r in counts if r['contrast'] == name and r['tolerance'] == 1e-12
                    and (r['cap'] == 0 or r['cap'] in (.25, 1.) and r['policy'] == 'trust')]
        gates.append(dict(contrast=name, passed=all(r['net'] > 0 for r in selected), cells=selected))
    result = dict(exact_baseline_replays=6, exact_direction_controls=len(controls), groups=summary,
                  max_solver_residual=max(r['relative_residual'] for r in new['solvers']),
                  max_mse_identity_error=max_identity, absolute_identity_gate=max_identity < 1e-12,
                  solver_certified_slopes=sum(r['certified'] for r in numerical),
                  max_parameter_norm_ratio=max(r['raw_parameter_norm_ratio'] for r in numerical),
                  gates=gates, new_scenes=0, adoption_established=False)
    for name, value in (('summary.json', result), ('counts.json', counts), ('per_pair.json', per_pair), ('numerical.json', numerical)):
        (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    lines = ['Paired damping intervention. 54 endpoints on six reused pairs; no new scenes.',
             f'Unchanged absolute identity gate1e-12: {max_identity < 1e-12}; maximum error={max_identity}.',
             'group arm field point_good linear_good finite_good lost_point_good rejected median_fit']
    lines += [' '.join(str(r[k]) for k in ('group', 'arm', 'field', 'point_good', 'linear_good', 'finite_good', 'lost_point_good', 'rejected', 'median_fit')) for r in summary]
    lines += ['', 'Frozen primary contrasts (positive net required in every cell):']
    lines += [str(r) for r in gates]
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n')
    figure(summary)
    print(json.dumps(result, indent=2))


def figure(rows):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    labels = [('standard', 'cd'), ('weak', 'cd'), ('standard', 'uot'), ('weak', 'uot')]
    x = np.arange(4)
    for ax, group in zip(axes, ('4DMatch', 'HF')):
        chosen = [next(r for r in rows if (r['group'], r['arm'], r['field']) == (group, a, f)) for a, f in labels]
        for offset, key, label in ((-.24, 'point_good', 'Point force'), (0., 'linear_good', 'Neural tangent'), (.24, 'finite_good', 'Accepted finite step')):
            ax.bar(x+offset, [r[key] for r in chosen], .24, label=label)
        ax.set(title=group, ylabel='States with task descent', xticks=x,
               xticklabels=['CD\nstandard', 'CD\nweak', 'UOT\nstandard', 'UOT\nweak'], ylim=(0, 39 if group == '4DMatch' else 20))
        ax.legend(fontsize=7)
    fig.suptitle('Same endpoints and motion bound; only field and ridge damping differ')
    fig.text(.5, .01, 'Own-objective trust, cap 1. Correlated stages from 6 reused pairs; diagnostic, not full-trajectory validation.', ha='center', fontsize=8)
    fig.tight_layout(rect=[0, .05, 1, .94]); fig.savefig(OUT/'paired_mechanism.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    main()
