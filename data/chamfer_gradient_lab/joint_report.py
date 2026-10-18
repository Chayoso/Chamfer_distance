"""Validate fixed-J alternation and retain all paired comparison outcomes."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fields_report import PAIRS, coverage, net_counts

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'paper_revision/joint_20261018'
METHODS = ('adam', 'cd_gd', 'cd1', 'cd4', 'uot1', 'uot4')


def read(folder, name):
    return [json.loads(x) for x in (folder/(name+'.txt')).read_text().splitlines()]


def index(rows, keys):
    result = {tuple(r[k] for k in keys): r for r in rows}
    assert len(rows) == len(result)
    return result


def main():
    rows = {n: read(OUT, n) for n in ('states', 'fields', 'inner', 'directions', 'finite')}
    states = {(p, l) for p in PAIRS for l in range(9)}
    keys = ('pair', 'level', 'method')
    coverage(rows['states'], keys[:2], states)
    coverage(rows['fields'], keys, {(*s, m) for s in states for m in ('cd', 'uot')})
    coverage(rows['inner'], ('pair', 'level', 'field'), {(*s, m) for s in states for m in ('cd', 'uot')})
    coverage(rows['directions'], keys, {(*s, m) for s in states for m in METHODS})
    coverage(rows['finite'], keys+('policy', 'cap'), {(*s, m, p, c) for s in states for m in METHODS for p in ('raw', 'trust', 'cd_trust') for c in (1/16, .25, 1.)})
    assert json.loads((OUT/'integrity.json').read_text())['exact_replays'] == list(PAIRS)
    d, f = index(rows['directions'], keys), index(rows['finite'], keys+('policy', 'cap'))
    old = ROOT/'paper_revision/fields_20261018/exact'
    old_d = index(read(old, 'directions'), keys)
    old_f = index(read(old, 'finite'), keys+('policy', 'cap'))
    for key, row in d.items():
        assert abs(row['output_norm']/row['motion_bound']-1) < 1e-10
        if row['method'] in ('adam', 'cd_gd'):
            assert row == old_d[key]
    for key, row in f.items():
        if row['method'] in ('adam', 'cd_gd'):
            assert {k: v for k, v in row.items() if k != 'mse_identity_error'} == old_f[key]
        if row['policy'] != 'raw' and row['accepted']:
            obj = 'cd' if row['policy'] == 'cd_trust' else 'objective'
            assert row[obj+'_change'] <= .1*row['fraction']*d[key[:3]][obj+'_derivative']+1e-14
            assert row['linearization_error'] <= .1+1e-12
    for label, source in (('standard', old), ('weak', ROOT/'paper_revision/mechanism_20261018/weak')):
        for r in read(source, 'directions'):
            if r['method'] == 'cd_ridge': d[r['pair'], r['level'], label+'_cd'] = r
        for r in read(source, 'finite'):
            if r['method'] == 'cd_ridge': f[r['pair'], r['level'], label+'_cd', r['policy'], r['cap']] = r
    violations, solve_count, residual_max = [], 0, 0.
    for row in rows['inner']:
        h = row['history']; assert len(h) == 5
        assert [r['iteration'] for r in h] == [1, 2, 3, 4, 5]
        assert len({r['solve']['damping'] for r in h[:4]}) == 1
        for r in h[:4]:
            solve = r['solve']; solve_count += 1
            assert solve['converged'] and solve['relative_residual'] <= 1e-8
            residual_max = max(residual_max, solve['relative_residual'])
            limit = 1e-10*max(1., abs(r['regularized_before']))
            for key in ('majorizer_change', 'reassignment_change'):
                if r[key] is not None and r[key] > limit:
                    violations.append(dict(pair=row['pair'], level=row['level'], field=row['field'], iteration=r['iteration'], kind=key, change=r[key], tolerance=limit))
        if h[-1]['final_reassignment_change'] > 1e-10*max(1., abs(h[-1]['final_objective'])):
            violations.append(dict(pair=row['pair'], level=row['level'], field=row['field'], kind='final_reassignment', change=h[-1]['final_reassignment_change']))
        if row['field'] == 'uot':
            assert all(r['plan']['converged'] and r['plan']['residual'] <= 1e-8 for r in h)
    assert solve_count == 432
    summary, per_pair, counts, gates = [], [], [], []
    for group, pairs in (('4DMatch', PAIRS[:4]), ('HF', PAIRS[4:])):
        for method in METHODS:
            ds = [d[p, l, method] for p in pairs for l in range(9)]
            fs = [f[p, l, method, 'trust', 1.] for p in pairs for l in range(9)]
            summary.append(dict(group=group, method=method, states=len(ds),
                tangent_good=sum(r['risk_derivative'] < -1e-12 for r in ds),
                finite_good=sum(r['risk_change'] < -1e-12 for r in fs), rejected=sum(not r['accepted'] for r in fs),
                mean_mse_change=float(np.mean([r['risk_change'] for r in fs])),
                cd_down_task_up=sum(r['cd_change'] < -1e-12 and r['risk_change'] > 1e-12 for r in fs)))
            for p in pairs:
                chosen = [f[p, l, method, 'trust', 1.] for l in range(9)]
                per_pair.append(dict(pair=p, method=method, finite_good=sum(r['risk_change'] < -1e-12 for r in chosen), mean_mse_change=float(np.mean([r['risk_change'] for r in chosen]))))
        for candidate in ('cd4', 'uot4'):
            bases = ['cd1', 'cd_gd', 'adam', 'standard_cd', 'weak_cd']
            if candidate == 'uot4': bases += ['uot1', 'cd4']
            for baseline in bases:
                for cap in (0., 1/16, .25, 1.):
                    for policy in (('raw',) if cap == 0 else ('raw', 'trust')):
                        for controller in (('trust',) if cap == 0 else ('trust', 'cd_trust')):
                            a = [d[p, l, baseline]['risk_derivative'] if cap == 0 else f[p, l, baseline, policy, cap]['risk_change'] for p in pairs for l in range(9)]
                            b = [d[p, l, candidate]['risk_derivative'] if cap == 0 else f[p, l, candidate, controller, cap]['risk_change'] for p in pairs for l in range(9)]
                            for eps in (1e-12, 1e-10, 1e-8):
                                counts.append(dict(group=group, candidate=candidate, baseline=baseline, cap=cap, base_policy=policy, controller=controller, tolerance=eps, **net_counts(a, b, eps)))
    identity_failures = [r for r in rows['finite'] if r['mse_identity_error'] >= 1e-12]
    numerical = not violations and not identity_failures
    for candidate in ('cd4', 'uot4'):
        selected = [r for r in counts if r['candidate'] == candidate and r['controller'] == 'trust' and r['cap'] in (0., .25, 1.) and r['tolerance'] == 1e-12]
        gates.append(dict(candidate=candidate, passed=numerical and all(r['net'] > 0 for r in selected),
            positive_cells=sum(r['net'] > 0 for r in selected), required_cells=len(selected), worst_net=min(r['net'] for r in selected)))
    result = dict(exact_replays=6, exact_baseline_direction_controls=108, groups=summary, solves=solve_count,
        max_solver_residual=residual_max, numerical_gate=numerical, mm_violations=violations,
        max_mse_identity_error=max(r['mse_identity_error'] for r in rows['finite']),
        identity_failures=len(identity_failures), accepted_identity_failures=sum(r['policy'] != 'raw' and r['accepted'] for r in identity_failures),
        gates=gates, new_scenes=0, adopted=False)
    for name, data in (('summary.json', result), ('counts.json', counts), ('per_pair.json', per_pair), ('identity_failures.json', identity_failures)):
        (OUT/name).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    lines = ['Fixed-J joint correspondence diagnostic; 54 correlated endpoints, six reused pairs.',
             'group method tangent_good finite_good rejected mean_mse_change']
    lines += [' '.join(str(r[k]) for k in ('group', 'method', 'tangent_good', 'finite_good', 'rejected', 'mean_mse_change')) for r in summary]
    lines += ['', 'Numerics: '+str({k: v for k, v in result.items() if k not in ('groups', 'gates')}), 'Frozen gates:']+[str(g) for g in gates]
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n'); figure(summary)
    print('\n'.join(lines))


def figure(rows):
    methods = ('cd1', 'cd4', 'uot1', 'uot4'); x = np.arange(4)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, group in zip(axes, ('4DMatch', 'HF')):
        selected = [next(r for r in rows if r['group'] == group and r['method'] == m) for m in methods]
        ax.bar(x-.18, [r['tangent_good'] for r in selected], .36, label='Tangent task descent')
        ax.bar(x+.18, [r['finite_good'] for r in selected], .36, label='Finite task descent')
        ax.set(title=group, xticks=x, xticklabels=['CD\n1 solve', 'CD\n4 solves', 'UOT\n1 solve', 'UOT\n4 solves'], ylabel='Endpoint count', ylim=(0, 39 if group == '4DMatch' else 20))
        ax.legend(fontsize=8)
    fig.suptitle('Refresh correspondences inside the same neural tangent model')
    fig.text(.5, .01, 'Own-objective trust, cap1; not independent scenes or complete registration trajectories.', ha='center', fontsize=8)
    fig.tight_layout(rect=[0, .05, 1, .94]); fig.savefig(OUT/'joint_comparison.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    main()
