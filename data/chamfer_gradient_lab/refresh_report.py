"""Separate algebra/reproducibility checks from actual refresh effectiveness."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fields_report import PAIRS, coverage, net_counts

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'paper_revision/refresh_audit_20261018'


def read(folder, name):
    return [json.loads(s) for s in (folder/(name+'.txt')).read_text().splitlines()]


def old_geometry():
    folder = ROOT/'paper_revision/joint_20261018'
    directions = {(r['pair'], r['level'], r['method']): r for r in read(folder, 'directions')}
    finite = {(r['pair'], r['level'], r['method'], r['policy']): r for r in read(folder, 'finite') if r['cap'] == 1}
    result = []
    for group, pairs in (('4DMatch', PAIRS[:4]), ('HF', PAIRS[4:])):
        for kind in ('cd', 'uot'):
            rows = [r for r in read(folder, 'inner') if r['pair'] in pairs and r['field'] == kind]
            virtual, real, fractions, ratios, errors = [], [], [], [], []
            for r in rows:
                key = (r['pair'], r['level'], kind+'4'); bound = directions[key]['motion_bound']
                f, raw = finite[key+('trust',)], finite[key+('raw',)]
                # Fourth solve matches at the THIRD virtual displacement.
                distance = r['history'][2]['output_norm']
                virtual.append(distance/bound); real.append(f['output_norm']/bound)
                fractions.append(f['fraction']); errors.append(raw['linearization_error'])
                if f['accepted']: ratios.append(distance/f['output_norm'])
            result.append(dict(group=group, field=kind, states=len(rows), accepted=len(ratios),
                virtual_rms_h_median=float(np.median(virtual)), accepted_rms_h_median=float(np.median(real)),
                virtual_to_actual_median=float(np.median(ratios)),
                accepted_fraction_median=float(np.median(fractions)),
                raw_cap1_linearization_median=float(np.median(errors))))
    return result


def main():
    folder = OUT/'verified'
    steps, controls, jacobians = [read(folder, name) for name in ('steps', 'controls', 'jacobians')]
    states = {(p, l) for p in PAIRS for l in range(9)}
    coverage(steps, ('pair', 'level', 'field', 'refresh', 'iteration'),
             {(*s, k, flag, i) for s in states for k in ('cd', 'uot') for flag in (False, True) for i in range(1, 5)})
    coverage(controls, ('pair', 'level', 'field', 'refresh'),
             {(*s, k, flag) for s in states for k in ('cd', 'uot') for flag in (False, True)})
    coverage(jacobians, ('pair', 'level'), states)
    assert json.loads((folder/'integrity.json').read_text())['exact_replays'] == list(PAIRS)
    for r in steps:
        assert r['solve']['converged'] and r['solve']['relative_residual'] <= 1e-8
        if r['field'] == 'uot': assert r['plan']['converged'] and r['plan']['residual'] <= 1e-8
        if r['fraction']:
            assert r['objective_change'] <= .1*r['fraction']*r['slope']+1e-14
            assert r['linearization_error'] <= .1+1e-12
        else: assert r['mse_change'] == r['objective_change'] == 0
    for s in states:
        for kind in ('cd', 'uot'):
            first = [r for r in steps if (r['pair'], r['level']) == s and r['field'] == kind and r['iteration'] == 1]
            assert {k: v for k, v in first[0].items() if k != 'refresh'} == {k: v for k, v in first[1].items() if k != 'refresh'}
            for flag in (False, True):
                path = [r for r in steps if (r['pair'], r['level'], r['field'], r['refresh']) == (*s, kind, flag)]
                assert len({r['solve']['damping'] for r in path}) == 1
                assert abs(sum(r['mse_change'] for r in path)-path[-1]['total_mse_change']) < 1e-12
    summaries, contrasts = [], []
    for group, pairs in (('4DMatch', PAIRS[:4]), ('HF', PAIRS[4:])):
        for kind in ('cd', 'uot'):
            for flag in (False, True):
                for iteration in (1, 4):
                    rows = [r for r in steps if r['pair'] in pairs and r['field'] == kind and r['refresh'] == flag and r['iteration'] == iteration]
                    summaries.append(dict(group=group, field=kind, refresh=flag, iteration=iteration,
                        states=len(rows), task_descents=sum(r['total_mse_change'] < -1e-12 for r in rows),
                        mean_mse_change=float(np.mean([r['total_mse_change'] for r in rows])),
                        cd_down=sum(r['total_cd_change'] < -1e-12 for r in rows),
                        objective_down=sum(r['total_objective_change'] < -1e-12 for r in rows),
                        rejected=sum(r['fraction'] == 0 for r in rows),
                        median_force_refresh=float(np.median([r['force_refresh_relative'] for r in rows]))))
            arms = [[r['total_mse_change'] for r in steps if r['pair'] in pairs and r['field'] == kind and r['refresh'] == flag and r['iteration'] == 4] for flag in (False, True)]
            a, b = arms
            contrasts.append(dict(group=group, field=kind, **net_counts(a, b, 1e-12),
                mean_mse_difference=float(np.mean(b)-np.mean(a)),
                diagnostic_support=bool(sum(v < -1e-12 for v in b) > sum(v < -1e-12 for v in a) and np.mean(b) < np.mean(a))))
    historical_cd = [r for r in controls if r['field'] == 'cd' and not r['refresh']]
    historical_uot = [r for r in controls if r['field'] == 'uot' and not r['refresh']]
    assert all(r['direction_exact'] and r['fraction_exact'] for r in historical_uot)
    old_d = {(r['pair'],r['level'],r['method']):r for r in read(ROOT/'paper_revision/joint_20261018','directions')}
    old_f = {(r['pair'],r['level'],r['method']):r for r in read(ROOT/'paper_revision/joint_20261018','finite') if r['policy']=='trust' and r['cap']==1}
    first_uot = [r for r in steps if r['field']=='uot' and not r['refresh'] and r['iteration']==1]
    controller = [dict(group=g,
        changed_fractions=sum(r['fraction'] != r['old_first_fraction'] for r in first_uot if r['pair'] in ps),
        mean_mse_difference=float(np.mean([r['total_mse_change']-old_f[r['pair'],r['level'],'uot1']['risk_change'] for r in first_uot if r['pair'] in ps])))
        for g,ps in (('4DMatch',PAIRS[:4]),('HF',PAIRS[4:]))]
    identity_failures = [r for r in steps if r['mse_identity_error'] >= 1e-12]
    result = dict(groups=summaries, contrasts=contrasts, mechanism=trajectory_mechanism(steps), old_geometry=old_geometry(), uot_controller_effect=controller, exact_baseline_replays=6,
        step_rows=len(steps), jacobians=len(jacobians),
        max_adjoint_error=max(r['adjoint_relative'] for r in jacobians),
        max_best_finite_difference=max(min(r['finite_difference_relative']) for r in jacobians),
        max_solver_residual=max(r['solve']['relative_residual'] for r in steps),
        max_mse_identity_error=max(r['mse_identity_error'] for r in steps), identity_failures=len(identity_failures),
        historical_cd_inexact_directions=sum(not r['direction_exact'] for r in historical_cd),
        historical_cd_inexact_fractions=sum(not r['fraction_exact'] for r in historical_cd),
        historical_cd_max_slope_difference=max(abs(r['slope_difference']) for r in historical_cd),
        historical_cd_max_controller_mse_difference=max(abs(r['old_controller_mse_difference']) for r in historical_cd),
        historical_cd_slope_sign_changes=int(sum(np.sign(r['slope_difference']+old_d[r['pair'],r['level'],'cd1']['risk_derivative']) != np.sign(old_d[r['pair'],r['level'],'cd1']['risk_derivative']) for r in historical_cd)),
        historical_uot_exact_controls=len(historical_uot), new_scenes=0, adopted=False)
    per_pair = [dict(pair=p, field=k, refresh=flag,
        task_descents=sum(r['total_mse_change'] < -1e-12 for r in steps if (r['pair'],r['field'],r['refresh'],r['iteration']) == (p,k,flag,4)),
        mean_mse_change=float(np.mean([r['total_mse_change'] for r in steps if (r['pair'],r['field'],r['refresh'],r['iteration']) == (p,k,flag,4)])))
        for p in PAIRS for k in ('cd','uot') for flag in (False,True)]
    for name, value in (('summary.json', result), ('per_pair.json', per_pair)):
        (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    lines = ['Actual accepted-state refresh audit;54 correlated states/six reused pairs.',
             'group field refresh iteration task_descents mean_mse_change objective_down rejected']
    lines += [' '.join(str(r[k]) for k in ('group','field','refresh','iteration','task_descents','mean_mse_change','objective_down','rejected')) for r in summaries]
    lines += ['Matched four-step contrasts:']+[str(r) for r in contrasts]+['Old virtual/actual geometry:']+[str(r) for r in result['old_geometry']]
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines)); figure(result)


def figure(result):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    old = result['old_geometry']; x = np.arange(4)
    axes[0].bar(x-.18, [r['virtual_rms_h_median'] for r in old], .36, label='Last virtual match')
    axes[0].bar(x+.18, [r['accepted_rms_h_median'] for r in old], .36, label='Actual accepted move')
    axes[0].set(yscale='log', xticks=x, xticklabels=['4D CD','4D UOT','HF CD','HF UOT'], title='Old fixed-J positions differ', ylabel='RMS movement / observed spacing')
    axes[0].tick_params(axis='x', labelsize=8); axes[0].legend(fontsize=7)
    for ax, group in zip(axes[1:], ('4DMatch', 'HF')):
        for flag, offset, label in ((False,-.18,'Frozen matching'),(True,.18,'Refreshed matching')):
            chosen = [next(r for r in result['groups'] if r['group']==group and r['field']==k and r['refresh']==flag and r['iteration']==4) for k in ('cd','uot')]
            ax.bar(np.arange(2)+offset,[r['task_descents'] for r in chosen],.36,label=label)
        ax.set(title=group+': four actual updates',xticks=[0,1],xticklabels=['CD','UOT'],ylabel='Task-descending endpoints',ylim=(0,39 if group=='4DMatch' else 20))
        ax.legend(fontsize=7)
    fig.text(.5,.01,'Both new arms rebuild J and share the same true observed objective. Counts are not independent scenes or final registration accuracy.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.06,1,1]);fig.savefig(OUT/'refresh_audit.png',dpi=160);plt.close(fig)


def trajectory_mechanism(steps):
    """Post-hoc direction diagnosis; no changes to trajectories or selection."""
    counts = []
    for group, pairs in (('4DMatch', PAIRS[:4]), ('HF', PAIRS[4:])):
        for kind in ('cd', 'uot'):
            for flag in (False, True):
                rows = [r for r in steps if r['pair'] in pairs and r['field'] == kind and r['refresh'] == flag]
                accepted = [r for r in rows if r['fraction'] > 0]
                bad = [r for r in accepted if r['mse_change'] > 1e-12]
                counts.append(dict(group=group, field=kind, refresh=flag,
                    accepted_steps=len(accepted), task_increasing_steps=len(bad),
                    task_increasing_with_positive_slope=sum(r['task_slope'] > 1e-12 for r in bad),
                    objective_down_steps=sum(r['objective_change'] < -1e-12 for r in accepted),
                    final_objective_down_task_up=sum(r['iteration'] == 4 and r['total_objective_change'] < -1e-12 and r['total_mse_change'] > 1e-12 for r in rows)))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, title in zip(axes, ('total_objective_change', 'total_mse_change'),
                              ('Observed UOT objective', 'GT correspondence MSE')):
        for flag, color, label in ((False, 'tab:blue', 'Frozen matching'), (True, 'tab:orange', 'Refreshed matching')):
            paths = []
            for pair in PAIRS[4:]:
                for level in range(9):
                    rows = sorted([r for r in steps if (r['pair'], r['level'], r['field'], r['refresh']) == (pair, level, 'uot', flag)], key=lambda r:r['iteration'])
                    path = [0]+[r[key] for r in rows]; paths.append(path)
                    ax.plot(range(5), path, color=color, alpha=.15, linewidth=.7)
            ax.plot(range(5), np.mean(paths, axis=0), 'o-', color=color, label=label, linewidth=2)
        ax.axhline(0, color='black', linewidth=.6)
        ax.set(title=title, xlabel='Actual update', ylabel='Change from initial endpoint', xticks=range(5))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0)); ax.legend(fontsize=8)
    fig.suptitle('HF: loss decreases while correspondence error usually increases')
    fig.text(.5,.01,'Thin: 18 correlated endpoints from 2 reused pairs. Thick: mean. GT is evaluation only; lower is better on both panels.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.05,1,.95]); fig.savefig(OUT/'refresh_trajectory.png',dpi=160); plt.close(fig)
    return counts


if __name__ == '__main__':
    main()
