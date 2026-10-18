"""Known rigid solver control and unchanged GD/Adam neural rejection gates."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fields_report as gates

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'paper_revision/fgr_20261018'
METHODS = ('cd_gd', 'cd_ridge', 'fgr_gd', 'fgr_ridge', 'adam')


def read(folder, name):
    return [json.loads(s) for s in (folder/(name+'.txt')).read_text().splitlines()]


def pose_report():
    data = json.loads((OUT/'evaluation.json').read_text())
    assert data['observed_sha256'] == hashlib.sha256((OUT/'observed.json').read_bytes()).hexdigest()
    rows = data['rows']
    assert [r['pair'] for r in rows] == list(gates.PAIRS)
    lines = ['Known FGR control; six reused pairs; preliminary screen is NOT adoption.',
             'pair identity_MSE centroid_MSE RANSAC_MSE FGR_MSE support mutual_precision top10_supported_coverage']
    for r in rows:
        a = r['descriptors']
        lines.append(' '.join(map(str, [r['pair'], *[r[m]['mse'] for m in ('identity','centroid','ransac','fgr')],
            a['support_fraction'], a['mutual_precision'], a['top10_supported_coverage']])))
    lines.append('Preliminary screen: '+str(data['preliminary_screen']))
    (OUT/'pose_summary.txt').write_text('\n'.join(lines)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    positions = np.arange(len(rows))
    for offset, method in enumerate(('identity','centroid','ransac','fgr')):
        axes[0].bar(positions+(offset-1.5)*.19, [r[method]['mse'] for r in rows], .19, label=method)
    axes[0].set(yscale='log', title='Standalone full-source error (lower is better)', ylabel='MSE')
    for offset, (key, label) in enumerate([('support_fraction','GT target support'),
        ('mutual_precision','Mutual precision'), ('top10_supported_coverage','Top10 coverage / support')]):
        axes[1].bar(positions+(offset-1)*.25, [r['descriptors'][key] for r in rows], .25, label=label)
    axes[1].set(title='Independent FPFH candidate audit', ylabel='Fraction', ylim=(0,1))
    for ax in axes:
        ax.set(xticks=positions, xticklabels=[str(r['pair']) for r in rows], xlabel='Pair (first four: 4D; last two: HF)')
        ax.legend(fontsize=8)
    fig.text(.5,.01,'GT used only after six poses freeze. Feature audit does not reproduce the native FGR tuple filter.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,1]); fig.savefig(OUT/'fgr_pose.png',dpi=160); plt.close(fig)


def neural_report():
    folder = OUT/'neural'
    rows = {name:read(folder,name) for name in ('states','fields','solvers','directions','finite')}
    states = {(p,l) for p in gates.PAIRS for l in range(9)}
    gates.coverage(rows['states'],('pair','level'),states)
    for name in ('fields','solvers'):
        gates.coverage(rows[name],('pair','level','method'),{(*s,m) for s in states for m in ('cd','fgr')})
    gates.coverage(rows['directions'],('pair','level','method'),{(*s,m) for s in states for m in METHODS})
    gates.coverage(rows['finite'],('pair','level','method','policy','cap'),
        {(*s,m,p,c) for s in states for m in METHODS for p in ('raw','trust','cd_trust') for c in (1/16,.25,1.)})
    assert json.loads((folder/'integrity.json').read_text())['exact_replays'] == list(gates.PAIRS)
    d = {(r['pair'],r['level'],r['method']):r for r in rows['directions']}
    f = {(r['pair'],r['level'],r['method'],r['policy'],r['cap']):r for r in rows['finite']}
    old = read(ROOT/'paper_revision/joint_20261018','directions')
    matched = [r for r in old if r['method'] in ('adam','cd_gd')]
    assert len(matched) == 108
    for r in matched:
        new = d[r['pair'],r['level'],r['method']]
        for key in ('risk_derivative','cd_derivative','motion_bound'):
            np.testing.assert_allclose(new[key],r[key],rtol=1e-10,atol=1e-12)
    assert all(r['converged'] and r['relative_residual'] <= 1e-8 for r in rows['solvers'])
    maximum = max(r['mse_identity_error'] for r in rows['states'])
    numerical = maximum < 1e-12
    for r in rows['directions']:
        assert abs(r['output_norm']/r['motion_bound']-1) < 1e-10
        if r['method'] != 'adam': assert r['objective_derivative'] <= 1e-12
    for r in rows['finite']:
        if r['policy'] != 'raw' and r['accepted']:
            objective = 'cd' if r['policy'] == 'cd_trust' else 'objective'
            slope = d[r['pair'],r['level'],r['method']][objective+'_derivative']
            assert r[objective+'_change'] <= .1*r['fraction']*slope+1e-14
            assert r['linearization_error'] <= .1+1e-12
    previous = gates.METHODS
    try:
        gates.METHODS = METHODS
        counts, screens = gates.assess(d,f)
    finally:
        gates.METHODS = previous
    for screen in screens:
        screen['direction_and_finite_conditions_passed'] = screen['pass_screen']
        screen['pass_screen'] = screen['pass_screen'] and numerical
    groups = []
    point_lookup = {(r['pair'],r['level'],r['method']):r for r in rows['fields']}
    for group, pairs in (('4DMatch',gates.PAIRS[:4]),('HF',gates.PAIRS[4:])):
        for method in METHODS:
            chosen = [f[p,l,method,'trust',1.] for p in pairs for l in range(9)]
            point = [r for r in rows['fields'] if r['pair'] in pairs and r['method']==method.rsplit('_',1)[0]]
            groups.append(dict(group=group,method=method,states=len(chosen),
                point_descents=sum(r['point_derivative'] < -1e-12 for r in point) if point else None,
                tangent_descents=sum(d[p,l,method]['risk_derivative'] < -1e-12 for p in pairs for l in range(9)),
                finite_descents=sum(r['risk_change'] < -1e-12 for r in chosen),
                cd_trust_descents=sum(f[p,l,method,'cd_trust',1.]['risk_change'] < -1e-12 for p in pairs for l in range(9)),
                trust_cd_increases=sum(r['cd_change'] > 1e-12 for r in chosen),
                helpful_proposal_to_harmful_tangent=sum(point_lookup[p,l,method.rsplit('_',1)[0]]['point_derivative'] < -1e-12
                    and d[p,l,method]['risk_derivative'] > 1e-12 for p in pairs for l in range(9)) if point else None,
                rejected=sum(not r['accepted'] for r in chosen),
                mean_mse_change=float(np.mean([r['risk_change'] for r in chosen]))))
    per_pair = [dict(pair=p,method=m,
        tangent_descents=sum(d[p,l,m]['risk_derivative'] < -1e-12 for l in range(9)),
        finite_descents=sum(f[p,l,m,'trust',1.]['risk_change'] < -1e-12 for l in range(9)),
        cd_trust_descents=sum(f[p,l,m,'cd_trust',1.]['risk_change'] < -1e-12 for l in range(9)))
        for p in gates.PAIRS for m in METHODS]
    result = dict(numerical_gate=numerical,exact_replays=6,matched_control_directions=108,
        states=54,solves=len(rows['solvers']),max_mse_identity_error=maximum,
        max_solver_residual=max(r['relative_residual'] for r in rows['solvers']),groups=groups,gates=screens,
        advance_to_native=[r['candidate'] for r in screens if r['pass_screen']],adopted=False,new_scenes=0)
    for name, data in (('summary.json',result),('counts.json',counts),('per_pair.json',per_pair)):
        (folder/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    lines = ['FGR anchors;54 correlated endpoints; no new scenes or adopted method.',
        'group method point_descents tangent_descents anchor_trust_descents CD_trust_descents rejected mean_mse_change']
    lines += [' '.join(str(r[k]) for k in ('group','method','point_descents','tangent_descents','finite_descents','cd_trust_descents','rejected','mean_mse_change')) for r in groups]
    lines += [str(r) for r in screens]
    lines += [f'Absolute MSE-identity gate: {numerical}; max error: {maximum}; threshold:1e-12.']
    (folder/'summary.txt').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))
    fig,axes = plt.subplots(1,2,figsize=(11,4.5))
    for ax,group in zip(axes,('4DMatch','HF')):
        chosen = [r for r in groups if r['group']==group]
        x = np.arange(len(chosen))
        ax.bar(x-.24,[r['tangent_descents'] for r in chosen],.24,label='Tangent task descent')
        ax.bar(x,[r['finite_descents'] for r in chosen],.24,label='Own-objective trust: task descent')
        ax.bar(x+.24,[r['cd_trust_descents'] for r in chosen],.24,label='Chamfer trust: task descent')
        ax.set(xticks=x,xticklabels=METHODS,title=group,ylabel='Endpoint count',ylim=(0,39 if group=='4DMatch' else 20))
        ax.legend(fontsize=8)
    fig.text(.5,.01,'Same motion bound, cap1. FGR own-objective trust can increase Chamfer. All policies and failed gates are retained.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(folder/'fgr_neural.png',dpi=160);plt.close(fig)


if __name__ == '__main__':
    pose_report()
    neural_report()
