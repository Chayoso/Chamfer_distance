"""Require identical non-MMD controls and compare RFF with exact Gaussian MMD."""
import json
import fields_report as report


def main():
    previous = report.OUT
    report.OUT = previous/'exact'
    report.main()
    matched = {}
    for name in ('fields', 'solvers', 'directions', 'finite'):
        old = [json.loads(v) for v in (previous/(name+'.txt')).read_text().splitlines()]
        new = report.read(name)
        old = [r for r in old if not r['method'].startswith('mmd')]
        new = [r for r in new if not r['method'].startswith('mmd')]
        assert old == new, 'Changed non-MMD controls: '+name
        matched[name] = len(new)
    old = json.loads((previous/'summary.json').read_text())
    new = json.loads((report.OUT/'summary.json').read_text())
    new['exact_kernel_control'] = new.pop('rff_diagnostics')
    new['identical_non_mmd_records'] = matched
    (report.OUT/'summary.json').write_text(json.dumps(new, indent=2)+'\n')
    comparison = []
    for original in old['groups']:
        if not original['method'].startswith('mmd'):
            continue
        exact = next(r for r in new['groups'] if r['method'] == original['method'] and r['group'] == original['group'])
        comparison.append(dict(group=original['group'], method=original['method'],
                               rff={k: original[k] for k in ('point_descent', 'linear_descent', 'finite_descent')},
                               exact={k: exact[k] for k in ('point_descent', 'linear_descent', 'finite_descent')}))
    (report.OUT/'estimator_comparison.json').write_text(json.dumps(comparison, indent=2)+'\n')
    (report.OUT/'estimator_comparison.txt').write_text('\n'.join(str(r) for r in comparison)+'\n')
    print(json.dumps(dict(identical_controls=matched, estimator_comparison=comparison), indent=2))


if __name__ == '__main__':
    main()
