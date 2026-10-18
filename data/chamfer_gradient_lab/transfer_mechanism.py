"""Exact unsquared-Chamfer witness: damping versus a missing output direction."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'paper_revision/mechanism_20261018'
P, Q = np.array([0., -1.5]), np.array([-1., .5])
FORCE, TASK = np.ones(2), P-Q


def chamfer(p):
    distance = abs(p[:, None]-Q[None, :])
    return distance.min(0).mean()+distance.min(1).mean()


def witness(mu, damping):
    kernel = np.diag([1., mu])
    raw = kernel@np.linalg.solve(kernel+damping*np.eye(2), FORCE)
    motion = .05*raw/np.linalg.norm(raw)
    after = P+motion
    change = float(((after-Q)**2).mean()-((P-Q)**2).mean())
    identity = float(TASK@motion+motion@motion/2)
    assert abs(change-identity) < 1e-14
    distance = abs(after[:, None]-Q[None, :])
    assert np.array_equal(distance.argmin(0), [1, 0])
    assert np.array_equal(distance.argmin(1), [1, 0])
    assert chamfer(after) < chamfer(P)
    return dict(mu=mu, damping=damping, point_slope=float(TASK@FORCE),
                raw_slope=float(TASK@raw), scaled_slope=float(TASK@motion),
                cd_change=float(chamfer(after)-chamfer(P)), mse_change=change,
                relative_fit_error=float(np.linalg.norm(raw-FORCE)/np.linalg.norm(FORCE)))


def main():
    rows = [witness(mu, damping) for mu in (.001, 0.) for damping in (.01, .0001)]
    assert rows[0]['mse_change'] > 0 > rows[1]['mse_change']
    assert rows[2]['mse_change'] > 0 and rows[3]['mse_change'] > 0
    threshold = .001/(1-2*.001)
    assert abs(witness(.001, threshold)['raw_slope']) < 1e-14
    (OUT/'toy_results.json').write_text(json.dumps(dict(rows=rows, threshold=threshold), indent=2)+'\n')
    lines = ['Exact Chamfer witness. All four moves lower actual unsquared Chamfer.',
             'mu lambda point_slope transferred_slope finite_MSE_change CD_change']
    lines += [' '.join(str(r[k]) for k in ('mu', 'damping', 'point_slope', 'scaled_slope', 'mse_change', 'cd_change')) for r in rows]
    (OUT/'toy_results.txt').write_text('\n'.join(lines)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    lambdas = np.geomspace(1e-6, 1., 201)
    for mu, label in ((.001, 'Full range: damping can reverse descent'), (0., 'Rank deficient: missing useful component')):
        data = [witness(mu, x) for x in lambdas]
        for ax, key in zip(axes, ('scaled_slope', 'mse_change')):
            ax.semilogx(lambdas, [r[key] for r in data], label=label)
    for ax in axes:
        ax.axhline(0, color='black', lw=.7)
        for value in (.0001, .01):
            ax.axvline(value, color='gray', lw=.7, linestyle=':')
        ax.set_xlabel('Ridge lambda (same Jacobian and point force)')
        ax.legend(fontsize=7)
    axes[0].set_title('Task slope after transfer; negative is favorable')
    axes[1].set_title('Actual finite MSE change; all moves lower CD')
    fig.suptitle('Two distinct points, unique NN assignments, fixed motion norm 0.05')
    fig.tight_layout(); fig.savefig(OUT/'exact_mechanism.png', dpi=160); plt.close(fig)
    print(json.dumps(rows))


if __name__ == '__main__':
    main()
