"""Identifiable rotated triangle: global matching, transfer, and CD rejection."""
import itertools
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent/'paper_revision/joint_20261018'


def geometry():
    p = np.array([[0., 0.], [1., 0.], [0., 3.]])
    p -= p.mean(0)
    rotation = np.array([[np.sqrt(3)/2, -.5], [.5, np.sqrt(3)/2]])
    return p, p@rotation.T


def loss(p, q):
    d = np.linalg.norm(p[:, None]-q[None, :], axis=2)
    return d.min(0).mean()+d.min(1).mean()


def experiment():
    p, q = geometry(); n = len(p)
    distance = np.linalg.norm(p[:, None]-q[None, :], axis=2)
    ai, bi = distance.argmin(1), distance.argmin(0)
    force = (q[ai]-p)/distance[np.arange(n), ai, None]/n
    np.add.at(force, bi, (q-p[bi])/distance[bi, np.arange(n), None]/n)
    # Enumerate every balanced assignment using observed clouds only.
    plans = list(itertools.permutations(range(n)))
    costs = [float(((p-q[list(plan)])**2).sum()) for plan in plans]
    chosen = plans[int(np.argmin(costs))]
    transport = (q[list(chosen)]-p)/n
    truth = q.copy(); task = 2*(p-truth)/n
    assert chosen == (0, 1, 2) and sorted(costs)[1]-min(costs) > 1.7
    assert np.linalg.norm(q.mean(0)-p.mean(0)) < 1e-15
    products = (transport*force).ravel(); b = products[2]; a = products.sum()-b
    assert .32 < a < .33 and -.062 < b < -.060 and a+b > 0
    rows = []
    for damping in (.01, .0001):
        eigenvalues = np.full(p.size, .001); eigenvalues[2] = 1.
        filt = eigenvalues/(eigenvalues+damping)
        raw_cd = (filt*force.ravel()).reshape(p.shape)
        raw_ot = (filt*transport.ravel()).reshape(p.shape)
        assert abs((task*raw_cd).sum()+2*(force*raw_ot).sum()) < 1e-14
        for method, raw in (('CD', raw_cd), ('OT', raw_ot)):
            motion = .01*raw/np.linalg.norm(raw)
            after = p+motion
            new_distance = np.linalg.norm(after[:, None]-q[None, :], axis=2)
            assert np.array_equal(new_distance.argmin(1), ai) and np.array_equal(new_distance.argmin(0), bi)
            change = float(((after-truth)**2).sum()/n-((p-truth)**2).sum()/n)
            assert abs(change-((task*motion).sum()+(motion*motion).sum()/n)) < 1e-14
            rows.append(dict(method=method, damping=damping, task_slope=float((task*motion).sum()),
                cd_slope=float(-(force*motion).sum()), mse_change=change,
                cd_change=float(loss(after, q)-loss(p, q)), motion=motion.tolist()))
    assert rows[0]['mse_change'] > 0 > rows[0]['cd_change']
    assert rows[1]['mse_change'] < 0 < rows[1]['cd_change']
    assert all(r['mse_change'] < 0 for r in rows[2:])
    return dict(rows=rows, assignments=plans, costs=costs, A=float(a), B=float(b),
                threshold=float(.001*(a+b)/(-.001*a-b)), centroid_motion_norm=0.)


def main():
    result = experiment(); p, q = geometry()
    (OUT/'rigid_witness.json').write_text(json.dumps(result, indent=2)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    ax.plot(*p[[0, 1, 2, 0]].T, 'o-', label='Source')
    ax.plot(*q[[0, 1, 2, 0]].T, 's--', label='Target: 30 degree rotation')
    for i in range(3):
        ax.annotate('', xy=q[i], xytext=p[i], arrowprops=dict(arrowstyle='->', color='gray'))
    ax.scatter([0], [0], marker='+', color='black'); ax.set_aspect('equal')
    ax.set_title('Unique rigid/OT correspondence; same centroid'); ax.legend(fontsize=7)
    ax = axes[1]; rows = result['rows']; x = np.arange(4)
    ax.bar(x-.18, [r['cd_change'] for r in rows], .36, label='Chamfer change')
    ax.bar(x+.18, [r['mse_change'] for r in rows], .36, label='Task MSE change')
    ax.axhline(0, color='black', lw=.7)
    ax.set(xticks=x, xticklabels=['CD\nstandard', 'OT\nstandard', 'CD\nweak', 'OT\nweak'], title='Equal motion norm: 0.01; negative is better')
    ax.legend(fontsize=8)
    fig.text(.5, .01, 'Constructed full-overlap rigid example; no partial-overlap or benchmark-performance guarantee.', ha='center', fontsize=8)
    fig.tight_layout(rect=[0, .05, 1, 1]); fig.savefig(OUT/'rigid_witness.png', dpi=160); plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
