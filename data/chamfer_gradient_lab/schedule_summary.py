"""Report the frozen optimizer control and check historical baseline equivalence."""
from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def read(path):
    row = json.loads(path.read_text())
    trace = pd.read_csv(path.with_name(path.stem+'_trace.txt'))
    assert row['calls_by_level'] == [51]*9 and len(trace) == 27
    t = (trace.level*50+trace.step).values
    e = trace.train_epe.values
    row.update(auc=float(np.sum(np.diff(t)*(e[1:]+e[:-1])/2)/450), initial_epe=float(e[0]))
    return row, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', default='paper_revision/refactor_20261013',
                        help='Experiment folder beneath this lab, containing scan/')
    args = parser.parse_args()
    out = (ROOT/args.folder).resolve()
    if not out.is_relative_to(ROOT.resolve()) or not (out/'scan').is_dir():
        raise ValueError('Provide an experiment folder with scan/ inside this lab')
    frames, traces, report = [], {}, {}
    for phase, expected in [('scan',48), ('transfer',16)]:
        if not (out/phase).exists():
            continue
        paths = sorted((out/phase).glob('*.json'))
        assert len(paths) == expected
        rows = []
        errors = []
        for path in paths:
            row, trace = read(path)
            rows.append(row)
            traces[(phase,row['pair'],row['seed'],row['method'])] = trace
            if phase == 'scan' and row['method'] == 'official':
                sub = 'ndp_scan' if row['seed'] == 0 else 'ndp_unsquared'
                old, old_trace = read(ROOT/'paper_revision/curvature_20261013'/sub/path.name)
                errors += [abs(row[k]-old[k]) for k in ['full_epe','held_epe','auc']]
                errors.append(float(np.abs(trace.values-old_trace.values).max()))
        frame = pd.DataFrame(rows)
        frame['group'] = np.where(frame.pair < (10 if phase == 'scan' else 200),
                                  'bunny' if phase == 'scan' else '4DMatch',
                                  'drill' if phase == 'scan' else 'HF controlled')
        assert not frame.duplicated(['pair','seed','method']).any()
        assert np.isfinite(frame[['full_epe','held_epe','auc','seconds']]).all().all()
        assert frame.groupby(['pair','seed']).initial_epe.agg(lambda s:s.max()-s.min()).max() < 1e-8
        means = frame.groupby(['group','method'])[['full_epe','held_epe','vis_epe','occ_epe','auc','seconds']].mean()
        changes = {metric:{group:float(means.loc[(group,'cosine'),metric]/means.loc[(group,'official'),metric]-1)
                           for group in frame.group.unique()} for metric in ['full_epe','auc']}
        paired = frame.pivot(index=['pair','seed'], columns='method', values='full_epe')
        wins = int((paired.cosine < paired.official).sum())
        ratio = float(frame[frame.method=='cosine'].seconds.mean()/frame[frame.method=='official'].seconds.mean())
        gate = all(v < 0 for metric in changes.values() for v in metric.values()) and wins > len(paired)/2 and ratio <= 1.25
        report[phase] = dict(new_fits=len(frame), paired_wins=wins, relative_change=changes,
                             seconds_ratio=ratio, gate=gate, baseline_max_error=max(errors, default=0))
        if errors:
            assert max(errors) < 1e-8, 'Refactor changed the historical baseline'
        frame.to_csv(out/f'{phase}_rows.txt', index=False)
        means.to_csv(out/f'{phase}_summary.txt')
        # Post-hoc diagnostic only: no checkpoint choice or feedback to training.
        disagreement = []
        for row in rows:
            trace = traces[(phase,row['pair'],row['seed'],row['method'])]
            delta = trace.groupby('level')[['cd_euclidean','train_epe']].diff().dropna()
            disagreement.append(dict(pair=row['pair'],seed=row['seed'],method=row['method'],
                intervals=len(delta), cd_down_epe_up=int(((delta.cd_euclidean<0)&(delta.train_epe>0)).sum())))
        pd.DataFrame(disagreement).to_csv(out/f'{phase}_disagreement.txt', index=False)
        frames.append(frame)
    report['adopted'] = all(report.get(p,{}).get('gate',False) for p in ['scan','transfer'])
    (out/'RESULTS.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    fig, axes = plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax, metric, title in zip(axes[:2], ['full_epe','auc'], ['Final correspondence EPE','Sampled trajectory EPE integral']):
        combined = pd.concat(frames)
        groups = list(combined.group.unique())
        for i, group in enumerate(groups):
            a = combined[combined.group == group].pivot(index=['pair','seed'],columns='method',values=metric)
            values = a.cosine.values/a.official.values
            ax.scatter(i+np.linspace(-.12,.12,len(values)), values, s=24)
        ax.axhline(1, color='black', ls='--')
        ax.set(xticks=range(len(groups)), xticklabels=groups, ylabel='Cosine / constant (lower is better)', title=title)
    for group, pairs, color in [('bunny',[4,5,6,7],'tab:blue'),('drill',[16,17,18,19],'tab:orange')]:
        for method, style in [('official','--'),('cosine','-')]:
            a = [traces[('scan',p,s,method)] for p in pairs for s in [0,1,2]]
            axes[2].plot(a[0].level*50+a[0].step, np.mean([b.train_epe for b in a],axis=0),
                         style, color=color, label=f'{group}, {method}')
    axes[2].set(xlabel='Optimizer updates', ylabel='Train EPE', title='Mean development trajectories')
    axes[2].legend(fontsize=8)
    fig.savefig(out/'schedule_results.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
