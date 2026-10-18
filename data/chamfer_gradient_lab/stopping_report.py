"""Summarize baseline convergence without treating correlated states as scenes."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from pipeline.model import ROOT

OUT = ROOT/'paper_revision/stopping_20261018'
PAIRS = (100, 101, 124, 125, 200, 201)


def read(name):
    return [json.loads(line) for line in (OUT/(name+'.txt')).read_text().splitlines()]


def summarize_path(rows):
    rows = sorted(rows, key=lambda r: r['step'])
    assert [r['step'] for r in rows] == list(range(101))
    first, last = rows[0], rows[-1]
    best = min(rows, key=lambda r: r['cd'])  # CD only; never select by GT.
    return dict(final_cd_gain=(first['cd']-last['cd'])/first['cd'],
                best_cd_gain=(first['cd']-best['cd'])/first['cd'], best_step=best['step'],
                final_mse_change=last['mse']-first['mse'],
                final_epe_change=last['epe']-first['epe'],
                best_cd_mse_change=best['mse']-first['mse'],
                initial_grad=first['grad_norm'], final_grad=last['grad_norm'])


def main():
    stops, paths, traces = read('stops'), read('continuation'), read('trajectory')
    integrity = json.loads((OUT/'integrity.json').read_text())
    assert integrity['exact_original_replays'] == list(PAIRS)
    assert len(stops) == 108 and len(paths) == 21816
    assert len({(r['pair'], r['mode'], r['level']) for r in stops}) == 108
    assert sum(r['mode'] == 'ceiling500' for r in traces) == 27000
    for stop in stops:
        observed = [r for r in traces if all(r[k] == stop[k] for k in ('pair', 'mode', 'level'))]
        assert [r['step'] for r in observed] == list(range(stop['evaluations']))
        assert observed[-1]['cd'] == stop['cached_cd']
        if stop['mode'] == 'original':
            previous, count, reason = 1e6, 0, 'iteration_cap'
            for row in observed:
                if row['cd'] < 1e-4:
                    reason = 'absolute_loss'
                else:
                    count += abs(previous-row['cd']) < .001*previous
                    if count >= 15:
                        reason = 'cumulative_small_changes'
                if reason != 'iteration_cap':
                    assert row['step'] == observed[-1]['step']
                previous = row['cd']
            assert reason == stop['reason'] and count == stop['small_change_count']
        assert stop['updates'] == stop['evaluations']-(stop['reason'] != 'iteration_cap')
    grouped = {}
    for row in paths:
        key = tuple(row[k] for k in ('pair', 'mode', 'level', 'lr_scale'))
        grouped.setdefault(key, []).append(row)
    assert len(grouped) == 216
    outcomes = [dict(zip(('pair', 'mode', 'level', 'lr_scale'), key), **summarize_path(rows))
                for key, rows in grouped.items()]
    with (OUT/'continuation_summary.txt').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(outcomes[0]))
        writer.writeheader(); writer.writerows(outcomes)
    summary = dict(pairs=6, independent_new_scenes=0, original_replays_exact=6,
                   total_full_fits=24, initial_protocol_fits=18, handoff_control_fits=6,
                   observed_level_endpoints=108,
                   continuation_paths=216, continuation_updates=21600, groups=[], endpoints=[])
    lines = ['BASELINE STOPPING AUDIT: development data, not new independent scenes.',
             'Instrumented times include diagnostics; no runtime superiority claim.']
    for mode in ('original', 'ceiling500'):
        for family, ids in [('4DMatch', PAIRS[:4]), ('HF', PAIRS[4:])]:
            selected = [r for r in stops if r['mode'] == mode and r['pair'] in ids]
            group = dict(mode=mode, family=family, pairs=len(ids), levels=len(selected),
                stop_reasons={reason:sum(r['reason'] == reason for r in selected)
                              for reason in sorted({r['reason'] for r in selected})},
                updates=sum(r['updates'] for r in selected),
                gradient_norm_range=[min(r['grad_norm'] for r in selected), max(r['grad_norm'] for r in selected)],
                negative_risk_cosines=sum(r['risk_cosine'] < 0 for r in selected),
                max_cached_output_lag=max(r['cached_output_lag'] for r in selected), arms=[])
            for scale in (1., .1):
                values = [r for r in outcomes if r['mode'] == mode and r['pair'] in ids and r['lr_scale'] == scale]
                arm = dict(lr_scale=scale, paths=len(values),
                    median_best_cd_gain=float(np.median([r['best_cd_gain'] for r in values])),
                    median_final_cd_gain=float(np.median([r['final_cd_gain'] for r in values])),
                    plateau_counts={str(eps):sum(r['best_cd_gain'] <= eps for r in values)
                                    for eps in (1e-6, 1e-4, 1e-3)},
                    final_cd_down_gt_up=sum(r['final_cd_gain'] > 1e-6 and r['final_mse_change'] > 1e-8 for r in values),
                    best_cd_down_gt_up=sum(r['best_cd_gain'] > 1e-6 and r['best_cd_mse_change'] > 1e-8 for r in values))
                group['arms'].append(arm)
            group['joint_plateau_counts'] = {str(eps):sum(all(summarize_path(grouped[(p, mode, level, scale)])['best_cd_gain'] <= eps
                for scale in (1., .1)) for p in ids for level in range(9)) for eps in (1e-6, 1e-4, 1e-3)}
            summary['groups'].append(group); lines.append(json.dumps(group))
    for pair in PAIRS:
        name = f'p{pair}_s0_official_w0.json'
        reference = ROOT/'paper_revision'/('replication_4d' if pair < 200 else 'replication_hf')
        row = dict(pair=pair)
        for mode, folder in [('fixed50', reference), ('original', OUT/'original'),
                             ('ceiling500', OUT/'ceiling500'), ('fresh500', OUT/'fresh500')]:
            result = json.loads((folder/name).read_text())
            if mode == 'fresh500':
                assert result['calls_by_level'] == [501]*9
            row[mode] = {key:result[key] for key in ('full_epe', 'held_epe', 'vis_epe', 'occ_epe', 'full_accs')}
        cds = []
        for folder, step in [(reference, 50), (OUT/'fresh500', 500)]:
            trace = np.loadtxt(folder/(name[:-5]+'_trace.txt'), delimiter=',', skiprows=1)
            assert trace[-1, 0] == 8 and trace[-1, 1] == step
            cds.append(float(trace[-1, 2]))
        row['budget_change'] = dict(sampled_cd_50=cds[0], sampled_cd_500=cds[1],
            relative_cd_change=cds[1]/cds[0]-1,
            relative_full_epe_change=row['fresh500']['full_epe']/row['fixed50']['full_epe']-1)
        summary['endpoints'].append(row); lines.append(json.dumps(row))
    summary['fresh500_cd_down_full_epe_up_pairs'] = sum(r['budget_change']['relative_cd_change'] < 0
        and r['budget_change']['relative_full_epe_change'] > 0 for r in summary['endpoints'])
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n')
    plot(summary, grouped, traces)
    required = ['ndp_stopping_probe.py', 'ndp_stopping_handoff.py', 'pipeline/stopping.py', 'stopping_report.py']
    files = {str((OUT/(n+'.txt')).relative_to(ROOT)) for n in ('stops', 'continuation', 'trajectory')}
    files.update(required)
    (OUT/'report_hashes.json').write_text(json.dumps({f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
                                                   for f in sorted(files)}, indent=2)+'\n')
    print(json.dumps(summary))


def plot(summary, grouped, traces):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
    colors = {'original':'#0072B2', 'ceiling500':'#D55E00'}
    for ax, pair in zip(axes.flat, PAIRS):
        for mode in colors:
            for scale, style in [(1., '-'), (.1, '--')]:
                rows = grouped[(pair, mode, 8, scale)]
                ax.plot([r['cd']/rows[0]['cd']-1 for r in rows],
                        [r['mse']/rows[0]['mse']-1 for r in rows], style,
                        color=colors[mode], label=f'{mode}, lr x{scale:g}', linewidth=1)
                ax.scatter(rows[-1]['cd']/rows[0]['cd']-1, rows[-1]['mse']/rows[0]['mse']-1,
                           color=colors[mode], marker='x', s=25)
        ax.axhline(0, color='grey', lw=.6); ax.axvline(0, color='grey', lw=.6)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_title(f'{"4DMatch" if pair < 200 else "HF"} pair {pair}, final level')
        ax.set_xlabel('Relative CD change'); ax.set_ylabel('Relative GT MSE change')
    axes.flat[0].legend(fontsize=7)
    fig.suptitle('100 Adam updates: upper-left = CD improves, GT worsens; x marks final state')
    fig.savefig(OUT/'continuation_trajectories.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    x = np.arange(6)
    for i, (key, label) in enumerate([('relative_cd_change', 'Sampled training CD'),
                                    ('relative_full_epe_change', 'Full-cloud GT EPE')]):
        axes[0].bar(x+(i-.5)*.35, [100*r['budget_change'][key] for r in summary['endpoints']], .35, label=label)
    axes[0].axhline(0, color='grey', lw=.8)
    axes[0].set_xticks(x, [str(p) for p in PAIRS]); axes[0].set_xlabel('Development pair')
    axes[0].set_ylabel('Change from 50 to 500 updates (%)'); axes[0].legend(fontsize=8)
    for mode in colors:
        gain = [max(summarize_path(grouped[(p, mode, level, s)])['best_cd_gain'] for s in (1., .1))
                for p in PAIRS for level in range(9)]
        axes[1].step(np.sort(gain), np.arange(1, 55)/54, where='post', label=mode, color=colors[mode])
    axes[1].set_xscale('symlog', linthresh=1e-6); axes[1].axvline(.001, color='grey', ls=':')
    axes[1].set_xlabel('Best relative CD gain across both fixed lr arms')
    axes[1].set_ylabel('Fraction of 54 correlated level endpoints'); axes[1].legend()
    fig.suptitle('Baseline budget audit; six reused development pairs')
    fig.savefig(OUT/'stopping_results.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    for ax, pair in zip(axes.flat, PAIRS):
        for mode in colors:
            rows = [r for r in traces if r['pair'] == pair and r['mode'] == mode
                    and r['level'] == 8 and 'risk_cosine' in r]
            ax.plot([r['step'] for r in rows], [r['risk_cosine'] for r in rows],
                    '.-', color=colors[mode], label=mode, linewidth=1)
        ax.axhline(0, color='grey', lw=.8); ax.axhspan(-1, 0, color='#D55E00', alpha=.07)
        ax.set_ylim(-1, 1); ax.set_title(f'Pair {pair}, final level')
        ax.set_xlabel('Adam updates'); ax.set_ylabel('Cosine of CD and GT gradients')
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Negative cosine: parameter steepest-CD descent is locally task-adverse')
    fig.savefig(OUT/'gradient_alignment.png', dpi=170); plt.close(fig)


if __name__ == '__main__':
    main()
