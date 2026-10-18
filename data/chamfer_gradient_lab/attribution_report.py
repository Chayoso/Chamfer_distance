"""Exact-replay attribution: proposal quality versus neural transfer."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fields_report import PAIRS, coverage

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'paper_revision/attribution_20261018'
TOL = 1e-10


def category(a, b):
    if min(abs(a), abs(b)) <= TOL: return 'near_zero'
    return ('helpful' if a < 0 else 'harmful')+'_'+('helpful' if b < 0 else 'harmful')


def summarize(rows):
    transitions = {k: 0 for k in ('helpful_helpful','helpful_harmful','harmful_helpful','harmful_harmful','near_zero')}
    force_to_proposal = dict(transitions)
    for r in rows:
        a = r['attribution']
        transitions[category(a['proposal_cosine'],a['projected_cosine'])] += 1
        force_to_proposal[category(a['force_cosine'],a['proposal_cosine'])] += 1
    median = lambda key: float(np.median([r['attribution'][key] for r in rows])) if rows else None
    return dict(rows=len(rows), proposal_to_projection=transitions, force_to_proposal=force_to_proposal,
        median_raw_transfer_error=median('raw_transfer_error'), median_weighted_transfer_error=median('weighted_transfer_error'),
        median_direction_gap=median('direction_gap'), certificate_count=sum(r['attribution']['directional_descent_certificate'] for r in rows))


def main():
    rows = [json.loads(s) for s in (OUT/'attribution.txt').read_text().splitlines()]
    old = [json.loads(s) for s in (ROOT/'paper_revision/refresh_audit_20261018/verified/steps.txt').read_text().splitlines()]
    keys = ('pair','level','field','refresh','iteration')
    coverage(rows, keys, {(p,l,k,f,i) for p in PAIRS for l in range(9) for k in ('cd','uot') for f in (False,True) for i in range(1,5)})
    expected = {tuple(r[k] for k in keys): r for r in old}
    assert json.loads((OUT/'integrity.json').read_text())['exact_replays'] == list(PAIRS)
    for r in rows:
        assert {k:v for k,v in r.items() if k != 'attribution'} == expected[tuple(r[k] for k in keys)]
        a = r['attribution']
        scale = max(1.,abs(a['projected_slope']),abs(a['proposal_slope']),abs(a['correction_slope']))
        assert a['slope_identity_error'] <= 1e-12*scale
        assert abs(a['correction_slope']-a['supported_correction_slope']-a['unsupported_motion_slope']) <= 1e-12*scale
        if abs(a['projected_cosine']) > TOL: assert a['projected_slope']*r['task_slope'] > 0
        if a['directional_descent_certificate']: assert a['projected_cosine'] < 0
    groups, first, failures = [], [], []
    for group, pairs in (('4DMatch', PAIRS[:4]), ('HF', PAIRS[4:])):
        for kind in ('cd','uot'):
            for flag in (False,True):
                chosen = [r for r in rows if r['pair'] in pairs and r['field']==kind and r['refresh']==flag]
                tag = dict(group=group,field=kind,refresh=flag)
                groups.append(dict(**tag,**summarize(chosen)))
                first.append(dict(**tag,**summarize([r for r in chosen if r['iteration']==1])))
                failures.append(dict(**tag,**summarize([r for r in chosen if r['fraction'] and r['mse_change']>1e-12])))
    per_pair = [dict(pair=p,field=k,**summarize([r for r in rows if r['pair']==p and r['field']==k and r['refresh'] and r['fraction'] and r['mse_change']>1e-12])) for p in PAIRS for k in ('cd','uot')]
    bad = [r for r in rows if r['refresh'] and r['fraction'] and r['mse_change']>1e-12]
    result = dict(exact_step_replays=len(rows),exact_full_baseline_replays=6,correlated_states=54,
        all_rows=groups,first_steps=first,task_increasing_accepted_steps=failures,refresh_failures=summarize(bad),per_pair_failures=per_pair,
        max_slope_identity_error=max(r['attribution']['slope_identity_error'] for r in rows),
        zero_mass_rows=sum(r['attribution']['zero_mass_rows'] for r in rows),
        min_positive_mass=min(r['attribution']['positive_mass_min'] for r in rows),new_scenes=0,adopted=False)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines = ['Proposal/transfer attribution; exact replay, not independent new observations.',
             'REFRESH accepted task-increasing steps: '+str(result['refresh_failures']),
             'group field refresh rows proposal_to_projection force_to_proposal']
    lines += [' '.join(str(r[k]) for k in ('group','field','refresh','rows','proposal_to_projection','force_to_proposal')) for r in failures]
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))
    figure(rows,failures)


def figure(rows, failures):
    fig,axes = plt.subplots(1,3,figsize=(14,4.4))
    selected = [r for r in rows if r['refresh'] and r['fraction']]
    for ax,kind in zip(axes[:2],('cd','uot')):
        for group,pairs,color in (('4DMatch',PAIRS[:4],'tab:blue'),('HF',PAIRS[4:],'tab:orange')):
            points = [r for r in selected if r['field']==kind and r['pair'] in pairs]
            ax.scatter([r['attribution']['proposal_cosine'] for r in points],
                       [r['attribution']['projected_cosine'] for r in points],s=14,alpha=.5,color=color,label=group)
        ax.axhline(0,color='black',lw=.7); ax.axvline(0,color='black',lw=.7)
        ax.set(title=kind.upper()+': accepted REFRESH steps',xlabel='Task cosine of point proposal d=f/m',ylabel='Task cosine after neural transfer u')
        ax.legend(fontsize=8)
    bad = [r for r in failures if r['refresh']]
    bottom = np.zeros(4)
    for key,color,label in (('harmful_harmful','tab:orange','Proposal already harmful'),('helpful_harmful','tab:red','Sign reversed in transfer'),('near_zero','gray','Near-zero direction')):
        counts = np.array([r['proposal_to_projection'][key] for r in bad])
        axes[2].bar(range(4),counts,bottom=bottom,color=color,label=label)
        for i,n in enumerate(counts):
            if n: axes[2].text(i,bottom[i]+n/2,str(n),ha='center',va='center',fontsize=9)
        bottom += counts
    axes[2].set(title='Attribution of task-increasing steps',xticks=range(4),xticklabels=['4D CD','4D UOT','HF CD','HF UOT'],ylabel='Accepted REFRESH steps')
    axes[2].set_ylim(0,max(bottom)*1.18)
    axes[2].tick_params(axis='x',labelsize=8);axes[2].legend(fontsize=7)
    fig.text(.5,.01,'Positive task cosine is harmful. GT is evaluation only. Correlated iterates from 6 reused pairs; these counts are not independent scenes.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(OUT/'attribution.png',dpi=160);plt.close(fig)


if __name__ == '__main__':
    main()
