"""Validate real mixed NDP updates against unchanged directional/finite gates."""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fields_report import PAIRS, coverage, net_counts

ROOT=Path(__file__).resolve().parent
PRECISE='--precise-metrics' in sys.argv
OUT=ROOT/'paper_revision/mix_20261018'/('precise' if PRECISE else 'finite')
METHODS=('cd_gd','cd_ridge','fgr_gd','fgr_ridge','fgr_gdmix','fgr_ridgemix','adam')


def read(folder,name):
    return [json.loads(s) for s in (folder/(name+'.txt')).read_text().splitlines()]


def main():
    rows={n:read(OUT,n) for n in ('states','fields','solvers','mixes','directions','finite')}
    states={(p,l) for p in PAIRS for l in range(9)}
    coverage(rows['states'],('pair','level'),states)
    for name in ('fields','solvers'):
        coverage(rows[name],('pair','level','method'),{(*s,m) for s in states for m in ('cd','fgr')})
    coverage(rows['mixes'],('pair','level','transfer'),{(*s,m) for s in states for m in ('gd','ridge')})
    coverage(rows['directions'],('pair','level','method'),{(*s,m) for s in states for m in METHODS})
    coverage(rows['finite'],('pair','level','method','policy','cap'),
        {(*s,m,p,c) for s in states for m in METHODS for p in ('raw','trust','cd_trust') for c in (1/16,.25,1.)})
    assert json.loads((OUT/'integrity.json').read_text())['exact_replays']==list(PAIRS)
    d={(r['pair'],r['level'],r['method']):r for r in rows['directions']}
    f={(r['pair'],r['level'],r['method'],r['policy'],r['cap']):r for r in rows['finite']}
    old=ROOT/'paper_revision/fgr_20261018/neural'
    control_counts={}
    for name,lookup,keys in [('directions',d,('pair','level','method')),
                             ('finite',f,('pair','level','method','policy','cap'))]:
        controls=read(old,name)
        for r in controls:
            new=lookup[tuple(r[k] for k in keys)]
            for k,v in r.items():
                if isinstance(v,(int,float)):
                    np.testing.assert_allclose(new[k],v,rtol=1e-10,atol=1e-12,err_msg=str((name,keys,k)))
        control_counts[name]=len(controls)
    assert all(r['converged'] and r['relative_residual']<=1e-8 for r in rows['solvers'])
    phase={(r['pair'],r['level']):r for r in json.loads((OUT.parent/'states.json').read_text())}
    for p,l in states:
        r=d[p,l,'fgr_ridgemix'];low,high=phase[p,l]['normalized_task_interval']
        assert low-1e-12<=r['risk_derivative']<=high+1e-12
        assert r['cd_derivative'] <= .5*d[p,l,'cd_gd']['cd_derivative']+1e-12
        assert int(np.sign(r['risk_derivative']))==phase[p,l]['classifications'][str(1e-12)]
    for r in rows['finite']:
        if r['policy']!='raw' and r['accepted']:
            key='cd' if r['policy']=='cd_trust' else 'objective'
            slope=d[r['pair'],r['level'],r['method']][key+'_derivative']
            assert r[key+'_change'] <= .1*r['fraction']*slope+1e-14
            assert r['linearization_error']<=.1+1e-12
    numerical=max(r['mse_identity_error'] for r in rows['finite'])<1e-12
    precision=[dict(pair=r['pair'],level=r['level'],method=r['method'],policy=r['policy'],cap=r['cap'],
        risk_change=r['risk_change'],float64_identity_error=r.get('float64_identity_error',r['mse_identity_error']),**r['precision_reference'])
        for r in rows['finite'] if r.get('float64_identity_error',r['mse_identity_error'])>1e-12]
    assert all(r['reference_identity_error']<1e-12 for r in precision)
    precision_parity={}
    if PRECISE:
        original=OUT.parent/'finite'
        for name in ('fields','solvers','mixes','directions'):
            assert rows[name]==read(original,name),('Changed proposals',name)
        previous=read(original,'finite');assert len(previous)==len(rows['finite'])
        for r,old_row in zip(rows['finite'],previous):
            for key,value in old_row.items():
                if key not in ('risk_change','mse_identity_error','precision_reference'):
                    assert r[key]==value,('Changed observed trial',r['pair'],r['level'],key)
            assert r['float64_risk_change']==old_row['risk_change']
            assert r['float64_identity_error']==old_row['mse_identity_error']
            for eps in (1e-12,1e-10,1e-8):
                category=lambda x: -1 if x < -eps else 1 if x > eps else 0
                assert category(r['risk_change'])==category(old_row['risk_change'])
        precision_parity=dict(exact_proposal_records=sum(len(rows[n]) for n in ('fields','solvers','mixes','directions')),
            exact_observed_trials=len(previous),unchanged_task_classifications=3*len(previous),
            evaluated_reference_rows=len(rows['finite']),historical_float64_gate=False,historical_failure_preserved=True)
    counts=[]
    for group,pairs in (('4DMatch',PAIRS[:4]),('HF',PAIRS[4:])):
        for baseline in ('adam','cd_gd','cd_ridge','fgr_gdmix'):
            for cap in (0.,1/16,.25,1.):
                for policy in (('raw',) if cap==0 else ('raw','trust')):
                    a=[d[p,l,baseline]['risk_derivative'] if cap==0 else f[p,l,baseline,policy,cap]['risk_change'] for p in pairs for l in range(9)]
                    b=[d[p,l,'fgr_ridgemix']['risk_derivative'] if cap==0 else f[p,l,'fgr_ridgemix','trust',cap]['risk_change'] for p in pairs for l in range(9)]
                    for eps in (1e-12,1e-10,1e-8):
                        counts.append(dict(group=group,baseline=baseline,cap=cap,base_policy=policy,tolerance=eps,**net_counts(a,b,eps)))
    primary=[r for r in counts if r['cap'] in (0,.25,1.) and r['tolerance']==1e-12]
    assert len(primary)==40
    original=[r for r in primary if r['baseline'] in ('adam','cd_gd')]
    groups=[]
    for group,pairs in (('4DMatch',PAIRS[:4]),('HF',PAIRS[4:])):
        for method in METHODS:
            chosen=[f[p,l,method,'trust',1.] for p in pairs for l in range(9)]
            groups.append(dict(group=group,method=method,states=len(chosen),
                tangent_descents=sum(d[p,l,method]['risk_derivative'] < -1e-12 for p in pairs for l in range(9)),
                finite_descents=sum(r['risk_change'] < -1e-12 for r in chosen),
                cd_trust_descents=sum(f[p,l,method,'cd_trust',1.]['risk_change'] < -1e-12 for p in pairs for l in range(9)),
                rejected=sum(not r['accepted'] for r in chosen),cd_increases=sum(r['cd_change']>1e-12 for r in chosen),
                mean_mse_change=float(np.mean([r['risk_change'] for r in chosen]))))
    per_pair=[dict(pair=p,descents=sum(f[p,l,'fgr_ridgemix','trust',1.]['risk_change'] < -1e-12 for l in range(9)),
        mean_mse_change=float(np.mean([f[p,l,'fgr_ridgemix','trust',1.]['risk_change'] for l in range(9)]))) for p in PAIRS]
    passed=all(r['net']>0 for r in primary)
    if PRECISE:
        assert counts==json.loads((OUT.parent/'finite/counts.json').read_text())
    summary=dict(exact_original_replays=6,states=54,solves=108,finite_rows=len(rows['finite']),controls=control_counts,
        original_gd_adam_gate=all(r['net']>0 for r in original),all_direction_finite_conditions=passed,
        positive_required_cells=sum(r['net']>0 for r in primary),required_cells=40,worst_net=min(r['net'] for r in primary),
        numerical_gate=numerical,max_mse_identity_error=max(r['mse_identity_error'] for r in rows['finite']),
        max_candidate_identity_error=max(r['mse_identity_error'] for r in rows['finite'] if r['method']=='fgr_ridgemix'),
        numerical_failure_rows=len(precision),max_reference_identity_error=max([r['reference_identity_error'] for r in precision],default=0.),
        max_solver_residual=max(r['relative_residual'] for r in rows['solvers']),groups=groups,per_pair=per_pair,
        necessary_screen_passed=passed and numerical,adopted=False,new_scenes=0,native_candidate_fits=0,
        precision_mode=PRECISE,precision_parity=precision_parity)
    for name,data in (('summary.json',summary),('counts.json',counts),('precision.json',precision)):
        (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    lines=['Actual mixed updates;54 reused correlated states; no new native candidate or scene.']
    lines += [str(r) for r in groups]+['Failed required direction/finite cells:']+[str(r) for r in primary if r['net']<=0]
    lines += ['Gates: '+str({k:v for k,v in summary.items() if k not in ('groups','per_pair')}),'Precision audit:']+[str(r) for r in precision]
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines));figure(groups)


def figure(groups):
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    methods=('cd_gd','adam','fgr_ridge','fgr_gdmix','fgr_ridgemix')
    for ax,group in zip(axes,('4DMatch','HF')):
        rows=[next(r for r in groups if r['group']==group and r['method']==m) for m in methods]
        x=np.arange(len(rows))
        ax.bar(x-.18,[r['tangent_descents'] for r in rows],.36,label='Tangent task descent')
        ax.bar(x+.18,[r['cd_trust_descents'] for r in rows],.36,label='CD-trust task descent')
        ax.set(xticks=x,xticklabels=['CD-GD','Adam','FGR ridge','GD mix','Ridge mix'],title=group,ylabel='Endpoint count',ylim=(0,39 if group=='4DMatch' else 20))
        ax.legend(fontsize=8)
    fig.text(.5,.01,'One fixed half-GD correction; matched motion bound, cap1. Full numerical and adoption gates are reported separately.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(OUT/'finite_mix.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
