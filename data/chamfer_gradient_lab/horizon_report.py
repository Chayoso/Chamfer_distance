"""Causal path diagnosis of frozen early/late interventions; not adoption gates."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'paper_revision/trajectory_horizon_20261018'
PAIRS=(100,101,124,125,200,201)


def reversal(rows,key,tolerance=1e-12):
    observed=[r for r in rows if key in r]
    initial=observed[0][key]
    if abs(initial)<=tolerance:return None
    later=next((r for r in observed[1:] if r[key]*np.sign(initial)<-tolerance),None)
    return None if later is None else dict(level=later['level'],step=later['step'],
        updates_since_intervention=later['updates_since_intervention'],gap=later[key],
        direction='benefit_to_harm' if initial<0 else 'harm_to_benefit')


def collect():
    assert json.loads((OUT/'integrity.json').read_text())['completed_fits']==24
    pairs=[];paths=[]
    for pair in PAIRS:
        for site in (0,8):
            stem=f'p{pair}_s0_official_w0';branches={}
            for arm in ('adam','fgr_ridgemix'):
                folder=OUT/'runs'/f'site{site}'/arm
                read=lambda suffix:json.loads((folder/(stem+suffix+'.json')).read_text())
                result,obs,updates,meta=read(''),read('_observations'),read('_updates'),read('_metadata')
                assert result['calls_by_level']==[51]*9 and len(updates)==450
                assert [(r['level'],r['step']) for r in obs]==[(l,s) for l in range(9) for s in range(51)]
                assert [(r['level'],r['step']) for r in updates]==[(l,s) for l in range(9) for s in range(50)]
                assert all(r['moment_step']==r['step']+1 for r in updates)
                assert all(np.isfinite(v) for r in obs for v in r.values() if isinstance(v,(int,float)))
                np.testing.assert_allclose([obs[-1]['held_epe'],obs[-1]['full_epe']],
                    [result['held_epe'],result['full_epe']],rtol=2e-6,atol=1e-8)
                controls=[r for r in updates if r['controlled']]
                assert [(r['level'],r['step']) for r in controls]==([] if arm=='adam' else [(site,49)])
                if arm=='adam':
                    ref=ROOT/'paper_revision/trajectory_mix_20261018/runs/r0/adam'
                    expected=json.loads((ref/(stem+'.json')).read_text())
                    assert {k:v for k,v in result.items() if k!='seconds'}=={k:v for k,v in expected.items() if k!='seconds'}
                    assert np.array_equal(np.loadtxt(folder/(stem+'_trace.txt'),delimiter=',',skiprows=1),
                                          np.loadtxt(ref/(stem+'_trace.txt'),delimiter=',',skiprows=1))
                branches[arm]=dict(result=result,observations=obs,updates=updates,metadata=meta,controls=controls)
            a,b=branches['adam'],branches['fgr_ridgemix'];after=site*51+50
            assert a['metadata']==b['metadata'] and a['observations'][:after]==b['observations'][:after]
            control=b['controls'][0];pre=b['observations'][after-1];post=b['observations'][after]
            assert control['valid'] and control['applied_exact']
            assert post['cd']-pre['cd']==control['cd_change']
            assert all(s['converged'] and s['relative_residual']<=1e-8 for s in control['solvers'].values())
            if control['accepted']:
                assert control['cd_change']<=.1*control['fraction']*control['cd_slope']+1e-14 and control['linearization_error']<=.1+1e-12
            else:assert control['cd_change']==control['fraction']==0
            path=[]
            for original,changed in zip(a['observations'][after:],b['observations'][after:]):
                row=dict(pair=pair,site=site,level=original['level'],step=original['step'],
                    updates_since_intervention=original['level']*50+original['step']-(site+1)*50)
                for key in ('train_mse','cd','held_epe','held_mse','full_epe','full_mse'):
                    if key in original:
                        row[key+'_gap']=changed[key]-original[key]
                        row[key+'_ratio']=changed[key]/original[key] if original[key] else None
                path.append(row)
            paths.extend(path)
            pairs.append(dict(pair=pair,site=site,group='4DMatch' if pair<200 else 'HF',
                accepted=control['accepted'],fraction=control['fraction'],beta=control['beta'],
                own_immediate_train_mse_change=post['train_mse']-pre['train_mse'],
                own_immediate_held_epe_change=post['held_epe']-pre['held_epe'],
                immediate_train_mse_gap=path[0]['train_mse_gap'],immediate_held_epe_gap=path[0]['held_epe_gap'],
                immediate_held_epe_ratio=path[0]['held_epe_ratio'],
                final_held_epe_gap=path[-1]['held_epe_gap'],final_train_mse_gap=path[-1]['train_mse_gap'],
                final_cd_gap=path[-1]['cd_gap'],final_held_epe_ratio=path[-1]['held_epe_ratio'],
                final_held_epe=b['result']['held_epe'],baseline_held_epe=a['result']['held_epe'],
                first_train_reversal=reversal(path,'train_mse_gap'),first_held_reversal=reversal(path,'held_epe_gap'),
                min_future_train_mse_gap=min(r['train_mse_gap'] for r in path),
                max_future_train_mse_gap=max(r['train_mse_gap'] for r in path),exact_prefix=True))
    return pairs,paths


def figure(pairs,paths):
    fig,axes=plt.subplots(1,3,figsize=(14,4.8))
    for ax,group in zip(axes[:2],(PAIRS[:4],PAIRS[4:])):
        for pair in group:
            rows=[r for r in paths if r['pair']==pair and r['site']==0]
            ax.plot([r['updates_since_intervention'] for r in rows],
                [100*(r['train_mse_ratio']-1) for r in rows],label=str(pair))
        ax.axhline(0,color='black',ls='--',lw=1)
        ax.set(title='Early intervention: '+('4DMatch' if group==PAIRS[:4] else 'HF'),
            xlabel='Remaining Adam updates since intervention',ylabel='Training MSE difference / plain Adam (%)')
        ax.legend(fontsize=8)
    index={(r['pair'],r['site']):r for r in pairs};ax=axes[2]
    for offset,site,key,label in ((-.25,0,'immediate_held_epe_ratio','Early: immediate'),
            (0,0,'final_held_epe_ratio','Early: final'),(.25,8,'final_held_epe_ratio','Late: immediate = final')):
        ax.bar(np.arange(6)+offset,[100*(index[p,site][key]-1) for p in PAIRS],.25,label=label)
    ax.axhline(0,color='black',ls='--',lw=1)
    ax.set(xticks=range(6),xticklabels=PAIRS,title='Held-point task effect',ylabel='Held EPE difference / plain Adam (%)')
    ax.legend(fontsize=8)
    fig.text(.5,.01,'One fixed intervention; every later update is plain Adam. Negative values favor the intervention. Six reused pairs; no GT selection.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(OUT/'horizon.png',dpi=160);plt.close(fig)


def main():
    pairs,paths=collect();groups=[]
    for site in (0,8):
        for group in ('4DMatch','HF'):
            rows=[r for r in pairs if r['site']==site and r['group']==group]
            groups.append(dict(site=site,group=group,pairs=len(rows),
                immediate_train_benefits=sum(r['immediate_train_mse_gap']<-1e-12 for r in rows),
                immediate_held_benefits=sum(r['immediate_held_epe_gap']<-1e-12 for r in rows),
                final_held_benefits=sum(r['final_held_epe_gap']<-1e-12 for r in rows),
                train_benefit_reversals=sum(r['first_train_reversal'] is not None and r['first_train_reversal']['direction']=='benefit_to_harm' for r in rows),
                held_benefit_reversals=sum(r['first_held_reversal'] is not None and r['first_held_reversal']['direction']=='benefit_to_harm' for r in rows),
                final_mean_held_epe_improvement=1-np.mean([r['final_held_epe'] for r in rows])/np.mean([r['baseline_held_epe'] for r in rows])))
    result=dict(completed_fits=24,exact_plain_replays=12,exact_prefixes=12,interventions=12,
        accepted=sum(r['accepted'] for r in pairs),new_scenes=0,adopted=False,adoption_test=False,groups=groups)
    for name,value in (('summary.json',result),('pairs.json',pairs),('paths.json',paths)):
        (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    lines=['Single-intervention diagnosis; twelve matched comparisons, six reused pairs. Not a new adoption test.']
    lines += [str(r) for r in groups]+['Every pair:']+[str(r) for r in pairs]
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines));figure(pairs,paths)


if __name__=='__main__':main()
