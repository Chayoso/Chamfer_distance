"""Actual finite movements, stage telescoping and a fixed descriptive anchor split."""
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ndp_continuation_geometry import ROOT,OUT,OLD,PAIRS,np,exact_reference

KINDS=('early_adam','global_7','global_15','late_adam')


def collect():
    assert json.loads((OUT/'integrity.json').read_text())['completed_replays']==6
    snapshots=[];moves=[];stages=[];max_error=0.;max_telescope=0.
    folder=OUT/'runs';reference=OLD/'runs/r0/restored_fgr_pcgmix'
    for pair in PAIRS:
        stem=f'p{pair}_s0_official_w0';read=lambda suffix:json.loads((folder/(stem+suffix+'.json')).read_text())
        result,updates,prefix,observations,movements=[read(s) for s in ('','_updates','_prefix','_snapshots','_movements')]
        exact_reference(folder,reference,stem,result,updates)
        assert prefix==json.loads((reference/(stem+'_prefix.json')).read_text())
        assert result['calls_by_level']==[51]*9
        assert [(s['level'],s['step']) for s in observations]==[(l,t) for l in range(9) for t in range(51)]
        assert [(m['level'],m['step']) for m in movements]==[(l,t) for l in range(9) for t in range(50)]
        assert [(s['level'],s['step']) for s in observations if 'held_epe' in s]==[(l,t) for l in range(9) for t in (16,50)]
        index={(s['level'],s['step']):s for s in observations}
        for m,u in zip(movements,updates):
            level,t=m['level'],m['step'];a,b=index[level,t],index[level,t+1]
            expected=f'global_{t}' if t in (7,15) else ('early_adam' if t<16 else 'late_adam')
            assert (u['level'],u['step'])==(level,t) and u['controlled']==(t in (7,15)) and m['kind']==expected
            assert u['moment_step']==t+1
            for key in ('fraction','beta','accepted','linearization_error'):assert m[key]==u.get(key)
            assert m['cd_change']==b['cd']-a['cd'] and m['task_epe_change']==b['task_epe']-a['task_epe']
            assert np.isfinite([m['cd_gradient_dot'],m['motion_norm']]).all()
            for key in ('task','anchor'):
                r=m[key];assert np.isfinite(list(r.values())).all() and r['quadratic']>=0
                assert abs(r['before']-a[key+'_mse'])<1e-12 and abs(r['after']-b[key+'_mse'])<1e-12
                assert r['change']==r['after']-r['before'] and r['identity_error']==abs(r['change']-r['linear']-r['quadratic'])
                assert r['identity_error']<1e-12;max_error=max(max_error,r['identity_error'])
        for level in range(9):
            for start in (0,16):
                selected=[m for m in movements if m['level']==level and m['step']>=start]
                for key in ('task','anchor'):
                    error=abs(math.fsum(m[key]['change'] for m in selected)-(index[level,50][key+'_mse']-index[level,start][key+'_mse']))
                    assert error<1e-12;max_telescope=max(max_telescope,error)
            a,b=index[level,16],index[level,50]
            stages.append(dict(pair=pair,group='4DMatch' if pair<200 else 'HF',level=level,
                cd_change=b['cd']-a['cd'],task_mse_change=b['task_mse']-a['task_mse'],
                task_epe_change=b['task_epe']-a['task_epe'],anchor_change=b['anchor_mse']-a['anchor_mse'],
                held_mse_change=b['held_mse']-a['held_mse'],held_epe_change=b['held_epe']-a['held_epe'],
                full_mse_change=b['full_mse']-a['full_mse'],full_epe_change=b['full_epe']-a['full_epe']))
        snapshots.extend(dict(s,pair=pair) for s in observations)
        moves.extend(dict(m,pair=pair,group='4DMatch' if pair<200 else 'HF') for m in movements)
    return snapshots,moves,stages,dict(max_mse_identity_error=max_error,max_telescope_error=max_telescope)


def counts(rows):
    task=lambda m:m['task']['change'];anchor=lambda m:m['anchor']['change']
    return dict(movements=len(rows),cd_down=sum(m['cd_change']<0 for m in rows),task_up=sum(task(m)>0 for m in rows),
        cd_down_task_up=sum(m['cd_change']<0 and task(m)>0 for m in rows),
        task_up_nonnegative_linear=sum(task(m)>0 and m['task']['linear']>=0 for m in rows),
        task_up_negative_linear=sum(task(m)>0 and m['task']['linear']<0 for m in rows),
        first_order_cd_down_actual_cd_up=sum(m['cd_gradient_dot']<0 and m['cd_change']>0 for m in rows),
        anchor_up=sum(anchor(m)>0 for m in rows),anchor_up_task_up=sum(anchor(m)>0 and task(m)>0 for m in rows),
        anchor_up_task_down=sum(anchor(m)>0 and task(m)<0 for m in rows),
        anchor_nonincrease_task_up=sum(anchor(m)<=0 and task(m)>0 for m in rows),
        anchor_near_zero_1e_minus12=sum(abs(anchor(m))<=1e-12 for m in rows),
        task_near_zero_1e_minus12=sum(abs(task(m))<=1e-12 for m in rows),
        sum_task_change=math.fsum(task(m) for m in rows),sum_anchor_change=math.fsum(anchor(m) for m in rows),
        sum_cd_change=math.fsum(m['cd_change'] for m in rows))


def figures(snapshots,moves,stages):
    fig,axes=plt.subplots(2,3,figsize=(12,6.5))
    for pair,ax in zip(PAIRS,axes.flat):
        selected=[s for s in snapshots if s['pair']==pair]
        for key,label in (('cd','Observed CD'),('task_mse','Task MSE'),('anchor_mse','Anchor MSE')):
            for level in range(9):
                part=[s for s in selected if s['level']==level]
                color={'cd':'C0','task_mse':'C1','anchor_mse':'C2'}[key]
                ax.plot([level*50+s['step'] for s in part],[s[key]/selected[0][key] for s in part],
                    color=color,lw=1,label=label if level==0 else None)
        for level in range(9):ax.axvline(level*50+16,color='gray',lw=.5,ls=':')
        ax.set(title=f'Pair{pair}',xlabel='Actual attempted updates',ylabel='Ratio to own initial value')
    axes[0,0].legend(fontsize=8)
    fig.text(.5,.01,'Six exact restored-path replays. Dotted: after second global event. Curves do not join across stage boundaries.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(OUT/'trajectories.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for i,pair in enumerate(PAIRS):
        selected=[m for m in moves if m['pair']==pair and m['kind']=='late_adam']
        axes[0].scatter([m['anchor']['change'] for m in selected],[m['task']['change'] for m in selected],
            s=7,alpha=.45,color=f'C{i}',label=str(pair))
        selected=[s for s in stages if s['pair']==pair]
        axes[1].scatter([s['anchor_change'] for s in selected],[s['held_mse_change'] for s in selected],
            s=24,color=f'C{i}',label=str(pair))
    for ax,title,ylabel in zip(axes,('Each actual late Adam update','Stage continuation16 to50'),('Training MSE change','Held-point MSE change')):
        ax.axhline(0,color='black',lw=.8);ax.axvline(0,color='black',lw=.8)
        ax.set_xscale('symlog',linthresh=1e-6);ax.set_yscale('symlog',linthresh=1e-6)
        ax.set(title=title,xlabel='Observed fixed-anchor MSE change',ylabel=ylabel)
    axes[1].legend(fontsize=8,ncol=2)
    fig.text(.5,.01,'Descriptive zero-sign split; no threshold tuning or guarded rerun. Axis scaling is for display only.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(OUT/'anchor_split.png',dpi=160);plt.close(fig)


def main():
    snapshots,moves,stages,numeric=collect();groups=[];pairs=[];stage_groups=[]
    for group in ('4DMatch','HF'):
        for kind in KINDS:groups.append(dict(group=group,kind=kind,**counts([m for m in moves if m['group']==group and m['kind']==kind])))
        part=[s for s in stages if s['group']==group]
        stage_groups.append(dict(group=group,stages=len(part),cd_down_task_up=sum(s['cd_change']<0 and s['task_mse_change']>0 for s in part),
            cd_down_held_up=sum(s['cd_change']<0 and s['held_mse_change']>0 for s in part),
            anchor_up_held_up=sum(s['anchor_change']>0 and s['held_mse_change']>0 for s in part),
            anchor_nonincrease_held_up=sum(s['anchor_change']<=0 and s['held_mse_change']>0 for s in part),
            anchor_up_held_down=sum(s['anchor_change']>0 and s['held_mse_change']<0 for s in part),
            train_down_held_up=sum(s['task_mse_change']<0 and s['held_mse_change']>0 for s in part),
            train_up_held_down=sum(s['task_mse_change']>0 and s['held_mse_change']<0 for s in part)))
    for pair in PAIRS:
        for kind in KINDS:pairs.append(dict(pair=pair,kind=kind,**counts([m for m in moves if m['pair']==pair and m['kind']==kind])))
    result=dict(exact_replays=6,exact_update_records=2700,observations=2754,actual_movements=2700,
        late_movements=1836,stages=54,numerical_gate=True,**numeric,groups=groups,stage_groups=stage_groups,
        posthoc_diagnostic=True,new_candidates=0,adopted=False,new_scenes=0)
    for name,value in (('summary.json',result),('pair_counts.json',pairs),('stage_changes.json',stages)):
        (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    lines=['Exact path diagnostic; no new candidate, selector, or adoption. Correlated updates are not independent samples.']
    lines += [str(numeric)]+[str(r) for r in groups]+['Stage continuation:']+[str(r) for r in stage_groups]
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines));figures(snapshots,moves,stages)


if __name__=='__main__':main()
