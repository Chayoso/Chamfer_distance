"""Observable selection on existing records; no training or runtime claim."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from trust_report import read
from lift_report import key,table
from capacity_report import BASES,verify_coverage,assess
from pipeline.selection import RULES,observed,select

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'paper_revision/selection_20261018'
PHASES={'weighted_20261015':('geo_soft','geo_unbalanced','feature_soft','feature_unbalanced'),
        'descriptor_20261015':('feature_soft','feature_unbalanced','descriptor_soft','descriptor_unbalanced')}
FILES=('directions','finite','states','solver','transfer')
COMPARISON={'cd_slope':'cd_slope_baselines','cd_trial':'cd_trial_baselines','ray_cd':'cd_trial_baselines'}
FIXED={r:'feature_soft_weighted' for r in RULES}


def label(row):
    base=(key(row),row['method'])
    return base+(row['policy'],row['cap']) if 'policy' in row else base


def merge(parts):
    """Shared records must agree except measured timing/memory."""
    merged={};errors=[]
    for rows in parts:
        assert len({label(r) for r in rows})==len(rows),'Duplicate input record'
        for row in rows:
            k=label(row)
            if k in merged:
                old=merged[k];assert set(old)==set(row),'Different record schema'
                for field,value in row.items():
                    if field in ('seconds','cuda_peak_bytes'):continue
                    previous=old[field]
                    if isinstance(value,(bool,str)) or value is None:
                        assert value==previous,'Control metadata differs'
                    else:
                        error=abs(value-previous);assert np.isfinite(error) and error<=1e-8,'Control value differs'
                        errors.append(error)
            else:merged[k]=row
    return merged,max(errors,default=0.)


def load_pool():
    parts={n:[] for n in FILES};paths=[]
    for phase,kinds in PHASES.items():
        folder=ROOT/'paper_revision'/phase/'probe'
        rows=[read(folder/(n+'.txt')) for n in FILES]
        integrity=json.loads((folder/'integrity.json').read_text())
        methods=tuple(k+'_'+t for k in kinds for t in ('gd','lift','weighted'))
        verify_coverage(*rows[:4],integrity,kinds,methods,tuple(k+'_'+t for k in kinds for t in ('lift','weighted')))
        keys={key(r) for r in rows[2]};labels=[label(r) for r in rows[4]]
        assert len(labels)==len(set(labels))==integrity['rows']['transfer']==648
        assert set(labels)=={(k,m) for k in keys for m in methods}
        for name,rs in zip(FILES,rows):parts[name].append(rs)
        paths.extend([folder/(n+'.txt') for n in FILES]+[folder/'integrity.json'])
    merged=[merge(parts[n]) for n in FILES]
    pools=[item[0] for item in merged]
    assert [len(p) for p in pools]==[1080,9720,324,648,972]
    assert all(np.isfinite(v) for p in pools for r in p.values() for v in r.values() if isinstance(v,float))
    assert all(r['converged'] and r['relative_residual']<=1e-8 for r in pools[3].values())
    assert all(r['converged'] and r['residual']<=1e-8 and r['mse_identity_error']<1e-12 for r in pools[2].values())
    return pools,max(item[1] for item in merged),paths


def choose(d,f,t,keys):
    choices=[];methods=sorted({m for k,m in d})
    for k in keys:
        observations=[observed(d[k,m],f[k,m,'trust',1.],t.get((k,m))) for m in methods]
        bases=[r for r in observations if r['method'] in BASES]
        for rule in RULES+('cd_slope_baselines','cd_trial_baselines'):
            baseline_only=rule.endswith('_baselines')
            selected=select(bases if baseline_only else observations,rule.removesuffix('_baselines'))
            choices.append(dict(pair=k[0],level=k[1],step=k[2],rule=rule,selected=selected))
    # No GT outcome enters this artifact or the selection function.
    (OUT/'choices.json').write_text(json.dumps(choices,indent=2)+'\n')
    return choices


def attach(d,f,choices):
    """Copy outcomes only after observable decisions are saved."""
    for choice in choices:
        k=key(choice);m=choice['selected'];name=choice['rule']
        direction=dict(d[k,m or 'actual_adam'],method=name)
        if m is None:
            for n in ('risk_derivative','objective_derivative','cd_derivative','output_norm'):direction[n]=0.
        d[k,name]=direction
        for policy in ('raw','trust','cd_trust'):
            for cap in (1/16,.25,1.):
                trial=dict(f[k,m or 'actual_adam',policy,cap],method=name)
                if m is None:
                    for n in ('risk_change','objective_change','cd_change','output_norm','linearization_error','fraction','attempts'):trial[n]=0.
                    trial['accepted']=False
                f[k,name,policy,cap]=trial


def figure(counts,choices):
    fig,axes=plt.subplots(2,2,figsize=(12,8));x=np.arange(len(RULES))
    for col,group in enumerate(('4DMatch','HF')):
        for shift,base,color,title in [(-.27,'actual_adam','#27886d','Adam'),(-.09,'parameter_gd','#c65353','GD'),
                                      (.09,None,'#537eac','baseline-only selection'),(.27,'feature_soft_weighted','#c89544','fixed feature + weighted')]:
            rows=[next(r for r in counts if r['group']==group and r['candidate']==m and r['baseline']==(base or COMPARISON[m]) and not r['periodic'] and r['cap']==1 and r['base_policy']=='trust' and r['controller']=='trust' and r['tolerance']==1e-12) for m in RULES]
            axes[0,col].bar(x+shift,[r['conservative_net'] for r in rows],.17,color=color,label=title)
        axes[0,col].axhline(0,color='black',lw=.6)
        axes[0,col].set(xticks=x,xticklabels=RULES,title=group+': GT evaluation at cap 1',ylabel='Conservative net rescued states')
        axes[0,col].legend(fontsize=8)
        bottom=np.zeros(3)
        for category,color in [('global','#537eac'),('actual_adam','#27886d'),('parameter_gd','#c65353'),('identity','#aaa')]:
            values=[sum(c['rule']==rule and ((c['pair']<200)==(group=='4DMatch')) and category==('identity' if c['selected'] is None else c['selected'] if c['selected'] in BASES else 'global') for c in choices) for rule in RULES]
            axes[1,col].bar(x,values,bottom=bottom,color=color,label=category);bottom+=values
        axes[1,col].set(xticks=x,xticklabels=RULES,title=group+': selected direction source',ylabel='Number of development states')
        axes[1,col].legend(fontsize=8)
    fig.suptitle('GT-free selection of 18 global proposals plus GD / Adam')
    fig.text(.5,.012,'Saved-record screen: 54 correlated states / 6 development pairs. No new fits, held-out scenes or runtime advantage.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.96]);fig.savefig(OUT/'selection_results.png',dpi=170);plt.close(fig)


def main():
    pools,control_error,paths=load_pool();d,f,states,solvers,t=pools
    keys=sorted({k for k,m in states});assert len(keys)==54
    choices=choose(d,f,t,keys);attach(d,f,choices)
    counts,gates=assess(d,f,keys,RULES,COMPARISON)
    fixed_counts,fixed_gates=assess(d,f,keys,RULES,FIXED)
    counts.extend(r for r in fixed_counts if r['baseline']=='feature_soft_weighted')
    for g,fg in zip(gates,fixed_gates):
        assert (g['candidate'],g['periodic'])==(fg['candidate'],fg['periodic'])
        g['fixed_proposal_gate']=fg['attribution_gate']
    table(OUT/'counts.txt',counts)
    safety=[]
    for rule in RULES:
        safety.append(dict(rule=rule,negative_cd_cap1=sum(f[k,rule,'trust',1.]['cd_change']<0 for k in keys),
            positive_cd_cap1=sum(f[k,rule,'trust',1.]['cd_change']>0 for k in keys),
            identity=sum(c['selected'] is None for c in choices if c['rule']==rule)))
        if rule!='cd_slope':assert all(f[k,rule,'trust',1.]['cd_change']<=0 for k in keys)
    for k in keys:
        assert d[k,'cd_slope']['cd_derivative']<=min(0.,*(d[k,m]['cd_derivative'] for m in BASES))
        assert f[k,'cd_trial','trust',1.]['cd_change']<=f[k,'cd_trial_baselines','trust',1.]['cd_change']
    summary=dict(numerical_gate=True,unique_development_states=54,reused_exact_replays=[100,101,124,125,200,201],
        new_replays=0,new_fits=0,new_independent_scenes=0,pool_rows=dict(zip(FILES,[1080,9720,324,648,972])),
        max_shared_control_difference=control_error,selection_uses_gt=False,screen='saved-record development only',
        native_replay_verified=False,total_selection_runtime_tested=False,gates=gates,safety=safety,
        advance_to_native=[m for m in RULES if all(all(g[n] for n in ('primary_gate','controlled_baseline_gate','attribution_gate','fixed_proposal_gate')) for g in gates if g['candidate']==m)],
        choice_counts={r:dict(Counter(c['selected'] or 'identity' for c in choices if c['rule']==r)) for r in RULES},
        primary_cap1=[r for r in counts if not r['periodic'] and r['controller']=='trust' and r['cap']==1 and r['base_policy']=='trust' and r['tolerance']==1e-12])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');figure(counts,choices)
    paths += [OUT/'PROTOCOL.txt',OUT/'PROOF.txt',ROOT/'pipeline/selection.py',Path(__file__)]
    paths=[p for p in paths if p.name not in ('PROTOCOL.txt','PROOF.txt','NUMERICAL_AMENDMENT.txt') or p.is_file()]
    (OUT/'report_hashes.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
