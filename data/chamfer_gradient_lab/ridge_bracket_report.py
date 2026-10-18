"""Fixed-plan damping effects and unchanged necessary rejection conditions."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fields_report import PAIRS, coverage, net_counts

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'paper_revision/ridge_bracket_20261018'
FACTORS = (.1,1.,10.)
LABELS = {.1:'low',1.:'base',10.:'high'}


def read(folder,name):
    return [json.loads(s) for s in (folder/(name+'.txt')).read_text().splitlines()]


def main():
    rows = read(OUT,'bracket'); states = {(p,l) for p in PAIRS for l in range(9)}
    coverage(rows,('pair','level','field','factor'),{(*s,k,f) for s in states for k in ('cd','uot') for f in FACTORS})
    assert json.loads((OUT/'integrity.json').read_text())['exact_replays']==list(PAIRS)
    old = {(r['pair'],r['level'],r['field']):r for r in read(ROOT/'paper_revision/attribution_20261018','attribution') if r['refresh'] and r['iteration']==1}
    lookup = {(r['pair'],r['level'],r['field'],r['factor']):r for r in rows}
    for r in rows:
        base = lookup[r['pair'],r['level'],r['field'],1.]
        if r['factor']==1.: assert {k:v for k,v in r.items() if k!='factor'}==old[r['pair'],r['level'],r['field']]
        assert r['solve']['converged'] and r['solve']['relative_residual']<=1e-8
        assert r['solve']['spectral_estimate']==base['solve']['spectral_estimate'] and r['plan']==base['plan']
        assert abs(r['solve']['damping']/base['solve']['damping']-r['factor'])<1e-12
        for k in ('force_slope','proposal_slope','force_cosine','proposal_cosine','proposal_norm','positive_mass_min'):
            assert r['attribution'][k]==base['attribution'][k]
        if r['fraction']:
            assert r['objective_change']<=.1*r['fraction']*r['slope']+1e-14 and r['linearization_error']<=.1+1e-12
        else: assert r['mse_change']==r['objective_change']==0
    d = {(r['pair'],r['level'],r['method']):r for r in read(ROOT/'paper_revision/joint_20261018','directions')}
    finite = {(r['pair'],r['level'],r['method'],r['policy']):r for r in read(ROOT/'paper_revision/joint_20261018','finite') if r['cap']==1}
    def value(p,l,name,cap,policy):
        if name in ('adam','cd_gd'):
            return d[p,l,name]['risk_derivative'] if cap==0 else finite[p,l,name,policy]['risk_change']
        kind,label = name.split('_');factor=next(f for f in FACTORS if LABELS[f]==label)
        r=lookup[p,l,kind,factor]
        return r['task_slope'] if cap==0 else r['mse_change']
    groups,counts,certificates,per_pair = [],[],[],[]
    for group,pairs in (('4DMatch',PAIRS[:4]),('HF',PAIRS[4:])):
        for kind in ('cd','uot'):
            for factor in FACTORS:
                chosen=[lookup[p,l,kind,factor] for p in pairs for l in range(9)]
                centers=[lookup[p,l,kind,1.] for p in pairs for l in range(9)]
                groups.append(dict(group=group,field=kind,factor=factor,states=len(chosen),
                    point_descents=sum(r['attribution']['proposal_slope']< -1e-12 for r in chosen),
                    tangent_descents=sum(r['task_slope']< -1e-12 for r in chosen),
                    finite_descents=sum(r['mse_change']< -1e-12 for r in chosen),
                    rejected=sum(r['fraction']==0 for r in chosen),mean_mse_change=float(np.mean([r['mse_change'] for r in chosen])),
                    median_weighted_error=float(np.median([r['attribution']['weighted_transfer_error'] for r in chosen])),
                    median_parameter_ratio=float(np.median([r['raw_parameter_norm']/b['raw_parameter_norm'] for r,b in zip(chosen,centers)])),
                    median_fraction=float(np.median([r['fraction'] for r in chosen])),
                    recovered_transfer_flips=sum(b['attribution']['proposal_cosine']< -1e-10 and b['attribution']['projected_cosine']>1e-10 and r['attribution']['projected_cosine']< -1e-10 for r,b in zip(chosen,centers))))
                for p in pairs:
                    selected=[lookup[p,l,kind,factor] for l in range(9)]
                    per_pair.append(dict(pair=p,field=kind,factor=factor,finite_descents=sum(r['mse_change']< -1e-12 for r in selected),mean_mse_change=float(np.mean([r['mse_change'] for r in selected]))))
                if factor==1.: continue
                certified = violations = 0
                for r,b in zip(chosen,centers):
                    diff=b['attribution']['weighted_transfer_error']-r['attribution']['weighted_transfer_error']
                    if factor>1:diff=-diff
                    allowance=b['solve']['relative_residual']+r['solve']['relative_residual']+1e-12
                    certified+=diff>allowance;violations+=diff< -allowance
                certificates.append(dict(group=group,field=kind,factor=factor,certified_residual_order=int(certified),violations=int(violations),states=len(chosen)))
                candidate=kind+'_'+LABELS[factor]; baselines=['adam','cd_gd',kind+'_base']
                if kind=='uot':baselines+=['cd_base','cd_'+LABELS[factor]]
                for baseline in baselines:
                    for cap in (0,1):
                        for policy in (('raw','trust') if cap and baseline in ('adam','cd_gd') else ('trust',)):
                            a=[value(p,l,baseline,cap,policy) for p in pairs for l in range(9)]
                            b=[value(p,l,candidate,cap,'trust') for p in pairs for l in range(9)]
                            for tolerance in (1e-12,1e-10,1e-8):
                                counts.append(dict(group=group,candidate=candidate,baseline=baseline,cap=cap,base_policy=policy,tolerance=tolerance,**net_counts(a,b,tolerance)))
    maximum=max(r['mse_identity_error'] for r in rows)
    numerical=maximum<1e-12 and not any(r['violations'] for r in certificates)
    gates=[]
    for kind in ('cd','uot'):
        for label in ('low','high'):
            candidate=kind+'_'+label;required=[r for r in counts if r['candidate']==candidate and r['tolerance']==1e-12]
            assert len(required)==(16 if kind=='cd' else 24)
            gates.append(dict(candidate=candidate,necessary_screen_passed=numerical and all(r['net']>0 for r in required),positive_cells=sum(r['net']>0 for r in required),required_cells=len(required),worst_net=min(r['net'] for r in required)))
    references=[dict(group=g,method=m,finite_descents=sum(finite[p,l,m,'trust']['risk_change']< -1e-12 for p in ps for l in range(9))) for g,ps in (('4DMatch',PAIRS[:4]),('HF',PAIRS[4:])) for m in ('adam','cd_gd')]
    result=dict(exact_original_replays=6,exact_central_controls=108,solves=len(rows),groups=groups,certificates=certificates,gates=gates,
        numerical_gate=numerical,max_mse_identity_error=maximum,max_solver_residual=max(r['solve']['relative_residual'] for r in rows),
        reused_references=references,required_next_candidates=[r['candidate'] for r in gates if r['necessary_screen_passed']],adopted=False,new_scenes=0)
    for name,data in (('summary.json',result),('counts.json',counts),('per_pair.json',per_pair)):
        (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    lines=['Fixed-state damping bracket;54 correlated initial endpoints, not a new native trajectory.',
           'group field factor tangent_descents finite_descents median_weighted_error median_parameter_ratio median_fraction recovered_transfer_flips']
    lines+=[' '.join(str(r[k]) for k in ('group','field','factor','tangent_descents','finite_descents','median_weighted_error','median_parameter_ratio','median_fraction','recovered_transfer_flips')) for r in groups]
    lines+=['Necessary rejection gates (not sufficient for adoption):']+[str(r) for r in gates]
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines));figure(result)


def figure(result):
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for col,group in enumerate(('4DMatch','HF')):
        for kind,color in (('cd','tab:blue'),('uot','tab:orange')):
            chosen=[r for r in result['groups'] if r['group']==group and r['field']==kind]
            axes[0,col].plot(FACTORS,[r['median_weighted_error'] for r in chosen],'o-',color=color,label=kind.upper())
            axes[1,col].plot(FACTORS,[r['finite_descents'] for r in chosen],'o-',color=color,label=kind.upper())
        for method,style in (('adam','--'),('cd_gd',':')):
            n=next(r['finite_descents'] for r in result['reused_references'] if r['group']==group and r['method']==method)
            axes[1,col].axhline(n,color='gray',linestyle=style,label=method)
        axes[0,col].set(title=group+': weighted fitting residual',ylabel='Median relative error')
        axes[1,col].set(title=group+': accepted task-descent states',ylabel='Endpoint count',ylim=(0,39 if group=='4DMatch' else 20))
        for ax in axes[:,col]:
            ax.set(xscale='log',xticks=FACTORS,xticklabels=['0.1','1','10'],xlabel='Damping / original damping');ax.legend(fontsize=8)
    fig.text(.5,.01,'Same initial states/plans; true-objective acceptance. Six reused pairs; no independent-scene or full-method improvement claim.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(OUT/'ridge_bracket.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
