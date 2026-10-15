"""Synthetic attribution fixture, not scientific performance data."""
from capacity_report import BASES,assess
from feature_report import METHODS,COMPARISON


def test_attribution_uses_the_matching_marginal_model():
    keys=[(100,0,25),(200,0,25)];d={};f={}
    for k in keys:
        for method in BASES+METHODS:
            risk=1. if method in BASES or method.startswith('geo_unbalanced') else -1.
            d[k,method]=dict(risk_derivative=risk)
            for policy in ('raw','trust','cd_trust'):
                for cap in (1/16,.25,1.):f[k,method,policy,cap]=dict(risk_change=risk)
    counts,gates=assess(d,f,keys,METHODS,COMPARISON)
    selected=[r for r in gates if r['candidate'].startswith('feature_')]
    assert selected and all(r['primary_gate'] and r['controlled_baseline_gate'] for r in selected)
    assert all(r['attribution_gate']==r['candidate'].startswith('feature_unbalanced') for r in selected)
    assert all(r['baseline']==COMPARISON[r['candidate']] for r in counts if r['baseline'] not in BASES)
