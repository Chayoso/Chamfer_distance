"""Synthetic audit fixtures; no rows here are scientific experiment outcomes."""
import copy
import pytest
from capacity_report import BASES,METHODS,KINDS,verify_coverage,assess


def complete_records():
    pairs=(100,101,124,125,200,201)
    states=[dict(pair=p,level=l,step=t) for p in pairs for l in (0,4,8) for t in (0,25,49)]
    dr=[dict(s,method=m) for s in states for m in BASES+METHODS]
    fr=[dict(r,policy=p,cap=c) for r in dr for p in ('raw','trust','cd_trust') for c in (1/16,.25,1.)]
    sr=[dict(s,method=m) for s in states for m in KINDS]
    solvers=[dict(s,method=m+'_lift') for s in states for m in KINDS]
    integrity=dict(exact_replays=list(pairs),rows=dict(directions=len(dr),finite=len(fr),states=len(sr),solver=len(solvers)))
    return dr,fr,sr,solvers,integrity


def test_complete_identity_grid_is_accepted():
    verify_coverage(*complete_records())


@pytest.mark.parametrize('table_index',[0,1,2,3])
def test_duplicated_row_cannot_replace_missing_result_at_equal_count(table_index):
    records=complete_records();records[table_index][-1]=copy.deepcopy(records[table_index][0])
    with pytest.raises(AssertionError,match='duplicated'):verify_coverage(*records)


def test_missing_family_and_false_replay_list_are_rejected():
    records=complete_records()
    for rows in records[:4]:
        for row in rows:
            if row['pair']>=200:row['pair']-=50
    with pytest.raises(AssertionError):verify_coverage(*records)
    records=complete_records();records[-1]['exact_replays'][-1]=100
    with pytest.raises(AssertionError,match='replay list'):verify_coverage(*records)


@pytest.mark.parametrize('baseline_risk',[-1.,1.])
def test_zero_update_is_never_rescue_and_losing_descent_counts_as_damage(baseline_risk):
    keys=[(100,0,25),(200,0,25)];d={};f={}
    for k in keys:
        for method in BASES+METHODS:
            risk=baseline_risk if method in BASES else 0.
            d[k,method]=dict(risk_derivative=risk)
            for policy in ('raw','trust','cd_trust'):
                for cap in (1/16,.25,1.):f[k,method,policy,cap]=dict(risk_change=risk)
    counts,gates=assess(d,f,keys)
    selected=[r for r in counts if r['baseline'] in BASES]
    assert selected and all(r['rescued']==0 for r in selected)
    assert all(r['uncertain_damage']==int(baseline_risk<0) for r in selected)
    assert all(not r['primary_gate'] and not r['controlled_baseline_gate'] for r in gates)
