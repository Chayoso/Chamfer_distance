"""GT separation, rejection handling, and distinct selector decisions."""
import copy
import pytest
from pipeline.selection import observed,select,RULES


def fixture():
    return [dict(method=m,cd_derivative=s,cd_change=c,accepted=True,
                 linearization_error=.01,ray_error=r) for m,s,c,r in (
        ('actual_adam',-3.,-.1,None),('parameter_gd',-1.,-.2,None),
        ('field_a',-2.,-.4,.8),('field_b',-.5,-.3,.2))]


def test_distinct_observable_criteria_and_baseline_only_controls():
    rows=fixture()
    assert [select(rows,r) for r in RULES]==['actual_adam','field_a','field_b']
    assert [select(rows[:2],r) for r in RULES]==['actual_adam','parameter_gd','parameter_gd']


def test_evaluation_and_state_identity_cannot_change_selector_inputs():
    d=dict(method='a',cd_derivative=-1.,risk_derivative=-100.,pair=100,level=0)
    f=dict(method='a',cd_change=-.1,accepted=True,linearization_error=.01,risk_change=-10.)
    t=dict(method='a',barycentric_ray_error=.4,task_cosine=1.)
    before=observed(d,f,t)
    d.update(risk_derivative=float('nan'),pair=201,level=8)
    f['risk_change']=1e9;t['task_cosine']=-1.
    assert observed(d,f,t)==before
    leaked=dict(before,risk_change=1.)
    with pytest.raises(ValueError,match='observable'):select([leaked],'cd_trial')


def test_finite_selection_never_uses_rejected_or_inconsistent_decrease():
    rows=fixture();rows[2].update(accepted=False,cd_change=-100.)
    rows[3].update(linearization_error=.2,cd_change=-200.)
    assert select(rows,'cd_trial')=='parameter_gd'
    assert select(rows,'ray_cd')=='parameter_gd'
    for r in rows:r.update(cd_change=1.,cd_derivative=1.)
    assert all(select(rows,rule) is None for rule in RULES)


def test_order_invariance_and_explicit_tie_breaking():
    rows=fixture();rows[3].update(cd_change=rows[2]['cd_change'])
    assert select(rows,'cd_trial')==select(rows[::-1],'cd_trial')=='field_a'
    with pytest.raises(ValueError,match='Duplicate'):select(rows+[copy.deepcopy(rows[0])],'cd_trial')


def test_invalid_numeric_inputs_and_unknown_rules_are_errors():
    rows=fixture();rows[2]['ray_error']=float('nan')
    with pytest.raises(ValueError,match='ray error'):select(rows,'ray_cd')
    with pytest.raises(ValueError,match='Unknown'):select(fixture(),'oracle')
