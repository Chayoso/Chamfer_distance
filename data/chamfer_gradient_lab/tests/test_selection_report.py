"""Reject incompatible saved states before any selector evaluation."""
import pytest
from selection_report import merge


def row(**values):
    return dict(pair=100,level=4,step=25,method='a',risk_change=.5,seconds=1.,**values)


def test_join_preserves_state_identity_and_ignores_only_timing():
    a=row();b=dict(a,seconds=9.)
    merged,error=merge([[a],[b]])
    assert len(merged)==1 and error==0
    with pytest.raises(AssertionError,match='Control value'):
        merge([[a],[dict(b,risk_change=.6)]])
    with pytest.raises(AssertionError,match='Duplicate'):
        merge([[a,a]])


def test_same_method_at_different_state_is_not_a_duplicate_control():
    a=row();b=dict(a,step=49,risk_change=-2.)
    merged,error=merge([[a],[b]])
    assert len(merged)==2 and error==0
