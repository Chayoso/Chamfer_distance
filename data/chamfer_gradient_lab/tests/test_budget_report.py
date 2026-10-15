"""A compute-limited rescue must survive both timing repetitions and tolerances."""
from budget_report import rescue


def test_rescue_requires_both_adam_runs_stalled_and_real_candidate_gain():
    assert rescue(.002,[.00001,.00002],.0001)
    assert not rescue(.002,[.00001,.003],.0001)
    assert not rescue(.0001,[0.,0.],.0001)
    assert not rescue(0.,[0.,0.],.0001)
    assert not rescue(.000100001,[.0001,.0001],.0001)
