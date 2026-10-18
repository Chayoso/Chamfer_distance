import pytest
from fields_report import net_counts, coverage


def test_rejection_is_not_a_rescue_and_can_lose_a_baseline_descent():
    counts = net_counts([1., -1., -1., 1.], [-1., 1., 0., 0.], 1e-12)
    assert counts == dict(rescued=1, damaged=1, lost=1, net=-1)


def test_duplicate_or_missing_endpoint_rejected():
    rows = [dict(pair=100, level=0), dict(pair=100, level=0)]
    with pytest.raises(AssertionError):
        coverage(rows, ('pair', 'level'), {(100, 0), (100, 1)})
