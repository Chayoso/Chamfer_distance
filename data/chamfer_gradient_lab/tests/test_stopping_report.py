import pytest
from stopping_report import summarize_path


def rows():
    return [dict(step=i, cd=1.+abs(i-25)/100, mse=abs(i-90), epe=1., grad_norm=.1)
            for i in range(101)]


def test_report_separates_best_cd_from_final_and_gt():
    result = summarize_path(rows())
    assert result['best_step'] == 25  # GT optimum is at 90, never used.
    assert result['best_cd_gain'] == pytest.approx(.2)
    assert result['final_cd_gain'] == pytest.approx(-.4)
    assert result['best_cd_mse_change'] == -25
    assert result['final_mse_change'] == -80


def test_incomplete_path_cannot_be_reported_as_complete():
    with pytest.raises(AssertionError):
        summarize_path(rows()[:-1])
