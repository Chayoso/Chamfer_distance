import numpy as np
import pytest
from pipeline.direction_mix import mix_fraction, normalized_slope_interval, interval_sign, feasible_task_range


def test_analytic_mix_matches_independent_linear_program():
    from scipy.optimize import linprog
    for a,c in [(2.,-1.),(-.9,-1.),(-.2,-1.),(0.,-3.),(10.,-.01)]:
        actual = mix_fraction(a,c)
        solution = linprog([1.],A_ub=[[c-a]],b_ub=[.5*c-a],bounds=[(0,1)],method='highs')
        assert solution.success
        assert actual == pytest.approx(solution.x[0],abs=1e-12)


def test_slope_certificates_contain_actual_random_vector_normalization():
    rng = np.random.default_rng(19381)
    for _ in range(100):
        g,h,a = rng.normal(size=(3,7)); c=-g/np.linalg.norm(g); a/=np.linalg.norm(a)
        sa,sc = g@a,g@c; beta=mix_fraction(sa,sc)
        v=(1-beta)*a+beta*c; normalized=v/np.linalg.norm(v)
        lower,upper=normalized_slope_interval(h@a,h@c,beta,sa,sc,1.,np.linalg.norm(g))
        assert lower-1e-12 <= h@normalized <= upper+1e-12
        assert g@normalized <= .5*sc+1e-12


def test_existence_at_open_cd_boundary_and_impossibility():
    # CD slope=1-2b, task slope=-2+3b: strict overlap exists for1/2<b<2/3.
    result=feasible_task_range(-2.,1.,1.,-1.)
    assert result['beta_min']==.5 and result['minimum']==-.5
    # A half-GD requirement forces b>=3/4, eliminating that overlap.
    assert feasible_task_range(-2.,1.,1.,-1.,.5)['minimum']==.25
    assert feasible_task_range(2.,1.,1.,-1.)['minimum']==1.


def test_ambiguous_sign_and_degenerate_reference_are_not_silently_accepted():
    assert interval_sign((-2e-12,-.5e-12),1e-12) is None
    assert interval_sign((-.5e-12,.5e-12),1e-12)==0
    assert interval_sign((-3e-12,-2e-12),1e-12)==-1
    with pytest.raises(ValueError): mix_fraction(1.,0.)
