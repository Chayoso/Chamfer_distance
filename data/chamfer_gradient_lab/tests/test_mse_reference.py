from fractions import Fraction
import numpy as np
from pipeline.mse_reference import mse_reference


def test_decimal_mse_difference_matches_exact_binary_rational_reference():
    p=np.array([[.1,1e4,-.3],[1e-5,2.,3.]])
    z=np.array([[.1+1e-10,1e4+1e-8,78.],[1e-4,12.,-.3]])
    y=np.array([[.4,1e4-3.,.2],[1.,-.5,2.]])
    expected=sum((Fraction(float(a))-Fraction(float(t)))**2-(Fraction(float(b))-Fraction(float(t)))**2
        for a,b,t in zip(z.ravel(),p.ravel(),y.ravel()))/len(p)
    actual=mse_reference(p,z,y)
    assert actual['reference_change']==float(expected)
    assert actual['reference_identity_error']<1e-65
