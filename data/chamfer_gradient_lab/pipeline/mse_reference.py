"""Independent high-precision finite MSE identity audit; never a step selector."""
from decimal import Decimal, localcontext


def mse_reference(start, after, truth):
    if start.shape != after.shape or start.shape != truth.shape or start.ndim != 2:
        raise ValueError('Equal two-dimensional point arrays required')
    with localcontext() as context:
        context.prec = 80
        direct = expanded = Decimal(0)
        for p,z,y in zip(start.ravel(),after.ravel(),truth.ravel()):
            p,z,y = (Decimal.from_float(float(v)) for v in (p,z,y))
            displacement = z-p
            direct += (z-y)**2-(p-y)**2
            expanded += 2*(p-y)*displacement+displacement**2
        n=Decimal(len(start))
        return dict(reference_change=float(direct/n),reference_identity_error=float(abs(direct-expanded)/n))
