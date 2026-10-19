"""Scalar certificates for a single observed-loss constrained two-ray mixture."""
import math


def mix_fraction(proposed_slope, descent_slope, retain=.5):
    """Least reference mixing that retains a fixed fraction of its loss descent."""
    if not all(math.isfinite(v) for v in (proposed_slope, descent_slope, retain)):
        raise ValueError('Finite slopes and retention required')
    if descent_slope >= 0 or not 0 < retain <= 1:
        raise ValueError('A strict reference descent and retention in(0,1] required')
    if proposed_slope <= retain*descent_slope:
        return 0.
    return (proposed_slope-retain*descent_slope)/(proposed_slope-descent_slope)


def normalized_mix_fraction(proposed_slope, descent_slope, cosine, retain=.5):
    """Least mixing after output normalization; degenerate rays keep the old rule."""
    upper=mix_fraction(proposed_slope,descent_slope,retain)
    if not math.isfinite(cosine) or abs(cosine)>1+1e-12:
        raise ValueError('A finite ray cosine in[-1,1] is required')
    if upper==0 or abs(cosine)>=1-1e-12:
        return upper
    lower=0.
    for _ in range(64):
        mid=(lower+upper)/2
        ratio=math.sqrt((1-2*mid)**2+2*(1+cosine)*mid*(1-mid))
        slope=((1-mid)*proposed_slope+mid*descent_slope)/ratio
        if slope<=retain*descent_slope:upper=mid
        else:lower=mid
    return upper


def normalized_slope_interval(task_a, task_c, beta, cd_a, cd_c, bound, cd_norm):
    """Bounds after norm-b normalization, assuming both input rays have norm b."""
    if not 0 <= beta <= 1 or bound <= 0 or cd_norm <= 0:
        raise ValueError('Valid mixture and positive geometric scales required')
    cd = (1-beta)*cd_a+beta*cd_c
    lower_norm_ratio = max(abs(1-2*beta), abs(cd)/(bound*cd_norm))
    if not 0 < lower_norm_ratio <= 1+1e-12:
        raise ValueError('Unavailable or inconsistent norm certificate')
    lower_norm_ratio = min(1., lower_norm_ratio)
    task = (1-beta)*task_a+beta*task_c
    return tuple(sorted((task, task/lower_norm_ratio)))


def interval_sign(interval, tolerance):
    lower, upper = interval
    if upper < -tolerance: return -1
    if lower > tolerance: return 1
    if lower >= -tolerance and upper <= tolerance: return 0
    return None


def feasible_task_range(task_a, task_c, cd_a, cd_c, retain=0.):
    """Extrema on the closed CD-feasible interval; GT-only existence diagnostic."""
    if cd_c >= 0 or not 0 <= retain <= 1:
        raise ValueError('Strict CD reference descent and retention in[0,1] required')
    beta_min = 0. if cd_a <= retain*cd_c else (cd_a-retain*cd_c)/(cd_a-cd_c)
    values = ((1-beta_min)*task_a+beta_min*task_c, task_c)
    return dict(beta_min=beta_min, minimum=min(values), maximum=max(values))
