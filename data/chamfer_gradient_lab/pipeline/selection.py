"""Small observable-only selectors; neither GT nor state identity is an input."""
import math

BASELINES=('actual_adam','parameter_gd')
RULES=('cd_slope','cd_trial','ray_cd')
FIELDS={'method','cd_derivative','cd_change','accepted','linearization_error','ray_error'}


def observed(direction,trial,transfer=None):
    """Explicit boundary between evaluation records and selector inputs."""
    assert direction['method']==trial['method']
    if transfer is not None:assert transfer['method']==direction['method']
    return dict(method=direction['method'],cd_derivative=direction['cd_derivative'],
        cd_change=trial['cd_change'],accepted=trial['accepted'],
        linearization_error=trial['linearization_error'],
        ray_error=None if transfer is None else transfer['barycentric_ray_error'])


def select(records,rule):
    """Return a method name or None (identity); ties use the method name."""
    if rule not in RULES:raise ValueError('Unknown selection rule')
    names=[r.get('method') for r in records]
    if len(set(names))!=len(names):raise ValueError('Duplicate proposal')
    for r in records:
        if set(r)!=FIELDS:raise ValueError('Only observable fields are allowed')
        if not isinstance(r['method'],str) or not isinstance(r['accepted'],bool):raise ValueError('Invalid proposal metadata')
        if not all(math.isfinite(r[n]) for n in ('cd_derivative','cd_change','linearization_error')):raise ValueError('Nonfinite observation')
        if r['linearization_error']<0:raise ValueError('Negative linearization error')
        if r['ray_error'] is not None and not (math.isfinite(r['ray_error']) and 0<=r['ray_error']<=1+1e-12):raise ValueError('Invalid ray error')
    if rule=='cd_slope':
        eligible=[r for r in records if r['cd_derivative']<0]
        score='cd_derivative'
    else:
        eligible=[r for r in records if r['accepted'] and r['linearization_error']<=.1 and r['cd_change']<0]
        score='cd_change'
        if rule=='ray_cd':
            proposals=[r for r in eligible if r['ray_error'] is not None and r['method'] not in BASELINES]
            if proposals:eligible=proposals;score='ray_error'
            else:eligible=[r for r in eligible if r['method'] in BASELINES]
    return min(eligible,key=lambda r:(r[score],r['method']))['method'] if eligible else None
