"""Point-set objectives shared by the NDP experiments."""
from .model import np, torch
from transport_direction import transport


def components(p, q):
    d = torch.cdist(p, q, compute_mode='donot_use_mm_for_euclid_dist')
    return d.min(1).values, d.min(0).values


def objective(p, q, method, weight, scale, directions):
    if method not in ('official', 'cosine', 'squared', 'trimmed', 'sw', 'uot'):
        raise ValueError(f'Unknown objective: {method}')
    a, b = components(p, q)
    base = a.mean()+b.mean()
    if method == 'squared': return a.square().mean()+b.square().mean()
    if method == 'trimmed':
        return a.topk(int(np.ceil(.7*len(a))), largest=False).values.mean()+b.topk(int(np.ceil(.7*len(b))), largest=False).values.mean()
    if method == 'sw':
        base = base+weight*((p@directions).sort(0).values-(q@directions).sort(0).values).square().mean()/scale
    if method == 'uot':
        # A fixed, uniformly sampled subset of the official random 2000-point sample.
        ot, _ = transport((p[:128]/scale).double(), (q[:128]/scale).double(), tau=.1)
        base = base+weight*scale*ot
    return base
