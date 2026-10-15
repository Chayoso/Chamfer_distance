"""GT-free cloned Adam continuation and scale-aware transfer diagnostics."""
import copy
import time
from .model import torch
from .controlled import cd


def tick(tensor):
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)
    return time.perf_counter()


def clone_adam(layer, history, group):
    clone = copy.deepcopy(layer)
    clone._forward_pre_hooks.clear()
    params = list(clone.parameters())
    if group.get('capturable') or group.get('fused') or group.get('amsgrad'):
        raise ValueError('Diagnostic supports the ordinary baseline Adam state')
    if len(params) != len(history):
        raise ValueError('Optimizer history does not match model parameters')
    optimizer = torch.optim.Adam([dict(group, params=params)])
    for p, (m, v, step) in zip(params, history):
        if step:
            optimizer.state[p] = dict(step=torch.tensor(float(step)),
                                      exp_avg=m.to(p).clone(), exp_avg_sq=v.to(p).clone())
    return clone, optimizer


def transfer_error(target, velocity):
    """Relative fit and best nonnegative-ray error; zero targets are undefined."""
    t, v = float(target.norm()), float(velocity.norm())
    if t == 0:
        return dict(relative_error=None, ray_error=None, cosine=None, ray_scale=None)
    dot = float((target*velocity).sum())
    scale = max(0., dot/(v*v)) if v else 0.
    return dict(relative_error=float((velocity-target).norm())/t,
                ray_error=float((scale*velocity-target).norm())/t,
                cosine=dot/(t*v) if v else 0., ray_scale=scale)


def adam_path(layer, inputs, target, history, group, budget, max_updates=10000):
    """Time setup and actual optimization; exclude returned diagnostic snapshots.

    Include the first evaluated state beyond budget to bracket a discrete update.
    The caller evaluates GT only AFTER this path has completed.
    """
    if budget <= 0:
        raise ValueError('A positive compute budget is required')
    start = tick(inputs)
    clone, optimizer = clone_adam(layer, history, group)
    elapsed = tick(inputs)-start
    records = []
    for step in range(max_updates+1):
        start = tick(inputs)
        output = clone(inputs)[0]
        value = cd(output, target)
        elapsed += tick(inputs)-start
        records.append(dict(step=step, seconds=elapsed, cd=float(value.detach()),
                            output=output.detach().clone()))
        if elapsed > budget:
            return records
        start = tick(inputs)
        optimizer.zero_grad()
        value.backward()
        optimizer.step()
        elapsed += tick(inputs)-start
    raise RuntimeError('Continuation update limit reached before its time budget')


def budget_bracket(records, budget):
    """Observable best CD through the first post-deadline evaluation; no GT."""
    within = [r for r in records if r['seconds'] <= budget]
    after = next(r for r in records if r['seconds'] > budget)
    eligible = within+[after]
    return dict(last=within[-1] if within else None, after=after,
                best=min(eligible, key=lambda r:r['cd']))
