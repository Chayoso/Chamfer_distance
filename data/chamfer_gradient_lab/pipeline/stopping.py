"""Observe the pinned stopping loop and continue independent Adam clones."""
import ast
import copy
from .model import OFFICIAL, official, torch


def stopping_environment(step_callback, end_callback, force_ceiling=False):
    env = official()
    tree = ast.parse((OFFICIAL/'model_registration.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    method = next(n for n in cls.body if n.name == 'optimize_deformation_pyramid')
    outer = next(n for n in ast.walk(method) if isinstance(n, ast.For)
                 and isinstance(n.target, ast.Name) and n.target.id == 'level')
    loop = next(n for n in outer.body if isinstance(n, ast.For))
    conditions = {'loss.item() < 0.0001',
                  'abs(loss_prev - loss.item()) < loss_prev * break_threshold_ratio',
                  'break_counter >= max_break_count'}
    found = [n for n in loop.body if isinstance(n, ast.If)
             and ast.unparse(n.test) in conditions]
    assert len(found) == 3, 'Pinned upstream stopping logic changed'
    step = ast.parse('audit_step(level, iter, loss, s_sample_warped, t_sample, '
                     'NDP.pyramid[level], src[:config.samples], tgt_mean)').body[0]
    loop.body.insert(loop.body.index(found[0]), step)
    if force_ceiling:
        loop.body = [n for n in loop.body if n not in found]
    end = ast.parse('audit_end(level, iter, break_counter, loss_prev, loss, '
                    'optimizer, NDP.pyramid[level], s_sample, t_sample, '
                    's_sample_warped, src[:config.samples], tgt_mean)').body[0]
    outer.body.insert(outer.body.index(loop)+1, end)
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    env.update(audit_step=step_callback, audit_end=end_callback)
    exec(compile(module, 'observed_original_stopping', 'exec'), env)
    return env, module


def continuation(layer, optimizer, inputs, target, loss_fn, steps=100, lr_scale=1.):
    """Fixed budget, unchanged history, no access to GT or outcome selection."""
    clone = copy.deepcopy(layer)
    opt = torch.optim.Adam(clone.parameters(), lr=optimizer.param_groups[0]['lr'])
    opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for group in opt.param_groups:
        group['lr'] *= lr_scale
    for step in range(steps+1):
        output = clone(inputs)[0]
        value = loss_fn(output, target)
        yield step, output, value, list(clone.parameters())
        if step < steps:
            opt.zero_grad()
            value.backward()
            opt.step()
