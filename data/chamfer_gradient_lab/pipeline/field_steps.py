"""Common observable motion scale and finite trials. No ground-truth inputs."""
import copy
from .model import torch
from .controlled import cd
from .trust import select_step


def scale_direction(delta, forward, output_norm):
    velocity = forward(delta)
    norm = float(velocity.norm())
    factor = output_norm/norm if norm > 0 else 0.
    return [v*factor for v in delta], velocity*factor


def finite_trials(layer, inputs, p, q, delta, velocity, objective, gradient, cd_gradient):
    trial = copy.deepcopy(layer)
    saved = [v.detach().clone() for v in layer.parameters()]
    live = list(trial.parameters())
    initial, cd_initial = float(objective(p)), float(cd(p, q))
    slope, cd_slope = float((gradient*velocity).sum()), float((cd_gradient*velocity).sum())
    cache = {}
    def evaluate(fraction, common=False):
        if fraction not in cache:
            with torch.no_grad():
                for parameter, start, step in zip(live, saved, delta):
                    parameter.copy_(start+fraction*step)
                output = trial(inputs)[0].detach()
                cache[fraction] = output, float(objective(output)), float(cd(output, q))
        output, value, common_value = cache[fraction]
        return output, common_value if common else value
    for cap in (1/16, .25, 1.):
        own, attempts = select_step(evaluate, p, velocity, initial, slope, cap)
        common, cd_attempts = select_step(lambda a: evaluate(a, True), p, velocity, cd_initial, cd_slope, cap)
        for policy, fraction, count in (('raw', cap, 1), ('trust', own, attempts), ('cd_trust', common, cd_attempts)):
            output, value = evaluate(fraction) if fraction else (p, initial)
            motion = output-p
            denominator = fraction*float(velocity.norm())
            error = float((motion-fraction*velocity).norm())/denominator if denominator else 0.
            yield dict(policy=policy, cap=cap, fraction=fraction, accepted=fraction > 0,
                       objective_change=value-initial, cd_change=float(cd(output, q))-cd_initial,
                       output_norm=float(motion.norm()), linearization_error=error, attempts=count), output
