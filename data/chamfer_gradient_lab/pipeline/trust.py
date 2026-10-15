"""Observed-loss and output-linearization backtracking; no task labels."""
import math
from .model import torch


def select_step(evaluate, start, velocity, loss, slope, cap=1., max_halvings=20):
    """Return (fraction, trial count); evaluate returns output and observed loss."""
    speed = float(velocity.norm())
    if slope >= 0 or speed == 0: return 0.,0
    if not (math.isfinite(slope) and math.isfinite(loss) and 0 < cap <= 1):
        raise ValueError('Invalid line-search input')
    count = 0
    for j in range(max_halvings+1):
        fraction = 2.**(-j)
        if fraction > cap: continue
        output,value = evaluate(fraction)
        count += 1
        if not torch.isfinite(output).all() or not math.isfinite(value): continue
        error = float((output-start-fraction*velocity).norm())/(fraction*speed)
        if value <= loss+.1*fraction*slope and error <= .1:
            return fraction,count
    return 0.,count
