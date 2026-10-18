"""Accepted nonlinear updates; compare frozen versus refreshed correspondence."""
import copy
from .model import torch
from .joint import correspondence, fixed_lift
from .lifting import jacobian_products, weighted_lift
from .field_steps import scale_direction
from .capacity import frozen_change
from .trust import select_step


def accepted_path(layer, inputs, q, kind, refresh, seed, bound, steps=4, observe=None):
    """No GT. Both arms rebuild J and use the same true observed objective."""
    model = copy.deepcopy(layer).double()
    initial = model(inputs)[0].detach()
    objective_match = correspondence(initial, q, kind)  # Fixed epsilon/anchor.
    def match(z):
        if kind != 'cd':
            return objective_match(z)
        # CUDA index_add otherwise changes low bits even for identical inputs.
        enabled = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            return objective_match(z)
        finally:
            torch.use_deterministic_algorithms(enabled)
    force0, mass0, value0, _ = match(initial)
    damping = None
    for iteration in range(1, steps+1):
        output = model(inputs)[0]; p = output.detach()
        params = list(model.parameters()); saved = [v.detach().clone() for v in params]
        forward, transpose = jacobian_products(output, params)
        total_force, mass, value, plan = match(p)
        current_force = total_force-mass[:, None]*(p-initial)
        frozen_force = force0-mass0[:, None]*(p-initial)
        force, weights = (current_force, mass) if refresh else (frozen_force, mass0)
        if damping is None:
            raw, solve = weighted_lift(forward, transpose, force, weights, seed, .0001, 2048)
            damping = solve['damping']
        else:
            raw, solve = fixed_lift(forward, transpose, force, weights, damping)
        if not solve['converged']:
            raise RuntimeError(f'Unconverged refresh solve: {solve}')
        raw_norm = float(sum(v.square().sum() for v in raw).sqrt())
        delta, velocity = scale_direction(raw, forward, bound)
        slope = float((-current_force*velocity).sum())
        # A separate trial clone preserves the graph used for JVP/VJP.
        trial = copy.deepcopy(model); trial_params = list(trial.parameters()); cache = {}
        def evaluate(alpha):
            if alpha not in cache:
                with torch.no_grad():
                    for live, start, d in zip(trial_params, saved, delta): live.copy_(start+alpha*d)
                    z = trial(inputs)[0].detach()
                error = float((z-p-alpha*velocity).norm())/(alpha*float(velocity.norm()))
                # Both tests are required, so omit expensive matching if geometry fails.
                objective = match(z)[2] if error <= .1 and torch.isfinite(z).all() else float('inf')
                cache[alpha] = z, objective
            return cache[alpha]
        fraction, attempts = select_step(evaluate, p, velocity, value, slope)
        after = evaluate(fraction)[0] if fraction else p
        final_value = evaluate(fraction)[1] if fraction else value
        old_fraction, old_output = None, None
        if iteration == 1:
            # Reproduce the old first-step frozen-plan controller independently.
            def old_evaluate(alpha):
                z, _ = evaluate(alpha)
                return z, float(frozen_change(z, p, force0, mass0))
            if kind == 'uot':
                old_fraction, _ = select_step(old_evaluate, p, velocity, 0., slope)
                old_output = evaluate(old_fraction)[0] if old_fraction else p
            else:
                old_fraction, old_output = fraction, after
        if observe is not None:
            observe(p, force, weights, forward(raw))  # After all acceptance decisions.
        with torch.no_grad():
            for live, start, d in zip(params, saved, delta): live.copy_(start+fraction*d)
        error = float((after-p-fraction*velocity).norm())/(fraction*bound) if fraction else 0.
        row = dict(iteration=iteration, fraction=fraction, attempts=attempts,
            objective=value, objective_change=final_value-value, total_objective_change=final_value-value0,
            slope=slope, linearization_error=error, displacement_norm=float((after-p).norm()),
            total_displacement_norm=float((after-initial).norm()), raw_parameter_norm=raw_norm,
            force_refresh_relative=float((current_force-frozen_force).norm()/frozen_force.norm().clamp_min(1e-30)),
            mass_refresh_relative=float((mass-mass0).norm()/mass0.norm()),
            old_first_fraction=old_fraction, solve=solve, plan=plan)
        yield row, initial, p, velocity, after, old_output
