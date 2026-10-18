"""Fixed-J correspondence/parameter alternation. No task labels or selection."""
from .model import torch
from .lifting import cg, weighted_lift
from .capacity import transport_plan


def correspondence(p, q, kind):
    initial = torch.cdist(p, q, compute_mode='donot_use_mm_for_euclid_dist')
    epsilon = float((initial.square().min(0).values.mean()/p.shape[1]).clamp_min(torch.finfo(p.dtype).eps))
    n, m = len(p), len(q)
    def match(z):
        distance = torch.cdist(z, q, compute_mode='donot_use_mm_for_euclid_dist')
        if kind == 'cd':
            a, ai = distance.min(1); b, bi = distance.min(0)
            if min(float(a.min()), float(b.min())) <= 0:
                raise RuntimeError('Zero NN distance in unsquared-CD majorization')
            wa, wb = 1/(n*a), 1/(m*b)
            mass = wa.clone().index_add_(0, bi, wb)
            force = wa[:, None]*(q[ai]-p)
            force.index_add_(0, bi, wb[:, None]*(q-p[bi]))
            return force, mass, float(a.mean()+b.mean()), dict(min_distance=min(float(a.min()), float(b.min())))
        if kind != 'uot':
            raise ValueError('Expected cd or uot')
        cost = .5*distance.square()
        plan, info = transport_plan(cost, epsilon, 'unbalanced')
        if not info['converged']:
            raise RuntimeError(f'Unconverged correspondence: {info}')
        row, col = plan.sum(1), plan.sum(0)
        def kl(value, reference):
            return (torch.xlogy(value, value/reference)-value+reference).sum()
        value = (cost*plan).sum()+epsilon*kl(plan, 1/(n*m))
        value += 2*epsilon*(kl(row, 1/n)+kl(col, 1/m))
        info.update(epsilon=epsilon, total_mass=float(row.sum()))
        return plan@q-row[:, None]*p, row, float(value), info
    return match


def fixed_lift(forward, transpose, force, mass, damping):
    root = mass.sqrt()[:, None]
    rhs = force/root.clamp_min(torch.finfo(force.dtype).tiny)
    if (mass < 0).any() or (force[mass == 0] != 0).any():
        raise ValueError('Invalid mass or unsupported force')
    def kernel(v):
        return root*forward(transpose(root*v))
    dual, info = cg(lambda v: kernel(v)+damping*v, rhs, maxiter=2048)
    return transpose(root*dual), dict(info, damping=damping)


def joint_path(p, q, forward, transpose, seed, kind, steps=4):
    """Total updates from p; fixed regularizer across changing plans/weights."""
    match = correspondence(p, q, kind)
    velocity, previous_penalty = torch.zeros_like(p), 0.
    initial_force = initial_mass = damping = None
    history, previous_upper = [], None
    for iteration in range(1, steps+1):
        force, mass, value, plan = match(p+velocity)
        if iteration == 1:
            initial_force, initial_mass = force.clone(), mass.clone()
            delta, solve = weighted_lift(forward, transpose, force, mass, seed, .0001, 2048)
            damping = solve['damping']
        else:
            delta, solve = fixed_lift(forward, transpose, force, mass, damping)
        if not solve['converged']:
            raise RuntimeError(f'Unconverged tangent solve: {solve}')
        new_velocity = forward(delta)
        penalty = .5*damping*float(sum(d.square().sum() for d in delta))
        change = float((.5*mass[:, None]*(new_velocity.square()-velocity.square())
                        -force*(new_velocity-velocity)).sum())+penalty-previous_penalty
        total = value+previous_penalty
        history.append(dict(iteration=iteration, objective=value, regularized_before=total,
            majorizer_change=change, reassignment_change=None if previous_upper is None else total-previous_upper,
            plan=plan, solve=solve, parameter_penalty=penalty, output_norm=float(new_velocity.norm())))
        previous_upper = total+change
        velocity, previous_penalty = new_velocity, penalty
        if iteration in (1, steps):
            yield iteration, delta, initial_force, initial_mass, list(history)
    _, _, value, info = match(p+velocity)
    history.append(dict(iteration=steps+1, final_objective=value,
                        final_reassignment_change=value+previous_penalty-previous_upper, plan=info))
    yield 0, None, initial_force, initial_mass, history
