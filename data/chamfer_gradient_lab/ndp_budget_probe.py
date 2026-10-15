"""Native finite trials, force transfer, and same-state time-matched Adam paths."""
import copy
import json
from pathlib import Path
import open3d  # Load before replay construction, as in the preceding feature study.
from pipeline.model import ROOT, np, torch, config
from pipeline.datasets import transfer
from pipeline.runner import fit
from pipeline.controlled import cd
from pipeline.capacity import plan_field, frozen_change
from pipeline.features import feature_penalty
from pipeline.lifting import jacobian_products, ridge_lift
from pipeline.trust import select_step
from pipeline.continuation import tick, clone_adam, transfer_error, adam_path, budget_bracket
from ndp_direction_audit import instrumented

OUT = ROOT/'paper_revision/budget_20261015'
KINDS = ('geo_soft', 'geo_unbalanced', 'feature_soft', 'feature_unbalanced')


def append(name, row):
    with (OUT/(name+'.txt')).open('a') as handle:
        handle.write(json.dumps(row, allow_nan=False)+'\n')


def assignments(output, target):
    distance = torch.cdist(output, target, compute_mode='donot_use_mm_for_euclid_dist')
    return distance.argmin(1), distance.argmin(0)


def switches(output, target, original):
    current = assignments(output, target)
    return [float((a != b).double().mean()) for a, b in zip(current, original)]


def fields(p, q):
    penalty, feature_seconds = None, 0.
    for name in KINDS:
        augmented = name.startswith('feature_')
        if augmented and penalty is None:
            start = tick(p); penalty, _ = feature_penalty(p, q)
            feature_seconds = tick(p)-start
        start = tick(p)
        force, mass, info = plan_field(p, q, name.split('_', 1)[1],
                                       penalty=penalty if augmented else None)
        seconds = tick(p)-start+(feature_seconds if augmented else 0.)
        if not info['converged']:
            raise RuntimeError(f'Transport did not converge: {name}: {info}')
        yield name, force, mass, seconds, info


def native_trial(before, inputs, q, anchor, force, mass, saved, delta, velocity, gp, policy, native):
    start = tick(inputs)
    trial = copy.deepcopy(before); trial._forward_pre_hooks.clear()
    params = list(trial.parameters())
    use_cd = force is None or policy == 'cd_trust'
    objective = (lambda z: cd(z, q)) if use_cd else (
        lambda z: frozen_change(z.double(), anchor, force, mass))
    gradient = gp if use_cd else mass[:, None]*(native-anchor)-force
    slope = float((gradient*velocity).sum()); initial = float(objective(native.to(inputs)))
    cache = {}
    def evaluate(h):
        with torch.no_grad():
            for p, a, d in zip(params, saved, delta):
                p.copy_(a+h*d)
            output = trial(inputs)[0]
            cache[h] = (output.detach().double(), float(objective(output)))
        return cache[h]
    h, attempts = select_step(evaluate, native, velocity, initial, slope)
    output, value = cache[h] if h else (native, initial)
    elapsed = tick(inputs)-start
    error = float((output-native-h*velocity).norm()/(h*velocity.norm())) if h else 0.
    return output, dict(policy=policy, fraction=h, attempts=attempts,
                        objective_change=value-initial, objective_slope=slope,
                        linearization_error=error, controller_seconds=elapsed)


def observe(before, after, inputs, q, gt, history, group, scale, dirs, key, records):
    base = dict(zip(('pair', 'level', 'step'), key))
    # Recompute the actual native update: no access to GT, and charge its cost.
    start = tick(inputs)
    calibration, optimizer = clone_adam(before, history, group)
    native = calibration(inputs)[0]
    initial_loss = cd(native, q)
    optimizer.zero_grad(); initial_loss.backward(); optimizer.step()
    calibration_seconds = tick(inputs)-start
    assert all(torch.equal(a, b) for a, b in zip(calibration.parameters(), after.parameters())), key
    native = native.detach().double(); native_cd = float(initial_loss.detach())
    original_assignment = assignments(native, q.double())
    start = tick(inputs)
    model = copy.deepcopy(before); model._forward_pre_hooks.clear(); model = model.double()
    params = list(model.parameters()); saved = [p.detach().clone() for p in params]
    p = model(inputs.double())[0]; target = q.double()
    value = cd(p, target); gp, = torch.autograd.grad(value, p, retain_graph=True)
    gp = gp.detach(); forward, transpose = jacobian_products(p, params)
    actual = [a.detach().double()-b for a, b in zip(after.parameters(), saved)]
    adam_velocity = forward(actual); speed = adam_velocity.norm()
    setup_seconds = tick(inputs)-start+calibration_seconds
    p = p.detach(); gt = gt.double()
    task = 2*(p-gt)/len(p); initial_risk = float((native-gt).square().sum()/len(p))
    point_grad_norm = float(gp.norm()); parameter_grad_norm = float(sum(g.square().sum() for g in transpose(gp)).sqrt())
    append('states', dict(base, native_double_discrepancy=float((native-p).abs().max()),
                         initial_cd=native_cd, initial_risk=initial_risk, point_grad_norm=point_grad_norm,
                         parameter_grad_norm=parameter_grad_norm, setup_seconds=setup_seconds,
                         calibration_seconds=calibration_seconds, exact_native_first_update=True))
    proposals = [('cd', -gp, None, 0., dict(converged=True, residual=0.))]+list(fields(p, target))
    outcomes = []
    for field, force, mass, field_seconds, transport in proposals:
        for lift in ('gd', 'lift'):
            method = field+'_'+lift; start = tick(inputs)
            if lift == 'gd':
                raw = transpose(force); solver = dict(converged=True, relative_residual=0., iterations=0, damping=0.)
            else:
                raw, solver = ridge_lift(forward, transpose, -force, 81000+key[0]*100+key[1]*10+key[2])
            solve_seconds = tick(inputs)-start
            if not solver['converged']:
                raise RuntimeError(f'Ridge did not converge: {base}, {method}')
            start = tick(inputs)
            raw_velocity = forward(raw); norm = raw_velocity.norm()
            delta = [d*speed/norm for d in raw] if norm > 1e-20 else [torch.zeros_like(d) for d in raw]
            velocity = forward(delta)
            direction_seconds = tick(inputs)-start
            record = dict(base, method=method, field_seconds=field_seconds, solve_seconds=solve_seconds,
                          direction_seconds=direction_seconds, damping=solver['damping'],
                          solver_residual=solver['relative_residual'], transport_residual=transport['residual'],
                          risk_derivative=float((task*velocity).sum()), cd_derivative=float((gp*velocity).sum()),
                          point_risk_derivative=float((task*force).sum()), output_norm=float(velocity.norm()))
            for label, output in [('raw', raw_velocity), ('normalized', velocity)]:
                record.update({label+'_'+k: v for k, v in transfer_error(force, output).items()})
            if mass is not None:
                valid = mass > 0; weight = mass[valid, None].sqrt()
                barycentric = force[valid]/mass[valid, None]
                record.update({'barycentric_'+k: v for k, v in transfer_error(weight*barycentric, weight*velocity[valid]).items()})
            append('transfer', record)
            common_seconds = setup_seconds+field_seconds+solve_seconds+direction_seconds
            for policy in ('trust', 'cd_trust'):
                output, info = native_trial(before, inputs, q, p, None if field == 'cd' else force,
                                            mass, saved, delta, velocity, gp, policy, native)
                displacement = output-native
                risk = float((output-gt).square().sum()/len(p))-initial_risk
                identity = float((2*(native-gt)*displacement+displacement.square()).sum()/len(p))
                row = dict(base, method=method, **info, budget=common_seconds+info['controller_seconds'],
                           cd_change=float(cd(output.to(inputs), q))-native_cd, risk_change=risk,
                           mse_identity_error=abs(risk-identity),
                           finite_ray_error=transfer_error(force, displacement)['ray_error'])
                row['forward_switch'], row['reverse_switch'] = switches(output, target, original_assignment)
                append('finite', row); outcomes.append(row)
    maximum = max(r['budget'] for r in outcomes)
    for repetition in (0, 1):
        path = adam_path(before, inputs, q, history, group, maximum)
        for item in path:
            item['risk_change'] = float((item['output'].double()-gt).square().sum()/len(p))-initial_risk
            append('adam', dict(base, repetition=repetition, update=item['step'], seconds=item['seconds'],
                                cd_change=item['cd']-native_cd, risk_change=item['risk_change']))
        for result in outcomes:
            chosen = budget_bracket(path, result['budget'])
            row = dict(base, method=result['method'], policy=result['policy'], repetition=repetition,
                       budget=result['budget'], within_updates=chosen['last']['step'] if chosen['last'] else 0,
                       within_evaluated=chosen['last'] is not None)
            for label in ('last', 'after', 'best'):
                item = chosen[label]
                if item is not None and 'switches' not in item:
                    item['switches'] = switches(item['output'].double(), target, original_assignment)
                row.update({label+'_'+k: (v if item is not None else None) for k, v in
                            dict(update=item['step'] if item else None, seconds=item['seconds'] if item else None,
                                 cd_change=item['cd']-native_cd if item else None,
                                 risk_change=item['risk_change'] if item else None).items()})
                row[label+'_forward_switch'], row[label+'_reverse_switch'] = item['switches'] if item else (None, None)
            append('comparison', row)
    (OUT/'progress.json').write_text(json.dumps(dict(base, maximum_budget=maximum), indent=2)+'\n')
    records.append(key)


def main():
    folder = OUT/'probe'; folder.mkdir(exist_ok=False)
    cfg = config(); cfg.budget = [50]*9
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32 = False
    replays, records = [], []
    for pair in (100, 101, 124, 125, 200, 201):
        assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) < 800_000_000
        x, y, truth, _ = transfer(pair)
        factory = lambda: instrumented(pair, x, y, truth, None, None, records, observer=observe)
        got = fit(pair, 0, 'official', 0, folder, loader=transfer, cfg=cfg, factory=factory)
        stem = f'p{pair}_s0_official_w0'
        ref = ROOT/'paper_revision'/('replication_4d' if pair < 200 else 'replication_hf')
        old = json.loads((ref/(stem+'.json')).read_text())
        for key, value in old.items():
            if key in got and key != 'seconds':
                assert value == got[key], (pair, key)
        assert np.array_equal(np.loadtxt(ref/(stem+'_trace.txt'), delimiter=',', skiprows=1),
                              np.loadtxt(folder/(stem+'_trace.txt'), delimiter=',', skiprows=1)), pair
        replays.append(pair)
        (OUT/'integrity.json').write_text(json.dumps(dict(exact_replays=replays, completed_states=records), indent=2)+'\n')
    assert len(records) == 54


if __name__ == '__main__':
    main()
