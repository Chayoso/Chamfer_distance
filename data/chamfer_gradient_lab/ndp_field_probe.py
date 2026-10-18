"""Frozen field comparison at all original stopping endpoints; clones only."""
import copy
import json
import open3d  # Preserve the established server import order.
from pipeline.model import ROOT, np, torch
from pipeline.datasets import transfer
from pipeline.runner import fit
from pipeline.controlled import cd
from pipeline.stopping import stopping_environment, continuation
from pipeline.lifting import jacobian_products, ridge_lift
from pipeline.continuation import transfer_error, tick
from pipeline.fields import build_fields
from pipeline.field_steps import scale_direction, finite_trials

OUT = ROOT/'paper_revision/fields_20261018'
PAIRS = (100, 101, 124, 125, 200, 201)


def append(name, row):
    with (OUT/(name+'.txt')).open('a') as handle:
        handle.write(json.dumps(row, allow_nan=False)+'\n')


def compare(pair, level, layer, opt, inputs, target, truth):
    # Native Adam reference is fixed before any GT evaluation or proposal.
    before = [v.detach().double().clone() for v in layer.parameters()]
    path = continuation(layer, opt, inputs, target, cd, steps=1)
    _, native_start, _, _ = next(path)
    _, native_after, _, native_params = next(path)
    adam = [v.detach().double()-start for v, start in zip(native_params, before)]
    native_start, native_after = native_start.detach(), native_after.detach()
    path.close()
    model = copy.deepcopy(layer).double()
    inputs, q, gt = inputs.double(), target.double(), truth.double()
    params = list(model.parameters())
    p = model(inputs)[0]
    loss = cd(p, q)
    gradient = torch.autograd.grad(loss, p, retain_graph=True)[0].detach()
    forward, transpose = jacobian_products(p, params)
    start = p.detach()
    task = 2*(start-gt)/len(start)
    risk = float((start-gt).square().sum()/len(start))
    seed = 93000+100*pair+level
    methods, fields = [], []
    begin = tick(p)
    for name, force, objective, info in build_fields(start, q, seed):
        velocity = force/force.norm().clamp_min(1e-30)
        fields.append(dict(pair=pair, level=level, method=name, **info,
                           point_derivative=float((task*force).sum()), force_norm=float(force.norm()),
                           point_task_cosine=float(-(task*velocity).sum()/task.norm().clamp_min(1e-30))))
        gd = transpose(force)
        delta, solver = ridge_lift(forward, transpose, -force, seed, maxiter=2048)
        append('solvers', dict(pair=pair, level=level, method=name, **solver))
        if not solver['converged']:
            raise RuntimeError(f'Unconverged ridge: {(pair, level, name, solver)}')
        methods.extend([(name+'_gd', gd, force, objective), (name+'_ridge', delta, force, objective)])
    field_seconds = tick(p)-begin
    h = fields[0]['spacing']
    motion_bound = len(start)**.5*h
    methods.append(('adam', adam, -gradient, lambda z: cd(z, q)))
    identity_error = 0.
    for name, raw, force, objective in methods:
        raw_velocity = forward(raw)
        error = transfer_error(force, raw_velocity)
        delta, velocity = scale_direction(raw, forward, motion_bound)
        append('directions', dict(pair=pair, level=level, method=name,
            risk_derivative=float((task*velocity).sum()), cd_derivative=float((gradient*velocity).sum()),
            objective_derivative=float((-force*velocity).sum()), output_norm=float(velocity.norm()),
            motion_bound=motion_bound, raw_parameter_norm=float(sum(v.square().sum() for v in raw).sqrt()),
            transfer=error))
        # The GT-free controller completes each choice before evaluating its task effect.
        for row, output in finite_trials(model, inputs, start, q, delta, velocity, objective, -force, gradient):
            displacement = output-start
            change = float((output-gt).square().sum()/len(start))-risk
            identity = float((task*displacement).sum()+displacement.square().sum()/len(start))
            identity_error = max(identity_error, abs(change-identity))
            append('finite', dict(pair=pair, level=level, method=name, **row, risk_change=change,
                                 epe_change=float((output-gt).norm(dim=1).mean()-(start-gt).norm(dim=1).mean())))
    for row in fields:
        append('fields', row)
    append('states', dict(pair=pair, level=level, cd=float(loss.detach()), mse=risk, spacing=h,
        mse_identity_error=identity_error, field_seconds=field_seconds,
        native_adam_cd_change=float(cd(native_after, target)-cd(native_start, target)),
        native_adam_mse_change=float((native_after-truth).square().sum(1).mean()-(native_start-truth).square().sum(1).mean()),
        precision_output_difference=float((start-native_start.double()).norm()),
        peak_cuda_bytes=torch.cuda.max_memory_allocated()))
    (OUT/'progress.json').write_text(json.dumps(dict(pair=pair, level=level))+'\n')


def environment(pair, truth):
    def end(level, iteration, count, previous, value, opt, layer, inputs, q, cached, ids, mean):
        compare(pair, level, layer, opt, inputs, q, truth[ids]-mean)
    return stopping_environment(lambda *args: None, end)[0]


def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    folder = OUT/'replays'
    folder.mkdir(exist_ok=False)
    assert not (OUT/'directions.txt').exists()
    completed = []
    for pair in PAIRS:
        assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) < 800_000_000
        torch.cuda.reset_peak_memory_stats()
        _, _, truth, _ = transfer(pair)
        result = fit(pair, 0, 'official', 0, folder, loader=transfer, factory=lambda: environment(pair, truth))
        stem = f'p{pair}_s0_official_w0'
        reference = ROOT/'paper_revision/stopping_20261018/reference'
        expected = json.loads((reference/(stem+'.json')).read_text())
        assert all(result[k] == v for k, v in expected.items() if k != 'seconds')
        assert np.array_equal(np.loadtxt(reference/(stem+'_trace.txt'), delimiter=',', skiprows=1),
                              np.loadtxt(folder/(stem+'_trace.txt'), delimiter=',', skiprows=1))
        completed.append(pair)
        (OUT/'integrity.json').write_text(json.dumps(dict(exact_replays=completed), indent=2)+'\n')
    assert len(completed) == 6


if __name__ == '__main__':
    main()
