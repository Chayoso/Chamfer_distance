"""Matched one/four-step correspondences in a fixed NDP tangent model."""
import ndp_field_probe as base
from pipeline.joint import joint_path
from pipeline.fields import geometry_scales
from pipeline.capacity import frozen_change


def compare(pair, level, layer, opt, inputs, target, truth):
    torch = base.torch
    before = [v.detach().double().clone() for v in layer.parameters()]
    path = base.continuation(layer, opt, inputs, target, base.cd, steps=1)
    _, native_start, _, _ = next(path)
    _, native_after, _, native_params = next(path)
    adam = [v.detach().double()-start for v, start in zip(native_params, before)]
    native_start, native_after = native_start.detach(), native_after.detach(); path.close()
    model = base.copy.deepcopy(layer).double()
    inputs, q, gt = inputs.double(), target.double(), truth.double()
    p = model(inputs)[0]; params = list(model.parameters())
    loss = base.cd(p, q)
    gradient = torch.autograd.grad(loss, p, retain_graph=True)[0].detach()
    forward, transpose = base.jacobian_products(p, params)
    start = p.detach(); task = 2*(start-gt)/len(start)
    risk = float((start-gt).square().sum()/len(start))
    h, radius = geometry_scales(start, q); bound = len(start)**.5*h
    methods = [('adam', adam, -gradient, lambda z: base.cd(z, q)),
               ('cd_gd', transpose(-gradient), -gradient, lambda z: base.cd(z, q))]
    begin = base.tick(p)
    for field in ('cd', 'uot'):
        for iteration, delta, force, mass, history in joint_path(start, q, forward, transpose, 93000+100*pair+level, field):
            if iteration == 0:
                base.append('inner', dict(pair=pair, level=level, field=field, history=history))
                continue
            if iteration == 1:
                if field == 'cd':
                    torch.testing.assert_close(force, -gradient, rtol=1e-10, atol=1e-12)
                base.append('fields', dict(pair=pair, level=level, method=field, spacing=h, radius=radius,
                    point_derivative=float((task*force).sum()), force_norm=float(force.norm()),
                    point_task_cosine=float(-(task*force).sum()/(task.norm()*force.norm()))))
            objective = (lambda z: base.cd(z, q)) if field == 'cd' else (
                lambda z, f=force, m=mass: frozen_change(z, start, f, m))
            methods.append((field+str(iteration), delta, force, objective))
    seconds = base.tick(p)-begin
    identity_error = 0.
    for name, raw, force, objective in methods:
        fit = base.transfer_error(force, forward(raw))
        delta, velocity = base.scale_direction(raw, forward, bound)
        base.append('directions', dict(pair=pair, level=level, method=name,
            risk_derivative=float((task*velocity).sum()), cd_derivative=float((gradient*velocity).sum()),
            objective_derivative=float((-force*velocity).sum()), output_norm=float(velocity.norm()),
            motion_bound=bound, raw_parameter_norm=float(sum(v.square().sum() for v in raw).sqrt()), transfer=fit))
        for row, output in base.finite_trials(model, inputs, start, q, delta, velocity, objective, -force, gradient):
            displacement = output-start
            change = float((output-gt).square().sum()/len(start))-risk
            identity = float((task*displacement).sum()+displacement.square().sum()/len(start))
            error = abs(change-identity); identity_error = max(identity_error, error)
            base.append('finite', dict(pair=pair, level=level, method=name, **row, risk_change=change,
                mse_identity_error=error, epe_change=float((output-gt).norm(dim=1).mean()-(start-gt).norm(dim=1).mean())))
    base.append('states', dict(pair=pair, level=level, cd=float(loss.detach()), mse=risk, spacing=h,
        mse_identity_error=identity_error, field_seconds=seconds,
        native_adam_cd_change=float(base.cd(native_after, target)-base.cd(native_start, target)),
        native_adam_mse_change=float((native_after-truth).square().sum(1).mean()-(native_start-truth).square().sum(1).mean()),
        precision_output_difference=float((start-native_start.double()).norm()), peak_cuda_bytes=torch.cuda.max_memory_allocated()))
    (base.OUT/'progress.json').write_text(base.json.dumps(dict(pair=pair, level=level))+'\n')


if __name__ == '__main__':
    base.OUT = base.ROOT/'paper_revision/joint_20261018'
    base.compare = compare
    base.main()
