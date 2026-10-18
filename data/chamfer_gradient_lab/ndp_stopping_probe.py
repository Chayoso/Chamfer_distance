"""Baseline-only original stopping, fixed ceiling, and post-stop continuation."""
import json
import open3d  # Preserve the established server import order.
from pipeline.model import ROOT, np, torch, config
from pipeline.datasets import transfer
from pipeline.runner import fit
from pipeline.controlled import cd
from pipeline.stopping import stopping_environment, continuation

OUT = ROOT/'paper_revision/stopping_20261018'
PAIRS = (100, 101, 124, 125, 200, 201)


def append(name, row):
    with (OUT/(name+'.txt')).open('a') as handle:
        handle.write(json.dumps(row, allow_nan=False)+'\n')


def measure(p, q, gt, value, params, detail, previous=None):
    risk = (p-gt).square().sum(1).mean()
    row = dict(cd=float(value.detach()), mse=float(risk.detach()),
               epe=float((p.detach()-gt).norm(dim=1).mean()))
    with torch.no_grad():
        distance = torch.cdist(p, q, compute_mode='donot_use_mm_for_euclid_dist')
        assignment = (distance.argmin(1), distance.argmin(0))
        if previous is not None:
            row['forward_switch'] = float((assignment[0] != previous[0]).float().mean())
            row['reverse_switch'] = float((assignment[1] != previous[1]).float().mean())
    if detail:
        g = torch.autograd.grad(value, params, retain_graph=True)
        r = torch.autograd.grad(risk, params, retain_graph=True)
        gn = sum(a.square().sum() for a in g).sqrt()
        rn = sum(a.square().sum() for a in r).sqrt()
        row.update(grad_norm=float(gn), risk_grad_norm=float(rn),
                   risk_cosine=float(sum((a*b).sum() for a, b in zip(g, r)) /
                                     (gn*rn).clamp_min(1e-30)))
    return row, assignment


def observer(pair, mode, truth):
    previous = {}

    def step(level, iteration, value, p, q, layer, ids, mean):
        row, assignment = measure(p, q, truth[ids]-mean, value,
                                  list(layer.parameters()), iteration % 25 == 0,
                                  previous.get(level))
        previous[level] = assignment
        append('trajectory', dict(pair=pair, mode=mode, level=level,
                                  step=iteration, **row))

    def end(level, iteration, count, loss_prev, value, opt, layer, inputs, q, cached, ids, mean):
        gt = truth[ids]-mean
        if mode == 'ceiling500':
            reason = 'iteration_cap'
        elif float(value) < 1e-4:
            reason = 'absolute_loss'
        elif count >= 15:
            reason = 'cumulative_small_changes'
        else:
            reason = 'iteration_cap'
        updates = max((int(state.get('step', 0)) for state in opt.state.values()), default=0)
        actual = layer(inputs)[0]
        initial, _ = measure(actual, q, gt, cd(actual, q), list(layer.parameters()), True)
        append('stops', dict(pair=pair, mode=mode, level=level, evaluations=iteration+1,
                            updates=updates, reason=reason, small_change_count=count,
                            cached_cd=float(value), cached_mse=float((cached-gt).square().sum(1).mean()),
                            cached_output_lag=float((actual.detach()-cached.detach()).norm()),
                            **initial))
        for scale in (1., .1):
            last = None
            for index, output, loss, params in continuation(layer, opt, inputs, q, cd, lr_scale=scale):
                row, last = measure(output, q, gt, loss, params, index % 25 == 0, last)
                if index == 0:
                    assert row['cd'] == initial['cd'] and row['mse'] == initial['mse']
                append('continuation', dict(pair=pair, mode=mode, level=level,
                                            lr_scale=scale, step=index, **row))
        (OUT/'progress.json').write_text(json.dumps(dict(pair=pair, mode=mode, level=level))+'\n')

    return stopping_environment(step, end, mode == 'ceiling500')[0]


def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert not (OUT/'trajectory.txt').exists(), 'Do not append to an earlier run'
    reference = OUT/'reference'; reference.mkdir(exist_ok=False)
    for mode in ('original', 'ceiling500'):
        (OUT/mode).mkdir(exist_ok=False)
    completed = []
    for pair in PAIRS:
        assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) < 800_000_000
        expected = fit(pair, 0, 'official', 0, reference, loader=transfer)
        _, _, truth, _ = transfer(pair)
        for mode in ('original', 'ceiling500'):
            factory = lambda: observer(pair, mode, truth)
            result = fit(pair, 0, 'official', 0, OUT/mode, loader=transfer, factory=factory)
            if mode == 'original':
                assert all(result[k] == v for k, v in expected.items() if k != 'seconds')
                name = f'p{pair}_s0_official_w0_trace.txt'
                assert np.array_equal(np.loadtxt(reference/name, delimiter=',', skiprows=1),
                                      np.loadtxt(OUT/mode/name, delimiter=',', skiprows=1))
            completed.append([pair, mode])
            (OUT/'integrity.json').write_text(json.dumps(dict(completed=completed,
                exact_original_replays=[p for p, m in completed if m == 'original']), indent=2)+'\n')
    assert len(completed) == 12


if __name__ == '__main__':
    main()
