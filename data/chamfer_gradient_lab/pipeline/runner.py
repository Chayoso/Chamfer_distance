"""One fit, diagnostic gradients and text result files."""
import csv
import json
import time
from types import SimpleNamespace
from .model import ROOT, np, torch, official, config
from .losses import components, objective
from .datasets import four_d_match as data


def metrics(pred, source, truth, overlap, sample_ids):
    """An empty evaluation group has no score; encode it as JSON null."""
    err = (pred-truth).norm(dim=1)
    rel = err/(truth-source).norm(dim=1).clamp_min(1e-20)
    held = torch.ones(len(source), device=source.device, dtype=torch.bool)
    held[sample_ids] = False
    result = {}
    for name, mask in [('full',torch.ones_like(held)), ('held',held), ('vis',overlap), ('occ',~overlap)]:
        count = int(mask.sum())
        result[name+'_points'] = count
        values = [err[mask].mean(), ((err[mask]<.025)|(rel[mask]<.025)).float().mean(),
                  ((err[mask]<.05)|(rel[mask]<.05)).float().mean(), (rel[mask]>.3).float().mean()] if count else [None]*4
        result.update({name+'_'+key:float(value) if value is not None else None
                       for key,value in zip(['epe','accs','accr','outlier'],values)})
    return result


def fit(pair, seed, method, weight, folder, *, loader=None, loss_fn=None,
        cfg=None, factory=None):
    """Explicit dependencies for new runs; defaults preserve historical callers."""
    loader = data if loader is None else loader
    loss_fn = objective if loss_fn is None else loss_fn
    cfg = config() if cfg is None else cfg
    factory = official if factory is None else factory
    torch.manual_seed(17000+seed)
    x, y, truth, overlap = loader(pair)
    scale = (x-x.mean(0)).square().sum(1).mean().sqrt()
    dirs = torch.as_tensor(np.random.default_rng(7000+pair).normal(size=(3,64)), device=x.device, dtype=x.dtype)
    dirs = dirs/dirs.norm(dim=0)
    env = factory(); build = env['Deformation_Pyramid']; state = {}; rows = []
    def constructor(**kwargs):
        net = build(**kwargs); state['net'] = net
        gen = torch.Generator().set_state(torch.get_rng_state())
        state['ids'] = torch.randperm(len(x), generator=gen)[:cfg.samples].to(x.device)
        return net
    def loss(p, q, **kwargs):
        p, q = p[0], q[0]
        value = loss_fn(p, q, method, weight, scale, dirs)
        if not torch.isfinite(value): raise RuntimeError('Nonfinite loss')
        level = next(i for i, layer in enumerate(state['net'].pyramid) if next(layer.parameters()).requires_grad)
        step = state.get('step', -1)+1 if state.get('level') == level else 0
        state.update(level=level, step=step)
        state.setdefault('calls_by_level', [0]*cfg.m)[level] += 1
        if step%25 == 0:
            gt = truth[state['ids']]-y.mean(0)
            risk = (p-gt).square().sum(1).mean()
            params = list(state['net'].pyramid[level].parameters())
            g = torch.autograd.grad(value, params, retain_graph=True)
            r = torch.autograd.grad(risk, params, retain_graph=True)
            gn = sum(a.square().sum() for a in g).sqrt()
            rn = sum(a.square().sum() for a in r).sqrt()
            cos = sum((a*b).sum() for a,b in zip(g,r))/(gn*rn).clamp_min(1e-30)
            a,b = components(p.detach(),q.detach())
            rows.append([level,step,float(value),float(a.mean()+b.mean()),float((p-gt).norm(dim=1).mean()),float(risk),float(gn),float(cos)])
        return value
    env.update(Deformation_Pyramid=constructor, compute_truncated_chamfer_distance=loss)
    instance = SimpleNamespace(config=cfg, device=x.device, src_pcd=x, tgt_pcd=y, landmarks=None)
    torch.cuda.synchronize(); start = time.perf_counter()
    pred, _, _ = env['optimize_deformation_pyramid'](instance)
    torch.cuda.synchronize(); seconds = time.perf_counter()-start
    assert torch.isfinite(pred).all()
    result = dict(pair=pair, seed=seed, method=method, weight=weight, seconds=seconds,
        source_points=len(x), target_points=len(y),
        calls_by_level=state['calls_by_level'], final_level=state['level'])
    result.update(metrics(pred, x, truth, overlap, state['ids']))
    serialized = json.dumps(result, indent=2, allow_nan=False)+'\n'
    name = f'p{pair:02d}_s{seed}_{method}_w{weight:g}'
    with (folder/(name+'_trace.txt')).open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['level','step','objective','cd_euclidean','train_epe','train_mse','grad_norm','risk_cosine']);writer.writerows(rows)
    (folder/(name+'.json')).write_text(serialized)
    if pair in [4,16] and seed==0:
        np.savetxt(folder/(name+'_prediction.txt'), pred[::max(1,len(pred)//1000)].detach().cpu().numpy(), fmt='%.8g')
    print(json.dumps(result), flush=True)
    return result
