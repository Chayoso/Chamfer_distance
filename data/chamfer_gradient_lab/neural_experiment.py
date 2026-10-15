"""Coordinate-MLP diagnostic on a fixed public 4DMatch subset; no checkpoints."""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from experiment import distances, chamfer, sliced_w2

ROOT = Path(__file__).resolve().parent
LOSSES = ['cd', 'unsquared', 'trimmed', 'cd_sw']
STEPS = 800


def model(seed, device='cuda:0'):
    torch.manual_seed(9000+seed)
    layers = []
    for i in range(8): layers.extend([torch.nn.Linear(3 if i==0 else 128, 128), torch.nn.ReLU()])
    layers.append(torch.nn.Linear(128, 3))
    network = torch.nn.Sequential(*layers).to(device=device, dtype=torch.float64)
    torch.nn.init.zeros_(network[-1].weight); torch.nn.init.zeros_(network[-1].bias)
    return network


def data(pair):
    with np.load(ROOT/f'assets/4dmatch_subset/{pair:02d}.npz', allow_pickle=False) as entry:
        source, target = entry['s_pc'].astype(float), entry['t_pc'].astype(float)
        truth = (source+entry['s2t_flow'])@entry['rot'].T+entry['trans'].reshape(1, 3)
        overlap = np.zeros(len(source), dtype=bool); overlap[entry['correspondences'][:, 0]] = True
    rng = np.random.default_rng(1000+pair)
    ids = rng.permutation(len(source)); target_ids = rng.permutation(len(target))[:512]
    if len(ids)<1024 or len(target_ids)<512: raise ValueError('Too few points for frozen protocol')
    train, held = ids[:512], ids[512:1536]
    center = source[train].mean(0); scale = np.sqrt(np.mean(np.sum((source[train]-center)**2, axis=1)))
    cast = lambda a: torch.as_tensor(a, dtype=torch.float64, device='cuda:0')
    x, y, h, tx, th = [cast((a-center)/scale) for a in [source[train], target[target_ids], source[held], truth[train], truth[held]]]
    dirs = cast(np.random.default_rng(4000+pair).normal(size=(3, 64))); dirs /= dirs.norm(dim=0)
    return x, y, h, tx, th, torch.as_tensor(overlap[held], device=x.device), dirs, scale, center


def loss(p, y, truth, directions, name):
    if name=='supervised': return (p-truth).square().sum(1).mean()
    if name=='cd_sw': return chamfer(p, y)+sliced_w2(p, y, directions)
    if name=='cd': return chamfer(p, y)
    d = distances(p, y); forward, reverse = d.min(1).values, d.min(0).values
    if name=='unsquared': return forward.clamp_min(1e-24).sqrt().mean()+reverse.clamp_min(1e-24).sqrt().mean()
    if name=='trimmed': return forward.topk(int(np.ceil(.7*len(p))), largest=False).values.mean()+reverse.topk(int(np.ceil(.7*len(y))), largest=False).values.mean()
    raise ValueError(name)


def gradient_probe(net, item, step):
    x, y, held, _, truth_h, *_ = item
    parameters = list(net.parameters())
    cd = chamfer(x+net(x), y)
    g = torch.autograd.grad(cd, parameters)
    risk = (held+net(held)-truth_h).square().sum(1).mean()
    r = torch.autograd.grad(risk, parameters)
    norm = torch.sqrt(sum(v.square().sum() for v in g)); rnorm = torch.sqrt(sum(v.square().sum() for v in r))
    dot = sum((a*b).sum() for a,b in zip(g,r))
    cosine = dot/(norm*rnorm).clamp_min(1e-30)
    predicted = -dot/norm.clamp_min(1e-30)
    saved = [p.detach().clone() for p in parameters]; rows = []
    with torch.no_grad():
        for size in [1e-4, 1e-5, 1e-6]:
            for p, old, v in zip(parameters, saved, g): p.copy_(old-size*v/norm.clamp_min(1e-30))
            new_cd = chamfer(x+net(x), y)
            new_risk = (held+net(held)-truth_h).square().sum(1).mean()
            rows.append([step, size, float(norm), float(rnorm), float(cosine), float(predicted),
                         float((new_risk-risk.detach())/size), float(cd.detach()-new_cd), float(new_risk-risk.detach())])
        for p, old in zip(parameters, saved): p.copy_(old)
    return rows


def fit(item, method, seed, rate, folder, name, visualize=False, probes=False):
    if sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) >= 800_000_000:
        raise RuntimeError('Task storage guard')
    x, y, held, tx, th, overlap, directions, scale, center = item
    net = model(seed); optimizer = torch.optim.Adam(net.parameters(), lr=rate)
    trace, diagnostic, clouds, path, fchanges, rchanges = [], [], [], 0., 0, 0
    torch.cuda.synchronize(); start = time.perf_counter()
    for step in range(STEPS+1):
        if probes and step in [0, 200, 400, STEPS]: diagnostic.extend(gradient_probe(net, item, step))
        with torch.no_grad():
            old = x+net(x); d = distances(old, y); a, b = d.argmin(1), d.argmin(0)
            if step % 20 == 0:
                prediction = held+net(held); error = (prediction-th).norm(dim=1)*scale
                trace.append([step, float(chamfer(old,y).sqrt()*scale), float(error.mean()), float(error.square().mean().sqrt()), path, fchanges, rchanges])
                if visualize and step in [0, 200, 400, STEPS]:
                    arrays = [torch.full((len(held),1),step,device=x.device), held*scale+torch.as_tensor(center,device=x.device), th*scale+torch.as_tensor(center,device=x.device), prediction*scale+torch.as_tensor(center,device=x.device), overlap[:,None]]
                    clouds.append(torch.cat(arrays,1).cpu().numpy())
        if step==STEPS: break
        optimizer.param_groups[0]['lr'] = rate*(.01+.99*(1+np.cos(np.pi*step/(STEPS-1)))/2)
        optimizer.zero_grad(); value = loss(x+net(x), y, tx, directions, method)
        if not torch.isfinite(value): raise RuntimeError(f'Nonfinite objective: {name}, step {step}')
        value.backward(); optimizer.step()
        with torch.no_grad():
            new = x+net(x); dn = distances(new,y)
            path += float((new-old).norm(dim=1).mean()*scale)
            fchanges += int((dn.argmin(1)!=a).sum()); rchanges += int((dn.argmin(0)!=b).sum())
    torch.cuda.synchronize(); seconds = time.perf_counter()-start
    objective = loss(x+net(x), y, tx, directions, method)
    grads = torch.autograd.grad(objective, list(net.parameters()))
    p = (x+net(x)).detach().requires_grad_(); cd = chamfer(p,y); raw, = torch.autograd.grad(cd,p)
    cd_value = chamfer(x+net(x),y); cdg = torch.autograd.grad(cd_value,list(net.parameters()))
    with torch.no_grad():
        error = (held+net(held)-th).norm(dim=1)*scale
        metrics = dict(epe=float(error.mean()), rmse=float(error.square().mean().sqrt()),
                       overlap_epe=float(error[overlap].mean()) if overlap.any() else float('nan'),
                       nonoverlap_epe=float(error[~overlap].mean()) if (~overlap).any() else float('nan'),
                       overlap_count=int(overlap.sum()), held_count=len(held), cd_rms=float(cd.detach().sqrt()*scale),
                       objective=float(objective.detach()), objective_gradient=float(torch.sqrt(sum(g.square().sum() for g in grads))),
                       raw_cd_gradient=float(raw.norm()), parameter_cd_gradient=float(torch.sqrt(sum(g.square().sum() for g in cdg))),
                       path=path, fchanges=fchanges, rchanges=rchanges, seconds=seconds, scale=scale)
    np.savetxt(folder/(name+'_trace.txt'),trace,header='step cd_rms epe rmse path fchanges rchanges',fmt='%.12g')
    if diagnostic: np.savetxt(folder/(name+'_probes.txt'),diagnostic,header='step size cd_g risk_g cosine risk_derivative numeric_risk_derivative cd_drop risk_change',fmt='%.12g')
    if clouds: np.savetxt(folder/(name+'_cloud.txt'),np.concatenate(clouds),header='step source_xyz truth_xyz prediction_xyz overlap',fmt='%.10g')
    return metrics


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('phase',choices=['validation','test']); args = parser.parse_args()
    torch.set_num_threads(2)
    folder = ROOT/'paper_revision'/('neural_'+args.phase); folder.mkdir(exist_ok=False)
    rates = json.loads((ROOT/'paper_revision/neural_rates.txt').read_text()) if args.phase=='test' else None
    with (folder/'metrics.txt').open('w',newline='') as handle:
        writer = None
        for pair in range(4) if rates is None else range(4,28):
            item = data(pair)
            for seed in [0] if rates is None else [0,1]:
                for method in LOSSES if rates is None else LOSSES+['supervised']:
                    for rate in [.001,.003,.008] if rates is None else [rates.get(method,rates['cd'])]:
                        name = f'p{pair:02d}_s{seed}_{method}_{rate:g}'
                        result = fit(item,method,seed,rate,folder,name, rates is not None and pair in [4,16] and seed==0, rates is not None)
                        row = dict(pair=pair,seed=seed,method=method,rate=rate,**result)
                        if writer is None: writer = csv.DictWriter(handle,fieldnames=list(row)); writer.writeheader()
                        writer.writerow(row); handle.flush(); print(name,result['epe'],flush=True)


if __name__=='__main__': main()
