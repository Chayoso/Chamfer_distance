"""Small Chamfer/coupling experiment. Save scalar TXT results only."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch


def distances(x, y):
    return (x[:, None] - y[None]).square().sum(-1)


def chamfer(x, y):
    d = distances(x, y)
    return d.min(1).values.mean() + d.min(0).values.mean()


def sliced_w2(x, y, directions):
    a, b = (x @ directions).sort(0).values, (y @ directions).sort(0).values
    if len(x) == len(y): return (a-b).square().mean()
    n, m = len(x), len(y)
    cuts = torch.cat([torch.arange(n+1,device=x.device)*m,torch.arange(m+1,device=x.device)*n]).unique(sorted=True)
    mid = (cuts[1:]+cuts[:-1]).double()/2
    return ((a[(mid/m).long()]-b[(mid/n).long()]).square().mean(1)*cuts.diff()/(n*m)).sum()


def density_chamfer(x, y, alpha=40.):
    d = distances(x,y); f, a = d.min(1); r, b = d.min(0)
    wf = (len(x)/len(y))/(torch.bincount(a)[a].to(x.dtype)+1e-6)
    wr = (len(y)/len(x))/(torch.bincount(b)[b].to(x.dtype)+1e-6)
    return ((1-wf*(-alpha*f).exp()).mean()+(1-wr*(-alpha*r).exp()).mean())/2


def deform(x):
    y = x * np.array([1.2,.8,1.])
    y[:,0] += .25*(x[:,2]**2-.64/3)
    y[:,1] += .2*x[:,0]*x[:,2]
    return y + np.array([.15,-.1,.05])


def target_points(name, n, rng, mesh_dir):
    if name == "star":
        angle = np.arange(10) * np.pi / 5
        vertices = np.where(np.arange(10) % 2 == 0, 1., .25)[:, None] * np.c_[np.cos(angle), np.sin(angle)]
        edge = np.roll(vertices, -1, axis=0) - vertices
        return vertices[np.arange(n) % 10] + rng.uniform(size=(n, 1)) * edge[np.arange(n) % 10]
    vertices, faces = [], []
    for line in (mesh_dir / f"{name}.obj").read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == "v":
            vertices.append([float(v) for v in fields[1:4]])
        elif fields and fields[0] == "f":
            ids = [int(v.split('/')[0]) for v in fields[1:]]
            ids = [i - 1 if i > 0 else len(vertices) + i for i in ids]
            faces.extend([[ids[0], ids[i], ids[i+1]] for i in range(1, len(ids)-1)])
    vertices = np.asarray(vertices)
    vertices -= (vertices.min(0) + vertices.max(0)) / 2
    vertices /= np.linalg.norm(vertices, axis=1).max()
    triangles = vertices[np.array(faces)]
    area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)
    triangles = triangles[rng.choice(len(triangles), n, p=area/area.sum())]
    u, v = np.sqrt(rng.uniform(size=(n, 1))), rng.uniform(size=(n, 1))
    return (1-u)*triangles[:, 0] + u*(1-v)*triangles[:, 1] + u*v*triangles[:, 2]


def operators(x, neighbors=8, strength=10., rank=32, hops=8):
    d = distances(x, x)
    d.fill_diagonal_(float('inf'))
    indices = d.topk(min(neighbors, len(x)-1), largest=False).indices
    adjacency = torch.zeros_like(d).scatter_(1, indices, 1)
    adjacency = torch.maximum(adjacency, adjacency.T)
    laplacian = torch.diag(adjacency.sum(1)) - adjacency
    eye = torch.eye(len(x), device=x.device, dtype=x.dtype)
    _, basis = torch.linalg.eigh(laplacian)
    basis = basis[:, :min(rank, len(x))]
    one = eye-laplacian/(2*adjacency.sum(1).max())
    return {"cd": eye, "adam": eye, "repulsion": eye, "dcd": eye, "assignment": eye,
            "translation": torch.ones_like(eye)/len(x), "multi_hop": torch.linalg.matrix_power(one,hops), "one_hop": one,
            "sobolev": torch.linalg.solve(eye+strength*laplacian, eye),
            "spectral": basis @ basis.T, "sw": eye, "cd_sw": eye}, adjacency.bool(), d.min(1).values.sqrt().median()


def objective(x, y, method, directions, repulsion_weight, spacing, sw_weight=1., alpha=40.):
    if method == 'dcd': return density_chamfer(x,y,alpha)
    if method == 'assignment':
        from scipy.optimize import linear_sum_assignment
        if len(x) != len(y): raise ValueError('Exact assignment requires N=M')
        d = distances(x,y)
        a,b = linear_sum_assignment(d.detach().cpu().numpy())
        return d[a,b].mean()
    loss = sliced_w2(x, y, directions) if method == "sw" else chamfer(x, y)
    if method == "cd_sw":
        loss = loss + sw_weight*sliced_w2(x, y, directions)
    if method == "repulsion":
        d = distances(x, x)
        d = d.masked_fill(torch.eye(len(x), device=x.device, dtype=torch.bool), float('inf'))
        local = d.topk(min(8, len(x)-1), largest=False).values
        loss = loss + repulsion_weight * (-local/(8*spacing**2)).exp().mean()
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=['star', 'bunny', 'dragon', 'spot', 'deform'], default='star')
    parser.add_argument('--mesh-dir', type=Path, default=Path('../../assets/meshes'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--methods', default='cd,adam,repulsion,one_hop,multi_hop,sobolev,spectral,dcd,sw,cd_sw,assignment')
    for name, default in [('n',256), ('m',0), ('steps',1000), ('seed',0), ('rank',32), ('projections',64), ('hops',8), ('log-every',50)]:
        parser.add_argument('--'+name, type=int, default=default)
    for name, default in [('lr',.2), ('strength',10.), ('repulsion-weight',.01), ('sw-weight',1.), ('alpha',40.), ('adam-lr',.01)]:
        parser.add_argument('--'+name, type=float, default=default)
    parser.add_argument('--no-norm-match', action='store_true')
    args = parser.parse_args()
    args.m = args.m or args.n
    if not 2 <= args.n <= 2048 or args.steps < 0:
        raise ValueError('This dense pilot supports 2 <= n <= 2048 and nonnegative steps')
    project = Path(__file__).resolve().parent
    if project not in args.output.resolve().parents:
        raise ValueError('Output must be inside this project under data')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    rng = np.random.default_rng(args.seed)
    held_rng = np.random.default_rng(args.seed+10000)
    y = target_points(args.case, args.m, rng, args.mesh_dir) if args.case != 'deform' else np.zeros((args.m,3))
    heldout = target_points(args.case, 4*args.n, held_rng, args.mesh_dir) if args.case != 'deform' else held_rng.normal(size=(4*args.n,3))
    x0 = rng.normal(size=(args.n, y.shape[1]))
    x0 = .8*x0/np.linalg.norm(x0, axis=1, keepdims=True)
    if args.case == 'star':
        theta = (np.arange(args.n)+rng.uniform())*2*np.pi/args.n
        x0 = .8*np.c_[np.cos(theta), np.sin(theta)]
    if args.case == 'deform':
        if args.m != args.n: raise ValueError('Correspondence diagnostic requires N=M')
        y = deform(x0)
        heldout = deform(.8*heldout/np.linalg.norm(heldout,axis=1,keepdims=True))
    tensor = lambda z: torch.tensor(z, dtype=torch.float64, device=args.device)
    x0, y, heldout = map(tensor, [x0,y,heldout])
    directions = tensor(rng.normal(size=(y.shape[1],args.projections)))
    directions /= directions.norm(dim=0)
    eval_directions = tensor(np.random.default_rng(args.seed+20000).normal(size=(y.shape[1],256)))
    eval_directions /= eval_directions.norm(dim=0)
    if x0.is_cuda: torch.cuda.synchronize()
    start = time.perf_counter()
    kernels, edges, spacing = operators(x0, strength=args.strength, rank=args.rank, hops=args.hops)
    if x0.is_cuda: torch.cuda.synchronize()
    setup_seconds = time.perf_counter()-start
    rest_lengths = distances(x0,x0)[edges].sqrt()
    metadata = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    metadata.update(torch=torch.__version__, numpy=np.__version__, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if args.case not in ('star','deform'):
        metadata['mesh_sha256'] = hashlib.sha256((args.mesh_dir/f'{args.case}.obj').read_bytes()).hexdigest()
    (args.output/'settings.txt').write_text(json.dumps(metadata,indent=2))
    rows = []
    for method in args.methods.split(','):
        kernel = kernels[method]
        if sum(p.stat().st_size for p in project.rglob('*') if p.is_file()) > 800_000_000:
            raise RuntimeError('Project exceeded the 800 MB stop threshold; no further runs')
        x = x0.clone().requires_grad_(True)
        path_length = torch.zeros(len(x),device=x.device,dtype=x.dtype)
        elapsed, stalled, evaluations, consecutive = 0., 0, 0, 0
        stop_reason, last_gradient, last_effective = '', float('nan'), float('nan')
        optimizer = torch.optim.Adam([x],lr=args.adam_lr) if method == 'adam' else None
        loss_fn = lambda z: objective(z,y,method,directions,args.repulsion_weight,spacing,args.sw_weight,args.alpha)
        previous_assignments = None
        for step in range(args.steps+1):
            if step % args.log_every == 0 or step == args.steps or stop_reason:
                with torch.no_grad():
                    d, internal = distances(x,y), distances(x,x)
                    forward, reverse = d.min(1).values, d.min(0).values
                    reverse_indices = d.argmin(0)
                    reverse_gradient = torch.zeros_like(x).index_add_(0,reverse_indices,2*(x[reverse_indices]-y)/len(y))
                    internal.fill_diagonal_(float('inf'))
                    lengths = distances(x,x)[edges].sqrt().clamp_min(1e-12)
                    held_d = distances(heldout,x); held = held_d.min(1).values
                    assignments = torch.cat([d.argmin(1),reverse_indices])
                    row = dict(method=method,step=step,cd_rms=float((forward.mean()+reverse.mean()).sqrt()),
                               forward_rms=float(forward.mean().sqrt()),reverse_rms=float(reverse.mean().sqrt()),
                               objective=float(loss_fn(x)),coverage=float((reverse<.05**2).double().mean()),
                               heldout_coverage=float((held<.05**2).double().mean()),
                               heldout_cd_rms=float((held.mean()+held_d.min(0).values.mean()).sqrt()),
                               eval_sw_rms=float(sliced_w2(x,y,eval_directions).sqrt()),
                               correspondence_rms=float((x-y).square().sum(1).mean().sqrt()) if args.case == 'deform' else '',
                               duplicates=float((internal.min(1).values.sqrt()<.1*spacing).double().mean()),
                               near_005=float((internal.min(1).values.sqrt()<.05*spacing).double().mean()),
                               near_020=float((internal.min(1).values.sqrt()<.2*spacing).double().mean()),
                               occupied_targets=float(d.argmin(1).unique().numel()/len(y)),
                               assignment_change='' if previous_assignments is None else float((assignments!=previous_assignments).double().mean()),
                               reverse_active=float((reverse_gradient.norm(dim=1)>1e-10).double().mean()),
                               propagated_active=float(((kernel@reverse_gradient).norm(dim=1)>1e-10).double().mean()),
                               cd_gradient_norm=float((2*(x-y[d.argmin(1)])/len(x)+reverse_gradient).norm()),
                               edge_strain=float((lengths/rest_lengths).log().abs().mean()),
                               path_excess=float((path_length-(x-x0).norm(dim=1)).mean()),seconds=elapsed,stalled=stalled,
                               gradient_norm=last_gradient,effective_norm=last_effective,evaluations=evaluations,
                               setup_seconds=setup_seconds,stop_reason=stop_reason or ('budget' if step == args.steps else ''))
                    rows.append(row)
                    previous_assignments = assignments.clone()
            if step == args.steps or stop_reason:
                break
            if x.is_cuda: torch.cuda.synchronize()
            start = time.perf_counter()
            loss = loss_fn(x)
            evaluations += 1
            gradient = torch.autograd.grad(loss,x)[0]
            if not torch.isfinite(gradient).all(): raise RuntimeError(f'Nonfinite gradient: {method}')
            with torch.no_grad():
                direction = kernel@gradient
                norm = direction.norm()
                last_gradient, last_effective = float(gradient.norm()), float(norm)
                if last_gradient < 1e-10 or last_effective < 1e-12:
                    stop_reason = 'small_gradient' if last_gradient < 1e-10 else 'small_projected_gradient'
                    if x.is_cuda: torch.cuda.synchronize()
                    elapsed += time.perf_counter()-start
                    continue
                if optimizer is not None:
                    optimizer.param_groups[0]['lr'] = args.adam_lr*max(.01,.5*(1+np.cos(np.pi*step/max(args.steps,1))))
                    old = x.clone(); x.grad = gradient; optimizer.step()
                    path_length += (x-old).norm(dim=1)
                    if x.is_cuda: torch.cuda.synchronize()
                    elapsed += time.perf_counter()-start
                    continue
                if norm < 1e-14*gradient.norm(): direction.zero_()
                elif not args.no_norm_match: direction *= gradient.norm()/norm.clamp_min(1e-30)
                direction *= len(x)
                slope, rate, accepted = float((gradient*direction).sum()), args.lr, False
                for _ in range(20):
                    if slope <= 1e-24: break
                    candidate = x-rate*direction
                    evaluations += 1
                    if float(loss_fn(candidate)) <= float(loss.detach())-1e-4*rate*slope:
                        path_length += (candidate-x).norm(dim=1)
                        x.copy_(candidate)
                        accepted = True
                        break
                    rate *= .5
                stalled += int(not accepted)
                consecutive = 0 if accepted else consecutive+1
                if consecutive >= 10: stop_reason = 'line_search_stalled'
            if x.is_cuda: torch.cuda.synchronize()
            elapsed += time.perf_counter()-start
        with (args.output/'metrics.txt').open('w',newline='') as file:
            writer = csv.DictWriter(file,fieldnames=list(rows[0]),delimiter='\t'); writer.writeheader(); writer.writerows(rows)
        print(json.dumps(rows[-1]),flush=True)


if __name__ == '__main__':
    main()
