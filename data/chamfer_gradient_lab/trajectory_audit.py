"""Fresh trajectories; reuse frozen losses/operators, write small tab-separated files."""
import csv
from pathlib import Path
import numpy as np
import torch
from experiment import chamfer, distances, target_points, deform, operators, sliced_w2

ROOT = Path(__file__).resolve().parent


def components(x, y):
    d = distances(x, y)
    a, b = d.argmin(1), d.argmin(0)
    gf = 2*(x-y[a])/len(x)
    gr = torch.zeros_like(x).index_add_(0, b, 2*(x[b]-y)/len(y))
    return gf, gr, a, b


def first_boundary(x, y, a, velocity):
    """Exact first forward assignment boundary on a unit straight ray (not flow)."""
    if velocity.norm() < 1e-14:
        return float('inf')
    velocity = velocity/velocity.norm()
    margin = distances(x, y)-((x-y[a])**2).sum(1)[:, None]
    slope = 2*(velocity[:, None]*(y[a, None]-y[None])).sum(-1)
    eligible = (slope < -1e-14) & (torch.arange(len(y), device=x.device)[None] != a[:, None])
    radii = torch.where(eligible, margin.clamp_min(0)/(-slope).clamp_min(1e-30), float('inf'))
    return float(radii.min())


def save(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0], delimiter='\t')
        writer.writeheader(); writer.writerows(rows)


def main():
    torch.set_num_threads(2)
    out = ROOT/'paper_revision'/'trajectories'
    out.mkdir(exist_ok=False)
    for case in ['star', 'bunny', 'deform']:
        for seed in range(200, 205):
            rng = np.random.default_rng(seed); held_rng = np.random.default_rng(seed+10000)
            y = target_points(case, 256, rng, ROOT/'assets'/'meshes') if case != 'deform' else np.zeros((256, 3))
            held = target_points(case, 1024, held_rng, ROOT/'assets'/'meshes') if case != 'deform' else held_rng.normal(size=(1024, 3))
            x0 = rng.normal(size=y.shape); x0 = .8*x0/np.linalg.norm(x0, axis=1, keepdims=True)
            if case == 'star':
                theta = (np.arange(256)+rng.uniform())*2*np.pi/256
                x0 = .8*np.c_[np.cos(theta), np.sin(theta)]
            if case == 'deform':
                y = deform(x0); held = deform(.8*held/np.linalg.norm(held, axis=1, keepdims=True))
            tensor = lambda z: torch.tensor(z, dtype=torch.float64, device='cuda:0')
            x0, y, held = map(tensor, [x0, y, held])
            directions = tensor(rng.normal(size=(y.shape[1], 64))); directions /= directions.norm(dim=0)
            kernels, _, _ = operators(x0)
            if case == 'star' and seed == 200:
                np.savetxt(out/'star_200_target.txt', y.cpu().numpy(), fmt='%.12g')
            for method in ['cd', 'adam', 'sobolev', 'sw']:
                assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) < 800_000_000
                x = x0.clone().requires_grad_(True)
                opt = torch.optim.Adam([x], lr=.05) if method == 'adam' else None
                loss_fn = (lambda z: sliced_w2(z, y, directions)) if method == 'sw' else (lambda z: chamfer(z, y))
                rows, points = [], []
                changes_f = changes_r = stalled = 0
                path = torch.zeros(len(x), device=x.device, dtype=x.dtype)
                last_cos = last_length = float('nan'); reason = ''
                for step in range(1001):
                    loss = loss_fn(x); g = torch.autograd.grad(loss, x)[0]
                    if g.norm() < 1e-10: reason = 'small_gradient'
                    with torch.no_grad():
                        gf, gr, a, b = components(x, y); gcd = gf+gr
                        if step % 10 == 0 or step == 1000 or reason:
                            denom = float(gf.norm()+gr.norm())
                            local_denom = gf.norm(dim=1)+gr.norm(dim=1)
                            active = local_denom > 1e-12
                            rows.append(dict(step=step, cd=float(chamfer(x, y).sqrt()),
                                held_cd=float(chamfer(x, held).sqrt()), gradient=float(gcd.norm()),
                                gf=float(gf.norm()), gr=float(gr.norm()), cancellation=float(gcd.norm())/max(denom, 1e-30),
                                opposing_cos=float((gf*gr).sum()/(gf.norm()*gr.norm()).clamp_min(1e-30)),
                                cancel_fraction=float((gcd.norm(dim=1)[active]/local_denom[active] < .05).double().mean()) if active.any() else float('nan'),
                                first_cd=first_boundary(x, y, a, -gcd), first_sobolev=first_boundary(x, y, a, -kernels['sobolev']@gcd),
                                forward_changes=changes_f, reverse_changes=changes_r, path=float(path.mean()),
                                excess=float((path-(x-x0).norm(dim=1)).mean()), last_cos_cd=last_cos, last_step_length=last_length,
                                correspondence=float((x-y).square().sum(1).mean().sqrt()) if case == 'deform' else '',
                                reason=reason or ('budget' if step == 1000 else '')))
                            if case == 'star' and seed == 200:
                                points.append(np.c_[np.full(len(x), step), np.arange(len(x)), x.cpu().numpy()])
                        if step == 1000 or reason: break
                        old = x.clone()
                        if opt is not None:
                            opt.param_groups[0]['lr'] = .05*max(.01, .5*(1+np.cos(np.pi*step/1000)))
                            x.grad = g; opt.step()
                        else:
                            direction = kernels[method]@g
                            direction *= len(x)*g.norm()/direction.norm().clamp_min(1e-30)
                            slope = float((g*direction).sum()); rate = .2; accepted = False
                            for _ in range(20):
                                if slope <= 1e-24: break
                                candidate = x-rate*direction
                                if loss_fn(candidate) <= loss.detach()-1e-4*rate*slope:
                                    x.copy_(candidate); accepted = True; break
                                rate *= .5
                            stalled = 0 if accepted else stalled+1
                            if stalled >= 10: reason = 'line_search_stalled'
                        delta = x-old; path += delta.norm(dim=1)
                        last_length = float(delta.norm())
                        last_cos = float(-(delta*gcd).sum()/(delta.norm()*gcd.norm()).clamp_min(1e-30))
                        dnew = distances(x, y)
                        changes_f += int((dnew.argmin(1) != a).sum())
                        changes_r += int((dnew.argmin(0) != b).sum())
                save(out/f'{case}_{seed}_{method}.txt', rows)
                if points: np.savetxt(out/f'{case}_{seed}_{method}_points.txt', np.concatenate(points), fmt='%.12g')
                print(case, seed, method, rows[-1], flush=True)


if __name__ == '__main__':
    main()
