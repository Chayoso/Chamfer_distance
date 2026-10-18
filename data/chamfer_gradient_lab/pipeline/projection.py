"""Read-only point-gradient snapshots and paired deformation measurements."""
from .model import torch


def point_snapshot(points,target,loss):
    gradient=torch.autograd.grad(loss,points,retain_graph=True)[0].detach()
    with torch.no_grad():
        distance=torch.cdist(points,target,compute_mode='donot_use_mm_for_euclid_dist')
        assert torch.equal(loss.detach(),distance.min(1).values.mean()+distance.min(0).values.mean())
        return dict(points=points.detach().cpu().clone(),gradient=gradient.cpu().clone(),
            forward=distance.argmin(1).cpu(),reverse=distance.argmin(0).cpu())


def projection_metrics(current,reference,initial):
    delta=current['points'].double()-reference['points'].double()
    squared=initial.square().sum();length=squared.sqrt()
    if not torch.isfinite(squared) or squared<=0:raise ValueError('A nonzero finite initial deviation is required')
    projection=(delta*initial).sum()/squared;norm=delta.norm()
    g,h=current['gradient'].double(),reference['gradient'].double();gn,hn=g.norm(),h.norm()
    return dict(projection=float(projection),norm_gain=float(norm/length),
        deformation_cosine=float((delta*initial).sum()/(norm*length)) if norm>0 else None,
        orthogonal_gain=float((delta-projection*initial).norm()/length),
        gradient_cosine=float((g*h).sum()/(gn*hn)) if gn*hn>0 else None,
        relative_gradient_change=float((g-h).norm()/hn) if hn>0 else None,
        gradient_norm=float(gn),baseline_gradient_norm=float(hn),initial_deviation_norm=float(length),
        forward_changed=float((current['forward']!=reference['forward']).sum())/len(current['forward']),
        reverse_changed=float((current['reverse']!=reference['reverse']).sum())/len(current['reverse']))
