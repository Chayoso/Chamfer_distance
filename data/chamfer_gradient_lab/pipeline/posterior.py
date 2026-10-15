"""Hard/soft reverse responsibilities; CPD-style fields, not a full EM fit."""
from .model import torch


def correspondence_field(output, target, soft, weight=0., uniform=False):
    if not 0<=weight<1 or (weight and not soft):raise ValueError('Outlier weight requires a soft field and 0<=w<1')
    output,target=output.detach(),target.detach()
    distance=torch.cdist(output,target,compute_mode='donot_use_mm_for_euclid_dist').square()
    closest,indices=distance.min(0)
    variance=(closest.mean()/output.shape[1]).clamp_min(torch.finfo(output.dtype).eps)
    retained=torch.ones(len(target),device=output.device,dtype=output.dtype)
    if soft:
        affinity=-distance/(2*variance)
        log_probability=affinity.log_softmax(0)
        probability=log_probability.exp()
        entropy=-(probability*log_probability).sum(0).mean()
        if weight:
            c=(2*torch.pi*variance)**(output.shape[1]/2)*weight/(1-weight)*len(output)/len(target)
            retained=torch.sigmoid(torch.logsumexp(affinity,0)-c.log())
            if uniform:retained=torch.ones_like(retained)*retained.mean()
            probability=probability*retained[None,:]
        mass=probability.sum(1)
        force=probability@target-mass[:,None]*output
    else:
        mass=torch.bincount(indices,minlength=len(output)).to(output.dtype)
        force=torch.zeros_like(output).index_add_(0,indices,target-output[indices])
        entropy=torch.zeros((),device=output.device,dtype=output.dtype)
    return force,mass,dict(variance=float(variance),entropy=float(entropy),
        mass_error=float(abs(mass.sum()-retained.sum())),max_mass=float(mass.max()),
        retained_fraction=float(retained.mean()),target_weight_std=float(retained.std(unbiased=False)))
