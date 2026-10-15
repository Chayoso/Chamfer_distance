"""Small log-domain KL transport solver; envelope gradients after convergence."""
import numpy as np
import torch
from experiment import distances


def transport(p, q, tau=1., epsilon=.05):
    cost = distances(p,q)
    with torch.no_grad():
        loga, logb = -np.log(len(p)), -np.log(len(q))
        kernel = loga+logb-cost.detach()/epsilon
        u = torch.zeros(len(p),device=p.device,dtype=p.dtype)
        v = torch.zeros(len(q),device=p.device,dtype=p.dtype)
        power = 1. if np.isinf(tau) else tau/(tau+epsilon)
        for iteration in range(100000):
            old = v.clone()
            u = power*(loga-torch.logsumexp(kernel+v[None,:],1))
            v = power*(logb-torch.logsumexp(kernel+u[:,None],0))
            if iteration%20==19 and float((v-old).abs().max())<1e-11: break
        logplan = kernel+u[:,None]+v[None,:]; plan = logplan.exp()
        row, col = plan.sum(1), plan.sum(0)
        entropy = (plan*(logplan-loga-logb)-plan).sum()+1
        if np.isinf(tau):
            residual = max(float((row-np.exp(loga)).abs().max()),float((col-np.exp(logb)).abs().max()))
            penalty = 0.
        else:
            lr, lc = torch.logsumexp(logplan,1)-loga, torch.logsumexp(logplan,0)-logb
            penalty = tau*((row*lr-row).sum()+(col*lc-col).sum()+2)
            residual = float((cost.detach()+epsilon*(logplan-loga-logb)+tau*(lr[:,None]+lc[None,:])).abs().max())
        if residual>1e-7: raise RuntimeError(f'Transport residual {residual}, tau={tau}')
    return (cost*plan).sum()+epsilon*entropy+penalty, dict(residual=residual,iterations=iteration+1,mass=float(plan.sum()))
