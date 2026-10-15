"""Standard entropic marginal controls; exact final residuals, no GT or RNG."""
from .model import np,torch


def refine_balance(logplan,loga,logb,tolerance,maxsteps=512):
    """Damped Newton on column-reduced balanced dual; line search on marginals."""
    n,m=logplan.shape;iterations=0
    identity=torch.eye(n,device=logplan.device,dtype=logplan.dtype)
    for index in range(maxsteps):
        plan=logplan.exp();row=plan.sum(1)
        residual=float((row*np.exp(-loga)-1).abs().max())
        if residual<=tolerance:break
        hessian=torch.diag(row)-m*(plan@plan.T)
        step=-torch.linalg.solve(hessian+(1e-10/n)*identity,row-np.exp(loga))
        step=step-step.mean();step=step*min(1.,10/float(step.abs().max().clamp_min(1e-30)))
        for back in range(21):
            trial=logplan+(2.**(-back))*step[:,None]
            trial=trial-torch.logsumexp(trial,0)[None,:]+logb
            error=float((torch.logsumexp(trial,1)-loga).expm1().abs().max())
            if error<residual:logplan=trial;iterations+=1;break
        else:break
    return logplan,iterations


def transport_plan(cost,epsilon,kind,maxiter=4096,tolerance=1e-8,newton=False):
    if kind not in ('soft','semi','unbalanced','balanced'):raise ValueError(kind)
    if epsilon<=0 or maxiter<1:raise ValueError('Positive epsilon and iteration budget required')
    cost=cost.detach();loga,logb=-np.log(cost.shape[0]),-np.log(cost.shape[1])
    u=cost.new_zeros(cost.shape[0]);v=cost.new_zeros(cost.shape[1]);tau=2*epsilon
    if kind=='soft':
        logplan=(-cost/epsilon).log_softmax(0)+logb
        return logplan.exp(),dict(iterations=0,residual=0.,converged=True,scaling_residual=0.,newton_iterations=0)
    previous=64*epsilon;iterations=0
    for factor in (64.,16.,4.,1.):
        eps=factor*epsilon;u=u*(previous/eps);v=v*(previous/eps);previous=eps
        kernel=loga+logb-cost/eps
        row_power=1. if kind=='balanced' else tau/(tau+eps)
        col_power=tau/(tau+eps) if kind=='unbalanced' else 1.
        for index in range(maxiter if factor==1 else 64):
            u=row_power*(loga-torch.logsumexp(kernel+v[None,:],1))
            v=col_power*(logb-torch.logsumexp(kernel+u[:,None],0));iterations+=1
            if factor==1 and (index%20==19 or index==maxiter-1):
                logplan=kernel+u[:,None]+v[None,:]
                lr=torch.logsumexp(logplan,1)-loga;lc=torch.logsumexp(logplan,0)-logb
                if kind=='balanced':residual=max(float(lr.expm1().abs().max()),float(lc.expm1().abs().max()))
                else:
                    stationarity=u[:,None]+v[None,:]+2*lr[:,None]
                    stationarity=stationarity+2*lc[None,:] if kind=='unbalanced' else stationarity-v[None,:]
                    residual=float(stationarity.abs().max())
                    if kind=='semi':residual=max(residual,float(lc.expm1().abs().max()))
                if residual<=tolerance:break
    scaling_residual=residual;newton_iterations=0
    if kind=='balanced' and newton and residual>tolerance:
        logplan,newton_iterations=refine_balance(logplan,loga,logb,tolerance)
        residual=max(float((torch.logsumexp(logplan,dim)-ref).expm1().abs().max()) for dim,ref in ((1,loga),(0,logb)))
    return logplan.exp(),dict(iterations=iterations,residual=residual,converged=residual<=tolerance,
                             scaling_residual=scaling_residual,newton_iterations=newton_iterations)


def plan_field(p,q,kind,maxiter=4096,newton=False,penalty=None,feature_only=False):
    p,q=p.detach(),q.detach()
    if feature_only:
        if penalty is None:raise ValueError('Feature-only matching requires a feature cost')
        cost=p.new_zeros(len(p),len(q));epsilon=1.
    else:
        cost=.5*torch.cdist(p,q,compute_mode='donot_use_mm_for_euclid_dist').square()
        epsilon=float((2*cost.min(0).values.mean()/p.shape[1]).clamp_min(torch.finfo(p.dtype).eps))
    if penalty is not None:
        assert penalty.shape==cost.shape and torch.isfinite(penalty).all()
        cost=cost+epsilon*penalty.detach()
    plan,info=transport_plan(cost,epsilon,kind,maxiter=maxiter,newton=newton)
    mass=plan.sum(1);force=plan@q-mass[:,None]*p
    info.update(epsilon=epsilon,total_mass=float(mass.sum()),max_source_relative_mass=float(mass.max()*len(p)),
                source_relative_std=float((mass*len(p)).std(unbiased=False)),
                target_relative_std=float((plan.sum(0)*len(q)).std(unbiased=False)))
    return force,mass,info


def frozen_change(z,p,force,mass):
    displacement=z-p
    return .5*(mass[:,None]*displacement.square()).sum()-(force*displacement).sum()
