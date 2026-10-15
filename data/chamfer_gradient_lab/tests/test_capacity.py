"""Analytic plans, an independent constrained optimizer, envelope and MM checks."""
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import softmax,expit
from pipeline.capacity import transport_plan,plan_field,frozen_change,refine_balance
from pipeline.posterior import correspondence_field


def value(cost,plan,epsilon,kind):
    n,m=plan.shape;r,c=plan.sum(1),plan.sum(0)
    kl=lambda x,y:(x*(x/y).log()-x+y).sum()
    total=(cost*plan).sum()+epsilon*kl(plan,1/(n*m))
    if kind in ('semi','unbalanced'):total=total+2*epsilon*kl(r,1/n)
    if kind=='unbalanced':total=total+2*epsilon*kl(c,1/m)
    return float(total)


def test_balanced_plan_has_analytic_two_by_two_solution():
    cost=torch.tensor([[.1,.7],[.6,.3]],dtype=torch.float64);eps=.2
    plan,info=transport_plan(cost,eps,'balanced')
    x=.5*expit(-float(cost[0,0]+cost[1,1]-cost[0,1]-cost[1,0])/(2*eps))
    torch.testing.assert_close(plan,plan.new_tensor([[x,.5-x],[.5-x,x]]),rtol=1e-7,atol=1e-9)
    assert info['converged'] and info['residual']<=1e-8


def test_scalar_unbalanced_mass_and_semi_conservation():
    cost=torch.tensor([[.7]],dtype=torch.float64);eps=.2
    plan,info=transport_plan(cost,eps,'unbalanced')
    assert info['converged'] and abs(float(plan)-np.exp(-.7/(5*eps)))<1e-9
    plan,info=transport_plan(cost,eps,'semi')
    assert info['converged'] and abs(float(plan)-1)<1e-12


def test_semi_plan_matches_independent_column_simplex_optimization():
    cost=np.array([[.1,.9,.6],[.4,.2,.5],[.8,.3,.1],[.9,.7,.4]])
    n,m=cost.shape;eps=.2;tau=2*eps
    def objective(flat):
        p=softmax(flat.reshape(n,m),axis=0)/m;r=p.sum(1)
        f=np.sum(cost*p)+eps*np.sum(p*np.log(p*n*m)-p+1/(n*m))+tau*np.sum(r*np.log(r*n)-r+1/n)
        gradient=cost+eps*np.log(p*n*m)+tau*np.log(r*n)[:,None]
        derivative=p*(gradient-m*(p*gradient).sum(0))
        return f,derivative.ravel()
    fit=minimize(objective,np.zeros(n*m),jac=True,method='L-BFGS-B',options={'gtol':1e-11,'ftol':1e-15,'maxiter':1000})
    assert fit.success
    plan,info=transport_plan(torch.tensor(cost),eps,'semi')
    assert info['converged']
    np.testing.assert_allclose(plan.numpy(),softmax(fit.x.reshape(n,m),axis=0)/m,rtol=1e-6,atol=1e-8)


def test_resolved_envelope_and_weighted_majorizer_at_fixed_epsilon():
    generator=torch.Generator().manual_seed(72)
    p=.3*torch.randn(4,2,generator=generator,dtype=torch.float64)
    q=.3*torch.randn(5,2,generator=generator,dtype=p.dtype)
    direction=torch.randn(p.shape,generator=generator,dtype=p.dtype);direction/=direction.norm();eps=.2
    cost=lambda x:.5*torch.cdist(x,q).square()
    for kind in ('soft','semi','unbalanced','balanced'):
        plan,info=transport_plan(cost(p),eps,kind,tolerance=1e-11)
        assert info['converged']
        mass=plan.sum(1);force=plan@q-mass[:,None]*p
        resolved=lambda z:value(cost(z),transport_plan(cost(z),eps,kind,tolerance=1e-11)[0],eps,kind)
        fd=(resolved(p+1e-5*direction)-resolved(p-1e-5*direction))/2e-5
        assert abs(fd-float((-force*direction).sum()))<1e-7
        z=p+.1*direction
        change=float(frozen_change(z,p,force,mass))
        assert abs(change-float(((cost(z)-cost(p))*plan).sum()))<1e-13
        assert resolved(z)-resolved(p)<=change+1e-10


def test_soft_control_matches_existing_responsibilities_and_scaling():
    p=torch.tensor([[0.,0.],[1.,.3],[.4,.9]],dtype=torch.float64)
    q=torch.tensor([[.2,.1],[.7,.8]],dtype=p.dtype)
    force,mass,info=plan_field(p,q,'soft');old,counts,details=correspondence_field(p,q,True)
    torch.testing.assert_close(force,old/len(q));torch.testing.assert_close(mass,counts/len(q))
    assert info['epsilon']==details['variance']
    for kind in ('soft','semi','unbalanced','balanced'):
        a,r,i=plan_field(p,q,kind);b,s,j=plan_field(3*p,3*q,kind)
        torch.testing.assert_close(b,3*a,rtol=1e-7,atol=1e-9);torch.testing.assert_close(r,s,rtol=1e-7,atol=1e-9)
        assert i['converged'] and j['converged']


def test_iteration_limit_records_nonconvergence():
    gen=torch.Generator().manual_seed(73)
    cost=torch.rand(10,13,generator=gen,dtype=torch.float64)
    _,info=transport_plan(cost,.001,'balanced',maxiter=1)
    assert not info['converged'] and info['iterations']==193


def test_balanced_plan_translation_invariance_and_centroid_quadratic():
    gen=torch.Generator().manual_seed(74)
    p=.2*torch.randn(3,2,generator=gen,dtype=torch.float64);q=.2*torch.randn(4,2,generator=gen,dtype=p.dtype)
    z=p.new_tensor([.15,-.23]);epsilon=.1
    costs=[.5*torch.cdist(x,q).square() for x in (p,p+z)]
    plans=[transport_plan(c,epsilon,'balanced',tolerance=1e-11)[0] for c in costs]
    torch.testing.assert_close(plans[0],plans[1],rtol=1e-9,atol=1e-11)
    change=value(costs[1],plans[1],epsilon,'balanced')-value(costs[0],plans[0],epsilon,'balanced')
    expected=float(z@(p.mean(0)-q.mean(0))+.5*z.square().sum())
    assert abs(change-expected)<1e-10


def test_cd_translation_plateau_blocks_a_correct_global_step():
    from pipeline.controlled import cd
    from pipeline.trust import select_step
    x=torch.tensor([[0.],[2.]],dtype=torch.float64);q=x+1
    z=x.new_tensor(-.5,requires_grad=True);p=x+z
    loss=cd(p,q);slope_cd,=torch.autograd.grad(loss,z)
    assert float(loss.detach())==2 and float(slope_cd)==0
    plan,info=transport_plan(.5*torch.cdist(p.detach(),q).square(),.2,'balanced',tolerance=1e-12)
    assert info['converged']
    mass=plan.sum(1);force=plan@q-mass[:,None]*p.detach();velocity=torch.ones_like(x)*1.5
    evaluate=lambda h:(p.detach()+h*velocity,float(frozen_change(p.detach()+h*velocity,p.detach(),force,mass)))
    h,_=select_step(evaluate,p.detach(),velocity,0.,float((-force*velocity).sum()))
    common=lambda h:(p.detach()+h*velocity,float(cd(p.detach()+h*velocity,q)))
    cd_h,_=select_step(common,p.detach(),velocity,2.,float(slope_cd))
    assert h==1 and cd_h==0 and (evaluate(h)[0]-q).square().sum()==0


def test_single_target_relaxations_share_conditional_bias_and_weighted_controller():
    p=torch.tensor([[0.,0.,0.],[3.,0.,0.]],dtype=torch.float64);q=p.new_tensor([[1.,0.,0.]])
    semi,r,a=plan_field(p,q,'semi');uot,s,b=plan_field(p,q,'unbalanced')
    total=b['total_mass'];share=expit(1.5)
    assert abs(float(r[0])-share)<1e-8 and float(semi.sum(0)[0])>0
    torch.testing.assert_close(uot,total*semi,rtol=1e-8,atol=1e-10)
    torch.testing.assert_close(s,total*r,rtol=1e-8,atol=1e-10)
    target=p+p.new_tensor([.2,0.,0.])
    torch.testing.assert_close(frozen_change(target,p,uot,s),total*frozen_change(target,p,semi,r),rtol=1e-8,atol=1e-10)


def test_balanced_reduced_hessian_and_newton_optimum():
    gen=torch.Generator().manual_seed(75);cost=torch.rand(5,7,generator=gen,dtype=torch.float64);eps=.2
    loga,logb=-np.log(5),-np.log(7);u=torch.randn(5,generator=gen,dtype=cost.dtype)*.1
    logits=loga-cost/eps+u[:,None];logplan=logits.log_softmax(0)+logb;plan=logplan.exp()
    dual=lambda v:(torch.logsumexp(loga-cost/eps+v[:,None],0)).mean()-v.mean()
    explicit=torch.diag(plan.sum(1))-7*plan@plan.T
    torch.testing.assert_close(explicit,torch.autograd.functional.hessian(dual,u),rtol=1e-11,atol=1e-12)
    result,iterations=refine_balance(logplan,loga,logb,1e-10)
    reference,info=transport_plan(cost,eps,'balanced',tolerance=1e-12)
    assert iterations>0 and info['converged']
    torch.testing.assert_close(result.exp(),reference,rtol=1e-7,atol=1e-10)


def test_newton_resolves_an_insufficient_scaling_budget():
    gen=torch.Generator().manual_seed(76);cost=torch.rand(10,13,generator=gen,dtype=torch.float64)
    _,initial=transport_plan(cost,.01,'balanced',maxiter=1)
    plan,info=transport_plan(cost,.01,'balanced',maxiter=1,newton=True)
    reference,baseline=transport_plan(cost,.01,'balanced')
    assert not initial['converged'] and info['converged'] and baseline['converged'] and info['newton_iterations']>0
    torch.testing.assert_close(plan,reference,rtol=1e-5,atol=1e-9)
