"""Finite acceptance conditions, not ground-truth improvement, are guaranteed."""
import numpy as np
import torch
from pipeline.trust import select_step


def test_linear_and_nonlinear_backtracking():
    p=torch.tensor([1.,0.],dtype=torch.float64);v=torch.tensor([-1.,0.],dtype=p.dtype)
    def linear(h):
        q=p+h*v
        return q,float(q.square().sum()/2)
    assert select_step(linear,p,v,.5,-1.) == (1.,1)
    def nonlinear(h):
        q=p+h*v-2*h*h*v
        return q,float(q.square().sum()/2)
    h,_=select_step(nonlinear,p,v,.5,-1.)
    assert h==1/32
    q,c=nonlinear(h)
    assert c<=.5-.1*h and (q-p-h*v).norm()<=.1*h*v.norm()
    # The unobserved task can still become worse under a valid observed-loss step.
    truth=torch.tensor([2.,0.],dtype=p.dtype)
    assert (q-truth).square().sum() > (p-truth).square().sum()


def test_rejection_and_identical_controls():
    p=torch.zeros(3,dtype=torch.float64);v=torch.ones_like(p)
    calls=[]
    def evaluate(h):
        calls.append(h)
        return p+h*v,1.
    assert select_step(evaluate,p,v,1.,1.)==(0.,0) and not calls
    assert select_step(evaluate,p,v,1.,-1.,cap=.25,max_halvings=4)==(0.,3)
    assert calls==[.25,.125,.0625]


def test_units_do_not_change_backtracking():
    results=[]
    for units in (.01,1.,100.):
        p=torch.tensor([1.,2.],dtype=torch.float64)*units
        v=-p/2
        def evaluate(h):
            q=p+h*v+2*h*h*v
            return q,float(q.square().sum())
        results.append(select_step(evaluate,p,v,float(p.square().sum()),float(2*(p*v).sum())))
    assert results[0]==results[1]==results[2]
