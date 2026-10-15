"""Independent finite-branch and perturbation checks for theory_v2.txt."""
from itertools import product, permutations
import numpy as np


def branch(p,q,w=1,a=None,b=None):
    n,m=len(p),len(q); d=((p[:,None]-q)**2).sum(-1)
    a=d.argmin(1) if a is None else np.array(a)
    b=d.argmin(0) if b is None else np.array(b)
    value=d[np.arange(n),a].mean()+w*d[b,np.arange(m)].mean()
    g=2*(p-q[a])/n
    np.add.at(g,b,2*w*(p[b]-q)/m)
    h=2/n+2*w*np.bincount(b,minlength=n)/m
    return value,g,h


def all_active(p,q,w=1):
    loss=branch(p,q,w)[0]; gradients=[]
    for a in product(range(len(q)),repeat=len(p)):
        for b in product(range(len(p)),repeat=len(q)):
            value,g,_=branch(p,q,w,a,b)
            if abs(value-loss)<1e-13: gradients.append(g)
    return np.array(gradients)


def test_enumerate_all_729_branches_true_minimum_vs_tie_stall():
    p=np.array([[-1.],[-1.],[5/6]]); q=np.array([[-1.],[.5],[1.5]])
    active=all_active(p,q)
    assert len(active)==2 and np.max(np.abs(active))<1e-14
    p=np.zeros((3,1)); q=np.arange(-1.,2.)[:,None]
    active=all_active(p,q)
    assert len(active)==27 and np.linalg.norm(branch(p,q)[1])==0
    assert np.max(np.linalg.norm(active,axis=(1,2)))>0
    s=.01
    assert abs(branch(np.array([[-s],[0],[s]]),q)[0]-(2/3-4*s/3+4*s*s/3))<1e-14


def test_open_weighted_family_in_3d_and_exact_growth():
    rng=np.random.default_rng(104)
    for w in [.2,1.,5.]:
        for _ in range(100):
            q=np.array([[-1.,0,0],[.5,0,0],[1.5,0,0]])+rng.uniform(-.015,.015,(3,3))
            a,b,c=q; t=w/(1+2*w); z=b+t*(c-b); D=np.linalg.norm(c-b)
            m=min(np.linalg.norm(a-b),np.linalg.norm(a-c),(1-2*t)*D,
                  np.linalg.norm(z-a)-t*D,np.linalg.norm(b-a)-t*D,
                  np.linalg.norm(c-a)-(1-t)*D,np.linalg.norm(z-a))
            assert m>0
            p=np.array([a,a,z]); delta=rng.normal(size=(3,3))
            delta*=m/(8*np.linalg.norm(delta,axis=1).max())
            expected=((delta[:2]**2).sum()+w*(delta[:2]**2).sum(1).min()
                      +(1+2*w)*(delta[2]**2).sum())/3
            optimum=w*(1+w)*D**2/(3*(1+2*w))
            assert abs(branch(p,q,w)[0]-optimum)<1e-14
            assert abs(branch(p+delta,q,w)[0]-optimum-expected)<1e-14
            assert np.linalg.norm(branch(p,q,w)[1])<1e-14


def test_global_quadratic_majorizer_including_ties_and_assignment_switches():
    rng=np.random.default_rng(105)
    for tied in [False,True]:
        for _ in range(40):
            p=rng.normal(size=(3,2)); q=rng.normal(size=(4,2))
            if tied: p[:]=p[0]
            value,g,h=branch(p,q); A=rng.normal(size=(3,3)); K=A@A.T+.1*np.eye(3)
            d=K@g; beta=2/3+2; stable=1/(beta*np.linalg.eigvalsh(K)[-1])
            for alpha in [stable,10*stable]:
                upper=value-alpha*(g*d).sum()+alpha**2*(h[:,None]*d*d).sum()/2
                assert branch(p-alpha*d,q)[0]<=upper+1e-12
            assert branch(p-stable*d,q)[0]<value


def test_32_separated_minima_and_their_local_growth():
    rng=np.random.default_rng(106); q=np.concatenate([np.array([-1.,.5,1.5])+10*j for j in range(5)])[:,None]
    for choices in product([0,1],repeat=5):
        p=np.concatenate([np.array([-1.,-1.,5/6] if bad else [-1.,.5,1.5])+10*j
                          for j,bad in enumerate(choices)])[:,None]
        value,g,_=branch(p,q)
        assert abs(value-sum(choices)*2/45)<1e-13 and np.linalg.norm(g)<1e-13
        delta=rng.uniform(-.01,.01,p.shape)
        assert branch(p+delta,q)[0]>=value+(delta**2).sum()/15-1e-13


def test_spd_local_contraction_bound_for_dense_coordinate_coupling():
    rng=np.random.default_rng(107); pstar=np.array([[-1.,0],[-1.,0],[5/6,0]])
    q=np.array([[-1.,0],[.5,0],[1.5,0]])
    for _ in range(30):
        A=rng.normal(size=(6,6)); K=A@A.T+np.eye(6); lam,U=np.linalg.eigh(K)
        Kroot=(U*np.sqrt(lam))@U.T; Kinvroot=(U/np.sqrt(lam))@U.T
        alpha=1/((2/3+2)*lam[-1]); bound=1-alpha*(2/3)*lam[0]
        u=rng.normal(size=6); u*=.01/(np.linalg.norm(u)*np.sqrt(lam[-1]))
        p=pstar+(Kroot@u).reshape(3,2)
        for _ in range(60):
            g=branch(p,q)[1].ravel(); p-=alpha*(K@g).reshape(3,2)
            next_u=Kinvroot@(p-pstar).ravel()
            assert np.linalg.norm(next_u)<=bound*np.linalg.norm(u)+1e-14
            u=next_u


def test_transport_path_and_nonzero_swapped_gradients_at_duplicates():
    rng=np.random.default_rng(108)
    for _ in range(30):
        p=rng.normal(size=(3,2)); q=rng.normal(size=(3,2)); perms=list(permutations(range(3)))
        cost=lambda x,pi:((x-q[list(pi)])**2).sum()/3
        pi=min(perms,key=lambda pi:cost(p,pi)); start=cost(p,pi)
        for s in np.linspace(0,1,21):
            x=(1-s)*p+s*q[list(pi)]
            assert abs(min(cost(x,tau) for tau in perms)-(1-s)**2*start)<1e-13
        p[1]=p[0]; pi=min(perms,key=lambda pi:cost(p,pi)); swap=list(pi)
        swap[0],swap[1]=swap[1],swap[0]
        assert abs(cost(p,pi)-cost(p,swap))<1e-13
        assert np.linalg.norm((p-q[list(pi)])-(p-q[swap]))>1e-5


def test_averaging_active_gradients_can_stall_at_a_strict_maximum():
    p=np.zeros((1,1)); q=np.array([[-1.],[1.]])
    active=all_active(p,q)
    assert len(active)==2 and np.linalg.norm(active.mean(0))==0
    assert np.min(np.linalg.norm(active,axis=(1,2)))==2
    assert branch(p+.01,q)[0]<branch(p,q)[0] and branch(p-.01,q)[0]<branch(p,q)[0]


def test_forward_flow_mode_formula_and_variance_delay():
    from scipy.linalg import expm
    H=np.array([[1.,-1,0],[-1,2,-1],[0,-1,1]]); K=np.linalg.inv(np.eye(3)+2*H)
    p=np.array([[-.7,.4],[.2,.9],[1.4,-.2]]); lam,U=np.linalg.eigh(H)
    for t in [.1,1.,5.]:
        actual=expm(-2*t*K/3)@p
        predicted=U@(np.exp(-2*t/(3*(1+2*lam)))[:,None]*(U.T@p))
        assert np.max(np.abs(actual-predicted))<1e-14
        assert np.linalg.norm(actual.mean(0)-np.exp(-2*t/3)*p.mean(0))<1e-14
        variance=((actual-actual.mean(0))**2).sum()/3
        initial=((p-p.mean(0))**2).sum()/3
        assert variance>=np.exp(-4*t/3)*initial


def test_open_one_sided_tie_trap_and_reverse_weight_transition():
    q=np.array([[-1.],[0.],[1.]]); pstar=np.array([[-1.],[0.],[.5]])
    rng=np.random.default_rng(110)
    for _ in range(30):
        p=np.array([[-1+rng.uniform(-.04,.04)],[rng.uniform(-.04,.04)],[rng.uniform(.31,.49)]])
        initial=p.copy()
        for step in range(40): p-=.2*branch(p,q)[1]
        assert np.max(np.abs(p-pstar-(1-4*.2/3)**40*(initial-pstar)))<1e-14
        assert p[2]<.5 and branch(p,q)[0]>1/6
    active=all_active(pstar,q)
    assert len(active)==2 and np.linalg.norm(branch(pstar,q)[1])<1e-14
    assert np.linalg.norm(active)>0
    for w in [.2,.5]:
        p=pstar.copy(); p[2]=w/(1+w)
        assert np.linalg.norm(all_active(p,q,w))<1e-14
        assert abs(branch(p,q,w)[0]-w/(3*(1+w)))<1e-14
    p=np.array([[-1.],[0.],[.4]])
    for _ in range(80): p-=.2*branch(p,q,w=2)[1]
    assert np.linalg.norm(p-q)<1e-12
