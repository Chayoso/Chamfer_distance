import numpy as np


def gradient(p,q):
    d=((p[:,None]-q[None])**2).sum(2);a=d.argmin(1);b=d.argmin(0)
    g=2*(p-q[a])/3
    for j,i in enumerate(b):g[i]+=2*(p[i]-q[j])/3
    return g,a,float(d.min(0).mean()+d.min(1).mean())


def test_open_three_dimensional_target_family_and_crossing():
    rng=np.random.default_rng(129);checked=0
    for _ in range(100):
        q=np.array([[-2.,1,0],[0,0,0],[1,0,0]])+.08*rng.normal(size=(3,3))
        a,b,c=q;m=(b+c)/2;d=np.linalg.norm(c-b);n=(c-b)/d
        gamma=min(np.linalg.norm(a-b),np.linalg.norm(a-c),d/2,np.linalg.norm(a-m)-d/2,np.linalg.norm(a-c)-d/2)
        assert gamma>0
        r=gamma/4;delta=r/12;star=np.array([a,b,m])
        tangent=rng.normal(size=(3,3));tangent-=np.outer(tangent@n,n);tangent*=delta/(1+np.linalg.norm(tangent))
        for rho,sfactor in [(0.,-3),(.5,-3),(.5,-1)]:
            z=tangent.copy();z[1]+=sfactor*delta*n;z[2]-=delta*n
            assert np.linalg.norm(z)<r
            p=star+z;k=(1-rho)*np.eye(3)+rho*np.ones((3,3));cross=None;alpha=.1
            for step in range(120):
                g,assignment,value=gradient(p,q)
                if assignment[2]==2:cross=step;break
                assert np.array_equal(assignment,[0,1,1])
                assert abs(value-d*d/6-(2/3)*np.sum((p-star)**2))<1e-13
                assert np.allclose(g,4*(p-star)/3,atol=1e-13)
                p-=alpha*k@g
            if rho and sfactor<-2:
                aa=1-4*alpha*(1-rho)/3;bb=1-4*alpha*(1+2*rho)/3
                predicted=next(t for t in range(1,120) if (bb/aa)**t<(-2-sfactor)/(1-sfactor))
                assert cross==predicted
            else:assert cross is None and np.linalg.norm(p-star)<np.linalg.norm(z)*.001
            checked+=1
        eps=1e-5*d;p=star.copy();p[2]+=eps*n
        assert gradient(p,q)[2]<d*d/6
    assert checked==300
