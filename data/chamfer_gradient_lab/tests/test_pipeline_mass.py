from scipy.optimize import linear_sum_assignment
from ndp_pipeline import np, torch, official


def test_transport_translation_identity_and_flow():
    rng=np.random.default_rng(63);p=rng.normal(size=(9,3));q=rng.normal(size=(9,3))+[.5,0,0]
    delta=p.mean(0)-q.mean(0)
    def ot(a,b):
        costs=((a[:,None]-b[None])**2).sum(2);i,j=linear_sum_assignment(costs);return costs[i,j].mean()
    for t in [0,.1,1,4]:
        u=-(1-np.exp(-2*t))*delta
        assert abs(ot(p+u,q)-(ot(p,q)-(1-np.exp(-4*t))*sum(delta**2)))<1e-12
    v=rng.normal(size=(3,13));v/=np.linalg.norm(v,axis=0);u=rng.normal(size=3);m=v@v.T/13
    sw=lambda a:((np.sort(a@v,axis=0)-np.sort(q@v,axis=0))**2).mean()
    assert abs(sw(p+u)-sw(p)-2*u@m@delta-u@m@u)<1e-12


def test_ndp_last_translation_bias_is_exact():
    torch.manual_seed(64);env=official()
    net=env['Deformation_Pyramid'](depth=3,width=128,device='cpu',k0=-8,m=9,
        rotation_format='axis_angle',nonrigidity_est=False,motion='SE3')
    for layer in net.pyramid:layer.double()
    x=torch.randn(23,3,dtype=torch.float64);before,_=net.warp(x)
    u=torch.tensor([.1,-.2,.3],dtype=torch.float64)
    with torch.no_grad():net.pyramid[-1].trn_branch.bias.add_(1000*u)
    after,_=net.warp(x)
    assert torch.allclose(after-before,u.expand_as(after),atol=1e-13,rtol=1e-13)
