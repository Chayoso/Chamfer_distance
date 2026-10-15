from ndp_prior import ENV, prior, torch, np


def test_infocd_formula_matches_public_gradient():
    torch.manual_seed(41)
    p=torch.randn(15,3,dtype=torch.float64,requires_grad=True);q=torch.randn(15,3,dtype=torch.float64)
    args=(1e-7,torch.tensor(1.),None)
    a=prior(p,q,'infocd_code',*args);b=prior(p,q,'infocd_paper',*args)
    ga=torch.autograd.grad(a,p)[0];gb=torch.autograd.grad(b,p)[0]
    assert torch.allclose(ga,gb,atol=1e-12,rtol=1e-10)


def test_dcd_hand_count_case():
    p=torch.tensor([[[.1,0,0],[.2,0,0],[3.,0,0]]]);q=torch.tensor([[[0.,0,0],[1.,0,0],[2.,0,0]]])
    expected=((1-np.exp(-np.array([.01,.04,1.]))/(np.array([2.,2.,1.])+1e-6)).mean()+
              (1-np.exp(-np.array([.01,.64,1.]))/(1+1e-6)).mean())/2
    assert abs(float(ENV['calc_dcd'](p,q,alpha=1)[0])-expected)<1e-7
