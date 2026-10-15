from replication_experiment import np,torch,loss,objective


def test_sw_decomposition_and_parameter_gradients():
    torch.manual_seed(19)
    x=torch.randn(19,3,dtype=torch.float64);q=torch.randn(19,3,dtype=torch.float64)
    theta=torch.randn(3,3,dtype=torch.float64,requires_grad=True);p=x@theta
    d=torch.randn(3,64,dtype=torch.float64);d/=d.norm(dim=0)
    scale=torch.tensor(1.3,dtype=torch.float64)
    base=loss(p,q,'official',0,scale,d);full=loss(p,q,'sw',.1,scale,d)
    centered=loss(p,q,'centered_sw',.1,scale,d);moment=loss(p,q,'moment_sw',.1,scale,d)
    assert torch.allclose(full,centered+moment-base,atol=1e-12,rtol=1e-12)
    a=torch.autograd.grad(full,theta,retain_graph=True)[0]
    b=torch.autograd.grad(centered+moment-base,theta)[0]
    assert torch.allclose(a,b,atol=1e-12,rtol=1e-12)
    shift=torch.tensor([.2,-.5,.1],dtype=torch.float64)
    c=loss(p+shift,q,'centered_sw',.1,scale,d)-loss(p+shift,q,'official',0,scale,d)
    assert torch.allclose(c,centered-base,atol=1e-12,rtol=1e-12)
