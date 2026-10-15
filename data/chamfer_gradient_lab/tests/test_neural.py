"""Meaningful checks of gradient probes and the partial-observation counterexample."""
import numpy as np
import torch
from neural_experiment import gradient_probe, model, loss
from experiment import chamfer, sliced_w2


def test_probe_restores_parameters_and_matches_directional_derivative():
    torch.manual_seed(720)
    net = torch.nn.Sequential(torch.nn.Linear(3, 5),torch.nn.Tanh(),torch.nn.Linear(5,3)).double()
    x,y,held,truth = [torch.randn(n,3,dtype=torch.float64) for n in [5,7,8,8]]
    saved = [p.detach().clone() for p in net.parameters()]
    rows = np.asarray(gradient_probe(net,(x,y,held,None,truth),0))
    assert all(torch.equal(a,b) for a,b in zip(saved,net.parameters()))
    assert abs(rows[-1,5]-rows[-1,6]) < 2e-5
    assert np.all(rows[:,7]>0)
    assert abs(rows[-1,7]/rows[-1,1]-rows[-1,2]) < 2e-5


def test_network_identity_initialization_and_loss_definitions():
    net = model(0,device='cpu'); x=torch.randn(9,3,dtype=torch.float64)
    assert torch.equal(net(x),torch.zeros_like(x))
    y=x+.2; dirs=torch.eye(3,dtype=x.dtype)
    assert loss(x,y,y,dirs,'cd')==chamfer(x,y)
    assert loss(x,y,y,dirs,'cd_sw')==chamfer(x,y)+sliced_w2(x,y,dirs)
    assert loss(x,y,y,dirs,'supervised') > 0
    assert loss(y,y,y,dirs,'unsquared') < 1e-10
    assert loss(y,y,y,dirs,'trimmed')==0


def test_partial_observation_closed_form_and_balanced_transport():
    x=torch.tensor([[-1.],[0.],[1.]],dtype=torch.float64)
    q=torch.tensor([[-1.],[0.]],dtype=torch.float64); direction=torch.ones(1,1,dtype=x.dtype)
    for theta in [-.49,-.3,-1/6,-.01,0.]:
        p=x+theta
        assert abs(float(chamfer(p,q))-(2*theta**2+2*theta/3+1/3)) < 1e-14
        assert abs(float(sliced_w2(p,q,direction))-(theta**2+theta+.5)) < 1e-14
    for weight in [0.,.1,1.,10.]:
        optimum=-(2/3+weight)/(4+2*weight)
        theta=torch.tensor(optimum,dtype=x.dtype,requires_grad=True)
        value=chamfer(x+theta,q)+weight*sliced_w2(x+theta,q,direction)
        gradient,=torch.autograd.grad(value,theta)
        assert abs(float(gradient))<1e-13
