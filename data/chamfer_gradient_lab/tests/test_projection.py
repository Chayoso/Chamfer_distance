import pytest
import torch
from pipeline.controlled import cd
from pipeline.projection import point_snapshot,projection_metrics


def test_point_snapshot_preserves_parameter_backward_and_inputs():
    x=torch.tensor([[0.,0,0],[2.,1,0],[0.,3,0]])
    shift=torch.nn.Parameter(torch.zeros(3));p=x+shift;q=x+.2
    loss=cd(p,q);expected=torch.autograd.grad(loss,shift,retain_graph=True)[0]
    snapshot=point_snapshot(p,q,loss)
    assert shift.grad is None and torch.equal(snapshot['points'],x)
    assert torch.equal(snapshot['forward'],torch.arange(3))
    loss.backward();torch.testing.assert_close(shift.grad,expected,rtol=0,atol=0)


def frame(points,gradient):
    return dict(points=torch.tensor([points],dtype=torch.float64),gradient=torch.tensor([gradient],dtype=torch.float64),
                forward=torch.tensor([0]),reverse=torch.tensor([0]))


def test_projection_distinguishes_rotation_amplification_and_reversal():
    base=frame([0,0,0],[1,0,0]);initial=torch.tensor([[1.,0,0]])
    rotated=projection_metrics(frame([0,3,0],[0,1,0]),base,initial)
    assert rotated['projection']==0 and rotated['norm_gain']==rotated['orthogonal_gain']==3
    assert rotated['deformation_cosine']==rotated['gradient_cosine']==0
    reversed=projection_metrics(frame([-2,0,0],[-1,0,0]),base,initial)
    assert reversed['projection']==-2 and reversed['norm_gain']==2 and reversed['deformation_cosine']==-1


def test_zero_gradients_are_explicit_and_zero_initial_deviation_is_rejected():
    base=frame([0,0,0],[0,0,0]);initial=torch.tensor([[1.,0,0]])
    row=projection_metrics(frame([0,0,0],[0,0,0]),base,initial)
    assert row['deformation_cosine'] is row['gradient_cosine'] is row['relative_gradient_change'] is None
    with pytest.raises(ValueError):projection_metrics(base,base,torch.zeros_like(initial))
