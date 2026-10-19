"""Native early-exit semantics and preservation of the caller's parameters."""
import copy
import pytest
import torch
from pipeline import ray_search as search
from pipeline.native_mix import directions,native_trials
from pipeline.field_steps import scale_direction


def Layer():
    from pipeline.model import official
    return official()['NDPLayer'](depth=2,width=8,k0=-2,m=1,rotation_format='axis_angle')


def test_search_matches_explicit_grid_and_does_not_mutate_model():
    torch.manual_seed(812);layer=Layer();before=copy.deepcopy(layer.state_dict())
    x=torch.randn(12,3);q=torch.randn(13,3);anchor=x.double()+.2
    result=search.first_accepted(layer,x,q,anchor,19)
    check=directions(layer,x,q,anchor,19,method='fgr_search');forward=check.pop('forward')
    a,c=check['directions']['fgr_pcg'],check['directions']['cd_gd'];chosen=None
    for j in range(9):
        delta,velocity=scale_direction([(1-j/8)*v+j/8*w for v,w in zip(a['delta'],c['delta'])],forward,check['bound'])
        trials=list(native_trials(layer,x,q,check['native'],check['initial'],delta,velocity,check['gradient'],caps=(1.,)))
        if trials[-1][0]['accepted']:chosen=j/8;break
    assert result['item']['beta']==chosen
    assert torch.equal(result['output'],trials[-1][1])
    assert len(result['attempted'])==(9 if chosen is None else int(chosen*8)+1)
    assert result['production_seconds']>=sum(r['controller_seconds'] for r in result['attempted'])
    assert all(torch.equal(v,before[k]) for k,v in layer.state_dict().items())
    with pytest.raises(TypeError):search.first_accepted(layer,x,q,anchor,19,truth=x)


def test_all_rejections_return_zero_direction_and_original_output(monkeypatch):
    torch.manual_seed(813);layer=Layer();x=torch.randn(12,3);q=torch.randn(13,3)
    def reject(layer,inputs,target,native,initial,delta,velocity,gradient,caps):
        yield dict(policy='trust',cap=1.,fraction=0.,accepted=False,attempts=21,cd_change=0.,linearization_error=0.,
            output_norm=0.,cd_slope=-1.,controller_seconds=0.),native
    monkeypatch.setattr(search,'native_trials',reject)
    result=search.first_accepted(layer,x,q,x.double()+.2,19)
    assert len(result['attempted'])==9 and result['item']['beta'] is None
    assert result['item']['velocity'].count_nonzero()==0 and all(v.count_nonzero()==0 for v in result['item']['delta'])
    assert torch.equal(result['output'],layer(x)[0].detach().double())
