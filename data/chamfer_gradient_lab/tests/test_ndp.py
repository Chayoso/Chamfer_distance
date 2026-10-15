"""Operator/metric checks required by the official-pipeline adapter."""
import ast
import json
import pytest
from types import SimpleNamespace
from typing import Union
from ndp_pipeline import np, torch, OFFICIAL, official, components, objective


def test_official_loss_value_and_gradient():
    def knn(x,y,**kwargs):
        d=(x[:,:,None,:]-y[:,None,:,:]).square().sum(-1)
        values,ids=d.min(2)
        return SimpleNamespace(dists=values[:,:,None],idx=ids[:,:,None])
    env=dict(torch=torch,Union=Union,Pointclouds=type('Unused',(),{}),knn_points=knn)
    tree=ast.parse((OFFICIAL/'model_loss.py').read_text())
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in
               ['_validate_chamfer_reduction_inputs','_handle_pointcloud_input','compute_truncated_chamfer_distance']]
    exec(compile(tree,'official_loss','exec'),env)
    torch.manual_seed(31)
    x=torch.randn(13,3,dtype=torch.float64,requires_grad=True);y=torch.randn(17,3,dtype=torch.float64)
    original=env['compute_truncated_chamfer_distance'](x[None],y[None],trunc=1e9)
    a,b=components(x,y);adapted=a.mean()+b.mean()
    assert torch.allclose(original,adapted,atol=1e-12,rtol=1e-12)
    assert torch.allclose(torch.autograd.grad(original,x)[0],torch.autograd.grad(adapted,x)[0],atol=1e-12,rtol=1e-12)


def test_official_sampling_capture_and_warp():
    env=official();torch.manual_seed(91)
    net=env['Deformation_Pyramid'](depth=3,width=128,device='cpu',k0=-8,m=9,
        rotation_format='axis_angle',nonrigidity_est=False,motion='SE3')
    gen=torch.Generator().set_state(torch.get_rng_state())
    assert torch.equal(torch.randperm(2345,generator=gen),torch.randperm(2345))
    x=torch.randn(21,3);a,_=net.warp(x);b=x
    for layer in net.pyramid: b,_=layer(b)
    assert torch.equal(a,b)
    assert torch.isfinite(a).all()


def test_uot_augmentation_gradient():
    torch.manual_seed(5)
    p=torch.randn(13,3,dtype=torch.float64,requires_grad=True);q=torch.randn(13,3,dtype=torch.float64)
    dirs=torch.randn(3,8,dtype=torch.float64);dirs/=dirs.norm(dim=0)
    d=torch.randn_like(p);d/=d.norm();scale=torch.tensor(1.7,dtype=torch.float64)
    v=objective(p,q,'uot',.1,scale,dirs);g=torch.autograd.grad(v,p)[0]
    eps=1e-5
    fd=(objective(p.detach()+eps*d,q,'uot',.1,scale,dirs)-objective(p.detach()-eps*d,q,'uot',.1,scale,dirs))/(2*eps)
    assert torch.allclose(fd,(g*d).sum(),rtol=1e-5,atol=1e-7)


def test_fixed_loop_matches_frozen_reference():
    """All outputs, sampled losses and observer calls, including zero-update levels."""
    from ndp_pipeline import ROOT, config
    from ndp_matched_budget import fixed_official
    tree = ast.parse((ROOT/'tests/fixtures/ndp_fixed_reference.py').read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fixed_official']
    namespace = dict(ast=ast, OFFICIAL=OFFICIAL, official=official)
    exec(compile(tree, 'frozen_reference', 'exec'), namespace)
    cfg = config()
    cfg.m, cfg.width, cfg.samples, cfg.budget = 3, 12, 11, [0, 3, 2]
    torch.manual_seed(29)
    x, y = torch.randn(19,3), torch.randn(23,3)
    for common_origin in (False, True):
        outputs = []
        for factory in (namespace['fixed_official'], fixed_official):
            torch.manual_seed(33)
            env = factory(common_origin=common_origin)
            losses, rates = [], []

            def loss(p, q, **kwargs):
                a, b = components(p[0], q[0])
                value = a.mean()+b.mean()
                losses.append(float(value.detach()))
                return value

            class Observer(torch.optim.Adam):
                def step(self, closure=None):
                    rates.append(self.param_groups[0]['lr'])
                    return super().step(closure)

            env.update(compute_truncated_chamfer_distance=loss, optim=SimpleNamespace(Adam=Observer))
            instance = SimpleNamespace(config=cfg, device='cpu', src_pcd=x, tgt_pcd=y, landmarks=None)
            pred, _, _ = env['optimize_deformation_pyramid'](instance)
            outputs.append((pred.detach(), losses, rates, torch.get_rng_state()))
        assert torch.equal(outputs[0][0], outputs[1][0])
        assert outputs[0][1:3] == outputs[1][1:3]
        assert torch.equal(outputs[0][3], outputs[1][3])
        assert len(outputs[1][1]) == 8 and len(outputs[1][2]) == 5
    cfg.schedule = 'cosine'
    losses.clear()
    rates.clear()
    env = fixed_official()
    env.update(compute_truncated_chamfer_distance=loss, optim=SimpleNamespace(Adam=Observer))
    env['optimize_deformation_pyramid'](instance)
    assert np.allclose(rates, [.01, .0055, .001, .01, .001], atol=1e-15, rtol=0)
    assert len(losses) == 8


def test_empty_evaluation_groups_are_explicit():
    from pipeline.runner import metrics
    x = torch.zeros(3,3)
    truth = torch.ones(3,3)
    for visible in (torch.ones(3,dtype=torch.bool), torch.zeros(3,dtype=torch.bool)):
        row = metrics(truth,x,truth,visible,torch.arange(3))
        assert row['held_points'] == 0 and row['held_epe'] is None
        absent = 'occ' if visible.all() else 'vis'
        assert row[absent+'_points'] == 0 and row[absent+'_epe'] is None
        assert row['full_epe'] == 0 and row['full_accs'] == 1
        json.dumps(row,allow_nan=False)


def test_unknown_objective_does_not_silently_run_chamfer():
    x = torch.zeros(3,3)
    with pytest.raises(ValueError, match='Unknown objective'):
        objective(x,x,'offical',0,1,torch.eye(3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU replay needs CUDA')
def test_modular_pipeline_replays_saved_scan_runs(tmp_path, monkeypatch):
    """Four saved runs exercise direct modules and historical dependency overrides."""
    from pipeline.model import ROOT, config
    from pipeline.datasets import scans
    from pipeline.optim import fixed_official
    from pipeline.runner import fit
    import ndp_experiment as legacy
    if not (ROOT/'assets/stanford_raw/04.npz').exists():
        pytest.skip('Captured scan assets unavailable')
    cfg = config()
    cfg.budget = [50]*9
    monkeypatch.setattr(legacy, 'config', lambda:cfg)
    monkeypatch.setattr(legacy, 'data', scans)
    monkeypatch.setattr(legacy, 'official', fixed_official)
    threads, tf32 = torch.get_num_threads(), torch.backends.cuda.matmul.allow_tf32
    try:
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        for pair in (4,16):
            for method in ('official','cosine'):
                cfg.schedule = 'constant' if method == 'official' else 'cosine'
                if method == 'official':
                    result = legacy.fit(pair,0,method,0,tmp_path)
                else:
                    result = fit(pair,0,method,0,tmp_path,cfg=cfg,loader=scans,factory=fixed_official)
                stem = f'p{pair:02d}_s0_{method}_w0'
                reference = ROOT/'paper_revision/refactor_20261013/scan'
                old = json.loads((reference/(stem+'.json')).read_text())
                for key, value in old.items():
                    if key not in ('seconds','schedule','optimizer_updates'):
                        assert result[key] == value, (pair,method,key)
                old_trace = np.loadtxt(reference/(stem+'_trace.txt'),delimiter=',',skiprows=1)
                new_trace = np.loadtxt(tmp_path/(stem+'_trace.txt'),delimiter=',',skiprows=1)
                assert np.array_equal(old_trace,new_trace)
                old_pred = np.loadtxt(reference/(stem+'_prediction.txt'))
                new_pred = np.loadtxt(tmp_path/(stem+'_prediction.txt'))
                assert np.array_equal(old_pred,new_pred)
    finally:
        torch.set_num_threads(threads)
        torch.backends.cuda.matmul.allow_tf32 = tf32
