"""Observe cloned update directions while exactly replaying six existing fits."""
import copy
import csv
import json
from types import SimpleNamespace
from pipeline.model import ROOT, np, torch, config
from pipeline.datasets import transfer
from pipeline.optim import fixed_official
from pipeline.runner import fit
from replication_experiment import loss
from ndp_prior import prior

OUT = ROOT/'paper_revision/directions_20261013'
METHODS = ['official', 'sw', 'centered_sw', 'uot', 'infocd_code']


def adam_direction(grads, history, group):
    """One negative Adam update, with current gradients and fixed previous state."""
    if group.get('weight_decay', 0) or group.get('amsgrad', False):
        raise ValueError('This diagnostic uses ordinary unregularized Adam')
    b1, b2 = group['betas']
    result = []
    for g, (m, v, step) in zip(grads, history):
        m1, v1 = b1*m+(1-b1)*g, b2*v+(1-b2)*g.square()
        result.append(-group['lr']*(m1/(1-b1**(step+1))) /
                      ((v1/(1-b2**(step+1))).sqrt()+group['eps']))
    return result


def objective(p, q, method, scale, dirs):
    fn = prior if method == 'infocd_code' else loss
    return fn(p, q, method, .1, scale, dirs)


def observe(before, after, inputs, q, gt, history, group, scale, dirs, key, records):
    before._forward_pre_hooks.clear()
    with torch.no_grad():
        original = before(inputs)[0].double()
    before = before.double()
    inputs, q, gt = inputs.double(), q.double(), gt.double()
    params = list(before.parameters())
    start = [p.detach().clone() for p in params]
    actual_delta = [b.detach().double()-a for a,b in zip(start,after.parameters())]
    p = before(inputs)[0]
    base = objective(p,q,'official',scale,dirs)
    risk = (p-gt).square().sum(1).mean()
    gp, = torch.autograd.grad(base,p,retain_graph=True)
    dummy = torch.zeros_like(p,requires_grad=True)
    vjp = torch.autograd.grad(p,params,grad_outputs=dummy,create_graph=True,retain_graph=True)

    def jvp(delta):
        return torch.autograd.grad(sum((a*b).sum() for a,b in zip(vjp,delta)),
                                   dummy,retain_graph=True)[0].detach()

    actual_velocity = jvp(actual_delta)
    norm = actual_velocity.norm()
    assert norm > 1e-20
    proposals = [('official','actual_adam',actual_delta)]
    for method in METHODS:
        value = objective(p,q,method,scale,dirs)
        grads = [g.detach() for g in torch.autograd.grad(value,params,retain_graph=True)]
        proposals.extend([(method,'gd',[-g for g in grads]),
                          (method,'adam_shared',adam_direction(grads,history,group))])
    trial = copy.deepcopy(before)
    target_params = list(trial.parameters())
    max_identity_error = 0.
    for method, optimizer, delta in proposals:
        raw = jvp(delta)
        raw_norm = raw.norm()
        assert raw_norm > 1e-20 and torch.isfinite(raw).all()
        multiplier = norm/raw_norm
        velocity = raw*multiplier
        delta = [d*multiplier for d in delta]
        slope = (2*(p.detach()-gt)*velocity).sum()/len(p)
        cosine = (velocity*actual_velocity).sum()/norm.square()
        records['directions'].append([*key,method,optimizer,float(raw_norm),float(multiplier),
            float(cosine),float(slope),float((gp*velocity).sum())])
        with torch.no_grad():
            for fraction in [1/16,1/4,1.]:
                for t,a,d in zip(target_params,start,delta): t.copy_(a+fraction*d)
                warped = trial(inputs)[0]
                displacement = warped-p.detach()
                change = (warped-gt).square().sum(1).mean()-risk.detach()
                exact = (2*(p.detach()-gt)*displacement+displacement.square()).sum()/len(p)
                max_identity_error = max(max_identity_error,float(abs(change-exact)))
                cd = objective(warped,q,'official',scale,dirs)-base.detach()
                error = (displacement-fraction*velocity).norm()/(fraction*norm)
                records['finite'].append([*key,method,optimizer,fraction,float(change),
                    float(cd),float(displacement.norm()),float(error)])
        if method == 'official' and optimizer == 'adam_shared':
            relative = (raw-actual_velocity).norm()/norm
    records['states'].append([*key,float((p.detach()-original).abs().max()),
                              float(norm),float(relative),max_identity_error])


def instrumented(pair, x, y, truth, scale, dirs, records, observer=observe):
    env = fixed_official()
    build, optimize = env['Deformation_Pyramid'], env['optimize_deformation_pyramid']
    state = {}

    def constructor(**kwargs):
        net = build(**kwargs)
        state['net'] = net
        gen = torch.Generator().set_state(torch.get_rng_state())
        state['ids'] = torch.randperm(len(x),generator=gen)[:2000].to(x.device)
        for i,layer in enumerate(net.pyramid):
            layer.register_forward_pre_hook(lambda module,inputs,i=i:state.update({i:inputs[0].detach()}))
        return net

    def optimizer(instance):
        original = env['compute_truncated_chamfer_distance']

        def observed(p, q, **kwargs):
            level = next(i for i,m in enumerate(state['net'].pyramid) if next(m.parameters()).requires_grad)
            step = state.get('step',-1)+1 if state.get('level') == level else 0
            state.update(level=level,step=step,target=q[0].detach())
            return original(p,q,**kwargs)

        env['compute_truncated_chamfer_distance'] = observed
        return optimize(instance)

    class Observer(torch.optim.Adam):
        def step(self, closure=None):
            level, step = state['level'], state['step']
            if level not in [0,4,8] or step not in [0,25,49]: return super().step(closure)
            layer = state['net'].pyramid[level]
            before = copy.deepcopy(layer)
            history = []
            for p in layer.parameters():
                h = self.state[p]
                history.append((h.get('exp_avg',torch.zeros_like(p)).double().clone(),
                                h.get('exp_avg_sq',torch.zeros_like(p)).double().clone(),
                                int(h.get('step',0))))
            value = super().step(closure)
            observer(before,layer,state[level],state['target'],truth[state['ids']]-y.mean(0),
                    history,self.param_groups[0],scale,dirs,(pair,level,step),records)
            return value

    env.update(Deformation_Pyramid=constructor,optimize_deformation_pyramid=optimizer,
               optim=SimpleNamespace(Adam=Observer))
    return env


def main():
    folder = OUT/'probe'
    folder.mkdir(exist_ok=False)
    cfg = config()
    cfg.budget = [50]*9
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    records = {k:[] for k in ['directions','finite','states']}
    headers = {
        'directions':'pair level step method optimizer raw_output_norm multiplier cosine_actual risk_derivative cd_derivative',
        'finite':'pair level step method optimizer fraction risk_change cd_change output_norm relative_linearization_error',
        'states':'pair level step float64_output_discrepancy actual_jvp_norm formula_actual_relative_error mse_identity_error'}
    replays = []
    for pair in [100,101,124,125,200,201]:
        assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) < 800_000_000
        x,y,truth,_ = transfer(pair)
        scale = (x-x.mean(0)).square().sum(1).mean().sqrt().double()
        dirs = torch.as_tensor(np.random.default_rng(7000+pair).normal(size=(3,64)),device=x.device,dtype=x.dtype)
        dirs = (dirs/dirs.norm(dim=0)).double()
        factory = lambda:instrumented(pair,x,y,truth,scale,dirs,records)
        got = fit(pair,0,'official',0,folder,loader=transfer,cfg=cfg,factory=factory)
        stem = f'p{pair}_s0_official_w0'
        ref = ROOT/'paper_revision'/('replication_4d' if pair < 200 else 'replication_hf')
        old = json.loads((ref/(stem+'.json')).read_text())
        for key in old:
            if key in got and key != 'seconds': assert old[key] == got[key], (pair,key)
        a = np.loadtxt(ref/(stem+'_trace.txt'),delimiter=',',skiprows=1)
        b = np.loadtxt(folder/(stem+'_trace.txt'),delimiter=',',skiprows=1)
        assert np.array_equal(a,b), pair
        replays.append(pair)
        for name,rows in records.items():
            with (folder/(name+'.txt')).open('w',newline='') as f:
                w=csv.writer(f); w.writerow(headers[name].split()); w.writerows(rows)
        (folder/'integrity.json').write_text(json.dumps(dict(exact_replays=replays,
            rows={k:len(v) for k,v in records.items()}),indent=2)+'\n')
    assert [len(records[k]) for k in ['states','directions','finite']] == [54,594,1782]


if __name__ == '__main__':
    main()
