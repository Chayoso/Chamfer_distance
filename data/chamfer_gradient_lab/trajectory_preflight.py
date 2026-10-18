"""Exact production/diagnostic parity at six existing, final-level NDP states."""
import ndp_field_probe as base
from global_registration_control import load
from pipeline.native_mix import directions, native_trials

OUT=base.ROOT/'paper_revision/trajectory_mix_20261018/preflight'


def environment(pair,truth):
    x=base.torch.as_tensor(load(pair)[0],device=truth.device,dtype=base.torch.float32)
    pose=base.torch.as_tensor(POSES[pair],device=x.device,dtype=base.torch.float64)
    def end(level,iteration,count,previous,value,opt,layer,inputs,q,cached,ids,mean):
        if level!=8:return
        anchor=x[ids].double()@pose[:3,:3].T+pose[:3,3]-mean.double()
        seed=93000+100*pair+level;full=directions(layer,inputs,q,anchor,seed)
        saved=[p.detach().double().clone() for p in layer.parameters()]
        path=base.continuation(layer,opt,inputs,q,base.cd,steps=1)
        next(path);_,_,_,params=next(path)
        adam=[p.detach().double()-v for p,v in zip(params,saved)];path.close()
        model=base.copy.deepcopy(layer).double()
        forward,_=base.jacobian_products(model(inputs.double())[0],list(model.parameters()))
        d,v=base.scale_direction(adam,forward,full['bound'])
        full['directions']['adam']=dict(delta=d,velocity=v)
        for method in ('adam','cd_ridge','fgr_gdmix','fgr_ridgemix'):
            limited=directions(layer,inputs,q,anchor,seed,method=method,adam_delta=adam)
            a,b=full['directions'][method],limited['directions'][method]
            assert all(base.torch.equal(v,w) for v,w in zip(a['delta'],b['delta']))
            assert base.torch.equal(a['velocity'],b['velocity'])
            def trials(result,item,caps):
                return list(native_trials(layer,inputs,q,result['native'],result['initial'],
                    item['delta'],item['velocity'],result['gradient'],caps=caps))[-2:]
            for old,new in zip(trials(full,a,(1/16,.25,1.)),trials(limited,b,(1.,))):
                assert {k:v for k,v in old[0].items() if k!='controller_seconds'}=={k:v for k,v in new[0].items() if k!='controller_seconds'}
                assert base.torch.equal(old[1],new[1])
            base.append('parity',dict(pair=pair,level=level,method=method,exact=True))
    return base.stopping_environment(lambda *args:None,end)[0]


if __name__=='__main__':
    POSES={r['pair']:r['transform'] for r in base.json.loads((base.ROOT/'paper_revision/fgr_20261018/observed.json').read_text())['rows']}
    OUT.mkdir(exist_ok=False)
    base.OUT=OUT;base.environment=environment;base.main()
    rows=[base.json.loads(s) for s in (OUT/'parity.txt').read_text().splitlines()]
    assert len(rows)==24 and all(r['exact'] for r in rows)
    (OUT/'complete.json').write_text(base.json.dumps(dict(exact_methods=24,exact_baseline_replays=6))+'\n')
