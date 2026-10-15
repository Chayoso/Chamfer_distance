"""Full native development fits for the frozen observable controller."""
import json
import csv
import sys
from pipeline.model import ROOT,np,torch,config
from pipeline.datasets import transfer
from pipeline.runner import fit
from pipeline.optim import fixed_official
from pipeline.controlled import controlled_factory

OUT=ROOT/'paper_revision/trust_20261013/adaptive/native'


def main():
    global OUT
    periodic='--periodic' in sys.argv
    if periodic:OUT=ROOT/'paper_revision/trust_20261013/periodic/native'
    restart='--restart' in sys.argv
    if restart:OUT=ROOT/'paper_revision/restart_20261013/native';periodic=True
    OUT.mkdir(exist_ok=False)
    cfg=config();cfg.budget=[50]*9
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    completed=[]
    for pair in [100,101,124,125,200,201]:
        for kind in (('adam','adam_restart','adam_trust','adaptive') if restart else ('adam','adam_trust','adaptive')):
            assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<800_000_000
            folder=OUT/kind;folder.mkdir(exist_ok=True);rows=[]
            actual_kind='adam' if kind=='adam_restart' else kind
            factory=fixed_official if kind=='adam' else lambda:controlled_factory(actual_kind,rows,periodic=periodic,restart=restart)
            result=fit(pair,0,'official',0,folder,loader=transfer,cfg=cfg,factory=factory)
            assert result['calls_by_level']==[51]*9
            if kind=='adam':
                stem=f'p{pair}_s0_official_w0';ref=ROOT/'paper_revision'/('replication_4d' if pair<200 else 'replication_hf')
                old=json.loads((ref/(stem+'.json')).read_text())
                for key in old:
                    if key in result and key!='seconds':assert old[key]==result[key],(pair,key)
                a=np.loadtxt(ref/(stem+'_trace.txt'),delimiter=',',skiprows=1)
                b=np.loadtxt(folder/(stem+'_trace.txt'),delimiter=',',skiprows=1)
                assert np.array_equal(a,b),pair
            else:
                assert len(rows)==450
                if restart:
                    assert sum(r['restarted'] for r in rows)==18
                    assert all(r['moment_step']==(0 if r['restarted'] else r['step']+1 if r['step']<25 else r['step']-25) for r in rows)
                with (folder/f'p{pair}_updates.txt').open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            completed.append(dict(pair=pair,kind=kind))
            (OUT/'integrity.json').write_text(json.dumps(dict(completed=completed),indent=2)+'\n')


if __name__=='__main__':main()
