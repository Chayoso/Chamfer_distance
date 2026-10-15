"""Remove spatial matching cost; retain transfer, finite controls, and samples."""
import json
import time
import open3d as o3d
from pipeline.model import ROOT,torch
from pipeline.capacity import plan_field
from pipeline.features import feature_penalty
from ndp_capacity_probe import run

OUT=ROOT/'paper_revision/descriptor_20261015'
KINDS=('feature_soft','feature_unbalanced','descriptor_soft','descriptor_unbalanced')


def build_fields(p,q,key):
    torch.cuda.synchronize();start=time.perf_counter()
    penalty,details=feature_penalty(p,q)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    record=dict(pair=key[0],level=key[1],step=key[2],seconds=elapsed,open3d=o3d.__version__,**details)
    with (OUT/'features.txt').open('a') as f:f.write(json.dumps(record)+'\n')
    for name in KINDS:
        torch.cuda.synchronize();start=time.perf_counter()
        force,mass,info=plan_field(p,q,name.split('_',1)[1],penalty=penalty,
                                   feature_only=name.startswith('descriptor_'))
        torch.cuda.synchronize();seconds=time.perf_counter()-start+elapsed
        yield name,force,mass,info,seconds


if __name__=='__main__':run(OUT,build_fields,weighted=True)
