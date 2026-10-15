"""Infinitesimal strain and a symmetric, damped neural parameter metric."""
from .model import torch
from .graph import graph_edges
from .lifting import cg


def strain_operator(inputs,output):
    row,col,weight,info=graph_edges(inputs)
    edge=(output[row]-output[col]).detach();length=edge.norm(dim=1)
    valid=length>torch.finfo(output.dtype).eps
    normal=edge/length.clamp_min(torch.finfo(output.dtype).eps)[:,None]
    normal=normal*valid[:,None]
    trace=2*(weight*normal.square().sum(1)).sum()
    coefficient=output.numel()/float(trace) if trace>0 else 0.
    weight=coefficient*weight

    def apply(vector):
        difference=vector[row]-vector[col]
        force=(weight*(difference*normal).sum(1))[:,None]*normal
        result=torch.zeros_like(vector)
        result.index_add_(0,row,force);result.index_add_(0,col,-force)
        return result

    info.update(valid_edges=int(valid.sum()),omitted_edges=int((~valid).sum()),
                mean_diagonal=float(coefficient*trace/output.numel()))
    return apply,info


def parameter_metric(forward,transpose,gradient,metric,damping):
    """Solve (J^T A J + gamma I)delta=-J^T g without forming J or A."""
    rhs=transpose(gradient);shapes=[r.shape for r in rhs];sizes=[r.numel() for r in rhs]
    flatten=lambda xs:torch.cat([x.reshape(-1) for x in xs])
    unpack=lambda flat:[x.reshape(shape) for x,shape in zip(flat.split(sizes),shapes)]
    calls=0
    def operator(flat):
        nonlocal calls
        calls+=1
        return flatten(transpose(metric(forward(unpack(flat)))))+damping*flat
    delta,info=cg(operator,-flatten(rhs),rtol=1e-10,maxiter=256)
    info.update(operator_calls=calls,damping=damping)
    return unpack(delta),info
