"""Reciprocal nearest-neighbor anchors and sparse displacement propagation."""
from .model import torch
from .lifting import cg


def reciprocal_field(output, target):
    distance = torch.cdist(output.detach(),target.detach(),compute_mode='donot_use_mm_for_euclid_dist')
    nearest = distance.argmin(1)
    mask = distance.argmin(0)[nearest] == torch.arange(len(output),device=output.device)
    return (target[nearest]-output).detach()*mask[:,None],mask


def graph_edges(points, k=8):
    """Directed entries for a symmetric kNN graph, normalized by mean degree."""
    n = len(points)
    if n < 2: raise ValueError('Graph needs at least two points')
    distance = torch.cdist(points.detach(),points.detach(),compute_mode='donot_use_mm_for_euclid_dist')
    distance.fill_diagonal_(float('inf'))
    values,neighbors = distance.topk(min(k,n-1),largest=False,dim=1)
    bandwidth = values[:,-1].median().clamp_min(torch.finfo(points.dtype).eps)
    row = torch.arange(n,device=points.device)[:,None].expand_as(neighbors).reshape(-1)
    col = neighbors.reshape(-1)
    weight = .5*torch.exp(-.5*(values.reshape(-1)/bandwidth).square())
    degree = 2*weight.sum()/n
    if degree <= 0: raise ValueError('Graph has zero total degree')
    weight = weight/degree
    return row,col,weight,dict(bandwidth=float(bandwidth),directed_edges=len(row))


def laplacian(points, k=8):
    """Symmetric kNN Laplacian normalized by mean degree; no dense solve."""
    row,col,weight,info=graph_edges(points,k)

    def apply(vector):
        differences = weight[:,None]*(vector[row]-vector[col])
        result = torch.zeros_like(vector)
        result.index_add_(0,row,differences)
        result.index_add_(0,col,-differences)
        return result

    return apply,info


def propagate(points, anchor_field, mask, epsilon=.01, maxiter=128):
    operator,info = laplacian(points)
    field,solver = cg(lambda z:mask[:,None]*z+operator(z)+epsilon*z,anchor_field.detach(),maxiter=maxiter)
    info.update(solver)
    return field,info
