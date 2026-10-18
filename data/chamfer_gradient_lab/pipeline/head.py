"""Exact affine readout subspace; regularized least squares, no task labels."""
from .model import torch


def translation_design(layer,inputs):
    if layer.nonrigidity_est:raise ValueError('Requires the unmodulated affine translation head')
    with torch.no_grad():
        features=layer.mlp(layer.input(layer.posenc(inputs)))
        return layer.mlp_scale*torch.cat([features,torch.ones_like(features[:,:1])],dim=1)


def translation_lift(layer,inputs,gradient,damping_ratio=.01):
    if not 0<damping_ratio<float('inf'):raise ValueError('A finite positive damping ratio is required')
    with torch.no_grad():
        design=translation_design(layer,inputs)
        gram=design.T@design
        damping=damping_ratio*torch.linalg.eigvalsh(gram)[-1]
        if damping<=0:raise ValueError('Degenerate translation design')
        matrix=gram+damping*torch.eye(len(gram),device=gram.device,dtype=gram.dtype)
        rhs=-(design.T@gradient)
        weights=torch.linalg.solve(matrix,rhs)
        residual=(matrix@weights-rhs).norm()
        relative=float(residual/rhs.norm().clamp_min(1e-30))
        delta=[weights[:-1].T if name=='trn_branch.weight' else weights[-1]
               if name=='trn_branch.bias' else torch.zeros_like(p) for name,p in layer.named_parameters()]
        return delta,dict(converged=relative<=1e-8,relative_residual=relative,iterations=1,
                          damping=float(damping),velocity=design@weights)


def head_preconditioner(layer,inputs,damping):
    """Apply (A A^T + damping I)^-1 using one small Cholesky factorization."""
    if not 0<damping<float('inf'):raise ValueError('A finite positive damping is required')
    with torch.no_grad():
        design=translation_design(layer,inputs);gram=design.T@design
        factor=torch.linalg.cholesky(gram+damping*torch.eye(len(gram),device=gram.device,dtype=gram.dtype))
    def apply(vector):
        weights=torch.cholesky_solve(design.T@vector,factor)
        return (vector-design@weights)/damping
    return apply
