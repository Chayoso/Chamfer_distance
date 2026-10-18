"""Exact affine readout subspace; regularized least squares, no task labels."""
from .model import torch


def translation_lift(layer,inputs,gradient,damping_ratio=.01):
    if layer.nonrigidity_est:raise ValueError('Requires the unmodulated affine translation head')
    if not 0<damping_ratio<float('inf'):raise ValueError('A finite positive damping ratio is required')
    with torch.no_grad():
        features=layer.mlp(layer.input(layer.posenc(inputs)))
        design=layer.mlp_scale*torch.cat([features,torch.ones_like(features[:,:1])],dim=1)
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
