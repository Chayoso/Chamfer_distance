"""Quadratic branch solves for a fixed linear deformation basis; no training state."""
import torch
from experiment import distances


def branch_value(x, y, u, coef, a, b, ridge):
    p = x + u @ coef
    return ((p-y[a]).square().sum()/len(x)
            + (p[b]-y).square().sum()/len(y) + ridge*coef.square().sum())


def branch_system(x, y, u, coef, ridge):
    p = x + u @ coef
    d = distances(p, y)
    a, b = d.argmin(1), d.argmin(0)
    weight = 1/len(x) + torch.bincount(b, minlength=len(x)).to(x.dtype)/len(y)
    rhs = (y[a]-x)/len(x)
    rhs.index_add_(0, b, (y-x[b])/len(y))
    matrix = u.T @ (weight[:, None]*u) + ridge*torch.eye(u.shape[1], device=u.device, dtype=u.dtype)
    center = torch.linalg.solve(matrix, u.T @ rhs)
    return center, matrix, a, b


def switch_proposal(x, y, u, coef, ridge):
    """Best certified bound among all single forward-label changes.

    Uses exact branch quadratics, including a nonzero current branch residual.
    A positive predicted drop guarantees descent in exact arithmetic only.
    Caller must verify the actual objective and apply a numerical tolerance.
    """
    center, matrix, a, b = branch_system(x, y, u, coef, ridge)
    solved = torch.linalg.solve(matrix, u.T)
    leverage = (u*solved.T).sum(1)
    p = x + u @ center
    margin = distances(p, y) - (p-y[a]).square().sum(1)[:, None]
    separation = distances(y[a], y)
    old = branch_value(x, y, u, coef, a, b, ridge)
    base = branch_value(x, y, u, center, a, b, ridge)
    gain = old-base-margin/len(x)+leverage[:, None]*separation/len(x)**2
    gain[torch.arange(len(x), device=x.device), a] = -torch.inf
    flat = gain.argmax()
    i, j = int(flat//len(y)), int(flat % len(y))
    candidate = center + solved[:, i, None]*(y[j]-y[a[i]])[None, :]/len(x)
    return candidate, float(gain[i, j]), i, j, float(leverage[i]/len(x))
