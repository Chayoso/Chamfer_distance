"""Official NDP model adapter and default configuration."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np  # Import before torch in the existing server environment.
import torch

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT/'assets/ndp_official'


def official():
    env = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, optim=torch.optim)
    for name in ['model_rigid_body.py', 'model_nets.py']:
        tree = ast.parse((OFFICIAL/name).read_text())
        tree.body = [x for x in tree.body if isinstance(x, (ast.FunctionDef, ast.ClassDef))]
        exec(compile(tree, str(OFFICIAL/name), 'exec'), env)
    tree = ast.parse((OFFICIAL/'model_registration.py').read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'Registration')
    method = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == 'optimize_deformation_pyramid')
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(OFFICIAL/'model_registration.py'), 'exec'), env)
    return env


def config():
    return SimpleNamespace(max_break_count=15, break_threshold_ratio=.001, depth=3,
        width=128, k0=-8, m=9, w_reg=0., rotation_format='axis_angle',
        motion_type='SE3', samples=2000, lr=.01, iters=500)
