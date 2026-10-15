"""Frozen pre-refactor loop, extracted without changing its function AST."""
import ast
from pipeline.model import OFFICIAL, official

def fixed_official(common_origin=False):
    env=official();tree=ast.parse((OFFICIAL/'model_registration.py').read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Registration')
    method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='optimize_deformation_pyramid')
    if common_origin:
        nodes=[x for x in ast.walk(method) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='src_mean' for t in x.targets)]
        assert len(nodes)==1;nodes[0].value=ast.parse('self.tgt_pcd.mean(dim=0, keepdims=True)',mode='eval').body
    loop=next(x for x in ast.walk(method) if isinstance(x,ast.For) and isinstance(x.target,ast.Name) and x.target.id=='iter')
    loop.iter=ast.parse('range(self.config.budget[level]+1)',mode='eval').body
    found=[];body=[]
    for node in loop.body:
        condition=ast.unparse(node.test) if isinstance(node,ast.If) else ''
        if condition=='loss.item() < 0.0001':
            node.test=ast.parse('iter == self.config.budget[level]',mode='eval').body;found.append('absolute')
        elif condition in ['abs(loss_prev - loss.item()) < loss_prev * break_threshold_ratio','break_counter >= max_break_count']:
            found.append(condition);continue
        body.append(node)
    assert len(found)==3
    loop.body=body;module=ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]))
    exec(compile(module,'matched_budget_control','exec'),env)
    return env
