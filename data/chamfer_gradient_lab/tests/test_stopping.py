import ast
import copy
import torch
from pipeline.model import OFFICIAL
from pipeline.stopping import continuation, stopping_environment


def test_original_ast_unchanged_after_removing_observers():
    original = ast.parse((OFFICIAL/'model_registration.py').read_text())
    method = next(n for n in ast.walk(original) if isinstance(n, ast.FunctionDef)
                  and n.name == 'optimize_deformation_pyramid')
    _, module = stopping_environment(lambda *a: None, lambda *a: None)
    class RemoveObservers(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                if node.value.func.id in ('audit_step', 'audit_end'):
                    return None
            return node
    cleaned = RemoveObservers().visit(module).body[0]
    assert ast.dump(cleaned) == ast.dump(method)


def test_ceiling_removes_only_original_stopping_checks():
    _, module = stopping_environment(lambda *a: None, lambda *a: None, True)
    text = ast.unparse(module)
    assert 'range(self.config.iters)' in text
    assert 'break_counter += 1' not in text and 'break\n' not in text
    assert 'optimizer.step()' in text and 's_sample = s_sample_warped.detach()' in text


def test_continuation_preserves_adam_history_and_original():
    class Layer(torch.nn.Linear):
        def forward(self, x):
            return super().forward(x), None
    torch.manual_seed(51)
    model = Layer(2, 2).double()
    x, y = torch.randn(7, 2).double(), torch.randn(7, 2).double()
    loss = lambda a, b: (a-b).square().mean()
    opt = torch.optim.Adam(model.parameters(), lr=.01)
    for _ in range(4):
        opt.zero_grad(); loss(model(x)[0], y).backward(); opt.step()
    before = copy.deepcopy(model.state_dict())
    path = [float(v.detach()) for _, _, v, _ in continuation(model, opt, x, y, loss, steps=3)]
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    for i in range(4):
        value = loss(model(x)[0], y)
        assert float(value.detach()) == path[i]
        if i < 3:
            opt.zero_grad(); value.backward(); opt.step()
