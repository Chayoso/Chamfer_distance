"""Fetch pinned NDP and historical loss controls; no dataset download."""
import ast
import hashlib
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '63e9e2767e3db96a12bae18146971adeddcb9a4e'
SOURCES = {
    'nets': ('3cc718cf5437dc5cab60ff067594ad1911951e641fda649ed0ccc8df4094254e',
             {'Deformation_Pyramid', 'NDPLayer', 'MLP'}),
    'rigid_body': ('65a450ba134713601b79f2a7ad2fc0a9112d1ebae2fd9d5af4624160db448c97', None),
    'loss': ('3bd8b3914658cbe8bd6f183a38e268412be56288cda898578b1e1ba9e52f1cf1',
             {'_validate_chamfer_reduction_inputs', '_handle_pointcloud_input',
              'compute_truncated_chamfer_distance'}),
    'registration': ('69a8856370ede457cad19d030c761bf58a7f0647629af17420d57c64ff8b6ed6',
                     {'Registration'}),
}


def main():
    specs = []
    for name, (digest, names) in SOURCES.items():
        url = f'https://raw.githubusercontent.com/rabbityl/DeformationPyramid/{COMMIT}/model/{name}.py'
        specs.append((f'ndp_official/model_{name}.py', url, digest, names))
    specs += [
        ('infocd_official/loss_utils.py',
         'https://raw.githubusercontent.com/ark1234/NeurIPS2023-InfoCD/6a50bfd31a907ae3bd31635833df000b4dacb592/codes/utils/loss_utils.py',
         'c675feaf9b582a8765fba8f9bdd2fd2a2aa135b83e70aeb011f0ab185f431464', {'calc_cd_like_InfoV2'}),
        ('dcd_official/model_utils.py',
         'https://raw.githubusercontent.com/wutong16/Density_aware_Chamfer_Distance/3025c42ef6d29d192525e3d8c5c3b339f1099455/utils_v2/model_utils.py',
         '3b99f27792a364d06cf623f45a365d4e9daeac8bf1298b77eb6eaf861158bc41', {'calc_dcd', 'calc_cd'}),
    ]
    for name, url, digest, names in specs:
        path = ROOT/'assets'/name
        with urlopen(url, timeout=30) as response:
            raw = response.read(1_000_000)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f'Upstream hash mismatch: {name}')
        tree = ast.parse(raw.decode('utf-8'))
        tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                     and (names is None or n.name in names)]
        if name.endswith('model_registration.py'):
            tree.body[0].body = [n for n in tree.body[0].body
                                 if isinstance(n, ast.FunctionDef)
                                 and n.name == 'optimize_deformation_pyramid']
        text = f'# Upstream: {url}\n# Original SHA256: {digest}\n'+ast.unparse(tree)+'\n'
        if path.exists():
            if path.read_text(encoding='utf-8') != text:
                raise FileExistsError(f'Refusing to replace different definitions: {path}')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding='utf-8')
        print(path.relative_to(ROOT))


if __name__ == '__main__':
    main()
