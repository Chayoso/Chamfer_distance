"""Exact-kernel numerical control: unchanged states, scales, transfer and gates."""
import ndp_field_probe as probe
from pipeline.fields import build_fields, gaussian_force, gaussian_objective


def exact_fields(p, q, seed):
    for name, force, objective, info in build_fields(p, q, seed):
        if name.startswith('mmd'):
            force = gaussian_force(p, q, info['bandwidth'])
            objective = gaussian_objective(q, info['bandwidth'])
            info.update(features=0, estimator='exact_gaussian',
                        exact_gradient_relative_error=0., exact_gradient_cosine=1.)
        yield name, force, objective, info


if __name__ == '__main__':
    probe.OUT = probe.OUT/'exact'
    probe.build_fields = exact_fields
    probe.main()
