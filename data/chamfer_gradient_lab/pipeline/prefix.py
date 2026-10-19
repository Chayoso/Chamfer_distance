"""GT-free observation of a shared optimization prefix; no update changes."""
from .model import torch
from .losses import objective
from .horizon import digest


def prefix_observer(base_factory,metadata,calls=17,loss_fn=objective):
    state={};metadata['outputs']=[]
    def factory():
        env=base_factory();build=env['Deformation_Pyramid'];optimize=env['optimize_deformation_pyramid']
        def constructor(**kwargs):
            net=build(**kwargs);instance=state['instance']
            generator=torch.Generator().set_state(torch.get_rng_state())
            ids=torch.randperm(len(instance.src_pcd),generator=generator)[:instance.config.samples]
            metadata.update(source_ids_sha256=digest([ids]),rng_after_construction=digest([torch.get_rng_state()]),
                initial_parameters_sha256=digest([p for layer in net.pyramid for p in layer.parameters()]))
            return net
        def run(instance):
            state['instance']=instance
            return optimize(instance)
        env.update(Deformation_Pyramid=constructor,optimize_deformation_pyramid=run)
        return env
    def loss(p,q,*args):
        value=loss_fn(p,q,*args)
        if len(metadata['outputs'])<calls:
            metadata['outputs'].append(dict(prediction=digest([p]),target=digest([q]),rng=digest([torch.get_rng_state()])))
        return value
    return factory,loss
