"""Fixed-budget NDP optimization; no experiment setup or data loading."""
import math
from .model import official, torch


def fixed_official(common_origin=False):
    """Plain fixed-budget loop for the landmark-free, unregularized experiments."""
    env = official()

    def optimize(instance, visualize=False, timer=None):
        cfg = instance.config
        if instance.landmarks is not None or cfg.w_reg != 0 or visualize or timer is not None:
            raise ValueError('This experiment adapter supports unregularized point sets only')
        if len(cfg.budget) != cfg.m or any(n < 0 for n in cfg.budget):
            raise ValueError('Provide one nonnegative update count per level')
        schedule = getattr(cfg, 'schedule', 'constant')
        if schedule not in ('constant', 'cosine'):
            raise ValueError('Unknown learning-rate schedule')
        net = env['Deformation_Pyramid'](depth=cfg.depth, width=cfg.width,
            device=instance.device, k0=cfg.k0, m=cfg.m, nonrigidity_est=False,
            rotation_format=cfg.rotation_format, motion=cfg.motion_type)
        x, y = instance.src_pcd.to(instance.device), instance.tgt_pcd
        target_mean = y.mean(dim=0, keepdims=True)
        source_mean = target_mean if common_origin else x.mean(dim=0, keepdims=True)
        x, y = x-source_mean, y-target_mean
        source_ids, target_ids = torch.randperm(len(x)), torch.randperm(len(y))
        sample, target = x[source_ids[:cfg.samples]], y[target_ids[:cfg.samples]]
        for level, updates in enumerate(cfg.budget):
            net.gradient_setup(optimized_level=level)
            optimizer = env['optim'].Adam(net.pyramid[level].parameters(), lr=cfg.lr)
            for step in range(updates+1):
                warped, _ = net.warp(sample, max_level=level, min_level=level)
                loss = env['compute_truncated_chamfer_distance'](warped[None], target[None], trunc=1e9)
                if step == updates:
                    break
                if schedule == 'cosine':
                    optimizer.param_groups[0]['lr'] = cfg.lr*(.1+.9*(1+math.cos(math.pi*step/max(1,updates-1)))/2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            sample = warped.detach()
        net.gradient_setup(optimized_level=-1)
        pred, _ = net.warp(x)
        return pred+target_mean, {}, timer

    env['optimize_deformation_pyramid'] = optimize
    return env
