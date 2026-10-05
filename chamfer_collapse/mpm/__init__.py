"""Differentiable Material Point Method (MPM) and the physics-only /
joint physics+Chamfer morphing baselines of Sec. 4 of the main paper.

Honesty note (see also the package README): Xu et al. 2025's morphing
algorithm is present in the paper only by citation -- the main text and
supplement describe *what* it optimizes (grid-level mass-density loss,
Eq. 4), the material model and grid parameters (Supplementary Table 1),
and the outer optimization schedule (Supplementary Table 2: 3
forward-backward passes / frame, 1 GD iteration / pass, up to 5
line-search iterations, adaptive step norm target 2500), but not the
inner control representation Xu et al. actually optimize. ``joint_opt.py``
reconstructs a control representation consistent with every parameter the
paper *does* specify (a per-frame, per-particle learnable velocity field,
optimized by backpropagating through the differentiable simulator) but it
is a reconstruction, not a transcription of Xu et al.'s code.
"""
