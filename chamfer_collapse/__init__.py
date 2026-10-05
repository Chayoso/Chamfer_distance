"""Reproduction of "On the Structural Failure of Chamfer Distance in 3D Shape
Optimization" (ECCV 2026 submission).

Sub-packages / modules
-----------------------
``chamfer``        Forward / reverse / two-sided Chamfer distance and the
                    per-point gradient of Eq. (1) in the main paper.
``metrics``         Hausdorff distance, F1-score, coverage, cluster fraction.
``regularizers``    Repulsion, Laplacian smoothness, volume preservation
                    (all translationally invariant, cf. Proposition 3).
``dcd``             Density-aware Chamfer Distance (Wu et al. 2021) as
                    described in the supplementary remark to Proposition 3.
``shapes``          Source/target point cloud generation (procedural
                    stand-ins for Bunny/Duck/Cow/Teapot/Dragon, with
                    transparent override by real meshes dropped into
                    ``assets/meshes``).
``dco``             Direct Chamfer Optimization (Sec. 3.2-3.4).
``experiment_2d``   The circle -> star cross-domain collapse experiment
                    (Sec. 3.5 / Supplementary Sec. 2).
``mpm``             Differentiable Material-Point-Method simulator and the
                    physics-only / joint physics+Chamfer morphing baselines
                    of Sec. 4.
"""

__version__ = "0.1.0"
