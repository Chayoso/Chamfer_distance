# Testing "On the Structural Failure of Chamfer Distance in 3D Shape Optimization"

This project **tests** the paper's claims; it does not assume them. A disagreement with the
paper is reported as a finding only when the component producing it has been verified against
ground truth under the paper's stated assumptions. Each result below carries one label:

- **VERIFIED** - checked against ground truth / the paper's fully specified setting.
- **SUPPORTED (weak)** - direction agrees with the paper, effect is small.
- **DIFFERS (conditions unmatched)** - we disagree with the paper, but our setup departs from the paper's, so this is *not* evidence against it.

Run everything in the Nix devShell: `cd src && nix develop --command <cmd>` (add `PYTHONPATH=.` for scripts).

```
python -m pytest tests -q                                       # 37 tests, ~15 s
python scripts/run_2d_experiment.py                             # 2D table + figure
python -m chamfer_collapse.experiments.run_all --preset quick   # smoke, ~2 min
python -m chamfer_collapse.experiments.run_all --preset default # ~26 min on CPU; overwrites results/*.csv
python scripts/dco_scaling_sweep.py 1000 4000 20000 89000       # DCO vs point count (~30 min)
python scripts/fetch_stanford_meshes.py                         # OPT-IN real Bunny/Dragon
```
`results/default_run_log.txt` holds the output of the `default` run reported below
(CSVs are overwritten by whichever preset ran last).

## Ground-truth validation of the components

| Component | Check | Result |
|---|---|---|
| Propositions 1-3, Cor. 1, CD identity | closed-form gradient/Hessian, deadlock, translation invariance (`tests/test_propositions.py`) | VERIFIED |
| MLS-MPM simulator | B-spline partition of unity, mass conservation, rest stays at rest, momentum conservation, elastic restoring force, drag damps energy | VERIFIED |
| Neo-Hookean stress | equals autograd of the energy density | VERIFIED |
| Differentiable simulation | float64 `gradcheck` through 3 substeps | VERIFIED |
| Joint-optimizer gradient route | injected (clamped) reverse-Chamfer gradient == direct autograd when clamp inactive; Chamfer term changes the control gradient | VERIFIED |
| Outer optimizer (line search, schedules) | schedules match Supp. Table 2; the algorithm itself is a **reconstruction** (paper only cites Xu et al. 2025) | UNVALIDATED vs. original |

## Results

**2D circle -> star** (fully specified by the paper): ordering Fourier >> DCO ~ DCO+repulsion,
repulsion barely changes clustering - VERIFIED. Two-sided CD ours/paper: DCO 0.353/0.352,
DCO+rep 0.350/0.299, Fourier 0.139/0.123 (coverage differs: 15/38, 15/50, 55/62%).

**3D, DEFAULT preset** (800 MPM particles, 4K DCO points, object scale 4; CD in scale-4 units):

| Claim | Ours | Label |
|---|---|---|
| Joint (physics+Chamfer) beats physics-only (Tables 3-4) | wins 16/20 pairs; margins 0.2-6% (sphere->bunny 0.2178 vs 0.2218); F1 slightly higher | SUPPORTED (weak) |
| More particles help the joint method on the dragon (Table 5) | 0.384 (base N) -> 0.354 (1.9K particles); F1 0.49 -> 0.57 | SUPPORTED |
| DCO is worse than physics-only (Tables 1, 2, 5) | DCO CD is 4-6x *lower* than physics (e.g. 0.0375 vs 0.2218) | DIFFERS (conditions unmatched): 4K vs paper's 89K points, physics-only at 800 particles / 8 frames vs 37K / 40 |
| Local regularizers don't fix DCO (Table 1) | all regularized variants are worse than plain DCO (0.13-0.32 vs 0.0375); repulsion weights are in scale-dependent units | DIFFERS (conditions unmatched) |

**DCO vs. point count** (paper's DCO settings, 16-unit scale; `results/dco_scaling_sweep.csv`):

| N source | fwd | rev | two-sided |
|---|---|---|---|
| 1K | 0.069 | 0.207 | 0.218 |
| 4K | 0.076 | 0.125 | 0.146 |
| 20K | 0.113 | 0.071 | 0.133 |
| 89K | 0.151 | 0.043 | 0.157 |

Forward error grows and reverse error shrinks with N, consistent with the many-to-one
mechanism, but two-sided DCO stays well below the physics-only values here (dividing by scale:
DCO ~0.010 vs physics ~0.055 in unit-normalized units). The paper reports DCO 0.286 > physics 0.217.
The paper's ordering therefore is **not** reproduced at the scales we can run; whether that is
because our physics-only baseline is under-resourced is untested (open item below).

## Physics-only vs. DCO at increasing scale (the test of the paper's central 3D claim)

Sphere -> bunny, CD divided by object scale (`results/physics_scaling_sweep_all.csv`,
`scripts/physics_scaling_sweep.py`). Control step calibrated on physics-only alone
(0.3-1.2 / 1.2-4.8); the 37K row uses the best-supported value 1.2 without recalibration.
Last row is the paper's own resources: 37K particles, 40 frames, 32^3 grid.

| Particles / frames | Physics-only | Joint | DCO (same N) | Physics / DCO | Joint / Physics |
|---|---|---|---|---|---|
| 800 / 8 | 0.0555 | 0.0544 | 0.0171 | 3.24 | 0.982 |
| 3K / 12 | 0.0369 | 0.0367 | 0.0102 | 3.63 | 0.995 |
| 8K / 16 | 0.0238 | 0.0238 | 0.0087 | 2.75 | 0.999 |
| 20K / 24 | 0.0174 | 0.0174 | 0.0084 | 2.06 | 1.000 |
| **37K / 40 (paper scale)** | **0.0141** | **0.0141** | **0.0088** | **1.59** | **1.000** |

**Verdict: the paper's ordering is not reproduced at its own scale.** The paper has DCO worse
than physics-only (0.286 vs 0.217, i.e. physics/DCO = 0.76); we get DCO about 1.6x *better*.
The gap closes steadily with scale (3.6 -> 1.6) but has not crossed 1 at the paper's resources.
The joint method gives no measurable gain over physics-only at 20K+ particles.

Two observations that localise the disagreement (the units assumption is unverified: that the
paper's CDs are in its 16-unit grid scale, i.e. divide by 16):
- Our physics-only CD at paper scale is 0.0141, the paper's 0.217/16 = 0.0136: the physics baseline agrees.
- Our DCO is 0.0088 (and 0.0098 at 89K points); the paper's 0.286/16 = 0.0179, about 1.8x worse than ours.
So the discrepancy sits in DCO, not in the physics baseline.

This is a **disagreement under conditions that are not fully matched**, not a refutation: the
bunny is a procedural stand-in, the joint optimizer is a reconstruction, and the paper does not
specify how DCO is embedded in its pipeline.

## Calibrations and assumptions made (all are free parameters the paper does not fix)
- Control step size: preset 0.01 left every MPM run motionless (CD barely moved). DEFAULT uses 0.6, chosen on **physics-only alone** (sweep 0.3-4.0) so the baseline is not handicapped.
- Chamfer reduction: `chamfer_reduction="mean"` (default) makes the Chamfer gradient ~3% of the physics gradient at the original step size; `"sum"` was tried and was not better. The paper does not state which.
- DCO lr is scaled with object size (`paper_equivalent_lr`): the paper's 0.01 is in 16-unit grid scale. Running smaller scales with the same absolute lr hides collapse.
- MPM particles are volume-sampled, DCO points surface-sampled; the paper does not say how physics results are evaluated.
- Bunny/duck/cow/teapot/dragon are **procedural stand-ins** unless real meshes are placed in `assets/meshes/` (or fetched with the opt-in script).
- `cluster_fraction` is pinned at 100% for the Fourier method by geometry (see `metrics.py`).

## Open items
1. Run the sweep with the real Stanford Bunny (`scripts/fetch_stanford_meshes.py`) to remove the stand-in-shape caveat; find out why our DCO is ~1.8x better than the paper's (units, DCO embedding, sampling).
2. Joint optimizer: validate against Xu et al. 2025's implementation if available.
3. `--preset paper` (32^3 grid, 37K-89K particles) is provided but untested; expect hours.
