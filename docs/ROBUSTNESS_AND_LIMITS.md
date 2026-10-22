# Mechanics robustness and qualification limits

**Subsequent correction:** [mixed MINI qualification](MIXED_MINI_RESULTS.md) reduced the unchanged finest near-incompressible error from 80.19% to 2.53%. The failed P1 results below are preserved historical evidence; the matrix-free P1 approximation remains susceptible to locking. Auto now selects standard/direct MINI for nearly incompressible neo-Hookean requests or rejects unsupported combinations.

## Implemented

Matrix-free Newton solves now cache F, inverse transpose, log determinant and (for Guccione) second-order anisotropic material coefficients once per linear solve. No fourth-order element tangent cache or global sparse tangent is retained. This exchanges O(elements) second-order storage for fewer repeated inverse/material evaluations. The uncached directional implementation remains as an independent regression path. New tests compare cached actions with assembled and uncached actions, including rotated fibres, active tension and follower pressure. Caller displacement/parameter mutation does not alter the frozen material state.

`settings.linear_preconditioner` accepts `jacobi` (default) or `block_jacobi` with `linear_solver="matrix_free_gmres"`. The block option uses nodal 3x3 **material** blocks; pressure terms remain in the true operator. Fixed components are eliminated from blocks. Symmetrized blocks are spectrally regularized using absolute eigenvalues and a 1e-12 global scale floor to produce a positive preconditioning approximation. This does not modify the physical tangent, residual, Newton line search or true-residual acceptance. Partial constraints, zero/negative block eigenvalues, both materials and active tension have regressions.

For this neo-Hookean law the small-strain Lamé lambda equals kappa. Results report `provenance.linearized_material_regime.isotropic_poisson_ratio` and a P1 locking-risk flag at nu>=0.49. Affected runs carry an explicit accuracy warning. Guccione returns null for an isotropic Poisson ratio and flag because it is anisotropic. This is a regime label, not an error estimator or an accuracy certificate.

## Observed accuracy failure

The direct solver was used for a traction-free outer shell with symmetry constraints, 0.1 Pa inner pressure, mu=1,000 Pa and lambda in {1,000;100,000;10,000,000} Pa. Each regime was refined through 72 / 330 / 1,548 nodes and 288 / 1,536 / 7,680 tets. Compare mean inner radial displacement against the independently evaluated small-strain Lamé sphere.

| lambda/mu | Small-strain nu | Level 1 error | Level 2 error | Level 3 error |
|---:|---:|---:|---:|---:|
| 1 | 0.25000 | 20.12% | 6.22% | 2.12% |
| 100 | 0.49505 | 29.62% | 14.32% | 8.94% |
| 10000 | 0.49995 | 85.85% | 84.99% | 80.19% |

All nine cases met their equilibrium tolerance, yet **five failed the 15% research regression screen**. The finest nearly incompressible case remained about 80.2% too stiff. Its equilibrium residual was about 1.3e-12 N. The low-lambda finest result was about 2.1% error. This behavior is consistent with displacement P1 volumetric locking; using a direct solve rules out GMRES nonconvergence as the explanation. Small deformation makes the linear reference appropriate as a numerical check, but this is not biological validation.

The 15% screen is a broad synthetic regression criterion, not a clinical accuracy standard. nu below 0.49 is not an accuracy guarantee: the coarse low-lambda mesh also failed. Warning presence does not determine a measured error for another mesh. Preconditioning can improve convergence but cannot correct the underlying approximation space. **Cached actions and preconditioning did not fix locking; the subsequent mixed formulation addresses it separately, as linked above.**

## Executed checks and replay

128 tests passed without skips in Edge Colab, including CardiInfer integration. Initial lint found an unbound loop-variable closure in the profiling helper; explicit binding corrected it before profiling. The initial and corrected logs are retained.

The **published notebook was executed end to end in Edge Colab** in a fresh, isolated virtual environment. It passed formatting, lint, compilation, all 128 tests without skips, paired operator comparisons, material characterization and block-preconditioned shell refinement. Python 3.13.16, NumPy 2.1.3, SciPy 1.16.3, Pydantic 2.14.1 and pytest 8.4.2 were pinned; all packages are recorded in environment.txt. Executed implementation: `9da8da3a395e9637f7911a10bc0c36b69b04b24a`. Subsequent evidence/docs commits preserve those source bodies.

| Nodes | Tetrahedra | Compressible shell error | GMRES iterations | Peak RSS | Wall time |
|---:|---:|---:|---:|---:|---:|
| 1,032 | 4,608 | 3.188% | 96 | 102.7 MiB | 2.89 s |
| 6,156 | 30,720 | 1.152% | 160 | 226.4 MiB | 26.12 s |
| 32,784 | 172,032 | 0.546% | 283 | 883.0 MiB | 212.18 s |

All three block-preconditioned fixed-outer shells converged in two Newton iterations with positive Jacobians. The independent reference and scope are the same compressible, neo-Hookean, 1 Pa shell described in [MATRIX_FREE_MECHANICS.md](MATRIX_FREE_MECHANICS.md). Paired six-action timings on a 1,032-node mesh observed cached/uncached median speedups of 1.72x for neo-Hookean and 1.73x for Guccione in this replay, with matching actions. Setup/cache storage are recorded separately. These are operator measurements in one environment, not universal solve-time improvements. The 212-second large run is not a controlled timing comparison against earlier Colab hardware.

[Verified evidence ZIP](../validation/robustness_2026_10_22/mechanics_completion_evidence.zip), [manifest](../validation/robustness_2026_10_22/MANIFEST.json), [executed notebook copy](../validation/robustness_2026_10_22/Mechanics_Completion_executed.ipynb), and [Colab screenshot](../validation/robustness_2026_10_22/colab_completion.png). ZIP SHA-256 `6edaa388de56fb5facaddcc561f81f24a7031e01c1418f3178c27caae0c781a2`; Git blob `57e687e63cd404c0c7eda96c640cc2947850dd4a`. The ZIP includes raw nodal coordinates/displacements for all 12 cases, logs, JUnit, source copies, package versions and full-field/mesh SHA-256 identities. Bulky element-level JSON is omitted and can be regenerated. The separate [initial diagnostics](../validation/robustness_2026_10_22/initial_diagnostics.json) preserves the original closure lint finding and its correction. The executed notebook copy pairs the published source with stdout captured from the actual Colab output.


The published [Colab notebook](../notebooks/Mechanics_Completion.ipynb) pins the implementation commit, numerical package versions and CardiInfer commit. Its isolated environment does not inherit Colab site packages. It runs tests, paired operator profiling, the nine-case material sweep, block-preconditioned shell refinement and evidence export. `software_checks_passed` and `material_accuracy_qualification_passed` are separate fields; the latter is false. Successful recipe execution must not be described as all material regimes passing.

## Remaining engineering

The separate mixed MINI implementation and frozen synthetic verification campaign are recorded above. Broader work remains: stability/pressure modes, incompressible patch/objectivity tests, consistent tangent checks, independent analytical refinement over material regimes, solver/preconditioner robustness and coupled model contracts. It is not sufficient to loosen Newton tolerances, increase kappa, change a flag or mask the failed cases.

Block Jacobi does not supply multigrid, domain decomposition or universal mesh/material robustness. Large-scale evidence remains confined to the tested compressible neo-Hookean shell and its constraints. Actual KIT needs qualified reference state, reviewed geometry/fibres/cavities/materials and boundary conditions. EDGAR export conventions remain unqualified. Human cellular physiology, Purkinje/atrial details, circulation/valve coupling, hospital workflow access and independent patient accuracy still need separate work. The system is not a fully personalized, clinically qualified whole-heart twin.
