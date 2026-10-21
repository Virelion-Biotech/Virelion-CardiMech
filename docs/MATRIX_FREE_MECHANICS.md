# Experimental matrix-free spatial mechanics

**Follow-up qualification:** cached material state and optional `linear_preconditioner="block_jacobi"` are now available. A separate direct-solver near-incompressible benchmark measured about 80.2% error at its finest tested mesh despite equilibrium convergence. The compressible shell results below do not establish accuracy in that regime. See [ROBUSTNESS_AND_LIMITS.md](ROBUSTNESS_AND_LIMITS.md) for the executed replay and unresolved locking defect.

The direct solver remains the default with the original 5,000-node/20,000-tetrahedron guard. Explicit larger-mesh research settings:

```json
{
  "linear_solver": "matrix_free_gmres",
  "mesh_profile": "large_experimental",
  "linear_relative_tolerance": 1e-8,
  "linear_max_iterations": 1200,
  "linear_restart": 60
}
```

These are MechanicsSimulationRequest.settings entries alongside existing output/Newton controls. The large_experimental profile accepts at most 40,000 nodes and 200,000 tetrahedra and requires the matrix-free solver. It is a computational research profile. Subject/SHA, units, connectivity, cavity orientation, inversion, rigid-mode and equilibrium checks remain enforced. No direct fallback occurs when GMRES fails.

The operator applies exact first-Piola directional derivatives, active-fibre stress and follower-pressure derivatives. A material Jacobi diagonal is formed in element batches. No global tangent or sparse factorization is built. Restart limits bound Krylov storage. The **unpreconditioned linear residual** must meet tolerance before accepting an increment; a claimed GMRES success with a wrong solution is rejected. Both SciPy 1.11 tol and later rtol interfaces are supported. The nonlinear line search and equilibrium criteria remain unchanged.

## Executed evidence

All numerical validation ran in the user's Edge Colab. Executed implementation source: `b6f048e3a50b8affe7a13bfe666edd143075372e`. All five changed source blobs were checked against that published commit.

**114 tests passed, no skips**, including CardiInfer integration, rotated anisotropic material tensor/finite-difference agreement, assembled/finite-difference global actions with follower pressure, diagonal agreement, true-residual/nonconvergence rejection and nonlinear direct/GMRES agreement for both materials with active tension. The nonlinear matrix-free tests reject global tangent assembly. Formatting, lint and compilation passed.

| Nodes | Tetrahedra | Mean inner radial displacement error | Peak process RSS | Wall time | Total GMRES iterations |
|---:|---:|---:|---:|---:|---:|
| 1,032 | 4,608 | 3.188% | 102.6 MiB | 3.57 s | 106 |
| 6,156 | 30,720 | 1.152% | 225.2 MiB | 36.11 s | 177 |
| 32,784 | 172,032 | 0.546% | 906.1 MiB | 358.47 s | 325 |

All cases converged in two Newton iterations, with positive Jacobians and equilibrium residual below 1.2e-14 N. Errors decreased strictly with angular/radial refinement and the finest result was below 1%. The standard profile correctly rejected both larger cases. The largest synthetic mesh exceeds KIT's node and tetrahedron counts; its geometry and boundary conditions differ.

The independent reference is the small-strain Lamé solution for a spherical shell with fixed outer boundary, 1 Pa inner pressure, mu=1,000 Pa and lambda=kappa=2,000 Pa. Finite-deformation and discretization effects both contribute to the error. Peak RSS is each separate process's Linux high-water mark including imports, parsing, checks, solve and output. Timings are one Colab run, with other Colab tests executing during part of the large run; no general performance claim follows.

[Verified evidence archive](../validation/matrix_free_2026_10_21/matrix_free_evidence.zip): SHA-256 `28f16137b659b81f1aab85a6af4c2e9898dd2c649836f50b7860cff77c0615eb`, Git blob `b1b99de47ed109c817f89aeccd481bfd0cef7736`. It contains setup/test/failure logs, JUnit, exact source copies, qualification records and all raw nodal coordinates/displacements. Bulky complete stress/strain/tetrahedron JSON and mesh JSON are identified by SHA-256 in the internal MANIFEST.json and omitted; regenerate with the pinned source. Python 3.13.16, NumPy 2.1.3, SciPy 1.16.3, Pydantic 2.14.1, pytest 8.4.2, Ruff 0.17.1; CardiInfer pin 9aabf87bdbee4c70b98229baa3bbea3fb896b490.

## Replay

In Colab with the pinned package installed, run each case in a separate process:

```bash
python scripts/validate_matrix_free.py --level 3 --layers 4 --output /content/shell-3-4
python scripts/validate_matrix_free.py --level 4 --layers 6 --output /content/shell-4-6
python scripts/validate_matrix_free.py --level 5 --layers 8 --output /content/shell-5-8
```

The script saves raw fields, geometry and qualification.json, checks equilibrium and a 10% per-case analytical bound. The executed campaign additionally checked strictly decreasing errors and a finest-case error below 1%. A consolidated notebook was not completed/replayed after browser control was interrupted; the retained script and executed campaign are the reproducible evidence.

## Diagnosed failures

The first targeted run had 12 passes and one failure: an overly broad source edit inserted settings names as extra positional arguments to finite_number. The tolerance call was corrected without loosening acceptance. Lint subsequently found an unused test variable, renamed _req. Failed and successful logs are retained. No component numerical defect or cross-repository conflict was found in this campaign; CardiInfer integration passed.

## Limits

Large-scale evidence covers this constrained, compressible, neo-Hookean, 1 Pa synthetic shell. Guccione and active tension have small-mesh numerical tests. Jacobi preconditioning may converge slowly or fail for other meshes, constraints, material contrasts or stiffness; no multigrid/domain-decomposition preconditioner is supplied. P1 displacement elements can lock near incompressibility. This does not qualify KIT patient geometry, unloaded-reference reconstruction, reviewed fibres/cavities, physiological loads/coupling or clinical accuracy. Shared-face mesh checks do not provide exhaustive three-dimensional overlap detection.
