# Mixed MINI locking correction — executed Colab qualification

The frozen synthetic qualification passed on 2026-10-22. On the unchanged
1,548-node/7,680-tetrahedron nearly incompressible shell, P1 displacement error
was **80.1939%** and MINI error was **2.5347%** (31.64-fold reduction).
This corrects the demonstrated locking defect on this benchmark. It does not
remove mesh error or qualify patient predictions.

Executed implementation: `beaf842e5f6518e74b69b03c40ece7f9ca8e0f93`.
The public notebook at `f29a3da335083c649c23b5f52221c95520bc64ef` ran unedited
end to end in an isolated Edge Colab environment. The full suite passed:
**146 tests, zero failures/errors/skips, 89.41% coverage**.

The separate independent linear-elastic assembly (no nonlinear material routine,
Newton or GMRES) reproduced **80.1945%** P1 error on the finest mesh. Together with
the direct nonlinear baseline, material derivative tests and the unchanged
geometry/loads/reference, this supports discretization locking as the root cause.

## All original cases, with the original 15% screen

mu=1,000 Pa, pressure=0.1 Pa, radii=0.02/0.03 m, traction-free outer surface,
normal slip on coordinate planes. Lambda is the existing neo-Hookean `kappa`.
All 18 baseline/comparison solves converged. No failure is removed from the table.

| Lambda (Pa) | Mesh level | Original P1 error | MINI error | MINI 15% screen |
|---|---|---|---|---|
| 1000 | 1 | 20.1225% | 19.9444% | **fail retained** |
| 1000 | 2 | 6.2157% | 6.0383% | pass |
| 1000 | 3 | 2.1246% | 1.9766% | pass |
| 100000 | 1 | 29.6218% | 24.9629% | **fail retained** |
| 100000 | 2 | 14.3173% | 7.5890% | pass |
| 100000 | 3 | 8.9413% | 2.5162% | pass |
| 1e+07 | 1 | 85.8491% | 25.0845% | **fail retained** |
| 1e+07 | 2 | 84.9872% | 7.6300% | pass |
| 1e+07 | 3 | 80.1939% | 2.5347% | pass |

Three coarse mixed meshes still fail the original screen. All three finest
mixed meshes pass; error decreases under refinement in every material regime.
The 15% screen is a research regression threshold, not a clinical standard.

## Additional frozen gates

All eight frozen acceptance checks passed. Including the additional cases,
**23/23 equilibrium solves converged**, with positive sampled quadrature J.
- Raising quadrature order 5 to 6 changed finest near-incompressible displacement
  by about `5.51e-13` relative; both measured about 2.5347% reference error.
- Ten-times-lower pressure gave 2.5428% error; normalized displacement changed
  by less than 1%, separating finite-strain/reference effects from locking.
- Raising lambda from 1e7 to 1e9 Pa gave 2.5349% error: no renewed bulk-dependent
  locking was observed in this test.
- A 100 Pa finite-load pair on level 2 converged with minimum sampled J above
  0.977. Quadrature changed displacement by about `1.49e-9` relative. This is a
  finite-load equilibrium/quadrature check, not an exact nonlinear continuum
  accuracy qualification; the listed analytical reference is small-strain.
- The maximum full uncondensed residual across the nine MINI baseline rows was
  about `2.95e-16 N`; the finite-load rows had about `8.67e-12 N`, within their
  frozen tolerances. Bubble and scaled pressure residuals are included.

## Evidence and reproducibility

[Evidence directory](../validation/mixed_locking_2026_10_22) contains the compact
archive, full manifest, qualification JSON, executed notebook and Colab screenshot.
The archive contains all 23 raw states (coordinates, tetrahedra, displacement,
pressure/bubble amplitudes where applicable, element J), complete stage logs,
JUnit, exact source copies and bulk-file identities. Its SHA-256 is
`c8e1de7580c046e68e8541040230cf84286c1db64aace5a3173481a79bf3cb36`.

Full element-field and mesh JSON are omitted from the compact archive; their
hashes and the complete state/geometry needed to regenerate them are retained.
A 21,729,648-byte full-field archive was also generated in Colab, with SHA-256
`e7599570a02d059bb452165ef973a1351352e427be1e2543f6ddc45c2611adbc`;
the notebook offered it as a browser download. The repository does not contain
that larger archive. The earlier interrupted pilot is not used as qualification.

## Supported use and limits

Use `settings.formulation="mixed_mini"` explicitly, with the standard mesh profile
and direct solver. Default `auto` selects it for neo-Hookean nu>=0.49 and rejects
unsupported solver/profile combinations; explicit `displacement_p1` preserves
legacy comparisons and warnings. Guccione requires explicit mixed selection.

The continuum volume energy is unchanged, using an exact determinant-pressure
dual on 0<J<e. Quadrature-averaged field definitions and pressure scaling are
explicit in outputs; surface/nodal displacement alone does not represent the
interior bubble field. See [derivation/error diagnosis](MIXED_MINI_DIAGNOSIS.md)
and [frozen criteria](MIXED_MINI_ACCEPTANCE.md).

The finest mixed case took about 193 seconds in this Colab run; this is not a
large-mesh capacity or universal timing claim. Exact incompressibility, scalable
mixed preconditioning, arbitrary/distorted meshes, heterogeneous interfaces,
finite-strain pressure stability beyond the tests, whole-heart physiology and
patient accuracy remain unqualified. Coarse failures remain real failures.
