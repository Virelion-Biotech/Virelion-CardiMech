# Mixed MINI correction: frozen acceptance and scope

Recorded before the first candidate numerical run (2026-10-22). Original failing
displacement P1 data remain archived unchanged. No material, pressure, mesh,
boundary condition or analytical reference is adjusted to improve the result.

The proposed method is finite-strain P1 + quartic interior displacement bubble /
continuous P1 pressure (MINI), with exact local Newton condensation of the bubble.
Independent pressure is a Legendre transform of the existing log(J) penalty.
The final candidate constrains J directly on the convex branch 0<J<e; the
first log(J)-pressure candidate failed the finite compressible patch. This
formulation revision does not change the acceptance criteria below.
It must reproduce the original pointwise material after pressure elimination.

Acceptance:
- Quadrature polynomial moments, mixed energy/stress/tangent/objectivity, full
  finite-difference Schur complement and bubble recovery pass.
- At the undeformed limit, bubble coupling controls every pressure-gradient mode
  on the connected cube; only a constant pressure mode may remain as bulk
  compliance tends to zero. This finite test is not a universal inf-sup proof.
- Finite isochoric and compressible affine patches, both materials with active
  stress, positive quadrature J and true uncondensed residual checks pass.
- Repeat all nine original shell cases with identical meshes/loads/materials.
  Preserve the original 15% accuracy screen on every row, including any remaining
  coarse-geometry failures. For all three material regimes the finest mesh must
  be within 15%; nearly incompressible finest error must improve at least 4-fold
  relative to the original 80.1939%. Report the full refinement trend.
- Increase quadrature 5 to 6 and require <1% change in finest near-incompressible
  displacement; repeat at lower pressure to distinguish finite-strain reference
  effects, and at higher pressure to exercise finite-strain equilibrium.
- Existing displacement/direct and matrix-free tests remain passing.

No exact-incompressible, large-mesh, heterogeneous-interface, dynamic, whole-heart
or patient accuracy claim follows. Initial implementation is an explicit
`settings.formulation="mixed_mini"` option with assembled sparse direct solve.
