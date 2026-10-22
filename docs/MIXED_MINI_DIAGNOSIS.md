# Nearly incompressible mechanics: diagnosis and candidate correction

Status on 2026-10-22: **synthetic qualification passed; see [executed results](MIXED_MINI_RESULTS.md)**.

The archived original case has 80.1939% inner radial displacement error on a
1,548-node/7,680-tetrahedron traction-free shell, despite direct-solver equilibrium.
Its small-strain neo-Hookean constants are mu=1,000 Pa and lambda=10,000,000 Pa
(the existing parameter named `kappa` is lambda for this material). Pressure is
0.1 Pa, radii are 0.02/0.03 m, with normal displacement fixed on coordinate planes.
These symmetry constraints permit the analytical radial solution. The reference
is the traction-free outer-boundary thick-sphere solution, not the separately
qualified clamped compressible shell.

A small force residual tests discrete equilibrium, not spatial approximation.
Element-wise P1 volume constraints can suppress deformation as lambda/mu grows.
The original direct solve excludes GMRES as the cause. The independent linear
elasticity diagnostic in `scripts/diagnose_locking.py` is prepared to separate
locking from nonlinear/reference effects; its run is still pending.

## Candidate method and unchanged continuum law

Use continuous nodal P1 displacement plus one quartic interior vector bubble per
tetrahedron, and continuous nodal P1 pressure. The bubble is
`256*L0*L1*L2*L3`, zero on every face. This is the MINI enrichment; the original
Stokes construction and its small-strain inf-sup result are described by
[Arnold, Brezzi and Fortin (1984)](https://doi.org/10.1007/BF02576171).
That result is not a blanket finite-strain or arbitrary-mesh stability guarantee.

Let `W0(F)=W(F;kappa=0)` and `U(J)=kappa/2*log(J)^2`. On the strictly convex branch
`0<J<e`, introduce `p=U'(J)=kappa*log(J)/J` and its exact dual `U_star(p)`:

`Pi(F,p)=W0(F)+p*(J-1)-U_star(p)`.

The principal real Lambert-W branch gives `J(p)=exp(-W(-p/kappa))` for
`p<kappa/e`, with `dJ/dp=J^2/[kappa*(1-log(J))]`. Pressure elimination recovers
the original energy, stress and consistent tangent. No bulk softening, penalty
reduction, fitted displacement correction or changed load is introduced.
The discrete pressure constraint is weak; it intentionally does not reimpose
one independent volume constraint in every tetrahedron.

Each Newton system condenses the three bubble increments exactly through a
3x3 Schur complement, then recovers and updates them. Acceptance uses the full
nodal displacement, bubble and scaled pressure residuals. The condensed residual
alone cannot terminate the solve. Pressure is scaled by a fixed shear-modulus /
length factor, giving all residual components units of N. The physical tangent
is neither clipped nor regularized.

Output retains nodal displacement, nodal dual pressure, bubble amplitudes,
quadrature-averaged deformation/strain/stress, minimum quadrature J, pressure
scale, residual breakdown and field/energy definitions. Nodal displacement alone
does not describe the full interior field. Dual pressure is not total hydrostatic
Cauchy stress. Energy is the integrated mixed potential, not a re-evaluation of
the element-wise displacement penalty.

## Errors encountered and corrected

- Initial lint: import order and unused test variable; corrected without numerical
  criterion changes.
- Order-4 tensor Duffy quadrature failed an independent degree-6 moment. It is now
  rejected; orders 5–7 are supported. Default remains 5, checked against 6 in the
  pending qualification campaign.
- The active neo-Hookean patch fixture lacked explicit fibre/sheet directions;
  the existing strict material contract correctly rejected it. Fixture corrected.
- The new mixed branch did not set the final QC tolerance in the wrapper; fixed
  by using the actual converged mixed-step tolerance.
- A first `p*log(J)-p^2/(2*kappa)` candidate failed a finite compressible Guccione
  patch. A log-determinant constraint adds a pressure-dependent negative bubble
  curvature. The determinant-dual formulation above avoids this term while
  preserving the same continuum volume energy on its stated convex branch.

## Executed verification

The first determinant-pressure candidate passed 17 Colab tests in 38.02 s.
That pilot's result was not retrieved when browser context became unavailable;
its raw runtime logs were not exported. The fresh final pinned notebook supersedes
that interrupted attempt and supplies the qualification evidence.

Implementation `beaf842e5f6518e74b69b03c40ece7f9ca8e0f93` passed the complete suite:
146 tests, no failures/errors/skips, 89.41% coverage. The independent linear
calculation reproduced the 80.19% P1 error; all frozen shell/refinement/quadrature/
load/higher-bulk gates then passed. The original finest nonlinear P1 error was
80.1939%; MINI reduced it to 2.5347%. Three coarse MINI rows still fail the
unchanged 15% screen. See [all cases and raw evidence](MIXED_MINI_RESULTS.md).

## Selection and limits

`settings.formulation` accepts `auto` (default), `displacement_p1` and `mixed_mini`.
Auto selects mixed MINI for neo-Hookean nu>=0.49 with the standard direct solver;
it rejects other solver/profile combinations in that regime unless the caller
explicitly opts into legacy displacement P1. Other auto cases preserve P1.
Guccione has no single isotropic Poisson ratio and requires explicit mixed
selection. Existing legacy P1 risk warnings remain.

This candidate supports finite positive bulk compliance, homogeneous parameters,
assembled direct solve and the standard mesh profile only. It rejects J>=e and
p>=kappa/e rather than silently crossing a nonconvex energy branch. Exact
incompressibility, scalable mixed preconditioning, heterogeneous interfaces,
pathological/deformed meshes and patient/whole-heart accuracy are unqualified.
The Colab accuracy gate passed. Release/integration still requires latest-head
CI and the HeartTwin pinned/current-main/operational checks; software success
does not establish patient prediction accuracy.
