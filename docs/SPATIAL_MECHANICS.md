# Spatial tetrahedral mechanics

Version 0.4.0 ships `scipy-tetra-v1`, an actual CPU total-Lagrangian P1 tetrahedral
finite-strain equilibrium solver. SciPy is optional: install `[spatial]`. Core
installation still supports the NumPy LV/PV reference backend.

The solver assembles element internal forces and exact material tangents, plus
follower pressure forces and their geometric tangent. Sparse Newton solves use
load stepping and residual backtracking. Inverted elements, singular systems,
failed line searches and unmet convergence tolerances fail before output publication.
Every result reports actual free-DOF equilibrium and positive deformation Jacobians.

## Mesh and request contract

`anatomy_ref.kind` must be `tetra_mechanics_mesh`; its URI identifies local JSON.
Declared SHA-256 and subject identity are checked against consumed bytes. JSON
requires `subject_id`, `units: "m"`, `nodes` (N by 3), `tetrahedra` (M by 4,
zero-based positive orientation) and `cavity_faces` (F by 3, or empty for solids).
The mesh must be face-connected, use every node and have no nonmanifold faces.
Cavity faces must be actual exterior faces, closed and consistently oriented
outward from cavity into wall. Open basal cavities require a suitable meshing and
boundary formulation; artificial pressure caps are not constructed.
Optional `fiber` and `sheet` arrays have one orthogonal direction pair per element.
They are mandatory for anisotropic or active material, normalized on input.
Mesh intersection/quality assessment beyond these checks remains the mesh producer's responsibility.

`parameters.material_model` is `neo_hookean` with exactly `mu,kappa`, or `guccione`
with exactly `c,bf,bt,bfs,kappa`. Stress coefficients require explicit `Pa` units.
Guccione exponents are dimensionless. `kappa` denotes the coefficient of the
log-volume penalty, not measured bulk modulus (at identity Neo-Hookean bulk
modulus is kappa + 2 mu / 3). Guccione is explicitly a compressible augmentation
of the existing anisotropic energy with kappa/2 log(J)^2; it is not the original
incompressible formulation. `active_model: "constant_fiber"` permits optional
nonnegative `active.tension` in Pa. Prescribed second-Piola fibre stress is
T f tensor f, pushed to first Piola as T (F f) tensor f. This is static loading,
not an excitation-contraction kinetics model.

Fixed BCs require `kind: "fixed"`, `unit: "m"` and metadata `nodes: [indices]`,
optional `components: [0,1,2]` and `displacement: [ux,uy,uz]` (defaults zero).
Constraints must eliminate all six rigid modes; constrained displacements ramp
with load fraction. One `kind: "pressure"`, `region: "cavity"`, `unit: "Pa"`,
nonnegative scalar `value` applies to all declared cavity faces. Other BCs,
waveforms, activation artifacts, observations and circulation are rejected.

Settings require `output_dir`; optional `load_steps` (default 5),
`max_iterations` (30), `absolute_tolerance_N` (1e-9), `relative_tolerance` (1e-8).
Only these settings are accepted. Maximum steps/iterations are 100 each;
tolerances are capped at 1e-3. Convergence uses abs_tol + rel_tol times each
step's initial free-force norm, reported in field load history.

The SHA-256 artifact `spatial-mechanics.json` contains reference mesh,
displacements, F, Green strain, Cauchy stress, element J, nodal residual/reaction
forces and load history. Forces at unconstrained DOFs are residuals; forces at
constrained DOFs are support reactions. Units are metres, Pa, N and joules.
`strain_energy_J` includes the prescribed active potential. Runs report
`software_checked`, `patient_validated: false`; run QC alone does not grant
empirical or universal numerical validation.

## Evidence and remaining work

`tests/test_fem.py` independently checks energy/stress/tangent derivatives for
both materials with active tension, frame objectivity, tangent symmetry, the
affine displacement patch, assembled force derivative, cavity volume gradient,
follower pressure tangent, nonlinear active equilibrium and failed-solve handling.
`scripts/validate_spatial.py` compares pressure-driven shell displacement with the
independent linear-elastic solution u(r)=A r+B/r² at small pressure. Refinement
reduces mean inner displacement error to 6.22%; committed report is
`validation/cpu/spatial-shell.json`. This verifies a restricted synthetic problem;
it is not evidence of patient accuracy, a ventricular benchmark pass, or adequate
mesh accuracy for arbitrary geometry/material/load combinations.

Displacement P1 elements can lock near incompressibility. No mixed pressure,
incompressible formulation, dynamic EP coupling, scar, contact, spatial
circulation, patient inverse-field calibration or patient myocardial validation
is provided. Spatial calibration preparation fails explicitly. Held-out measured
CMR/echo displacement/strain and pressure data, registered anatomy, boundary
conditions and independent benchmark/cohort validation are required before
patient predictions can be represented as validated.

Primary formulation and verification references:
- Guccione et al. (1991), [passive myocardial material properties](https://pubmed.ncbi.nlm.nih.gov/2020175/).
- Land et al. (2015), [verification of cardiac mechanics software](https://pmc.ncbi.nlm.nih.gov/articles/PMC4707707/), DOI 10.1098/rspa.2015.0641.
No claim is made that this implementation passes the Land benchmark suite.

Local release checks: 90 tests passed, 87.55% total coverage; Ruff checks and
formatting passed; source/wheel builds and installed-wheel reference/spatial
shell verification passed on Python 3.12 CPU. No hosted CI result is implied.
