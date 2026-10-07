# CardiMech 0.3.0 CPU audit

## Scope and reproducibility

This release verifies the dependency-light LV reference model, constitutive point
utilities, artifact contracts and actual CardiInfer subprocess integration. It
neither solves continuum mechanics nor establishes patient or clinical accuracy.
The anatomy reference is tracked as identity/provenance; the LV backend does not
consume mesh geometry. No GPU or patient data is required for these checks.

Baseline: `048216c65a8d141136b554958023e0839152b919`, 24 passing tests, approximately
79% statement coverage. The audit reproduces six original failures before repair:
NaN material parameters, nonorthogonal material frames, NaN output series,
fractional cycle counts, silently ignored active parameters, and fabricated
periodic convergence. Regression tests preserve these cases.

Run the commands in README. Optional SciPy is needed only for the independent
adaptive integration/recovery harness; production reference mechanics remains
NumPy/Pydantic only. The report JSON files include SHA-256 of source and harness
files. CI independently regenerates and uploads reports for Python 3.10–3.13,
checks the installed wheel outside the source tree, and tests actual CardiInfer
at pinned commit `15a7bd5bf4c23784245b361c017a376572766254`.

## Scientific evidence

| Check | Result | Meaning |
|---|---|---|
| HO-style six shear modes, 31 strains each | 186 cases; maximum energy error 1.67e-16 kPa | Independent scalar invariant formulas agree at J=1 |
| Four material laws under superposed rotation | Maximum energy difference 1.87e-14 | Frame objectivity on the specified deformation |
| Four laws at identity | Zero energy | Reference configuration consistency |
| Neo-Hookean analytic stress vs central energy differences | Maximum discrepancy 8.40e-10 | Nine stress components agree within 1e-7 |
| Windkessel manufactured exponential decay | Errors 0.17229, 0.08586, 0.04286 mmHg for 20, 10, 5 ms | First-order explicit Euler refinement |
| Windkessel discrete mass balance | Maximum residual 1.05e-14 mL | Implemented arterial conservation |
| Full LV vs independent DOP853 integration | Endpoint maximum coordinate errors 0.18337, 0.09001, 0.04526 for 1, 0.5, 0.25 ms | Numerical error decreases with refinement; coordinates are volume in mL and pressure in mmHg, not a physical combined norm |
| Twenty-cycle periodic QC | Volume residual 7.25e-5 mL; pressure residual 1.20e-4 mmHg | Meets separately specified 0.001 tolerances |
| Same-model scalar parameter recovery | Truth 2.1; estimate 2.09999999 | Optimization recovers synthetic elastance, not independent physiological validation |
| CardiInfer integration | Schema, real external command, 8-particle ABC smoke run | Executable integration, not a calibrated posterior accuracy claim |

The LV adaptive solver independently implements pressure/flows/mass balance;
it shares only the prescribed activation utility. The decay test independently
uses a closed-form analytic solution. The six-shear test independently computes
scalar invariant formulas. These are complementary numerical checks, not external
experimental validation. Fixed test tolerances precede pass/fail checks.

## Data and model provenance

Eight HO parameters are taken from the Holzapfel and Ogden (2009) row of Table 2
in *In vivo estimation of passive biomechanical properties of human myocardium*:
https://pmc.ncbi.nlm.nih.gov/articles/PMC6096751/ . Stress coefficients are kPa.
`kappa=1000 kPa` is an explicitly chosen verification penalty, not a published
measurement. The 186 deformation cases are generated synthetic benchmark inputs;
no experimental shear curves or patient data are claimed or copied.

The compressible utility is an explicitly defined variant: I1bar for its isotropic
term, full fiber/sheet invariants, tension-only directional extensions and a
quadratic J penalty. It is not presented as a unique canonical compressible HO law.
Volume-preserving shear cases do not validate compressible physiological behavior.
For discussion of myocardial isochoric formulations, see Balaban et al. (2016):
https://link.springer.com/article/10.1007/s10237-016-0780-7 .

The calibration example embeds a same-model synthetic EDV. Its source and scope
are declared in observation metadata; it is runnable without a nonexistent patient
file. Gaussian/Student-t likelihoods, standard-deviation units and observed/model
array alignment require experiment-specific judgment before interpreting inference.

## Repairs and compatibility

Strict finite JSON and model boundary revalidation reject NaN/Infinity, duplicate
keys, malformed result arrays, output-ID duplication and mutated invalid requests.
Atomic writes use unique temporary files; serialization failure preserves previous
contents. Activation normalization is cached and Hill terms are numerically stable;
floating cycle-boundary remainders are snapped within floating-point precision.

The LV solver rejects unsupported controls, parameter names, activation models,
spatial BCs and circulation state/mode requests. Passive pressure fails beyond an
exponent of 50 instead of silently saturating. It reports the actual aligned time
step and periodic convergence, and optionally makes convergence mandatory. The
2 ms benchmark step failed the stability gate and was rejected; validation uses
finer steps without loosening that gate. Spatial plugins must provide QC and true
convergence; wrong subject/backend/provenance identities fail at the service boundary.

HO volumetric energy differs from 0.2.0 because its isotropic term is now isochoric.
Stroke work now closes the PV polygon and adds joule conversion. Unsupported
calibration observation mappings cannot silently discard volume/displacement data.
Noise aliases, proper-vs-distance backend selection, observation metadata retention,
unit checks and shared forward output directories are repaired. CLI failures use
stderr/nonzero exit status; successful stdout remains strict JSON.

## Remaining limits

There is no bundled spatial FE solve, mesh refinement study, patient-specific wall
stress validation, multi-parameter identifiability proof, held-out cohort agreement
or clinical calibration. The spherical strain and Laplace stress summaries are
surrogates; thick-wall ratios produce warnings. Intrinsic afterload remains when
stack circulation coupling is disabled and is explicitly identified. Generic plugin
contracts cannot prove an external solver's scientific validity. Published parameter
values alone cannot substitute for experiments.

## Local release checks

75 tests pass with 86.78% statement coverage (1,210 statements). Lint, formatting,
source fingerprints, strict calibration example validation, both scientific
reports, source/wheel builds and installed-wheel reference validation pass.
The real CardiInfer test includes a command evaluation and an ABC run; sampler
artifacts are isolated in a temporary test directory. Hosted CI is checked after
publication and its run is linked in the release verification note.

## Publication verification

Release implementation commit: `f0390418486ca520df356d85f1e891f5524ef920`.
The final documentation commit triggers all six hosted jobs: four Python versions,
installed-wheel validation and pinned CardiInfer integration. Current runs:
https://github.com/Virelion-Biotech/Virelion-CardiMech/actions/workflows/ci.yml .
