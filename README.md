# Virelion-CardiMech

[![CI](https://github.com/Virelion-Biotech/Virelion-CardiMech/actions/workflows/ci.yml/badge.svg)](https://github.com/Virelion-Biotech/Virelion-CardiMech/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue.svg)](LICENSE)

**CardiMech is the patient-specific cardiac mechanics and electromechanics layer for the Virelion HeartTwin stack.**

It combines stable mechanics contracts with a dependency-light reference model and explicit plugin boundaries for high-fidelity finite-element and multiphysics solvers.

## What is built

- typed anatomy, observation, material, loading, circulation, QC, simulation, and calibration contracts;
- a deterministic NumPy LV pressure-volume + Windkessel reference backend (`numpy-lumped-v1`);
- passive/active constitutive utilities for Neo-Hookean, Mooney-Rivlin, Guccione, and Holzapfel-Ogden style models;
- periodic active-tension activation utilities;
- pressure-volume metrics, stroke work, spherical strain and Laplace-stress reference summaries;
- file-backed artifacts with SHA-256 provenance;
- CardiInfer-ready calibration problem generation from mechanics observations;
- plugin discovery through the `cardimech.backends` entry-point group;
- an audited external ecosystem map rather than silent vendoring of heavyweight solvers;
- deterministic reference validation and regression tests.

> `numpy-lumped-v1` is deliberately **not** a finite-element solver. It is an integration/reference model for contract tests, rapid sweeps, inverse-loop plumbing, and deterministic regression before escalation to a spatial mechanics backend.

## Architecture

```text
CardiAnatomy
 mesh + regions + fibres + scar
          |
          +---------------- CardiEP activation/repolarization
          |                              |
          v                              v
     MechanicsSimulationRequest / observations
          |
          +-- material law / active law / loading / circulation
          |
          +--> numpy-lumped-v1 (fast reference)
          |
          +--> plugin: fenicsx-pulse / Ambit / CardioMechanics / other solver
          |
          v
 displacement / strain / stress / PV / hemodynamic artifacts
          |
          +--> HeartTwin canonical state
          |
          +--> CardiInfer likelihood + posterior calibration
          |
          +--> CardiEval / CardiTrace
```

## Install

```bash
python -m pip install -e .
```

Development:

```bash
python -m pip install -e '.[dev]'
pytest -q
ruff check src tests
cardimech validate-reference
```

Optional mesh I/O helpers can be installed with:

```bash
python -m pip install -e '.[io]'
```

## Fast reference simulation

```json
{
  "subject_id": "S1",
  "anatomy_ref": {
    "artifact_id": "s1-mechanics-geometry",
    "kind": "volume_mesh",
    "uri": "file:///data/S1/mesh.vtu"
  },
  "backend": "numpy-lumped-v1",
  "parameters": {
    "passive": {"v0_ml": 10.0, "a_mmHg": 0.08, "b": 0.055},
    "active": {"emax_mmHg_per_ml": 2.1, "rise_s": 0.07, "decay_s": 0.24},
    "source": "fixed"
  },
  "circulation": {
    "enabled": true,
    "model": "windkessel_3e",
    "parameters": {"r_systemic": 1.0, "c_arterial": 1.5}
  },
  "settings": {"cycles": 5, "dt_s": 0.001, "output_dir": "runs/S1"}
}
```

```bash
cardimech simulate request.json
```

The backend returns a typed `MechanicsSimulationResult` containing PV/hemodynamic time series, EDV, ESV, stroke volume, EF, peak LV pressure, stroke work, reference strain/stress summaries, QC, and provenance.

## CardiInfer calibration handoff

CardiMech does not duplicate Bayesian inference. It converts mechanics observations into a solver-neutral forward-model problem for CardiInfer:

```bash
cardimech prepare-calibration calibration.json
```

Supported automatic observation mappings include EDV, ESV, EF, peak pressure, pressure/volume curves, strain curves, wall thickness, and stroke work. PV loops require separate pressure and volume terms; displacement/other observations require explicit output paths supported by the selected backend. The generated bundle contains priors, likelihood terms, and a canonical forward-template payload for `mechanics.simulate`.

## High-fidelity backends

Heavy solvers remain optional. A plugin implements `MechanicsBackend` and registers under the `cardimech.backends` entry-point group. This keeps PETSc/MPI/FEniCSx or external binaries out of the core environment while preserving one HeartTwin-facing contract.

The research map tracks projects such as `fenicsx-pulse`, `pulse`, `simcardems2`, `Ambit`, `CardioMechanics`, `Chaste`, `ModularCirc`, `cardiac_benchmark`, `cardiac-geometriesx`, and `fenicsx-ldrb`. Their code and licenses remain upstream unless an explicit compatible adapter is added.

See `docs/RESEARCH_MAP.md`, `docs/BACKENDS.md`, and `THIRD_PARTY_NOTICES.md`.

## HeartTwin capabilities

- `mechanics.health`
- `mechanics.backends`
- `mechanics.materials`
- `mechanics.simulate`
- `mechanics.prepare_calibration`
- `mechanics.validate.reference`
- `mechanics.ecosystem`

HeartTwin should require a CardiAnatomy artifact that passes the `mechanics` readiness gate before publication-grade spatial mechanics. `activation_ref` is the explicit CardiEP handoff. Mechanics observations can originate from CMR/echo/MyoTrace or other measured artifacts and flow to CardiInfer through the calibration bundle.

## Validation ladder

1. schema/software invariants;
2. deterministic reference regression;
3. constitutive benchmark problems and mesh/material/BC consistency;
4. spatial-solver convergence and manufactured/analytic benchmarks;
5. synthetic parameter recovery through CardiInfer;
6. held-out imaging/hemodynamic agreement;
7. external patient/cohort validation.

Passing the reference validation establishes only software behavior. It does not establish myocardial material validity, patient-specific calibration, physiological fidelity, clinical utility, or regulatory suitability.

## License

AGPL-3.0-or-later. Third-party tools retain their own licenses and terms.

## CPU verification and constitutive points (0.3.0)

```bash
python -m pip install -e '.[dev,validation]'
python -m pytest -q --cov=cardimech
python scripts/validate_cpu.py
python scripts/validate_lv.py
cardimech material-point examples/material_point.json --output /tmp/material.json
cardimech simulate examples/reference_request.json --output /tmp/mechanics.json
cardimech prepare-calibration examples/calibration_request.json --output /tmp/bundle.json
```

`material-point` evaluates energy and Green strain for four constitutive families,
plus exact first Piola and Cauchy stress for Neo-Hookean. Energy and stress use the
input parameter stress unit; deformation gradients are dimensionless. This utility
is independent of the LV backend, which evaluates PV/elastance parameters rather
than continuum material laws. The HO-style compressible variant now uses an
isochoric isotropic invariant and full directional invariants, with a quadratic
volume penalty. This avoids residual isotropic stress at identity; it changes
volumetric behavior compared with 0.2.0.

JSON rejects non-finite numbers and duplicate keys. Output files use atomic
replacement and SHA-256 fingerprints. Reference settings, parameter names, spatial
boundary conditions, unsupported circulation modes and state artifacts now fail
explicitly. `cycles` must be an integer; the time grid aligns exactly to cycles and
reports the effective step. `qc.converged` reports actual periodic convergence.
Set `require_periodic_convergence=true` and explicit pressure/volume tolerances to
make that a required QC gate. Default basic QC is not proof of steady state.

Disabling stack circulation coupling leaves the reference model's intrinsic
afterload, declared in warnings/provenance. Wall stress is a spherical thin-wall
approximation; a thickness/radius warning explains when this approximation is
outside a conventional thin-wall range. Stroke work closes the sampled PV polygon
and is also reported in joules; a non-periodic run is not a steady-cycle work estimate.

Calibration defaults to Gaussian likelihood when uncertainty is supplied (legacy
`scale` becomes `sigma`) and RMSE otherwise. Proper likelihoods default to
Metropolis; distance terms default to ABC. Mixed semantics and incompatible noise
are rejected. Inline observed data and observation paths survive the handoff;
physical units must match output units. Time/region alignment and noise validation
remain the caller's responsibility. The example is explicitly synthetic. Forward
evaluations remove shared `output_dir` to prevent concurrent calibration artifacts
from overwriting one another.

See [the audit and scientific evidence](docs/CPU_AUDIT.md). No GPU is needed for
these verified workflows; spatial FE solving and empirical validation remain open.
