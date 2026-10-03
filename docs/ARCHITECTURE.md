# CardiMech architecture

CardiMech owns the mechanics-specific contract and solver orchestration layer between personalized anatomy, electrophysiology, measured deformation/hemodynamics, inference, and HeartTwin.

## Design rules

1. **One stable Virelion contract, many solvers.** Core code must not force PETSc/MPI/FEniCSx on every HeartTwin deployment.
2. **Fast reference != high fidelity.** `numpy-lumped-v1` exists for deterministic plumbing and regression and is always labelled non-spatial.
3. **Anatomy remains upstream.** Geometry, fibres, scar and coordinate frames are CardiAnatomy responsibilities.
4. **EP remains upstream.** Activation/repolarization fields are accepted through `activation_ref`; CardiMech does not reimplement EP.
5. **Inference remains downstream.** CardiMech exposes predictions and prepares likelihood mappings; CardiInfer owns posterior inference.
6. **CFD remains separate.** Detailed blood flow belongs in CardiFlow. CardiMech only owns the circulation necessary to load/couple myocardial mechanics.
7. **Every solver declares fidelity and limitations.** Availability is not a validation claim.

## Core layers

```text
models.py         versioned public contracts
materials.py      constitutive reference utilities
activation.py     active-tension timing utilities
pv.py             pressure-volume and reduced-order mechanics metrics
circulation.py    dependency-light 0D reference circulation
backends.py       backend protocol + plugin discovery
reference_backend.py
                  deterministic reduced-order mechanics backend
calibration.py    observation -> CardiInfer likelihood contract
service.py        backend registry, QC and subject guards
api.py            HeartTwin-native facade
validation.py     deterministic software-reference suite
```

## Spatial backend contract

A high-fidelity backend receives the unchanged `MechanicsSimulationRequest` and is responsible for resolving `anatomy_ref`, optional `activation_ref`, material parameters, BCs, circulation settings, solver settings, and output artifacts. It must return a `MechanicsSimulationResult` and must not silently claim empirical validation.

Recommended spatial outputs are displacement, deformation gradient, strain, stress, cavity volume, cavity pressure and solver diagnostics with explicit coordinate frame and units.

## Coupling strategy

CardiEP -> CardiMech is an explicit artifact handoff, allowing staggered electromechanics by default. Tightly coupled plugins may internally perform stronger coupling but should report that algorithm in provenance.

Mechanics <-> circulation coupling is solver-specific. The built-in reference backend advances LV volume and arterial pressure explicitly with diode valve laws; high-fidelity adapters can use monolithic or partitioned coupling.

## Calibration strategy

Measured mechanics evidence is represented as `MechanicsObservation`. `prepare_calibration` maps each observation to a named mechanics output and emits a CardiInfer-compatible prior/likelihood/forward-template bundle. This prevents a second inference framework from growing inside CardiMech.
