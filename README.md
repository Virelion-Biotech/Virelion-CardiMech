# Virelion-CardiMech

Patient-specific cardiac mechanics and electromechanics layer for the Virelion HeartTwin stack.

CardiMech defines stable contracts for passive and active myocardial mechanics, loading and boundary conditions, electromechanical coupling inputs, chamber pressure-volume outputs, lumped circulation hooks, solver quality-control, and mechanics calibration handoff to CardiInfer.

## Scope

CardiMech owns:
- myocardial material and active-tension parameter sets;
- mechanics domains built from CardiAnatomy artifacts;
- pressure, volume, preload, afterload, and support constraints;
- optional activation inputs from CardiEP;
- forward mechanics simulations and solver diagnostics;
- pressure-volume and deformation outputs;
- coupling interfaces to 0D circulation;
- calibration-ready contracts for CardiInfer.

CardiMech does **not** synthesize patient mechanics without an explicit backend. Missing numerical solvers fail closed.

## Quick start

```bash
python -m pip install -e '.[dev]'
pytest -q
cardimech doctor
```

## Scientific boundary

Passing schema, solver, or numerical checks does not establish myocardial material validity, patient-specific calibration, physiological fidelity, or clinical usefulness.

## License

AGPL-3.0-or-later.
