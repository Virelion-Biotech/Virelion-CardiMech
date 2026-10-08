# Mechanics backends

## Built-in: `numpy-lumped-v1`

Purpose: deterministic integration tests, fast sweeps, inverse-loop smoke tests, and software reference outputs.

It implements a nonlinear passive LV pressure-volume relation, periodic active elastance, diode mitral/aortic valves, arterial compliance/systemic resistance, PV metrics, and spherical strain/stress surrogates.

It does **not** solve continuum mechanics, consume tetrahedral element fields, model spatial fibre stress, or establish patient-specific mechanics.

## Plugin protocol

A backend exposes:

```python
class MechanicsBackend(Protocol):
    name: str
    def available(self) -> bool: ...
    def describe(self) -> dict[str, object]: ...
    def simulate(self, request: MechanicsSimulationRequest) -> MechanicsSimulationResult: ...
```

Register it with a Python entry point:

```toml
[project.entry-points."cardimech.backends"]
my_backend = "my_package.backend:backend"
```

Plugins should keep heavyweight dependencies inside their own package/environment and should return file-backed artifacts for large spatial fields.

## Adapter targets

- `fenicsx-pulse`: natural first FEniCSx myocardial mechanics adapter.
- `Ambit`: strong candidate for multiphysics and mechanics/0D or FSI workflows.
- `CardioMechanics`: useful external-process target for mature electromechanics/circulation workflows.
- `Chaste`: mature electromechanics benchmark/reference ecosystem.
- `simcardems2`: useful architecture for staggered EP-mechanics coupling.

No external solver is silently vendored by CardiMech. Deployment must review its license, binary/runtime requirements, numerical configuration, and validation evidence.

## Built-in spatial CPU backend: `scipy-tetra-v1` (0.4.0)

Finite-strain P1 tetrahedral static mechanics with compressible Neo-Hookean or
Guccione material, prescribed fibre tension, displacement constraints and closed
cavity follower pressure. Install `[spatial]`; see [contract, numerical evidence
and limitations](SPATIAL_MECHANICS.md). This backend is numerically exercised on
synthetic meshes and is not patient validated.
