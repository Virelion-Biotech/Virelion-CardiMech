from __future__ import annotations

import math

import numpy as np

from .activation import periodic_hill_activation
from .materials import holzapfel_ogden_energy, neo_hookean_energy
from .models import ArtifactRef, MechanicalParameterSet, MechanicsSimulationRequest
from .pv import passive_pressure_mmHg
from .reference_backend import ReferenceLumpedBackend


def run_reference_validation() -> dict[str, object]:
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}

    identity = np.eye(3)
    nh = neo_hookean_energy(identity, mu=1.0, kappa=100.0)
    ho = holzapfel_ogden_energy(
        identity,
        a=0.1,
        b=5.0,
        af=1.0,
        bf=5.0,
        as_=0.5,
        bs=5.0,
        afs=0.2,
        bfs=5.0,
        kappa=100.0,
    )
    checks["zero_energy_identity"] = abs(nh) < 1e-12 and abs(ho) < 1e-12

    t = np.linspace(0.0, 1.6, 1601)
    activation = periodic_hill_activation(t, cycle_length_s=0.8)
    checks["activation_bounded"] = bool(np.min(activation) >= 0.0 and np.max(activation) <= 1.0)
    checks["activation_periodic"] = bool(np.max(np.abs(activation[:801] - activation[800:])) < 2e-2)

    volumes = np.linspace(20.0, 160.0, 100)
    pressures = passive_pressure_mmHg(volumes, v0_ml=10.0, a_mmHg=0.08, b=0.055)
    checks["passive_pv_monotone"] = bool(np.all(np.diff(pressures) > 0.0))

    request = MechanicsSimulationRequest(
        subject_id="reference",
        anatomy_ref=ArtifactRef(artifact_id="fixture", kind="volume_mesh", uri="memory://fixture"),
        backend="numpy-lumped-v1",
        parameters=MechanicalParameterSet(
            passive={"v0_ml": 10.0, "a_mmHg": 0.08, "b": 0.055},
            active={"emax_mmHg_per_ml": 2.1},
            source="fixed",
        ),
        settings={"cycles": 4, "dt_s": 0.001, "inline_series": False},
    )
    backend = ReferenceLumpedBackend()
    a = backend.simulate(request)
    b = backend.simulate(request)
    checks["reference_qc_passed"] = bool(a.qc and a.qc.passed)
    checks["reference_deterministic"] = a.scalar_outputs == b.scalar_outputs
    checks["reference_finite"] = all(math.isfinite(value) for value in a.scalar_outputs.values())
    metrics.update({f"reference_{key}": value for key, value in a.scalar_outputs.items()})

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "validation_scope": "software/reference invariants only; not physiological or clinical validation",
    }
