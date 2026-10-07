import numpy as np
import pytest
from test_reference_backend import request

from cardimech.materials import guccione_energy, neo_hookean_energy
from cardimech.models import MechanicalParameterSet, MechanicsSimulationResult
from cardimech.reference_backend import ReferenceLumpedBackend


def test_nonfinite_material_parameter_rejected():
    with pytest.raises(ValueError):
        neo_hookean_energy(np.eye(3), mu=float("nan"), kappa=100.0)


def test_material_frame_must_be_orthogonal():
    with pytest.raises(ValueError, match="orthogonal"):
        guccione_energy(np.eye(3), c=1.0, bf=1.0, bt=1.0, bfs=1.0, fiber=[1, 0, 0], sheet=[1, 1, 0])


def test_result_rejects_nonfinite_series():
    with pytest.raises(ValueError):
        MechanicsSimulationResult(
            subject_id="S",
            backend="test",
            parameters=MechanicalParameterSet(),
            series={"x": [float("nan")]},
        )


def test_fractional_cycle_count_rejected():
    req = request()
    req.settings["cycles"] = 2.5
    with pytest.raises(ValueError, match="integer"):
        ReferenceLumpedBackend().simulate(req)


def test_reference_unknown_parameter_cannot_be_silently_ignored():
    req = request()
    req.parameters.active["invented_tension"] = 2.0
    with pytest.raises(ValueError, match="Unsupported"):
        ReferenceLumpedBackend().simulate(req)


def test_periodic_convergence_is_not_fabricated():
    req = request()
    req.settings.update(
        cycles=2, cycle_volume_tolerance_ml=1e-15, cycle_pressure_tolerance_mmHg=1e-15
    )
    result = ReferenceLumpedBackend().simulate(req)
    assert result.qc.converged is False


@pytest.mark.parametrize(
    "settings",
    [
        {"cycles": True},
        {"dt_s": 0},
        {"dt_s": 1.0},
        {"max_steps": 10},
        {
            "cycles": 2,
            "require_periodic_convergence": True,
            "cycle_volume_tolerance_ml": 1e-15,
            "cycle_pressure_tolerance_mmHg": 1e-15,
        },
        {"invented": 1},
        {"inline_series": "false"},
        {"wall_thickness_cm": 0},
    ],
)
def test_unsafe_solver_controls_fail(settings):
    from cardimech.service import CardiMechService

    req = request()
    req.settings.update(settings)
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        CardiMechService(load_plugins=False).simulate(req)


def test_hashed_artifact_round_trip_and_tamper(tmp_path):
    from cardimech.artifacts import sha256_file, write_json_artifact
    from cardimech.handoff import HandoffError, load_json_artifact

    ref = write_json_artifact(
        tmp_path,
        filename="activation.json",
        artifact_id="ep",
        kind="activation_map",
        payload={"activation_ms": [0.0, 20.0], "units": "ms"},
    )
    assert ref.sha256 == sha256_file(tmp_path / "activation.json")
    assert load_json_artifact(ref)["activation_ms"] == [0.0, 20.0]
    (tmp_path / "activation.json").write_text("{}")
    with pytest.raises(HandoffError, match="SHA-256"):
        load_json_artifact(ref)
    with pytest.raises(ValueError, match="filename"):
        write_json_artifact(tmp_path, filename="../bad.json", artifact_id="a", kind="b", payload={})


def test_result_alignment_and_mutation_checked():
    from cardimech.service import CardiMechService

    req = request()
    req.settings["dt_s"] = float("nan")
    with pytest.raises(ValueError):
        CardiMechService(load_plugins=False).simulate(req)
    with pytest.raises(ValueError, match="aligned"):
        MechanicsSimulationResult(
            subject_id="S",
            backend="test",
            parameters=MechanicalParameterSet(),
            series={"time_s": [0, 1], "x": [2]},
        )


def test_spatial_nonconvergence_rejected():
    from cardimech.models import MechanicsQC
    from cardimech.service import CardiMechService, ReadinessError

    class Fixture:
        name = "fixture"

        def available(self):
            return True

        def describe(self):
            return {"spatial": True}

        def simulate(self, req):
            return MechanicsSimulationResult(
                subject_id=req.subject_id,
                backend=self.name,
                parameters=req.parameters,
                qc=MechanicsQC(passed=True, converged=False, checks={"finite": True}),
            )

    service = CardiMechService(load_plugins=False)
    service.register_backend(Fixture())
    req = request()
    req.backend = "fixture"
    with pytest.raises(ReadinessError, match="convergence"):
        service.simulate(req)
