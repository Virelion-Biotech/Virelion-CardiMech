import pytest

from cardimech import (
    ArtifactRef,
    CardiMechService,
    MechanicalParameterSet,
    MechanicsSimulationRequest,
    MechanicsSimulationResult,
    MechanicsQC,
)
from cardimech.service import ReadinessError
from cardimech.backends import BackendUnavailable


def test_service_has_reference_backend() -> None:
    status = CardiMechService().backend_status()
    assert any(item["name"] == "numpy-lumped-v1" and item["available"] for item in status)


def test_service_fails_closed_without_backend() -> None:
    request = MechanicsSimulationRequest(
        subject_id="S1",
        anatomy_ref=ArtifactRef(artifact_id="mesh", kind="volume_mesh", uri="file:///mesh.vtu"),
        backend="missing",
        parameters=MechanicalParameterSet(passive={"stiffness": 1.0}, active={"tension": 1.0}),
    )
    with pytest.raises(BackendUnavailable):
        CardiMechService().simulate(request)


class _FailingQCBackend:
    name = "failing-qc"

    def available(self) -> bool:
        return True

    def describe(self) -> dict[str, object]:
        return {"name": self.name, "available": True}

    def simulate(self, request: MechanicsSimulationRequest) -> MechanicsSimulationResult:
        return MechanicsSimulationResult(
            subject_id=request.subject_id,
            backend=self.name,
            parameters=request.parameters,
            qc=MechanicsQC(
                passed=False,
                converged=False,
                checks={"stable_time_step": False},
                metrics={"max_volume_step_ml": 6.2},
                errors=["Reference solver QC failed"],
            ),
        )


def test_service_qc_failure_preserves_diagnostics() -> None:
    service = CardiMechService(load_plugins=False)
    service.register_backend(_FailingQCBackend())
    request = MechanicsSimulationRequest(
        subject_id="S1",
        anatomy_ref=ArtifactRef(
            artifact_id="mesh",
            kind="volume_mesh",
            uri="memory://mesh",
        ),
        backend="failing-qc",
        parameters=MechanicalParameterSet(
            passive={"v0_ml": 10.0},
            active={"emax_mmHg_per_ml": 2.0},
        ),
    )
    with pytest.raises(ReadinessError) as excinfo:
        service.simulate(request)
    message = str(excinfo.value)
    assert "stable_time_step" in message
    assert "max_volume_step_ml" in message
    assert "6.2" in message
