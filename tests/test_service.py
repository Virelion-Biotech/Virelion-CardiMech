import pytest

from cardimech import (
    ArtifactRef,
    CardiMechService,
    MechanicalParameterSet,
    MechanicsSimulationRequest,
)
from cardimech.backends import BackendUnavailable


def test_service_fails_closed_without_backend() -> None:
    request = MechanicsSimulationRequest(
        subject_id="S1",
        anatomy_ref=ArtifactRef(
            artifact_id="mesh",
            kind="volume_mesh",
            uri="file:///mesh.vtu",
        ),
        backend="missing",
        parameters=MechanicalParameterSet(
            passive={"stiffness": 1.0},
            active={"tension": 1.0},
        ),
    )
    with pytest.raises(BackendUnavailable):
        CardiMechService().simulate(request)
