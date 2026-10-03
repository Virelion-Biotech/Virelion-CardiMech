import sys

from cardimech import ArtifactRef, MechanicalParameterSet, MechanicsSimulationRequest
from cardimech.subprocess_backend import SubprocessMechanicsBackend


def test_subprocess_backend_contract(tmp_path) -> None:
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text(
        """
import json, sys
request = json.load(open(sys.argv[1]))
result = {
    "contract_version": "2.0",
    "subject_id": request["subject_id"],
    "backend": "test-external",
    "parameters": request["parameters"],
    "outputs": [],
    "scalar_outputs": {"edv_ml": 120.0},
    "series": {},
    "qc": {"passed": True, "converged": True, "checks": {"fixture": True}},
    "validation_status": "software_checked",
    "warnings": [],
    "provenance": {"fixture": True}
}
json.dump(result, open(sys.argv[2], "w"))
""".strip()
        + "\n",
        encoding="utf-8",
    )
    backend = SubprocessMechanicsBackend(
        name="test-external",
        command=[sys.executable, str(wrapper), "{request}", "{output}"],
        executable=sys.executable,
        timeout_s=10.0,
    )
    request = MechanicsSimulationRequest(
        subject_id="S1",
        anatomy_ref=ArtifactRef(artifact_id="a", kind="mesh", uri="memory://a"),
        backend="test-external",
        parameters=MechanicalParameterSet(source="fixed"),
    )
    result = backend.simulate(request)
    assert result.subject_id == "S1"
    assert result.scalar_outputs["edv_ml"] == 120.0
    assert result.qc is not None and result.qc.passed
