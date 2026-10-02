from cardimech import ArtifactRef, MechanicalParameterSet, MechanicsSimulationRequest
from cardimech.reference_backend import ReferenceLumpedBackend


def request() -> MechanicsSimulationRequest:
    return MechanicsSimulationRequest(
        subject_id="S1",
        anatomy_ref=ArtifactRef(artifact_id="a", kind="reference_geometry", uri="memory://a"),
        backend="numpy-lumped-v1",
        parameters=MechanicalParameterSet(
            passive={"v0_ml": 10.0, "a_mmHg": 0.08, "b": 0.055},
            active={"emax_mmHg_per_ml": 2.1, "rise_s": 0.07, "decay_s": 0.24},
            source="fixed",
        ),
        settings={"cycles": 5, "dt_s": 0.001, "inline_series": True},
    )


def test_reference_backend_is_deterministic_and_physically_ordered() -> None:
    backend = ReferenceLumpedBackend()
    first = backend.simulate(request())
    second = backend.simulate(request())
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.qc is not None and first.qc.passed
    assert first.scalar_outputs["edv_ml"] > first.scalar_outputs["esv_ml"] > 0
    assert 0 < first.scalar_outputs["ejection_fraction"] < 1
    assert first.scalar_outputs["stroke_work_mmHg_ml"] > 0
    assert max(first.series["activation"]) <= 1.0
    assert min(first.series["activation"]) >= 0.0


def test_reference_backend_step_guard() -> None:
    req = request().model_copy(update={"settings": {"cycles": 5, "dt_s": 1e-7, "max_steps": 100}})
    try:
        ReferenceLumpedBackend().simulate(req)
    except ValueError as exc:
        assert "max_steps" in str(exc)
    else:
        raise AssertionError("Expected oversized reference solve to be rejected")
