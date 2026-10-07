from cardimech.api import MechanicsAPI
from cardimech.forward import ForwardEnvelopeError, normalize_simulation_payload


def envelope() -> dict:
    return {
        "entity_id": "S1",
        "subject_id": "S1",
        "parameters": {
            "passive.a_mmHg": 0.08,
            "passive.b": 0.055,
            "active.emax_mmHg_per_ml": 2.1,
        },
        "context": {
            "cardimech_request": {
                "subject_id": "S1",
                "anatomy_ref": {
                    "artifact_id": "a",
                    "kind": "reference_geometry",
                    "uri": "memory://a",
                },
                "backend": "numpy-lumped-v1",
                "parameters": {"passive": {"v0_ml": 10.0}, "active": {}, "source": "prior"},
                "settings": {"cycles": 4, "dt_s": 0.001, "inline_series": False},
            }
        },
        "observations": [],
    }


def test_cardiinfer_envelope_is_normalized_and_runs() -> None:
    result = MechanicsAPI().simulate(envelope())
    assert result["subject_id"] == "S1"
    assert result["backend"] == "numpy-lumped-v1"
    assert result["qc"]["passed"] is True
    assert result["parameters"]["passive"]["a_mmHg"] == 0.08


def test_forward_envelope_cannot_overwrite_arbitrary_paths() -> None:
    payload = envelope()
    payload["parameters"] = {"settings.output_dir": 1.0}
    try:
        normalize_simulation_payload(payload)
    except ForwardEnvelopeError as exc:
        assert "namespace" in str(exc) or "Unsupported" in str(exc)
    else:
        raise AssertionError("Unsafe forward parameter path was accepted")
