import json

from cardimech.hearttwin_cli import dispatch, main


def _reference_request() -> dict:
    return {
        "anatomy_ref": {
            "artifact_id": "a",
            "kind": "reference_geometry",
            "uri": "memory://a",
        },
        "backend": "numpy-lumped-v1",
        "parameters": {
            "passive": {"v0_ml": 10.0, "a_mmHg": 0.08, "b": 0.055},
            "active": {"emax_mmHg_per_ml": 2.1},
            "source": "fixed",
        },
        "settings": {"cycles": 4, "dt_s": 0.001, "inline_series": False},
    }


def test_hearttwin_dispatch_health_and_reference_validation() -> None:
    assert dispatch("mechanics.health", {})["service"] == "CardiMech"
    assert dispatch("mechanics.validate.reference", {})["passed"] is True


def test_hearttwin_generic_envelope_runs_mechanics() -> None:
    result = dispatch(
        "mechanics.simulate",
        {
            "entity_id": "S1",
            "context": {"cardimech_request": _reference_request()},
            "observations": [],
        },
    )
    assert result["subject_id"] == "S1"
    assert result["backend"] == "numpy-lumped-v1"
    assert result["qc"]["passed"] is True


def test_hearttwin_generic_envelope_prepares_calibration() -> None:
    request = {
        **_reference_request(),
        "observations": [
            {
                "observation_id": "edv",
                "kind": "end_diastolic_volume",
                "artifact": {
                    "artifact_id": "edv",
                    "kind": "scalar",
                    "uri": "memory://edv",
                },
                "unit": "mL",
                "uncertainty": {"scale": 4.0},
            }
        ],
        "parameter_bounds": {"passive.a_mmHg": [0.01, 0.5]},
    }
    request.pop("parameters")
    result = dispatch(
        "mechanics.prepare_calibration",
        {
            "entity_id": "S1",
            "context": {"cardimech_calibration_request": request},
            "observations": [],
        },
    )
    assert result["subject_id"] == "S1"
    assert result["model_capability"] == "mechanics.simulate"


def test_hearttwin_cli_reads_standard_env(monkeypatch, capsys) -> None:
    monkeypatch.setenv("HEARTTWIN_CAPABILITY", "mechanics.health")
    monkeypatch.setenv("HEARTTWIN_PAYLOAD", "{}")
    monkeypatch.delenv("HEARTTWIN_PAYLOAD_STDIN", raising=False)
    assert main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["service"] == "CardiMech"


def test_hearttwin_cli_rejects_unknown_capability(monkeypatch, capsys) -> None:
    monkeypatch.setenv("HEARTTWIN_CAPABILITY", "mechanics.nope")
    monkeypatch.setenv("HEARTTWIN_PAYLOAD", "{}")
    assert main() == 1
    assert "does not support" in capsys.readouterr().err
