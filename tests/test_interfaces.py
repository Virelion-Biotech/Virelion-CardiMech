import json
import os
import subprocess
import sys

import pytest

from cardimech.calibration import prepare_calibration
from cardimech.models import MechanicsCalibrationRequest
from cardimech.serialization import strict_loads, write_json


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_strict_json(raw):
    with pytest.raises(ValueError):
        strict_loads(raw)


def test_atomic_serialization_failure_preserves_file(tmp_path):
    path = tmp_path / "data.json"
    path.write_text("original")
    with pytest.raises(ValueError):
        write_json(path, {"x": float("nan")})
    assert path.read_text() == "original"


def test_cli_material_point_and_error(tmp_path):
    path = tmp_path / "point.json"
    output = tmp_path / "output.json"
    path.write_text(
        json.dumps(
            {
                "model": "neo_hookean",
                "F": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "parameters": {"mu": 3.0, "kappa": 100.0},
            }
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cardimech.cli",
            "material-point",
            str(path),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["energy"] == 0
    path.write_text('{"x":NaN}')
    result = subprocess.run(
        [sys.executable, "-m", "cardimech.cli", "simulate", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1 and not result.stdout and "Nonstandard" in result.stderr


def test_forward_process(tmp_path):
    payload = {
        "subject_id": "synthetic",
        "anatomy_ref": {
            "artifact_id": "mesh",
            "kind": "reference_geometry",
            "uri": "memory://synthetic",
        },
        "backend": "numpy-lumped-v1",
        "parameters": {"source": "fixed"},
        "settings": {"cycles": 2, "dt_s": 0.001},
    }
    result = subprocess.run(
        [sys.executable, "-m", "cardimech.forward_cli"],
        env={**os.environ, "HEARTTWIN_PAYLOAD": json.dumps(payload)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["qc"]["passed"] and data["subject_id"] == "synthetic"
    assert len(data["series"]["time_s"]) == 801


def calibration_payload():
    return {
        "subject_id": "synthetic",
        "anatomy_ref": {
            "artifact_id": "mesh",
            "kind": "reference_geometry",
            "uri": "memory://synthetic",
        },
        "observations": [
            {
                "observation_id": "peak",
                "kind": "peak_pressure",
                "artifact": {"artifact_id": "obs", "kind": "scalar", "uri": "memory://synthetic"},
                "uncertainty": {"scale": 2.0},
                "metadata": {"observed": 100.0},
            }
        ],
        "parameter_bounds": {"active.emax_mmHg_per_ml": [1.0, 3.0]},
        "settings": {"output_dir": "ignored"},
    }


def test_calibration_noise_backend_and_observed():
    bundle = prepare_calibration(MechanicsCalibrationRequest.model_validate(calibration_payload()))
    assert bundle.cardiinfer_request["backend"] == "native-metropolis-v1"
    assert bundle.likelihood[0]["noise_parameters"] == {"sigma": 2.0}
    assert bundle.likelihood[0]["metadata"]["observed"] == 100.0
    assert "output_dir" not in bundle.forward_template["settings"]


@pytest.mark.parametrize("kind", ["pv_loop", "displacement_field", "other"])
def test_ambiguous_observation_rejected(kind):
    payload = calibration_payload()
    payload["observations"][0]["kind"] = kind
    with pytest.raises(ValueError, match="explicit model_output"):
        prepare_calibration(MechanicsCalibrationRequest.model_validate(payload))


def test_abc_rejects_posterior_terms():
    payload = calibration_payload()
    payload["settings"]["inference_backend"] = "native-abc-smc-v1"
    with pytest.raises(ValueError, match="ABC"):
        prepare_calibration(MechanicsCalibrationRequest.model_validate(payload))


@pytest.mark.parametrize(
    "command", ["doctor", "backends", "materials", "ecosystem", "validate-reference"]
)
def test_cli_in_process_json(command, monkeypatch, capsys):
    from cardimech.cli import main

    monkeypatch.setattr(sys, "argv", ["cardimech", command])
    assert main() == 0
    assert json.loads(capsys.readouterr().out)


def test_cli_error_is_stderr(monkeypatch, capsys, tmp_path):
    from cardimech.cli import main

    monkeypatch.setattr(sys, "argv", ["cardimech", "simulate", str(tmp_path / "missing.json")])
    assert main() == 1
    output = capsys.readouterr()
    assert not output.out and output.err


def test_forward_command_missing_and_invalid_payload(monkeypatch, capsys):
    from cardimech.forward_cli import main

    monkeypatch.delenv("HEARTTWIN_PAYLOAD", raising=False)
    assert main() == 2
    monkeypatch.setenv("HEARTTWIN_PAYLOAD", '{"subject_id":NaN}')
    assert main() == 1
    assert "Nonstandard" in capsys.readouterr().err


@pytest.mark.parametrize("weight", [0, -1, True])
def test_invalid_likelihood_weight(weight):
    payload = calibration_payload()
    payload["observations"][0]["metadata"]["weight"] = weight
    with pytest.raises(ValueError, match="weight"):
        prepare_calibration(MechanicsCalibrationRequest.model_validate(payload))


@pytest.mark.parametrize("path", ["settings.dt_s", "passive.unknown", "arbitrary"])
def test_invalid_calibration_parameter_path(path):
    payload = calibration_payload()
    payload["parameter_bounds"] = {path: [1.0, 3.0]}
    with pytest.raises(ValueError):
        prepare_calibration(MechanicsCalibrationRequest.model_validate(payload))


def test_reference_unit_conversion_cannot_be_implicit():
    payload = calibration_payload()
    payload["observations"][0]["unit"] = "kPa"
    with pytest.raises(ValueError, match="unit"):
        prepare_calibration(MechanicsCalibrationRequest.model_validate(payload))
