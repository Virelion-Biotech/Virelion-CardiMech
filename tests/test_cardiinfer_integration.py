"""Contract + actual external command evaluation against pinned CardiInfer in CI."""

import sys

import pytest

pytest.importorskip("cardiinfer")
from cardiinfer.forward import ForwardModelClient
from cardiinfer.models import InferenceRequest

from cardimech.calibration import prepare_calibration
from cardimech.models import MechanicsCalibrationRequest


def test_real_cardiinfer_contract_and_forward_evaluation(tmp_path):
    request = MechanicsCalibrationRequest.model_validate(
        {
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
                    "artifact": {
                        "artifact_id": "obs",
                        "kind": "scalar",
                        "uri": "memory://synthetic",
                    },
                    "uncertainty": {"scale": 2.0},
                    "metadata": {"observed": 100.0},
                }
            ],
            "parameter_bounds": {"active.emax_mmHg_per_ml": [1.0, 3.0]},
            "settings": {
                "cycles": 2,
                "dt_s": 0.001,
                "forward_model": {
                    "mode": "command",
                    "command": [sys.executable, "-m", "cardimech.forward_cli"],
                },
            },
        }
    )
    bundle = prepare_calibration(request)
    native = InferenceRequest.model_validate(bundle.cardiinfer_request)
    client = ForwardModelClient.from_request(native)
    result = client.evaluate(
        {"active.emax_mmHg_per_ml": 2.1},
    )
    assert result["qc"]["passed"]
    assert result["scalar_outputs"]["peak_lv_pressure_mmHg"] > 0
    from cardiinfer.api import InferAPI

    payload = bundle.cardiinfer_request
    payload["backend"] = "native-abc-smc-v1"
    payload["likelihood"][0]["discrepancy"] = "rmse"
    payload["likelihood"][0]["noise_parameters"] = {}
    payload["likelihood"][0]["metadata"]["observed"] = result["scalar_outputs"][
        "peak_lv_pressure_mmHg"
    ]
    payload["priors"][0]["bounds"] = [2.0, 2.2]
    payload["sampler_settings"] = {
        "n_particles": 8,
        "n_generations": 1,
        "initial_oversample": 1,
        "output_dir": str(tmp_path),
    }
    payload["seed"] = 42
    inferred = InferAPI().infer(payload)
    assert inferred["subject_id"] == "synthetic"
    assert inferred["backend"] == "native-abc-smc-v1"
    assert inferred["posterior_samples"]
