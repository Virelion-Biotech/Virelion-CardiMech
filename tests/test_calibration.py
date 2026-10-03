from cardimech import ArtifactRef, MechanicsCalibrationRequest, MechanicsObservation
from cardimech.calibration import prepare_calibration


def test_prepare_calibration_maps_observation_and_parameters() -> None:
    request = MechanicsCalibrationRequest(
        subject_id="S1",
        anatomy_ref=ArtifactRef(artifact_id="mesh", kind="volume_mesh", uri="memory://mesh"),
        backend="numpy-lumped-v1",
        observations=[
            MechanicsObservation(
                observation_id="edv",
                kind="end_diastolic_volume",
                artifact=ArtifactRef(artifact_id="edv-art", kind="scalar", uri="memory://edv"),
                unit="mL",
                uncertainty={"scale": 4.0},
            )
        ],
        parameter_bounds={"passive.a_mmHg": (0.01, 0.5), "active.emax_mmHg_per_ml": (0.5, 5.0)},
    )
    bundle = prepare_calibration(request)
    assert bundle.model_capability == "mechanics.simulate"
    assert bundle.likelihood[0]["model_output"] == "scalar_outputs.edv_ml"
    assert {item["name"] for item in bundle.priors} == {"passive.a_mmHg", "active.emax_mmHg_per_ml"}
    assert bundle.forward_template["backend"] == "numpy-lumped-v1"
    assert bundle.model_context["forward_model"]["command"] == ["cardimech-forward"]
    assert bundle.cardiinfer_request["model_capability"] == "mechanics.simulate"
