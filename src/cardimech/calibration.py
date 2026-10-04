from __future__ import annotations

from .models import MechanicsCalibrationBundle, MechanicsCalibrationRequest

_OUTPUT_BY_OBSERVATION = {
    "end_diastolic_volume": "scalar_outputs.edv_ml",
    "end_systolic_volume": "scalar_outputs.esv_ml",
    "ejection_fraction": "scalar_outputs.ejection_fraction",
    "peak_pressure": "scalar_outputs.peak_lv_pressure_mmHg",
    "pressure_curve": "series.lv_pressure_mmHg",
    "volume_curve": "series.lv_volume_ml",
    "pv_loop": "series.lv_pressure_mmHg",
    "strain_curve": "series.circumferential_strain",
    "stroke_work": "scalar_outputs.stroke_work_mmHg_ml",
    "wall_thickness": "scalar_outputs.wall_thickness_cm",
    "displacement_field": "outputs.displacement_field",
    "other": "scalar_outputs",
}


def prepare_calibration(request: MechanicsCalibrationRequest) -> MechanicsCalibrationBundle:
    initial = request.initial_parameters or None
    passive = {} if initial is None else dict(initial.passive)
    active = {} if initial is None else dict(initial.active)
    priors = []
    for name, (lower, upper) in sorted(request.parameter_bounds.items()):
        unit = None if initial is None else initial.units.get(name)
        priors.append(
            {
                "name": name,
                "distribution": "uniform",
                "bounds": [float(lower), float(upper)],
                "unit": unit,
            }
        )

    likelihood = []
    for observation in request.observations:
        output_path = str(observation.metadata.get("model_output") or _OUTPUT_BY_OBSERVATION[observation.kind])
        discrepancy = str(
            observation.metadata.get("discrepancy", request.settings.get("discrepancy", "rmse"))
        )
        noise = dict(observation.uncertainty)
        likelihood.append(
            {
                "term_id": observation.observation_id,
                "observation_ref": observation.artifact.model_dump(mode="json"),
                "model_output": output_path,
                "discrepancy": discrepancy,
                "noise_parameters": noise,
                "weight": float(observation.metadata.get("weight", 1.0)),
                "metadata": {
                    "region": observation.region,
                    "unit": observation.unit,
                    "observation_kind": observation.kind,
                },
            }
        )

    template = {
        "subject_id": request.subject_id,
        "anatomy_ref": request.anatomy_ref.model_dump(mode="json"),
        "activation_ref": (
            None
            if request.activation_ref is None
            else request.activation_ref.model_dump(mode="json")
        ),
        "backend": request.backend,
        "parameters": {
            "passive": passive,
            "active": active,
            "units": {} if initial is None else dict(initial.units),
            "source": "prior",
            "material_model": "holzapfel_ogden" if initial is None else initial.material_model,
            "active_model": "periodic_hill" if initial is None else initial.active_model,
        },
        "boundary_conditions": [item.model_dump(mode="json") for item in request.boundary_conditions],
        "circulation": request.circulation.model_dump(mode="json"),
        "observations": [item.model_dump(mode="json") for item in request.observations],
        "settings": {
            **{key: value for key, value in request.settings.items() if key not in {"discrepancy", "forward_model", "inference_backend", "sampler_settings", "seed"}},
            "inline_series": True,
        },
    }
    forward_model = request.settings.get("forward_model") or {
        "mode": "command",
        "command": ["cardimech-forward"],
        "timeout_s": 120.0,
    }
    model_context = {
        "forward_model": forward_model,
        "cardimech_request": template,
    }
    cardiinfer_request = {
        "subject_id": request.subject_id,
        "model_service": "CardiMech",
        "model_capability": "mechanics.simulate",
        "backend": str(request.settings.get("inference_backend", "native-abc-smc-v1")),
        "priors": priors,
        "likelihood": likelihood,
        "model_context": model_context,
        "sampler_settings": dict(request.settings.get("sampler_settings") or {}),
        "seed": request.settings.get("seed"),
    }
    return MechanicsCalibrationBundle(
        subject_id=request.subject_id,
        forward_template=template,
        priors=priors,
        likelihood=likelihood,
        model_context=model_context,
        cardiinfer_request=cardiinfer_request,
        notes=[
            "CardiMech prepares the forward-model contract; posterior inference belongs in CardiInfer.",
            "Observation/model alignment and empirical noise calibration remain experiment-specific.",
        ],
    )
