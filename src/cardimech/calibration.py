from __future__ import annotations

from .models import MechanicsCalibrationBundle, MechanicsCalibrationRequest
from .serialization import finite_number

_OUTPUT_BY_OBSERVATION = {
    "end_diastolic_volume": "scalar_outputs.edv_ml",
    "end_systolic_volume": "scalar_outputs.esv_ml",
    "ejection_fraction": "scalar_outputs.ejection_fraction",
    "peak_pressure": "scalar_outputs.peak_lv_pressure_mmHg",
    "pressure_curve": "series.lv_pressure_mmHg",
    "volume_curve": "series.lv_volume_ml",
    "strain_curve": "series.circumferential_strain",
    "stroke_work": "scalar_outputs.stroke_work_mmHg_ml",
    "wall_thickness": "scalar_outputs.wall_thickness_cm",
}


def prepare_calibration(request: MechanicsCalibrationRequest) -> MechanicsCalibrationBundle:
    request = MechanicsCalibrationRequest.model_validate(request.model_dump(mode="json"))
    initial = request.initial_parameters or None
    passive = {} if initial is None else dict(initial.passive)
    active = {} if initial is None else dict(initial.active)
    priors = []
    for name, (lower, upper) in sorted(request.parameter_bounds.items()):
        parts = name.split(".")
        if len(parts) != 2 or parts[0] not in {"passive", "active", "circulation"} or not parts[1]:
            raise ValueError("Parameter bounds require passive.*, active.*, or circulation.* paths")
        if request.backend == "numpy-lumped-v1":
            supported = {
                "passive": {"v0_ml", "a_mmHg", "b"},
                "active": {"emax_mmHg_per_ml", "rise_s", "decay_s", "onset_s"},
                "circulation": {
                    "r_mitral",
                    "r_aortic",
                    "r_systemic",
                    "c_arterial",
                    "p_atrium_mmHg",
                    "p_venous_mmHg",
                    "initial_lv_volume_ml",
                    "initial_arterial_pressure_mmHg",
                },
            }
            if parts[1] not in supported[parts[0]]:
                raise ValueError(f"Unsupported reference parameter: {name}")
            if parts[0] == "circulation" and not request.circulation.enabled:
                raise ValueError("Circulation priors require enabled coupling")
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
        output_path = str(
            observation.metadata.get("model_output")
            or _OUTPUT_BY_OBSERVATION.get(observation.kind, "")
        )
        if not output_path:
            raise ValueError(
                f"{observation.kind} requires an explicit model_output; PV loops need separate pressure and volume terms"
            )
        expected_units = {
            "end_diastolic_volume": {"mL", "ml"},
            "end_systolic_volume": {"mL", "ml"},
            "volume_curve": {"mL", "ml"},
            "peak_pressure": {"mmHg"},
            "pressure_curve": {"mmHg"},
            "ejection_fraction": {"1", "fraction"},
            "strain_curve": {"1", "fraction"},
            "wall_thickness": {"cm"},
            "stroke_work": {"mmHg*mL", "mmHg*ml"},
        }
        if (
            request.backend == "numpy-lumped-v1"
            and observation.unit is not None
            and not observation.metadata.get("model_output")
            and observation.unit not in expected_units.get(observation.kind, set())
        ):
            raise ValueError("Observation unit does not match reference output; convert explicitly")
        discrepancy = str(
            observation.metadata.get(
                "discrepancy",
                request.settings.get(
                    "discrepancy", "gaussian" if observation.uncertainty else "rmse"
                ),
            )
        )
        noise = dict(observation.uncertainty)
        if discrepancy == "gaussian" and "scale" in noise:
            scale = noise.pop("scale")
            if any(value != scale for value in noise.values()):
                raise ValueError("Conflicting Gaussian noise aliases")
            noise["sigma"] = scale
        allowed = {
            "gaussian": {"sigma", "sd"},
            "student_t": {"df", "scale", "sigma", "sd"},
            "rmse": set(),
            "mae": set(),
            "normalized_rmse": set(),
            "correlation": set(),
            "cosine": set(),
            "huber": {"delta"},
        }
        if discrepancy not in allowed or set(noise) - allowed[discrepancy]:
            raise ValueError("Unsupported discrepancy or incompatible noise parameters")
        aliases = [noise[name] for name in ("scale", "sigma", "sd") if name in noise]
        if aliases and any(value != aliases[0] for value in aliases):
            raise ValueError("Conflicting noise scale aliases")
        weight = finite_number(
            observation.metadata.get("weight", 1.0), "observation weight", strictly_positive=True
        )
        if any(value <= 0 for value in noise.values()):
            raise ValueError("Noise parameters must be positive")
        likelihood.append(
            {
                "term_id": observation.observation_id,
                "observation_ref": observation.artifact.model_dump(mode="json"),
                "model_output": output_path,
                "discrepancy": discrepancy,
                "noise_parameters": noise,
                "weight": weight,
                "metadata": {
                    **observation.metadata,
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
        "boundary_conditions": [
            item.model_dump(mode="json") for item in request.boundary_conditions
        ],
        "circulation": request.circulation.model_dump(mode="json"),
        "observations": [item.model_dump(mode="json") for item in request.observations],
        "settings": {
            **{
                key: value
                for key, value in request.settings.items()
                if key
                not in {
                    "discrepancy",
                    "forward_model",
                    "inference_backend",
                    "sampler_settings",
                    "seed",
                    "output_dir",
                }
            },
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
    proper = {item["discrepancy"] in {"gaussian", "student_t"} for item in likelihood}
    if len(proper) != 1:
        raise ValueError("Cannot mix posterior likelihoods with distance discrepancies")
    default_backend = "native-metropolis-v1" if True in proper else "native-abc-smc-v1"
    selected_backend = str(request.settings.get("inference_backend", default_backend))
    if selected_backend == "native-abc-smc-v1" and True in proper:
        raise ValueError("ABC requires distance discrepancies, not posterior likelihoods")
    cardiinfer_request = {
        "subject_id": request.subject_id,
        "model_service": "CardiMech",
        "model_capability": "mechanics.simulate",
        "backend": selected_backend,
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
