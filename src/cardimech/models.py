from __future__ import annotations

import math
import string
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    kind: str
    uri: str
    sha256: str | None = Field(default=None, min_length=64, max_length=64)
    coordinate_frame: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        lowered = value.lower()
        if any(character not in string.hexdigits for character in lowered):
            raise ValueError("sha256 must contain exactly 64 hexadecimal characters")
        return lowered


class MechanicsObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation_id: str
    kind: Literal[
        "end_diastolic_volume",
        "end_systolic_volume",
        "ejection_fraction",
        "peak_pressure",
        "pressure_curve",
        "volume_curve",
        "pv_loop",
        "strain_curve",
        "displacement_field",
        "wall_thickness",
        "stroke_work",
        "other",
    ]
    artifact: ArtifactRef
    region: str | None = None
    unit: str | None = None
    uncertainty: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_uncertainty(self) -> MechanicsObservation:
        for name, value in self.uncertainty.items():
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError(f"Observation uncertainty {name!r} must be finite and non-negative")
        return self


class MechanicalParameterSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passive: dict[str, float] = Field(default_factory=dict)
    active: dict[str, float] = Field(default_factory=dict)
    units: dict[str, str] = Field(default_factory=dict)
    source: Literal["prior", "calibrated", "fixed", "unknown"] = "unknown"
    material_model: str = "holzapfel_ogden"
    active_model: str = "periodic_hill"

    @model_validator(mode="after")
    def require_finite_values(self) -> MechanicalParameterSet:
        for group_name, values in (("passive", self.passive), ("active", self.active)):
            for name, value in values.items():
                if not math.isfinite(float(value)):
                    raise ValueError(f"{group_name} parameter {name!r} must be finite")
        return self


class BoundaryCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    boundary_id: str
    kind: Literal[
        "pressure",
        "traction",
        "fixed",
        "spring",
        "volume",
        "pericardial",
        "robin",
        "custom",
    ]
    region: str
    value: float | None = None
    unit: str | None = None
    waveform_ref: ArtifactRef | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_value_or_waveform(self) -> BoundaryCondition:
        if (
            self.value is None
            and self.waveform_ref is None
            and self.kind not in {"fixed", "custom"}
        ):
            raise ValueError("Boundary condition requires a scalar value or waveform reference")
        if self.value is not None and not math.isfinite(float(self.value)):
            raise ValueError("Boundary condition value must be finite")
        return self


class CirculationCoupling(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    model: Literal["none", "windkessel_3e", "closed_loop_0d", "external"] = "none"
    parameters: dict[str, float] = Field(default_factory=dict)
    state_ref: ArtifactRef | None = None

    @model_validator(mode="after")
    def validate_enabled_model(self) -> CirculationCoupling:
        if self.enabled and self.model == "none":
            raise ValueError("Enabled circulation coupling requires a model")
        for name, value in self.parameters.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"Circulation parameter {name!r} must be finite")
        return self


class MechanicsQC(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    converged: bool | None = None
    checks: dict[str, bool] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_status(self) -> MechanicsQC:
        if self.passed and (
            self.errors
            or self.converged is False
            or any(not value for value in self.checks.values())
        ):
            raise ValueError("passed=True is inconsistent with failed mechanics QC")
        return self


class MechanicsSimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject_id: str
    anatomy_ref: ArtifactRef
    backend: str
    parameters: MechanicalParameterSet
    boundary_conditions: list[BoundaryCondition] = Field(default_factory=list)
    activation_ref: ArtifactRef | None = None
    observations: list[MechanicsObservation] = Field(default_factory=list)
    circulation: CirculationCoupling = Field(default_factory=CirculationCoupling)
    settings: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_ids(self) -> MechanicsSimulationRequest:
        boundary_ids = [item.boundary_id for item in self.boundary_conditions]
        if len(boundary_ids) != len(set(boundary_ids)):
            raise ValueError("Boundary-condition IDs must be unique")
        observation_ids = [item.observation_id for item in self.observations]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("Mechanics observation IDs must be unique")
        return self


class MechanicsSimulationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: str = "2.0"
    subject_id: str
    backend: str
    parameters: MechanicalParameterSet
    outputs: list[ArtifactRef] = Field(default_factory=list)
    scalar_outputs: dict[str, float] = Field(default_factory=dict)
    series: dict[str, list[float]] = Field(default_factory=dict)
    qc: MechanicsQC | None = None
    validation_status: Literal[
        "unvalidated",
        "software_checked",
        "numerically_checked",
        "empirically_checked",
    ] = "unvalidated"
    warnings: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)


class MechanicsCalibrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject_id: str
    anatomy_ref: ArtifactRef
    activation_ref: ArtifactRef | None = None
    observations: list[MechanicsObservation]
    backend: str = "numpy-lumped-v1"
    parameter_bounds: dict[str, tuple[float, float]]
    initial_parameters: MechanicalParameterSet | None = None
    boundary_conditions: list[BoundaryCondition] = Field(default_factory=list)
    circulation: CirculationCoupling = Field(default_factory=CirculationCoupling)
    settings: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_problem(self) -> MechanicsCalibrationRequest:
        if not self.observations:
            raise ValueError("At least one mechanics observation is required")
        ids = [item.observation_id for item in self.observations]
        if len(ids) != len(set(ids)):
            raise ValueError("Mechanics observation IDs must be unique")
        for name, bounds in self.parameter_bounds.items():
            lo, hi = bounds
            if not (math.isfinite(float(lo)) and math.isfinite(float(hi)) and lo < hi):
                raise ValueError(f"Invalid parameter bounds for {name!r}")
        return self


class MechanicsCalibrationBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: str = "1.0"
    subject_id: str
    model_service: str = "CardiMech"
    model_capability: str = "mechanics.simulate"
    forward_template: dict[str, Any]
    priors: list[dict[str, Any]]
    likelihood: list[dict[str, Any]]
    model_context: dict[str, Any] = Field(default_factory=dict)
    cardiinfer_request: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
