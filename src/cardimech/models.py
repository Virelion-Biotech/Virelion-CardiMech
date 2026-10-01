from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    kind: str
    uri: str
    sha256: str | None = Field(default=None, min_length=64, max_length=64)
    coordinate_frame: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MechanicalParameterSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passive: dict[str, float] = Field(default_factory=dict)
    active: dict[str, float] = Field(default_factory=dict)
    units: dict[str, str] = Field(default_factory=dict)
    source: Literal["prior", "calibrated", "fixed", "unknown"] = "unknown"


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
        "custom",
    ]
    region: str
    value: float | None = None
    unit: str | None = None
    waveform_ref: ArtifactRef | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_value_or_waveform(self) -> "BoundaryCondition":
        if self.value is None and self.waveform_ref is None and self.kind not in {"fixed", "custom"}:
            raise ValueError("Boundary condition requires a scalar value or waveform reference")
        return self


class CirculationCoupling(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    model: Literal["none", "zero_d", "external"] = "none"
    parameters: dict[str, float] = Field(default_factory=dict)
    state_ref: ArtifactRef | None = None

    @model_validator(mode="after")
    def validate_enabled_model(self) -> "CirculationCoupling":
        if self.enabled and self.model == "none":
            raise ValueError("Enabled circulation coupling requires a model")
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
    def consistent_status(self) -> "MechanicsQC":
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
    circulation: CirculationCoupling = Field(default_factory=CirculationCoupling)
    settings: dict[str, Any] = Field(default_factory=dict)


class MechanicsSimulationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: str = "1.0"
    subject_id: str
    backend: str
    parameters: MechanicalParameterSet
    outputs: list[ArtifactRef] = Field(default_factory=list)
    scalar_outputs: dict[str, float] = Field(default_factory=dict)
    qc: MechanicsQC | None = None
    validation_status: Literal[
        "unvalidated",
        "software_checked",
        "numerically_checked",
        "empirically_checked",
    ] = "unvalidated"
    provenance: dict[str, Any] = Field(default_factory=dict)
