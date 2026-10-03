import pytest

from cardimech import (
    BoundaryCondition,
    CirculationCoupling,
    MechanicalParameterSet,
    MechanicsObservation,
    MechanicsQC,
)


def test_pressure_boundary_requires_value_or_waveform() -> None:
    with pytest.raises(ValueError):
        BoundaryCondition(boundary_id="lv-pressure", kind="pressure", region="lv_endocardium")


def test_enabled_circulation_requires_model() -> None:
    with pytest.raises(ValueError):
        CirculationCoupling(enabled=True, model="none")


def test_qc_cannot_pass_with_failed_checks() -> None:
    with pytest.raises(ValueError):
        MechanicsQC(passed=True, converged=True, checks={"finite_displacement": False})


def test_parameter_values_must_be_finite() -> None:
    with pytest.raises(ValueError):
        MechanicalParameterSet(passive={"bad": float("nan")})


def test_observation_uncertainty_must_be_nonnegative() -> None:
    with pytest.raises(ValueError):
        MechanicsObservation(
            observation_id="obs",
            kind="end_diastolic_volume",
            artifact={"artifact_id": "a", "kind": "scalar", "uri": "memory://a"},
            uncertainty={"scale": -1.0},
        )
