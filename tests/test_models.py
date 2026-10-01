import pytest

from cardimech import (
    BoundaryCondition,
    CirculationCoupling,
    MechanicsQC,
)


def test_pressure_boundary_requires_value_or_waveform() -> None:
    with pytest.raises(ValueError):
        BoundaryCondition(
            boundary_id="lv-pressure",
            kind="pressure",
            region="lv_endocardium",
        )


def test_enabled_circulation_requires_model() -> None:
    with pytest.raises(ValueError):
        CirculationCoupling(enabled=True, model="none")


def test_qc_cannot_pass_with_failed_checks() -> None:
    with pytest.raises(ValueError):
        MechanicsQC(
            passed=True,
            converged=True,
            checks={"finite_displacement": False},
        )
