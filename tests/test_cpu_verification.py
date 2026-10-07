import importlib.util
from pathlib import Path

import numpy as np
import pytest

from cardimech.activation import periodic_hill_activation
from cardimech.materials import material_point, mooney_rivlin_energy, neo_hookean_energy
from cardimech.pv import passive_pressure_mmHg, stroke_work_mmHg_ml


def test_independent_cpu_benchmarks():
    path = Path(__file__).resolve().parents[1] / "scripts/validate_cpu.py"
    spec = importlib.util.spec_from_file_location("validate_cpu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.validate()["passed"]


@pytest.mark.parametrize("gamma", [-0.4, -0.1, 0, 0.1, 0.4])
def test_isotropic_analytic_shear(gamma):
    F = np.eye(3)
    F[0, 1] = gamma
    assert neo_hookean_energy(F, mu=3.0, kappa=100.0) == pytest.approx(1.5 * gamma**2, abs=1e-14)
    assert mooney_rivlin_energy(F, c10=2.0, c01=1.0, kappa=100.0) == pytest.approx(
        3 * gamma**2, abs=1e-14
    )


def test_phase_wrap_and_material_point():
    assert np.array_equal(
        periodic_hill_activation(np.arange(21) * 0.8, cycle_length_s=0.8), np.zeros(21)
    )
    result = material_point("neo_hookean", np.eye(3), {"mu": 3.0, "kappa": 100.0})
    assert np.array_equal(result["first_piola_stress"], np.zeros((3, 3)))


def test_closed_polygon_work_and_model_range():
    assert (
        stroke_work_mmHg_ml(np.array([10.0, 20.0, 20.0, 10.0]), np.array([0.0, 0.0, 5.0, 5.0]))
        == 50.0
    )
    assert stroke_work_mmHg_ml(np.array([10.0, 20.0]), np.array([5.0, 5.0])) == 0.0
    with pytest.raises(ValueError, match="model range"):
        passive_pressure_mmHg(1000.0, v0_ml=1.0, a_mmHg=1.0, b=1.0)
