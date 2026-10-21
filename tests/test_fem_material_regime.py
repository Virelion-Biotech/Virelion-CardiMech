"""Derived material-regime labels must not masquerade as accuracy qualification."""

import numpy as np
import pytest
from test_fem_krylov import clamped_request

from cardimech.fem_materials import linearized_regime
from cardimech.service import CardiMechService


@pytest.mark.parametrize("lam", [1000.0, 100000.0, 10000000.0])
def test_regime_matches_lame_constants(lam):
    result = linearized_regime("neo_hookean", {"mu": 1000, "kappa": lam})
    np.testing.assert_allclose(result["isotropic_poisson_ratio"], lam / (2 * (lam + 1000)))
    assert result["p1_locking_risk"] == (lam / (2 * (lam + 1000)) >= 0.49)


def test_anisotropic_regime_is_not_falsely_isotropic():
    result = linearized_regime("guccione", {"kappa": 1e6})
    assert result["isotropic_poisson_ratio"] is None and result["p1_locking_risk"] is None


def test_nearly_incompressible_run_carries_accuracy_warning(tmp_path):
    req, _, _ = clamped_request(tmp_path)
    req.parameters.passive["kappa"] = 100000
    req.settings["formulation"] = "displacement_p1"
    result = CardiMechService(load_plugins=False).simulate(req)
    assert result.qc.converged
    assert result.provenance["linearized_material_regime"]["p1_locking_risk"]
    assert any("equilibrium convergence does not establish accuracy" in s for s in result.warnings)
