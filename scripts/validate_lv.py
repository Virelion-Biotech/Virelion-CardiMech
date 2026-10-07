"""Independent adaptive integration and synthetic parameter recovery on CPU."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from cardimech import MechanicsAPI
from cardimech.activation import periodic_hill_activation
from cardimech.serialization import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def validate():
    api = MechanicsAPI()
    payload = read_json(ROOT / "examples/reference_request.json")
    payload["settings"] = {"cycles": 5, "dt_s": 0.001, "inline_series": True}

    # Independently implement pressure and mass balance, sharing only prescribed activation.
    def rhs(t, y):
        v, pa = y
        act = float(periodic_hill_activation([t], cycle_length_s=0.8, rise_s=0.07, decay_s=0.24)[0])
        pressure = 0.08 * np.expm1(0.055 * (v / 10 - 1)) + act * 2.1 * max(v - 10, 0)
        qm = max((8 - pressure) / 0.01, 0)
        qa = max((pressure - pa) / 0.015, 0)
        qs = max(pa - 5, 0)
        return [qm - qa, (qa - qs) / 1.5]

    y = [120.0, 75.0]
    for cycle in range(5):
        solution = solve_ivp(
            rhs,
            (cycle * 0.8, (cycle + 1) * 0.8),
            y,
            method="DOP853",
            rtol=1e-10,
            atol=1e-10,
            max_step=0.002,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        y = solution.y[:, -1]
    errors = []
    for dt in [0.001, 0.0005, 0.00025]:
        payload["settings"]["dt_s"] = dt
        result = api.simulate(payload)
        endpoint = np.array(
            [result["series"]["lv_volume_ml"][-1], result["series"]["arterial_pressure_mmHg"][-1]]
        )
        errors.append(float(np.max(np.abs(endpoint - y))))
    check = {
        "dt_s": [0.001, 0.0005, 0.00025],
        "max_terminal_state_errors": errors,
        "reference_terminal_volume_ml": float(y[0]),
        "reference_terminal_arterial_pressure_mmHg": float(y[1]),
        "tolerance_finest": 0.1,
        "passed": errors[0] > errors[1] > errors[2] and errors[2] < 0.1,
    }
    payload["settings"] = {
        "cycles": 20,
        "dt_s": 0.001,
        "inline_series": False,
        "require_periodic_convergence": True,
        "cycle_volume_tolerance_ml": 0.001,
        "cycle_pressure_tolerance_mmHg": 0.001,
    }
    periodic = api.simulate(payload)
    # Same-model synthetic recovery tests identifiability/implementation, not empirical validity.
    payload["settings"] = {"cycles": 3, "dt_s": 0.001, "inline_series": False}
    truth = api.simulate(payload)["scalar_outputs"]["peak_lv_pressure_mmHg"]

    def objective(emax):
        payload["parameters"]["active"]["emax_mmHg_per_ml"] = float(emax)
        predicted = api.simulate(payload)["scalar_outputs"]["peak_lv_pressure_mmHg"]
        return (predicted - truth) ** 2

    recovery = minimize_scalar(
        objective, bounds=(1.5, 2.7), method="bounded", options={"xatol": 1e-7}
    )
    recovered = float(recovery.x)
    recovery_check = {
        "truth_emax_mmHg_per_ml": 2.1,
        "recovered": recovered,
        "evaluations": int(recovery.nfev),
        "tolerance": 1e-4,
        "passed": bool(recovery.success and abs(recovered - 2.1) < 1e-4),
    }
    checks = {
        "adaptive_ode_comparison": check,
        "periodic_convergence": {
            "passed": periodic["qc"]["converged"],
            "metrics": periodic["qc"]["metrics"],
        },
        "synthetic_recovery": recovery_check,
    }
    sources = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "src/cardimech").glob("*.py"))
    }
    sources["scripts/validate_lv.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {
        "scope": "Independent numerical verification and same-model synthetic recovery; no clinical validation",
        "checks": checks,
        "passed": all(c["passed"] for c in checks.values()),
        "source_sha256": sources,
    }


if __name__ == "__main__":
    result = validate()
    write_json(ROOT / "validation/cpu/lv_results.json", result)
    print(result["checks"])
    raise SystemExit(0 if result["passed"] else 1)
