"""Reproducible analytic verification; no experimental or patient validation."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from cardimech.circulation import WindkesselParameters, simulate_lv_windkessel
from cardimech.materials import (
    guccione_energy,
    holzapfel_ogden_energy,
    mooney_rivlin_energy,
    neo_hookean_energy,
    neo_hookean_first_piola,
)
from cardimech.serialization import write_json

ROOT = Path(__file__).resolve().parents[1]
HO = {
    "a": 0.059,
    "b": 8.023,
    "af": 18.472,
    "bf": 16.026,
    "as_": 2.481,
    "bs": 11.120,
    "afs": 0.216,
    "bfs": 11.436,
    "kappa": 1000.0,
}
SOURCE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC6096751/"


def validate():
    checks = {}
    errors = []
    for i, j in itertools.permutations(range(3), 2):
        for g in np.linspace(-0.3, 0.3, 31):
            F = np.eye(3)
            F[i, j] = g
            exact = HO["a"] / (2 * HO["b"]) * np.expm1(HO["b"] * g * g)
            if j == 0:
                exact += HO["af"] / (2 * HO["bf"]) * np.expm1(HO["bf"] * g**4)
            if j == 1:
                exact += HO["as_"] / (2 * HO["bs"]) * np.expm1(HO["bs"] * g**4)
            if {i, j} == {0, 1}:
                exact += HO["afs"] / (2 * HO["bfs"]) * np.expm1(HO["bfs"] * g * g)
            errors.append(abs(holzapfel_ogden_energy(F, **HO) - exact))
    checks["six_shear_modes"] = {
        "samples": 186,
        "max_absolute_error_kpa": float(max(errors)),
        "tolerance_kpa": 1e-11,
        "passed": max(errors) < 1e-11,
    }
    laws = [
        (neo_hookean_energy, {"mu": 3.0, "kappa": 100.0}),
        (mooney_rivlin_energy, {"c10": 2.0, "c01": 1.0, "kappa": 100.0}),
        (guccione_energy, {"c": 2.0, "bf": 8.0, "bt": 3.0, "bfs": 4.0}),
        (holzapfel_ogden_energy, HO),
    ]
    theta = 0.47
    Q = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    F = np.array([[1.1, 0.1, 0.02], [0, 0.95, 0.04], [0.01, 0, 1.03]])
    err = max(abs(law(Q @ F, **params) - law(F, **params)) for law, params in laws)
    zero = max(abs(law(np.eye(3), **params)) for law, params in laws)
    checks["objectivity_and_identity"] = {
        "max_error": err,
        "identity_energy": zero,
        "tolerance": 1e-10,
        "passed": err < 1e-10 and zero < 1e-12,
    }
    h = 1e-6
    numeric = np.empty((3, 3))
    for i, j in itertools.product(range(3), repeat=2):
        delta = np.zeros((3, 3))
        delta[i, j] = h
        numeric[i, j] = (
            neo_hookean_energy(F + delta, mu=3.0, kappa=100.0)
            - neo_hookean_energy(F - delta, mu=3.0, kappa=100.0)
        ) / (2 * h)
    err = float(np.max(np.abs(numeric - neo_hookean_first_piola(F, mu=3.0, kappa=100.0))))
    checks["neo_hookean_stress_gradient"] = {
        "max_absolute_error": err,
        "tolerance": 1e-7,
        "passed": err < 1e-7,
    }
    errors = []
    residuals = []
    params = WindkesselParameters(
        p_atrium_mmHg=0.0,
        p_venous_mmHg=5.0,
        initial_lv_volume_ml=100.0,
        initial_arterial_pressure_mmHg=80.0,
    )
    for dt in [0.02, 0.01, 0.005]:
        t = np.linspace(0, 1, round(1 / dt) + 1)
        result = simulate_lv_windkessel(time_s=t, pressure_fn=lambda t, v: 0.0, params=params)
        expected = 5 + 75 * np.exp(-t / 1.5)
        errors.append(float(np.max(np.abs(result["arterial_pressure_mmHg"] - expected))))
        residuals.append(
            float(
                np.max(
                    np.abs(
                        np.diff(result["arterial_pressure_mmHg"]) * 1.5
                        + np.diff(t) * result["systemic_flow_ml_s"][:-1]
                    )
                )
            )
        )
    checks["windkessel_decay_conservation"] = {
        "dt_s": [0.02, 0.01, 0.005],
        "errors_mmHg": errors,
        "max_balance_residual_ml": max(residuals),
        "passed": errors[0] > errors[1] > errors[2]
        and 1.9 < errors[0] / errors[1] < 2.1
        and max(residuals) < 1e-10,
    }
    for check in checks.values():
        check["passed"] = bool(check["passed"])
    sources = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "src/cardimech").glob("*.py"))
    }
    sources["scripts/validate_cpu.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {
        "scope": "Analytic software verification, not experimental or clinical validation",
        "parameter_source": {
            "url": SOURCE,
            "table": "Table 2, Holzapfel and Ogden (2009) row",
            "stress_unit": "kPa",
            "kappa": "1000 kPa chosen for verification, not published row",
            "parameters": HO,
        },
        "checks": checks,
        "passed": all(c["passed"] for c in checks.values()),
        "source_sha256": sources,
    }


if __name__ == "__main__":
    report = validate()
    write_json(ROOT / "validation/cpu/results.json", report)
    print(json.dumps(report["checks"], indent=2, allow_nan=False))
    raise SystemExit(0 if report["passed"] else 1)
