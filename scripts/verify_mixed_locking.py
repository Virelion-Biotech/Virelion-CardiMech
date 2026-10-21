"""Frozen side-by-side locking, quadrature and load qualification campaign."""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path

from qualify_material_regimes import qualify


def verify(output):
    output.mkdir(parents=True, exist_ok=True)
    baseline = qualify(output / "baseline")
    mixed = qualify(output / "mixed", "mixed_mini")
    q6 = qualify(output / "quadrature-6", "mixed_mini", (3,), 6, (1e7,))[0]
    low = qualify(output / "low-load", "mixed_mini", (3,), 5, (1e7,), 0.01)[0]
    stiff = qualify(output / "higher-bulk", "mixed_mini", (3,), 5, (1e9,))[0]
    high5 = qualify(output / "finite-load-5", "mixed_mini", (2,), 5, (1e7,), 100.0)[0]
    high6 = qualify(output / "finite-load-6", "mixed_mini", (2,), 6, (1e7,), 100.0)[0]
    all_cases = baseline + mixed + [q6, low, stiff, high5, high6]
    checks = {"all_equilibria_passed": all(r.get("equilibrium_passed", False) for r in all_cases)}
    if checks["all_equilibria_passed"]:
        finest = [r for r in mixed if r["level"] == 3]
        old = next(r for r in baseline if r["level"] == 3 and r["lambda_Pa"] == 1e7)
        new = next(r for r in finest if r["lambda_Pa"] == 1e7)

        def change(a, b):
            return abs(
                a["observed_inner_displacement_m"] - b["observed_inner_displacement_m"]
            ) / abs(b["observed_inner_displacement_m"])

        checks.update(
            finest_all_regimes_within_original_15_percent=all(
                r["accuracy_within_15_percent"] for r in finest
            ),
            near_incompressible_error_improved_at_least_fourfold=new["relative_displacement_error"]
            < old["relative_displacement_error"] / 4,
            refinement_errors_decrease=all(
                all(a > b for a, b in pairwise(es))
                for es in [
                    [r["relative_displacement_error"] for r in mixed if r["lambda_Pa"] == lam]
                    for lam in (1e3, 1e5, 1e7)
                ]
            ),
            finest_quadrature_change_below_one_percent=change(q6, new) < 0.01,
            low_load_scaled_displacement_change_below_one_percent=abs(
                low["observed_inner_displacement_m"] * 10 / new["observed_inner_displacement_m"] - 1
            )
            < 0.01,
            higher_bulk_error_within_original_15_percent=stiff["accuracy_within_15_percent"],
            finite_load_quadrature_change_below_one_percent=change(high6, high5) < 0.01,
        )
    report = {
        "acceptance": checks,
        "passed": all(checks.values()),
        "baseline": baseline,
        "mixed": mixed,
        "quadrature_6": q6,
        "low_load": low,
        "higher_bulk": stiff,
        "finite_load_5": high5,
        "finite_load_6": high6,
        "scope": "synthetic homogeneous material, finite bulk compliance, assembled direct MINI only; no patient accuracy qualification",
        "coarse_accuracy_failures_retained": [
            r for r in mixed if not r.get("accuracy_within_15_percent", False)
        ],
    }
    (output / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "QUALIFICATION", json.dumps({"passed": report["passed"], "acceptance": checks}), flush=True
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    if not verify(p.parse_args().output)["passed"]:
        raise SystemExit(1)
