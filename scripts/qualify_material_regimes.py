"""Characterize P1 locking using an independent traction-free thick-sphere limit.

Accuracy failures are retained as results, separately from equilibrium failures.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cardimech.artifacts import write_json_artifact
from cardimech.fem_examples import shell_mesh
from cardimech.fem_materials import linearized_regime
from cardimech.models import MechanicsSimulationRequest
from cardimech.service import CardiMechService


def qualify(output):
    output.mkdir(parents=True, exist_ok=True)
    records = []
    a, b, pressure, mu = 0.02, 0.03, 0.1, 1000.0
    for lam in [1000.0, 100000.0, 10000000.0]:
        B = pressure / (4 * mu * (1 / a**3 - 1 / b**3))
        A = 4 * mu * B / ((3 * lam + 2 * mu) * b**3)
        exact = A * a + B / a**2
        for level in [1, 2, 3]:
            directory = output / f"lambda-{lam:g}-level-{level}"
            directory.mkdir()
            record = {
                "lambda_Pa": lam,
                "mu_Pa": mu,
                "pressure_Pa": pressure,
                "level": level,
                "analytic_inner_displacement_m": exact,
                "material_regime": linearized_regime("neo_hookean", {"mu": mu, "kappa": lam}),
            }
            start = time.perf_counter()
            try:
                payload = shell_mesh(level, layers=level + 3)
                nodes = np.asarray(payload["nodes"])
                ref = write_json_artifact(
                    directory,
                    filename="mesh.json",
                    artifact_id="synthetic-shell",
                    kind="tetra_mechanics_mesh",
                    payload=payload,
                )
                bcs = [
                    {
                        "boundary_id": f"slip-{axis}",
                        "kind": "fixed",
                        "region": "symmetry",
                        "unit": "m",
                        "metadata": {
                            "nodes": np.flatnonzero(abs(nodes[:, axis]) < 1e-14).tolist(),
                            "components": [axis],
                        },
                    }
                    for axis in range(3)
                ]
                bcs.append(
                    {
                        "boundary_id": "pressure",
                        "kind": "pressure",
                        "region": "cavity",
                        "unit": "Pa",
                        "value": pressure,
                    }
                )
                req = MechanicsSimulationRequest(
                    subject_id=payload["subject_id"],
                    anatomy_ref=ref,
                    backend="scipy-tetra-v1",
                    parameters={
                        "material_model": "neo_hookean",
                        "active_model": "constant_fiber",
                        "passive": {"mu": mu, "kappa": lam},
                        "units": {"mu": "Pa", "kappa": "Pa"},
                    },
                    boundary_conditions=bcs,
                    settings={
                        "output_dir": str(directory),
                        "load_steps": 1,
                        "absolute_tolerance_N": 1e-10,
                    },
                )
                result = CardiMechService(load_plugins=False).simulate(req)
                data = json.loads((directory / "spatial-mechanics.json").read_text())
                inner = np.unique(payload["cavity_faces"])
                u = np.asarray(data["displacement"])
                observed = float(np.einsum("ij,ij->i", u[inner], nodes[inner] / a).mean())
                error = abs(observed - exact) / exact
                record.update(
                    nodes=len(nodes),
                    tetrahedra=len(payload["tetrahedra"]),
                    observed_inner_displacement_m=observed,
                    relative_displacement_error=error,
                    accuracy_within_15_percent=error < 0.15,
                    equilibrium_passed=result.qc.converged,
                    residual_N=result.scalar_outputs["residual_N"],
                    warnings=result.warnings,
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                record.update(
                    equilibrium_passed=False, error_type=type(exc).__name__, error=str(exc)
                )
            record["elapsed_seconds"] = time.perf_counter() - start
            records.append(record)
            (output / "regime-results.json").write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps(record), flush=True)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = qualify(args.output)
    # Execution failures cause nonzero exit; accuracy failures remain visible data.
    if not all(r["equilibrium_passed"] for r in records):
        raise SystemExit(1)
