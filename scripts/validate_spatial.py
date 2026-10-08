"""Independent small-strain thick-shell verification of the finite-strain solver."""

from __future__ import annotations

import hashlib
import json
import platform
import tempfile
from pathlib import Path

import numpy as np
import scipy

from cardimech.artifacts import write_json_artifact
from cardimech.fem_examples import shell_mesh
from cardimech.models import MechanicsSimulationRequest
from cardimech.service import CardiMechService


def verify(levels=(0, 1, 2)):
    records = []
    a, b, pressure, mu, lam = 0.02, 0.03, 0.1, 1000.0, 1000.0
    B = pressure / (4 * mu * (1 / a**3 - 1 / b**3))
    A = 4 * mu * B / ((3 * lam + 2 * mu) * b**3)
    exact = A * a + B / a**2
    with tempfile.TemporaryDirectory() as directory:
        for level in levels:
            payload = shell_mesh(level, layers=level + 3, inner=a, outer=b)
            ref = write_json_artifact(
                directory,
                filename="mesh.json",
                artifact_id="mesh",
                kind="tetra_mechanics_mesh",
                payload=payload,
            )
            nodes = np.array(payload["nodes"])
            # Slip constraints on coordinate planes preserve exact radial displacement.
            bcs = []
            for axis in range(3):
                bcs.append(
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
                )
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
                settings={"output_dir": directory, "load_steps": 1, "absolute_tolerance_N": 1e-12},
            )
            result = CardiMechService(load_plugins=False).simulate(req)
            fields = json.loads((Path(directory) / "spatial-mechanics.json").read_text())
            u = np.array(fields["displacement"])
            inner = np.unique(payload["cavity_faces"])
            radial = np.einsum("ij,ij->i", u[inner], nodes[inner] / a)
            measured = float(radial.mean())
            error = abs(measured - exact) / exact
            records.append(
                {
                    "refinement": level,
                    "nodes": len(nodes),
                    "elements": len(payload["tetrahedra"]),
                    "analytic_inner_displacement_m": exact,
                    "mean_inner_displacement_m": measured,
                    "relative_error": error,
                    "residual_N": result.scalar_outputs["residual_N"],
                }
            )
    passed = (
        all(
            records[i + 1]["relative_error"] < records[i]["relative_error"]
            for i in range(len(records) - 1)
        )
        and records[-1]["relative_error"] < 0.15
    )
    return {
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "source_sha256": {
            name: hashlib.sha256(
                (Path(__file__).parents[1] / "src" / "cardimech" / name).read_bytes()
            ).hexdigest()
            for name in ("fem_backend.py", "fem_materials.py", "fem_mesh.py", "fem_examples.py")
        },
        "benchmark": "linear elastic thick sphere, small pressure limit of compressible neo-Hookean FEM",
        "parameters": {
            "inner_m": a,
            "outer_m": b,
            "pressure_Pa": pressure,
            "mu_Pa": mu,
            "lambda_Pa": lam,
        },
        "records": records,
        "passed": passed,
        "patient_validated": False,
        "limits": "Synthetic closed shell; analytic linear limit, not patient anatomy or Land benchmark.",
    }


if __name__ == "__main__":
    report = verify()
    Path("validation/cpu").mkdir(parents=True, exist_ok=True)
    Path("validation/cpu/spatial-shell.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)
