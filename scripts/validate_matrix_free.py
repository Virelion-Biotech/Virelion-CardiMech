"""Synthetic clamped-shell refinement, capacity and residual qualification.

Run each case as a separate process to report its peak RSS, not a running peak
across multiple meshes. No patient anatomy or physiological validation.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

from cardimech.artifacts import write_json_artifact
from cardimech.fem_examples import shell_mesh
from cardimech.fem_mesh import load_mesh
from cardimech.models import MechanicsSimulationRequest
from cardimech.service import CardiMechService


def validate(level, layers, output, preconditioner="jacobi"):
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    payload = shell_mesh(level, layers=layers)
    nodes = np.asarray(payload["nodes"])
    radii = np.linalg.norm(nodes, axis=1)
    ref = write_json_artifact(
        output,
        filename="mesh.json",
        artifact_id="synthetic-shell",
        kind="tetra_mechanics_mesh",
        payload=payload,
    )
    n, ne = len(nodes), len(payload["tetrahedra"])
    default_rejection = None
    if n > 5000 or ne > 20000:
        try:
            load_mesh(ref, payload["subject_id"])
        except ValueError as exc:
            default_rejection = str(exc)
        assert default_rejection, "Standard mesh size guard was not enforced"
    request = MechanicsSimulationRequest(
        subject_id=payload["subject_id"],
        anatomy_ref=ref,
        backend="scipy-tetra-v1",
        parameters={
            "material_model": "neo_hookean",
            "active_model": "constant_fiber",
            "passive": {"mu": 1000, "kappa": 2000},
            "units": {"mu": "Pa", "kappa": "Pa"},
        },
        boundary_conditions=[
            {
                "boundary_id": "outer",
                "kind": "fixed",
                "region": "outer",
                "unit": "m",
                "metadata": {"nodes": np.flatnonzero(np.isclose(radii, 0.03)).tolist()},
            },
            {
                "boundary_id": "cavity",
                "kind": "pressure",
                "region": "cavity",
                "unit": "Pa",
                "value": 1.0,
            },
        ],
        settings={
            "output_dir": str(output),
            "linear_solver": "matrix_free_gmres",
            "mesh_profile": "large_experimental",
            "load_steps": 1,
            "max_iterations": 10,
            "absolute_tolerance_N": 1e-10,
            "relative_tolerance": 1e-8,
            "linear_relative_tolerance": 1e-8,
            "linear_max_iterations": 1200,
            "linear_preconditioner": preconditioner,
        },
    )
    print(
        json.dumps(
            {"phase": "solve", "nodes": n, "tetrahedra": ne, "default_guard": default_rejection}
        ),
        flush=True,
    )
    result = CardiMechService(load_plugins=False).simulate(request)
    fields = json.loads((output / "spatial-mechanics.json").read_text())
    displacement = np.asarray(fields["displacement"])
    inner = np.isclose(radii, 0.02)
    radial = np.einsum("ni,ni->n", displacement[inner], nodes[inner] / radii[inner, None])
    # Independent small-strain Lamé sphere with u(outer)=0 and inner pressure=1 Pa.
    # Compressible neo-Hookean linearizes to lambda=kappa, shear modulus=mu.
    ri, ro, mu, lam = 0.02, 0.03, 1000, 2000
    expected = (ro**3 / ri**2 - ri) / (3 * lam + 2 * mu + 4 * mu * ro**3 / ri**3)
    observed = float(radial.mean())
    relative_error = abs(observed - expected) / expected
    report = {
        "level": level,
        "layers": layers,
        "nodes": n,
        "tetrahedra": ne,
        "elapsed_seconds": time.perf_counter() - start,
        "peak_rss_KiB_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "mean_inner_radial_displacement_m": observed,
        "linear_Lame_displacement_m": expected,
        "relative_error_against_small_strain_Lame": relative_error,
        "qualification": "synthetic fixed-outer shell, neo-Hookean, 1 Pa; not patient qualification",
        "qc": result.qc.model_dump(mode="json"),
        "provenance": result.provenance,
        "load_history": fields["load_history"],
        "linear_history": fields["linear_history"],
        "default_guard_rejection": default_rejection,
    }
    (output / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    assert result.qc.converged and relative_error < 0.1
    print(json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=3, choices=[2, 3, 4, 5])
    parser.add_argument("--layers", type=int, default=4, choices=range(3, 9))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preconditioner", choices=["jacobi", "block_jacobi"], default="jacobi")
    arguments = parser.parse_args()
    validate(arguments.level, arguments.layers, arguments.output, arguments.preconditioner)
