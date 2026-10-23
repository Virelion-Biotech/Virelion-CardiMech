"""Compare the candidate with a frozen MINI implementation, in the same process.

Run in Colab. Timings are observations, not portable performance guarantees.
The reference is loaded from git without modifying either implementation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from cardimech.artifacts import write_json_artifact
from cardimech.fem_examples import shell_mesh
from cardimech.fem_mesh import load_mesh
from cardimech.fem_mixed import MiniAssembler

REFERENCE = "b2dacf79a8d0cdb93fbd34e5d78b19d4ee06d534"


def compare(output, levels=(1, 2, 3), repeats=3):
    output.mkdir(parents=True, exist_ok=True)
    source = subprocess.check_output(
        ["git", "show", f"{REFERENCE}:src/cardimech/fem_mixed.py"], text=True
    )
    reference_path = output / "reference_fem_mixed.py"
    reference_path.write_text(source)
    spec = importlib.util.spec_from_file_location("cardimech._mini_reference", reference_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    rows = []
    for model in ("neo_hookean", "guccione"):
        pars = (
            {"mu": 1000.0, "kappa": 1e7}
            if model == "neo_hookean"
            else {"c": 400, "bf": 8, "bt": 2, "bfs": 3, "kappa": 1e7}
        )
        for level in levels if model == "neo_hookean" else (1,):
            payload = shell_mesh(level, layers=level + 3)
            ref = write_json_artifact(
                output,
                filename=f"mesh-{model}-{level}.json",
                artifact_id=f"mesh-{model}-{level}",
                kind="tetra_mechanics_mesh",
                payload=payload,
            )
            mesh = load_mesh(ref, payload["subject_id"])
            old, new = (cls(mesh, model, pars) for cls in (module.MiniAssembler, MiniAssembler))
            rng = np.random.default_rng(714)
            u = mesh.nodes @ np.array([[0.01, 0.003, 0], [0, -0.005, 0.002], [0, 0, 0.004]])
            q = rng.normal(0, 1e-5, len(mesh.nodes))
            bubble = rng.normal(0, 1e-7, (len(mesh.elements), 3))
            args = (u, q, bubble, 75.0, 1.0)
            timings = {"reference": [], "candidate": []}
            # Alternate order to reduce systematic warm-up / scheduling bias.
            for i in range(repeats):
                for name, engine in (
                    (("reference", old), ("candidate", new))
                    if i % 2 == 0
                    else (("candidate", new), ("reference", old))
                ):
                    start = time.perf_counter()
                    value = engine.assemble(*args)
                    timings[name].append(time.perf_counter() - start)
                    if name == "reference":
                        a = value
                    else:
                        b = value
            delta = b.tangent - a.tangent
            tangent_relative_error = np.linalg.norm(delta.data) / np.linalg.norm(a.tangent.data)
            assert tangent_relative_error < 1e-10
            dx = rng.normal(0, 1e-7, 4 * len(mesh.nodes))
            for actual, expected in [
                (b.residual, a.residual),
                (b.bubble_residual, a.bubble_residual),
                (b.condensed_residual, a.condensed_residual),
                (b.constraint, a.constraint),
                (b.bubble_increment(dx), a.bubble_increment(dx)),
                *zip(b.fields, a.fields),
            ]:
                np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
            assert b.minimum_J == a.minimum_J
            row = {
                "model": model,
                "level": level,
                "nodes": len(mesh.nodes),
                "tetrahedra": len(mesh.elements),
                "tangent_relative_error": float(tangent_relative_error),
                "all_state_checks_passed": True,
                "seconds": timings,
                "median_speedup": float(
                    np.median(timings["reference"]) / np.median(timings["candidate"])
                ),
            }
            rows.append(row)
            (output / "comparison.json").write_text(
                json.dumps({"reference_commit": REFERENCE, "rows": rows}, indent=2) + "\n"
            )
            print(json.dumps(row), flush=True)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    if args.repeats < 1:
        p.error("--repeats must be positive")
    compare(args.output, repeats=args.repeats)
