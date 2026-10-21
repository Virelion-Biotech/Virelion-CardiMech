"""Paired same-process operator timings; enforce action agreement, not speed."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from cardimech.artifacts import write_json_artifact
from cardimech.fem_examples import shell_mesh
from cardimech.fem_krylov import tangent_action
from cardimech.fem_linearization import FrozenTangent
from cardimech.fem_mesh import load_mesh


def profile():
    records = []
    with tempfile.TemporaryDirectory() as directory:
        payload = shell_mesh(3, layers=4)
        ref = write_json_artifact(
            directory,
            filename="mesh.json",
            artifact_id="shell",
            kind="tetra_mechanics_mesh",
            payload=payload,
        )
        mesh = load_mesh(ref, payload["subject_id"])
        rng = np.random.default_rng(991)
        u = rng.normal(scale=1e-6, size=mesh.nodes.shape)
        vectors = rng.normal(size=(6, mesh.nodes.size))
        for model, parameters in [
            ("neo_hookean", {"mu": 1000, "kappa": 2000}),
            ("guccione", {"c": 400, "bf": 8, "bt": 2, "bfs": 3, "kappa": 2000}),
        ]:
            start = time.perf_counter()
            frozen = FrozenTangent(mesh, u, model, parameters, 20, 3)
            setup = time.perf_counter() - start

            def old(v, model=model, parameters=parameters):
                return tangent_action(mesh, u, model, parameters, 20, 3, v)

            for v in vectors:
                np.testing.assert_allclose(frozen.apply(v), old(v), rtol=1e-11, atol=1e-11)
            timings = {"uncached": [], "cached": []}
            for trial in range(4):
                order = [("uncached", old), ("cached", frozen.apply)]
                if trial % 2:
                    order.reverse()
                for name, operation in order:
                    start = time.perf_counter()
                    for v in vectors:
                        operation(v)
                    timings[name].append(time.perf_counter() - start)
            records.append(
                {
                    "model": model,
                    "nodes": len(mesh.nodes),
                    "elements": len(mesh.elements),
                    "six_action_seconds": timings,
                    "setup_seconds": setup,
                    "cached_material_bytes": frozen.cached_material_bytes,
                    "median_action_speedup": float(
                        np.median(timings["uncached"]) / np.median(timings["cached"])
                    ),
                }
            )
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = profile()
    args.output.write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))
