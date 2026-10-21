"""Synthetic bounded-assembly stress comparison; no patient data or guard bypass."""
import importlib.util
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

from cardimech.artifacts import write_json_artifact
from cardimech.fem_backend import assemble
from cardimech.fem_examples import shell_mesh
from cardimech.fem_mesh import load_mesh

root = Path(__file__).parent
spec = importlib.util.spec_from_file_location("cardimech.legacy_profile", root/"legacy_fem_backend.py")
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
payload = shell_mesh(4, layers=3)
ref = write_json_artifact(root, filename="synthetic-stress-mesh.json", artifact_id="synthetic-stress",
                          kind="tetra_mechanics_mesh", payload=payload)
mesh = load_mesh(ref, payload["subject_id"])
u = np.random.default_rng(78).normal(0, 1e-7, mesh.nodes.shape)
report = {"synthetic_only": True, "nodes": len(mesh.nodes), "tetrahedra": len(mesh.elements),
          "within_unchanged_mesh_limits": len(mesh.nodes)<=5000 and len(mesh.elements)<=20000,
          "measurement": "tracemalloc peak allocation during assembly, not process RSS",
          "baseline_commit": "5dc14d886ccbda5a1398b863f9ae7821bbffa18b", "materials": {}}
for model, parameters in [
        ("neo_hookean", {"mu": 1000, "kappa": 2000}),
        ("guccione", {"c": 400, "bf": 8, "bt": 2, "bfs": 3, "kappa": 2000})]:
    results, profiles = [], {}
    for name, function in [("legacy", legacy.assemble), ("bounded", assemble)]:
        tracemalloc.start()
        start = time.perf_counter()
        result = function(mesh, u, model, parameters, 20, 2)
        elapsed = time.perf_counter()-start
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results.append(result)
        profiles[name] = {"seconds": elapsed, "peak_traced_bytes": peak}
    old, new = results
    delta = old[1]-new[1]
    max_k = float(np.max(abs(delta.data), initial=0))
    max_r = float(np.max(abs(old[0]-new[0])))
    max_fields = [float(np.max(abs(a-b))) for a,b in zip(old[2],new[2],strict=True)]
    assert max_k<1e-9 and max_r<1e-11 and max(max_fields)<1e-10
    profiles["maximum_tangent_difference"] = max_k
    profiles["maximum_residual_difference_N"] = max_r
    profiles["maximum_field_differences"] = max_fields
    profiles["peak_traced_allocation_ratio"] = profiles["bounded"]["peak_traced_bytes"]/profiles["legacy"]["peak_traced_bytes"]
    report["materials"][model] = profiles
(root/"mechanics_stress.json").write_text(json.dumps(report, indent=2, allow_nan=False))
print(json.dumps(report, indent=2))
