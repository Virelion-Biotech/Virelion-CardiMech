"""Bounded element assembly, residual-only line searches and unchanged mesh guards."""

import numpy as np
import pytest
from test_fem import request

from cardimech import fem_backend
from cardimech.fem_backend import assemble, pressure_load
from cardimech.fem_examples import shell_mesh
from cardimech.fem_materials import constitutive
from cardimech.fem_mesh import load_mesh


@pytest.mark.parametrize(
    "model,parameters",
    [
        ("neo_hookean", {"mu": 1000, "kappa": 2000}),
        ("guccione", {"c": 400, "bf": 8, "bt": 2, "bfs": 3, "kappa": 2000}),
    ],
)
def test_bounded_assembly_and_residual_only_are_equivalent(
    tmp_path, monkeypatch, model, parameters
):
    payload = shell_mesh(0)
    payload["fiber"] = [[1.0, 0.0, 0.0]] * len(payload["tetrahedra"])
    payload["sheet"] = [[0.0, 1.0, 0.0]] * len(payload["tetrahedra"])
    req = request(tmp_path, payload)
    mesh = load_mesh(req.anatomy_ref, req.subject_id)
    u = np.random.default_rng(34).normal(scale=1e-6, size=mesh.nodes.shape)
    expected_r, expected_k, expected_fields = assemble(
        mesh, u, model, parameters, 30, 2, chunk_size=4096
    )
    calls = []
    original = fem_backend.constitutive

    def tracked(*args, **kwargs):
        calls.append((len(args[0]), kwargs["compute_tangent"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(fem_backend, "constitutive", tracked)
    actual_r, actual_k, fields = assemble(mesh, u, model, parameters, 30, 2, chunk_size=7)
    assert max(n for n, _ in calls) <= 7
    np.testing.assert_allclose(actual_r, expected_r, atol=1e-14)
    np.testing.assert_allclose(actual_k.toarray(), expected_k.toarray(), atol=1e-10)
    for actual, expected in zip(fields, expected_fields, strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-13)
    calls.clear()
    residual, k, _ = assemble(
        mesh, u, model, parameters, 30, 2, chunk_size=7, compute_tangent=False
    )
    assert k is None and calls and all(not tangent for _, tangent in calls)
    np.testing.assert_allclose(residual, expected_r, atol=1e-14)
    force, derivative = pressure_load(mesh.nodes + u, mesh.cavity, 2, compute_tangent=False)
    assert derivative is None
    np.testing.assert_allclose(force, pressure_load(mesh.nodes + u, mesh.cavity, 2)[0], atol=1e-14)


@pytest.mark.parametrize(
    "model,parameters",
    [
        ("neo_hookean", {"mu": 1000, "kappa": 2000}),
        ("guccione", {"c": 400, "bf": 8, "bt": 2, "bfs": 3, "kappa": 2000}),
    ],
)
def test_material_residual_only_retains_inversion_and_finite_guards(model, parameters):
    f = np.array([np.diag([1.02, 0.99, 1.03])])
    fiber, sheet = np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]])
    full = constitutive(f, model, parameters, fiber, sheet, 20)
    lean = constitutive(f, model, parameters, fiber, sheet, 20, compute_tangent=False)
    assert lean[2] is None
    for index in [0, 1, 3]:
        np.testing.assert_allclose(lean[index], full[index], atol=1e-13)
    f[0, 0, 0] = -1
    with pytest.raises(ValueError, match="inversion"):
        constitutive(f, model, parameters, fiber, sheet, 20, compute_tangent=False)


@pytest.mark.parametrize("chunk", [0, -1, True, 1.5, 4097])
def test_invalid_chunk_control_rejected(tmp_path, chunk):
    req = request(tmp_path, shell_mesh(0))
    mesh = load_mesh(req.anatomy_ref, req.subject_id)
    with pytest.raises(ValueError, match="chunk_size"):
        assemble(
            mesh,
            np.zeros_like(mesh.nodes),
            "neo_hookean",
            {"mu": 1000, "kappa": 1000},
            0,
            0,
            chunk_size=chunk,
        )


def test_patient_size_guard_remains_enforced(tmp_path):
    payload = shell_mesh(0)
    payload["nodes"] += [[1.0, 1.0, 1.0]] * (5001 - len(payload["nodes"]))
    req = request(tmp_path, payload)
    with pytest.raises(ValueError, match="5000"):
        load_mesh(req.anatomy_ref, req.subject_id)
