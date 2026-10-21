"""Independent assembled/finite-difference and nonlinear checks of matrix-free FEM."""

import json

import numpy as np
import pytest
from test_fem import request

from cardimech.fem_backend import assemble
from cardimech.fem_examples import shell_mesh
from cardimech.fem_krylov import solve_increment, stress_action, tangent_action, tangent_diagonal
from cardimech.fem_materials import constitutive
from cardimech.fem_mesh import load_mesh
from cardimech.models import BoundaryCondition
from cardimech.service import CardiMechService

MATERIALS = [
    ("neo_hookean", {"mu": 1000, "kappa": 2000}),
    ("guccione", {"c": 400, "bf": 8, "bt": 2, "bfs": 3, "kappa": 2000}),
]


@pytest.mark.parametrize("model,parameters", MATERIALS)
def test_directional_material_matches_tensor_and_finite_difference(model, parameters):
    rng = np.random.default_rng(48)
    F = np.eye(3)[None] + rng.normal(scale=0.02, size=(9, 3, 3))
    dF = rng.normal(size=F.shape)
    fiber = np.tile([1., 0., 0.], (9, 1))
    sheet = np.tile([0., 1., 0.], (9, 1))
    action = stress_action(F, dF, model, parameters, fiber, sheet, 30)
    C = constitutive(F, model, parameters, fiber, sheet, 30)[2]
    np.testing.assert_allclose(action, np.einsum("eijkl,ekl->eij", C, dF), rtol=1e-12, atol=1e-10)
    epsilon = 1e-6
    plus = constitutive(F + epsilon*dF, model, parameters, fiber, sheet, 30)[1]
    minus = constitutive(F - epsilon*dF, model, parameters, fiber, sheet, 30)[1]
    np.testing.assert_allclose(action, (plus-minus)/(2*epsilon), rtol=2e-7, atol=1e-5)


@pytest.mark.parametrize("model,parameters", MATERIALS)
def test_global_action_and_diagonal_include_follower_pressure(tmp_path, model, parameters):
    req = request(tmp_path, shell_mesh(1))
    mesh = load_mesh(req.anatomy_ref, req.subject_id)
    rng = np.random.default_rng(73)
    u = rng.normal(scale=1e-6, size=mesh.nodes.shape)
    v = rng.normal(size=mesh.nodes.size)
    _, K, _ = assemble(mesh, u, model, parameters, 20, 4)
    actual = tangent_action(mesh, u, model, parameters, 20, 4, v)
    np.testing.assert_allclose(actual, K@v, rtol=1e-11, atol=1e-10)
    np.testing.assert_allclose(tangent_diagonal(mesh, u, model, parameters, 20), K.diagonal(), rtol=1e-12, atol=1e-12)
    epsilon = 1e-8
    plus = assemble(mesh, u + epsilon*v.reshape(u.shape), model, parameters, 20, 4, compute_tangent=False)[0]
    minus = assemble(mesh, u - epsilon*v.reshape(u.shape), model, parameters, 20, 4, compute_tangent=False)[0]
    np.testing.assert_allclose(actual, (plus-minus)/(2*epsilon), rtol=1e-6, atol=1e-6)


def clamped_request(tmp_path):
    req = request(tmp_path, shell_mesh(0))
    mesh = load_mesh(req.anatomy_ref, req.subject_id)
    outer = np.flatnonzero(np.isclose(np.linalg.norm(mesh.nodes, axis=1), 0.03)).tolist()
    req.boundary_conditions = [
        BoundaryCondition(boundary_id="outer", kind="fixed", region="outer", unit="m", metadata={"nodes": outer}),
        BoundaryCondition(boundary_id="cavity", kind="pressure", region="cavity", unit="Pa", value=3),
    ]
    return req, mesh, np.setdiff1d(np.arange(mesh.nodes.size), np.array(outer)[:, None]*3 + np.arange(3))


def test_reduced_linear_solve_and_real_nonconvergence(tmp_path):
    from scipy.sparse.linalg import spsolve

    req, mesh, free = clamped_request(tmp_path)
    u = np.zeros_like(mesh.nodes)
    r, K, _ = assemble(mesh, u, "neo_hookean", {"mu":1000,"kappa":2000}, 0, 3)
    actual, report = solve_increment(mesh, u, "neo_hookean", {"mu":1000,"kappa":2000}, 0, 3, free, -r[free], relative_tolerance=1e-10)
    np.testing.assert_allclose(actual, spsolve(K[free][:,free], -r[free]), rtol=1e-8, atol=1e-12)
    assert report["residual_N"] <= report["tolerance_N"]*(1+1e-6)
    with pytest.raises(ValueError, match="GMRES"):
        solve_increment(mesh, u, "neo_hookean", {"mu":1000,"kappa":2000}, 0, 3, free, np.random.default_rng(1).normal(size=len(free)), max_iterations=1, restart=1)


def test_false_gmres_success_rejected_by_true_residual(tmp_path, monkeypatch):
    import scipy.sparse.linalg

    _, mesh, free = clamped_request(tmp_path)
    monkeypatch.setattr(scipy.sparse.linalg, "gmres", lambda A, b, **kwargs: (np.zeros_like(b), 0))
    with pytest.raises(ValueError, match="true residual"):
        solve_increment(mesh, np.zeros_like(mesh.nodes), "neo_hookean", {"mu":1000,"kappa":2000}, 0, 0, free, np.ones(len(free)))


def test_nonlinear_matrix_free_matches_direct(tmp_path):
    req, _, _ = clamped_request(tmp_path)
    service = CardiMechService(load_plugins=False)
    direct = service.simulate(req)
    original = json.loads((tmp_path/"spatial-mechanics.json").read_text())
    req.settings.update(linear_solver="matrix_free_gmres", linear_relative_tolerance=1e-10)
    iterative = service.simulate(req)
    result = json.loads((tmp_path/"spatial-mechanics.json").read_text())
    np.testing.assert_allclose(result["displacement"], original["displacement"], rtol=1e-7, atol=1e-11)
    assert iterative.qc.converged and direct.qc.converged and result["linear_history"]
    assert iterative.provenance["linear_solver"] == "matrix_free_gmres"


@pytest.mark.parametrize("settings", [
    {"mesh_profile":"large_experimental"},
    {"linear_solver":"invalid"},
    {"linear_solver":"matrix_free_gmres","linear_restart":101},
    {"linear_solver":"matrix_free_gmres","linear_relative_tolerance":1e-3},
    {"linear_solver":"matrix_free_gmres","linear_max_iterations":True},
    {"linear_relative_tolerance":1e-8},
])
def test_unsupported_controls_fail_before_solve(tmp_path, settings):
    req, _, _ = clamped_request(tmp_path)
    req.settings.update(settings)
    with pytest.raises((ValueError, TypeError)):
        CardiMechService(load_plugins=False).simulate(req)
