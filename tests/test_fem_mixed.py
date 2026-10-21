"""Independent checks of mixed energy, condensation, stability and exact fields."""

import json
from math import factorial

import numpy as np
import pytest
from test_fem import request

from cardimech.fem_examples import cube_mesh, shell_mesh
from cardimech.fem_materials import constitutive
from cardimech.fem_mesh import load_mesh
from cardimech.fem_mixed import MiniAssembler, mixed_material, tetra_rule, volume_dual
from cardimech.models import BoundaryCondition
from cardimech.service import CardiMechService


@pytest.mark.parametrize("order", [5, 6, 7])
def test_quadrature_exact_monomials(order):
    bary, weights = tetra_rule(order)
    assert np.all(weights > 0)
    for powers in [(0, 0, 0, 0), (2, 1, 0, 0), (1, 1, 2, 2), (6, 0, 0, 0)]:
        exact = 6 * np.prod([factorial(p) for p in powers]) / factorial(sum(powers) + 3)
        assert weights @ np.prod(bary**powers, axis=1) == pytest.approx(exact, rel=1e-12)


@pytest.mark.parametrize(
    "model,pars",
    [
        ("neo_hookean", {"mu": 1000, "kappa": 1e7}),
        ("guccione", {"c": 400, "bf": 8, "bt": 2, "bfs": 3, "kappa": 1e7}),
    ],
)
def test_mixed_legendre_material_and_objectivity(model, pars):
    F = np.array([[[1.02, 0.03, 0], [0.01, 0.99, 0.02], [0, 0, 1.01]]])
    fiber, sheet = np.array([[1.0, 0, 0]]), np.array([[0.0, 1, 0]])
    p = pars["kappa"] * np.log(np.linalg.det(F)) / np.linalg.det(F)
    w, phys, P, C, _J, A = mixed_material(F, p, model, pars, fiber, sheet, 50)
    direct = constitutive(F, model, pars, fiber, sheet, 50)
    np.testing.assert_allclose(w, direct[0], rtol=1e-12)
    np.testing.assert_allclose(phys, direct[0], rtol=1e-12)
    np.testing.assert_allclose(P, direct[1], rtol=1e-12)
    # Eliminating local pressure recovers exactly the original penalty tangent.
    np.testing.assert_allclose(
        C
        + (pars["kappa"] * (1 - np.log(np.linalg.det(F))) / np.linalg.det(F) ** 2)[
            :, None, None, None, None
        ]
        * np.einsum("eij,ekl->eijkl", A, A),
        direct[2],
        rtol=1e-12,
    )
    eps = 1e-6
    for i in range(3):
        for j in range(3):
            d = np.zeros_like(F)
            d[0, i, j] = eps
            plus = mixed_material(F + d, p, model, pars, fiber, sheet, 50)
            minus = mixed_material(F - d, p, model, pars, fiber, sheet, 50)
            np.testing.assert_allclose((plus[0] - minus[0]) / (2 * eps), P[:, i, j], rtol=2e-7)
            np.testing.assert_allclose(
                (plus[2] - minus[2]) / (2 * eps), C[:, :, :, i, j], rtol=2e-7, atol=1e-4
            )
    Q = np.array([[0, -1.0, 0], [1.0, 0, 0], [0, 0, 1.0]])
    rotated = mixed_material(Q @ F, p, model, pars, fiber, sheet, 50)
    np.testing.assert_allclose(rotated[0], w, atol=1e-9)
    np.testing.assert_allclose(rotated[2], Q @ P, atol=1e-8)


def test_condensation_against_full_finite_difference(tmp_path):
    req = request(tmp_path, cube_mesh())
    mesh = load_mesh(req.anatomy_ref, req.subject_id)
    # Isolate one tetrahedron; assembly needs geometry, not the closed mesh contract.
    tet = mesh.elements[0]
    mesh.nodes = mesh.nodes[tet]
    mesh.elements = np.array([[0, 1, 2, 3]])
    mesh.gradients, mesh.volumes = mesh.gradients[:1], mesh.volumes[:1]
    mesh.fiber, mesh.sheet = mesh.fiber[:1], mesh.sheet[:1]
    nn = 4
    engine = MiniAssembler(mesh, "neo_hookean", {"mu": 1000.0, "kappa": 1e7})
    rng = np.random.default_rng(86)
    state = rng.normal(0, 1e-6, 19)

    def get(x, tangent=False):
        return engine.assemble(
            x[:12].reshape(4, 3), x[12:16], x[16:].reshape(1, 3), 25, 0, tangent=tangent
        )

    def residual(x):
        a = get(x)
        return np.r_[a.residual, a.bubble_residual.ravel()]

    h = 1e-8
    full = np.column_stack(
        [
            (residual(state + h * np.eye(19)[i]) - residual(state - h * np.eye(19)[i])) / (2 * h)
            for i in range(19)
        ]
    )
    a = get(state, True)
    expected = full[:16, :16] - full[:16, 16:] @ np.linalg.solve(full[16:, 16:], full[16:, :16])
    np.testing.assert_allclose(a.tangent.toarray(), expected, atol=2e-7, rtol=2e-6)
    r = residual(state)
    dx = rng.normal(0, 1e-6, 16)
    db = a.bubble_increment(dx).ravel()
    np.testing.assert_allclose(full[16:, :16] @ dx + full[16:, 16:] @ db + r[16:], 0, atol=1e-10)
    assert nn == len(mesh.nodes)


def test_pressure_has_only_constant_null_mode_at_incompressible_limit(tmp_path):
    req = request(tmp_path, cube_mesh())
    mesh = load_mesh(req.anatomy_ref, req.subject_id)
    engine = MiniAssembler(mesh, "neo_hookean", {"mu": 1000.0, "kappa": 1e30})
    a = engine.assemble(
        np.zeros_like(mesh.nodes), np.zeros(len(mesh.nodes)), np.zeros((48, 3)), 0, 0
    )
    # Bubble-pressure Schur coupling alone must control all pressure gradients.
    pressure = -a.tangent[mesh.nodes.size :, mesh.nodes.size :].toarray()
    eig = np.linalg.eigvalsh(pressure)
    assert abs(eig[0]) < 1e-12 * eig[-1]
    assert eig[1] > 1e-4 * eig[-1]


@pytest.mark.parametrize("kappa", [1000, 1e7])
@pytest.mark.parametrize("model", ["neo_hookean", "guccione"])
@pytest.mark.parametrize("isochoric", [True, False])
def test_mixed_affine_patch(tmp_path, kappa, model, isochoric):
    payload = cube_mesh()
    payload["fiber"] = [[1, 0, 0]] * 48
    payload["sheet"] = [[0, 1, 0]] * 48
    nodes = np.asarray(payload["nodes"])
    req = request(tmp_path, payload)
    req.settings.update(formulation="mixed_mini")
    req.parameters.passive["kappa"] = kappa
    if model == "guccione":
        payload["fiber"] = [[1, 0, 0]] * 48
        payload["sheet"] = [[0, 1, 0]] * 48
        req = request(tmp_path, payload)
        req.settings.update(formulation="mixed_mini")
        req.parameters.material_model = model
        req.parameters.passive = {"c": 400, "bf": 8, "bt": 2, "bfs": 3, "kappa": kappa}
        req.parameters.units = {"c": "Pa", "kappa": "Pa"}
    req.parameters.active = {"tension": 50}
    req.parameters.units["tension"] = "Pa"
    # Exactly isochoric finite stretch; also checks no hydrostatic forcing error.
    G = np.diag([1.02, 1 / 1.02 if isochoric else 1.01, 1.0]) - np.eye(3)
    for node, x in enumerate(nodes):
        if np.any((x == 0) | (x == 0.02)):
            req.boundary_conditions.append(
                BoundaryCondition(
                    boundary_id=str(node),
                    kind="fixed",
                    region="boundary",
                    unit="m",
                    metadata={"nodes": [node], "displacement": (G @ x).tolist()},
                )
            )
    out = CardiMechService(load_plugins=False).simulate(req)
    d = json.loads((tmp_path / "spatial-mechanics.json").read_text())
    np.testing.assert_allclose(d["displacement"], nodes @ G.T, atol=2e-11)
    np.testing.assert_allclose(d["bubble_displacement_m"], 0, atol=2e-11)
    assert out.qc.converged and out.provenance["formulation"] == "mixed_mini"
    assert d["minimum_quadrature_J"] > 0
    expected_p = kappa * np.log(np.linalg.det(np.eye(3) + G)) / np.linalg.det(np.eye(3) + G)
    np.testing.assert_allclose(
        d["mixed_pressure_Pa"], expected_p, atol=max(1e-5, abs(expected_p) * 1e-7)
    )


def test_explicit_unsupported_mixed_controls(tmp_path):
    req = request(tmp_path, shell_mesh(0))
    req.settings.update(formulation="mixed_mini", linear_solver="matrix_free_gmres")
    with pytest.raises(ValueError, match="mixed_mini currently requires"):
        CardiMechService(load_plugins=False).simulate(req)


def test_dual_domain_and_consistency():
    pars = {"mu": 1000.0, "kappa": 1e7}
    with pytest.raises(ValueError, match="5..7"):
        tetra_rule(4)
    with pytest.raises(ValueError, match="convex"):
        volume_dual(np.array([pars["kappa"] / np.e]), pars["kappa"])
    for j in [0.2, 0.9, 1.0, 1.1, 2.0]:
        p = pars["kappa"] * np.log(j) / j
        jp, compliance, _ = volume_dual(np.array([p]), pars["kappa"])
        np.testing.assert_allclose(jp, j, atol=1e-14)
        np.testing.assert_allclose(
            compliance, j * j / (pars["kappa"] * (1 - np.log(j))), rtol=1e-12
        )


def test_near_incompressible_auto_and_legacy_opt_in(tmp_path):
    req = request(tmp_path, cube_mesh())
    req.parameters.passive["kappa"] = 1e7
    req.boundary_conditions = [
        BoundaryCondition(
            boundary_id="all",
            kind="fixed",
            region="boundary",
            unit="m",
            metadata={"nodes": list(range(27))},
        )
    ]
    out = CardiMechService(load_plugins=False).simulate(req)
    assert out.provenance["formulation"] == "mixed_mini"
    assert out.provenance["requested_formulation"] == "auto"
    req.settings["linear_solver"] = "matrix_free_gmres"
    with pytest.raises(ValueError, match="Nearly incompressible default requires"):
        CardiMechService(load_plugins=False).simulate(req)
    req.settings["formulation"] = "displacement_p1"
    out = CardiMechService(load_plugins=False).simulate(req)
    assert out.provenance["formulation"] == "displacement_p1"
    assert any("equilibrium convergence does not establish accuracy" in w for w in out.warnings)
