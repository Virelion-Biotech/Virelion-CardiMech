import json

import numpy as np
import pytest

from cardimech.artifacts import write_json_artifact
from cardimech.fem_backend import assemble, pressure_load
from cardimech.fem_examples import cube_mesh, shell_mesh
from cardimech.fem_materials import constitutive
from cardimech.fem_mesh import cavity_volume, load_mesh
from cardimech.models import BoundaryCondition, MechanicsSimulationRequest
from cardimech.service import CardiMechService


def request(tmp_path, payload):
    ref = write_json_artifact(
        tmp_path,
        filename="mesh.json",
        artifact_id="mesh",
        kind="tetra_mechanics_mesh",
        payload=payload,
    )
    return MechanicsSimulationRequest(
        subject_id=payload["subject_id"],
        anatomy_ref=ref,
        backend="scipy-tetra-v1",
        parameters={
            "material_model": "neo_hookean",
            "active_model": "constant_fiber",
            "passive": {"mu": 1000, "kappa": 1000},
            "units": {"mu": "Pa", "kappa": "Pa"},
        },
        settings={"output_dir": str(tmp_path), "load_steps": 2},
    )


@pytest.mark.parametrize(
    "model,parameters",
    [
        ("neo_hookean", {"mu": 1000, "kappa": 2000}),
        ("guccione", {"c": 400, "bf": 8, "bt": 2, "bfs": 3, "kappa": 2000}),
    ],
)
def test_material_derivatives_and_objectivity(model, parameters):
    F = np.array([[[1.03, 0.02, 0], [0.01, 0.98, 0.01], [0, 0, 1.01]]])
    fiber = np.array([[1.0, 0, 0]])
    sheet = np.array([[0.0, 1, 0]])
    energy, P, C, _ = constitutive(F, model, parameters, fiber, sheet, 100)
    eps = 1e-6
    for k in range(3):
        for l in range(3):
            d = np.zeros_like(F)
            d[0, k, l] = eps
            wp, pp, *_ = constitutive(F + d, model, parameters, fiber, sheet, 100)
            wm, pm, *_ = constitutive(F - d, model, parameters, fiber, sheet, 100)
            np.testing.assert_allclose((wp - wm) / (2 * eps), P[:, k, l], rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(
                (pp - pm) / (2 * eps), C[:, :, :, k, l], rtol=1e-6, atol=1e-5
            )
    Q = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    rotated = constitutive(Q @ F, model, parameters, fiber, sheet, 100)
    np.testing.assert_allclose(rotated[0], energy, atol=1e-10)
    np.testing.assert_allclose(rotated[1], Q @ P, atol=1e-10)
    np.testing.assert_allclose(C, C.transpose(0, 3, 4, 1, 2), atol=1e-9)


def test_affine_patch(tmp_path):
    payload = cube_mesh()
    req = request(tmp_path, payload)
    nodes = np.array(payload["nodes"])
    G = np.diag([0.02, -0.01, 0.03])
    for node, x in enumerate(nodes):
        if np.any((x == 0) | (x == 0.02)):
            req.boundary_conditions.append(
                BoundaryCondition.model_validate(
                    {
                        "boundary_id": str(node),
                        "kind": "fixed",
                        "region": "boundary",
                        "unit": "m",
                        "metadata": {"nodes": [node], "displacement": (G @ x).tolist()},
                    }
                )
            )
    req = MechanicsSimulationRequest.model_validate(req.model_dump())
    result = CardiMechService(load_plugins=False).simulate(req)
    fields = json.loads((tmp_path / "spatial-mechanics.json").read_text())
    np.testing.assert_allclose(fields["displacement"], nodes @ G.T, atol=1e-11)
    np.testing.assert_allclose(
        fields["deformation_gradient"], np.broadcast_to(np.eye(3) + G, (48, 3, 3)), atol=1e-9
    )
    assert result.qc.converged and result.validation_status == "software_checked"


def test_follower_pressure_and_assembly_derivatives(tmp_path):
    req = request(tmp_path, shell_mesh(0))
    mesh = load_mesh(req.anatomy_ref, req.subject_id)
    x = mesh.nodes.copy()
    force, tangent = pressure_load(x, mesh.cavity, 2)
    eps = 1e-8
    for dof in [0, 7, 17]:
        plus = x.copy()
        minus = x.copy()
        plus.ravel()[dof] += eps
        minus.ravel()[dof] -= eps
        grad = (cavity_volume(plus, mesh.cavity) - cavity_volume(minus, mesh.cavity)) / (2 * eps)
        assert force[dof] == pytest.approx(2 * grad, rel=1e-7, abs=1e-12)
        fp, _ = pressure_load(plus, mesh.cavity, 2)
        fm, _ = pressure_load(minus, mesh.cavity, 2)
        np.testing.assert_allclose(
            (fp - fm) / (2 * eps), tangent[:, dof].toarray().ravel(), atol=1e-10
        )
    u = np.random.default_rng(4).normal(0, 1e-6, x.shape)
    _, K, _ = assemble(mesh, u, "neo_hookean", {"mu": 1000, "kappa": 1000}, 0, 2)
    for dof in [0, 7, 17]:
        plus = u.copy()
        minus = u.copy()
        plus.ravel()[dof] += eps
        minus.ravel()[dof] -= eps
        rp, _, _ = assemble(mesh, plus, "neo_hookean", {"mu": 1000, "kappa": 1000}, 0, 2)
        rm, _, _ = assemble(mesh, minus, "neo_hookean", {"mu": 1000, "kappa": 1000}, 0, 2)
        np.testing.assert_allclose(
            (rp - rm) / (2 * eps), K[:, dof].toarray().ravel(), rtol=1e-6, atol=1e-7
        )


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda r: r.settings.update(unsupported=True), "settings"),
        (lambda r: r.parameters.units.update(mu="kPa"), "Pa"),
        (lambda r: r.parameters.passive.update(extra=1), "exact"),
        (lambda r: None, "rigid"),
    ],
)
def test_reject_unsupported(tmp_path, change, match):
    req = request(tmp_path, cube_mesh())
    change(req)
    with pytest.raises(ValueError, match=match):
        CardiMechService(load_plugins=False).simulate(req)
    assert not (tmp_path / "spatial-mechanics.json").exists()


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda m: m.update(units="mm"), "coordinates"),
        (
            lambda m: m["tetrahedra"][0].__setitem__(slice(0, 2), m["tetrahedra"][0][1::-1]),
            "orientation",
        ),
        (lambda m: m["cavity_faces"].pop(), "closed"),
        (lambda m: m["cavity_faces"][0].reverse(), "outward"),
    ],
)
def test_invalid_mesh(tmp_path, mutation, match):
    payload = shell_mesh(0)
    mutation(payload)
    req = request(tmp_path, payload)
    with pytest.raises(ValueError, match=match):
        load_mesh(req.anatomy_ref, req.subject_id)


def test_shell_refinement():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "spatial_validation", Path(__file__).parents[1] / "scripts/validate_spatial.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.verify()["passed"]


@pytest.mark.parametrize("model", ["neo_hookean", "guccione"])
def test_active_equilibrium(tmp_path, model):
    payload = cube_mesh()
    payload["fiber"] = [[1, 0, 0]] * 48
    payload["sheet"] = [[0, 1, 0]] * 48
    req = request(tmp_path, payload)
    req.parameters.material_model = model
    if model == "guccione":
        req.parameters.passive = {"c": 400, "bf": 8, "bt": 2, "bfs": 3, "kappa": 1000}
        req.parameters.units = {"c": "Pa", "kappa": "Pa"}
    req.parameters.active = {"tension": 50}
    req.parameters.units["tension"] = "Pa"
    for axis in range(3):
        req.boundary_conditions.append(
            BoundaryCondition(
                boundary_id=str(axis),
                kind="fixed",
                region="symmetry",
                unit="m",
                metadata={"nodes": [0], "components": [axis]},
            )
        )
    req.boundary_conditions.extend(
        [
            BoundaryCondition(
                boundary_id="x-face",
                kind="fixed",
                region="symmetry",
                unit="m",
                metadata={"nodes": [18], "components": [1, 2]},
            ),
            BoundaryCondition(
                boundary_id="y-face",
                kind="fixed",
                region="symmetry",
                unit="m",
                metadata={"nodes": [6], "components": [2]},
            ),
        ]
    )
    result = CardiMechService(load_plugins=False).simulate(req)
    assert result.qc.converged and result.scalar_outputs["maximum_displacement_m"] > 0
    req.settings.update(max_iterations=1, load_steps=1)
    (tmp_path / "spatial-mechanics.json").unlink()
    with pytest.raises(ValueError, match="converge"):
        CardiMechService(load_plugins=False).simulate(req)
    assert not (tmp_path / "spatial-mechanics.json").exists()
