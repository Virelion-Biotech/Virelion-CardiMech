"""Check cached actions and block preconditioning against assembled independent paths."""

import numpy as np
import pytest
from test_fem import request
from test_fem_krylov import MATERIALS

from cardimech.fem_backend import assemble
from cardimech.fem_examples import shell_mesh
from cardimech.fem_krylov import tangent_action
from cardimech.fem_linearization import FrozenTangent, block_inverse, material_blocks
from cardimech.fem_mesh import load_mesh


@pytest.mark.parametrize("model,parameters", MATERIALS)
def test_cached_action_matches_full_and_freezes_state(tmp_path, monkeypatch, model, parameters):
    payload = shell_mesh(1)
    frames = np.linalg.qr(
        np.random.default_rng(81).normal(size=(len(payload["tetrahedra"]), 3, 3))
    )[0]
    payload["fiber"] = frames[:, :, 0].tolist()
    payload["sheet"] = frames[:, :, 1].tolist()
    req = request(tmp_path, payload)
    mesh = load_mesh(req.anatomy_ref, req.subject_id)
    rng = np.random.default_rng(225)
    u = rng.normal(scale=1e-6, size=mesh.nodes.shape)
    vector = rng.normal(size=mesh.nodes.size)
    parameters = dict(parameters)
    frozen = FrozenTangent(mesh, u, model, parameters, 35, 3)
    _, K, _ = assemble(mesh, u, model, parameters, 35, 3)
    expected = K @ vector
    np.testing.assert_allclose(frozen.apply(vector), expected, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        frozen.apply(vector), tangent_action(mesh, u, model, parameters, 35, 3, vector), atol=1e-11
    )
    u[:] = np.nan
    parameters["kappa"] = np.nan
    monkeypatch.setattr(
        np.linalg,
        "inv",
        lambda *args, **kwargs: pytest.fail("Inverse recomputed during cached action"),
    )
    np.testing.assert_allclose(frozen.apply(vector), expected, rtol=1e-11, atol=1e-11)
    assert frozen.cached_material_bytes > 0


@pytest.mark.parametrize("model,parameters", MATERIALS)
def test_material_blocks_and_partial_constraints(tmp_path, model, parameters):
    req = request(tmp_path, shell_mesh(0))
    mesh = load_mesh(req.anatomy_ref, req.subject_id)
    u = np.random.default_rng(9).normal(scale=1e-6, size=mesh.nodes.shape)
    _, K, _ = assemble(mesh, u, model, parameters, 40, 0)
    blocks = material_blocks(mesh, u, model, parameters, 40)
    full = K.toarray()
    for node, block in enumerate(blocks):
        np.testing.assert_allclose(
            block, full[3 * node : 3 * node + 3, 3 * node : 3 * node + 3], atol=1e-12
        )
    free = np.flatnonzero(np.arange(mesh.nodes.size) % 4 != 0)
    precondition = block_inverse(blocks, free)
    rhs = np.random.default_rng(7).normal(size=len(free))
    expected = np.zeros(mesh.nodes.size)
    for node, block in enumerate(blocks):
        components = [c for c in range(3) if 3 * node + c in free]
        if components:
            ids = np.array([np.flatnonzero(free == 3 * node + c)[0] for c in components])
            expected[3 * node + np.array(components)] = np.linalg.solve(
                block[np.ix_(components, components)], rhs[ids]
            )
    np.testing.assert_allclose(precondition(rhs), expected[free], rtol=1e-11, atol=1e-12)


def test_block_regularization_is_positive_and_finite():
    blocks = np.array([np.diag([-2.0, 0.0, 3.0]), np.diag([1.0, 2.0, 3.0])])
    apply = block_inverse(blocks, np.arange(6))
    v = np.arange(1.0, 7.0)
    assert np.isfinite(apply(v)).all() and v @ apply(v) > 0


def test_invalid_cached_states_rejected(tmp_path):
    req = request(tmp_path, shell_mesh(0))
    mesh = load_mesh(req.anatomy_ref, req.subject_id)
    with pytest.raises(ValueError, match="Finite"):
        FrozenTangent(
            mesh, np.full_like(mesh.nodes, np.nan), "neo_hookean", {"mu": 1000, "kappa": 2000}, 0, 0
        )
    frozen = FrozenTangent(
        mesh, np.zeros_like(mesh.nodes), "neo_hookean", {"mu": 1000, "kappa": 2000}, 0, 0
    )
    with pytest.raises(ValueError, match="Finite"):
        frozen.apply(np.full(mesh.nodes.size, np.nan))
