"""Frozen second-order material state and nodal block preconditioning.

No global sparse tangent or element fourth-order tangent cache is retained.
"""

from __future__ import annotations

import numpy as np

from .fem_materials import constitutive

CHUNK = 512


class FrozenTangent:
    """Cache deformation-dependent coefficients once per Newton linear solve."""

    def __init__(self, mesh, u, model, parameters, tension, pressure):
        self.mesh = mesh
        self.model = model
        self.parameters = dict(parameters)
        self.tension, self.pressure = tension, pressure
        u = np.array(u, dtype=float, copy=True)
        if u.shape != mesh.nodes.shape or not np.isfinite(u).all():
            raise ValueError("Finite nodal displacement required for linearization")
        self.coordinates = mesh.nodes + u
        self.fiber = mesh.fiber.copy()
        ne = len(mesh.elements)
        shapes = {"F": (ne, 3, 3), "A": (ne, 3, 3), "logJ": (ne,)}
        if model == "guccione":
            shapes.update(basis=(ne, 3, 3), M=(ne, 3, 3), S=(ne, 3, 3), scale=(ne,))
            bf, bt, bfs = (parameters[key] for key in ("bf", "bt", "bfs"))
            self.weights = np.array([[bf, bfs, bfs], [bfs, bt, bt], [bfs, bt, bt]])
        self.state = {name: np.empty(shape) for name, shape in shapes.items()}
        for start in range(0, ne, CHUNK):
            stop = start + CHUNK
            cells, g = mesh.elements[start:stop], mesh.gradients[start:stop]
            F = np.eye(3) + np.einsum("eai,eaj->eij", u[cells], g)
            _, _, _, J = constitutive(
                F,
                model,
                parameters,
                mesh.fiber[start:stop],
                mesh.sheet[start:stop],
                tension,
                compute_tangent=False,
            )
            self.state["F"][start:stop] = F
            self.state["A"][start:stop] = np.linalg.inv(F).transpose(0, 2, 1)
            self.state["logJ"][start:stop] = np.log(J)
            if model == "guccione":
                fiber, sheet = mesh.fiber[start:stop], mesh.sheet[start:stop]
                basis = np.stack((fiber, sheet, np.cross(fiber, sheet)), axis=2)
                local = (
                    basis.transpose(0, 2, 1)
                    @ (0.5 * (F.transpose(0, 2, 1) @ F - np.eye(3)))
                    @ basis
                )
                M = self.weights * local
                scale = parameters["c"] * np.exp(np.sum(M * local, axis=(1, 2)))
                S = basis @ (scale[:, None, None] * M) @ basis.transpose(0, 2, 1)
                for name, values in [("basis", basis), ("M", M), ("scale", scale), ("S", S)]:
                    self.state[name][start:stop] = values
        if not all(np.isfinite(x).all() for x in self.state.values()):
            raise ValueError("Nonfinite cached material state")
        self.cached_material_bytes = sum(x.nbytes for x in self.state.values())

    def apply(self, vector):
        mesh = self.mesh
        vector = np.asarray(vector, dtype=float).reshape(mesh.nodes.shape)
        if not np.isfinite(vector).all():
            raise ValueError("Finite tangent direction required")
        result = np.zeros_like(mesh.nodes)
        kappa = self.parameters["kappa"]
        for start in range(0, len(mesh.elements), CHUNK):
            stop = start + CHUNK
            cells, g = mesh.elements[start:stop], mesh.gradients[start:stop]
            state = {key: values[start:stop] for key, values in self.state.items()}
            F, A, logJ = state["F"], state["A"], state["logJ"]
            dF = np.einsum("eai,eaj->eij", vector[cells], g)
            dlogJ = np.einsum("eij,eij->e", A, dF)
            dA = -A @ dF.transpose(0, 2, 1) @ A
            if self.model == "neo_hookean":
                mu = self.parameters["mu"]
                dP = (
                    mu * dF
                    + (kappa * dlogJ)[:, None, None] * A
                    + (kappa * logJ - mu)[:, None, None] * dA
                )
            else:
                basis, M, scale, S = (state[key] for key in ("basis", "M", "scale", "S"))
                dE = 0.5 * (F.transpose(0, 2, 1) @ dF + dF.transpose(0, 2, 1) @ F)
                dLocal = basis.transpose(0, 2, 1) @ dE @ basis
                dq = 2 * np.sum(M * dLocal, axis=(1, 2))
                dS = (
                    basis
                    @ (scale[:, None, None] * (self.weights * dLocal + dq[:, None, None] * M))
                    @ basis.transpose(0, 2, 1)
                )
                dP = (
                    dF @ S
                    + F @ dS
                    + (kappa * dlogJ)[:, None, None] * A
                    + (kappa * logJ)[:, None, None] * dA
                )
            fiber = self.fiber[start:stop]
            dP += self.tension * np.einsum("ei,ej->eij", np.einsum("eij,ej->ei", dF, fiber), fiber)
            np.add.at(result, cells, np.einsum("e,eij,eaj->eai", mesh.volumes[start:stop], dP, g))
        for start in range(0, len(mesh.cavity), CHUNK):
            faces = mesh.cavity[start : start + CHUNK]
            a, b, c = self.coordinates[faces].transpose(1, 0, 2)
            va, vb, vc = vector[faces].transpose(1, 0, 2)
            darea = (np.cross(vb - va, c - a) + np.cross(b - a, vc - va)) / 2
            np.add.at(result, faces.ravel(), np.repeat(-self.pressure * darea / 3, 3, axis=0))
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite cached tangent action")
        return result.ravel()


def material_blocks(mesh, u, model, parameters, tension):
    """Exact 3x3 nodal diagonal blocks of the material tangent, excluding pressure."""
    blocks = np.zeros((len(mesh.nodes), 3, 3))
    for start in range(0, len(mesh.elements), CHUNK):
        stop = start + CHUNK
        cells, g = mesh.elements[start:stop], mesh.gradients[start:stop]
        F = np.eye(3) + np.einsum("eai,eaj->eij", u[cells], g)
        _, _, C, _ = constitutive(
            F, model, parameters, mesh.fiber[start:stop], mesh.sheet[start:stop], tension
        )
        local = np.einsum("e,eaj,eijkl,eal->eaik", mesh.volumes[start:stop], g, C, g)
        np.add.at(blocks, cells, local)
    if not np.isfinite(blocks).all():
        raise ValueError("Nonfinite nodal tangent blocks")
    return blocks


def block_inverse(blocks, free):
    """SPD approximation on free components, with explicit spectral regularization."""
    mask = np.zeros(blocks.shape[:2], dtype=float)
    mask.ravel()[free] = 1
    restricted = (blocks + blocks.transpose(0, 2, 1)) / 2 * mask[:, :, None] * mask[:, None, :]
    restricted += np.eye(3)[None] * (1 - mask)[:, :, None]
    eigenvalues, vectors = np.linalg.eigh(restricted)
    floor = max(float(np.abs(eigenvalues).max()), 1.0) * 1e-12
    inverse = (vectors / np.maximum(np.abs(eigenvalues), floor)[:, None, :]) @ vectors.transpose(
        0, 2, 1
    )
    if not np.isfinite(inverse).all():
        raise ValueError("Nonfinite block preconditioner")

    def apply(vector):
        full = np.zeros(blocks.shape[:2])
        full.ravel()[free] = vector
        return np.einsum("nij,nj->ni", inverse, full).ravel()[free]

    return apply
