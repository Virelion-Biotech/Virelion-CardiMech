"""Experimental bounded-storage Newton tangent actions and checked GMRES solves."""

from __future__ import annotations

import inspect
import math

import numpy as np

from .fem_materials import constitutive

CHUNK = 512


def stress_action(F, dF, model, parameters, fiber, sheet, tension):
    """Exact directional derivative of the declared first-Piola material law."""
    J = np.linalg.det(F)
    if np.any(J <= 0) or not np.isfinite(J).all():
        raise ValueError("Element inversion/nonfinite tangent state")
    A = np.linalg.inv(F).transpose(0, 2, 1)
    logJ = np.log(J)
    dlogJ = np.einsum("eij,eij->e", A, dF)
    dA = -A @ dF.transpose(0, 2, 1) @ A
    kappa = parameters["kappa"]
    if model == "neo_hookean":
        mu = parameters["mu"]
        dP = mu * dF + (kappa * dlogJ)[:, None, None] * A + (kappa * logJ - mu)[:, None, None] * dA
    elif model == "guccione":
        basis = np.stack((fiber, sheet, np.cross(fiber, sheet)), axis=2)
        local = basis.transpose(0, 2, 1) @ (0.5 * (F.transpose(0, 2, 1) @ F - np.eye(3))) @ basis
        bf, bt, bfs = (parameters[key] for key in ("bf", "bt", "bfs"))
        weights = np.array([[bf, bfs, bfs], [bfs, bt, bt], [bfs, bt, bt]])
        M = weights * local
        scale = parameters["c"] * np.exp(np.sum(M * local, axis=(1, 2)))
        S = basis @ (scale[:, None, None] * M) @ basis.transpose(0, 2, 1)
        dE = 0.5 * (F.transpose(0, 2, 1) @ dF + dF.transpose(0, 2, 1) @ F)
        dLocal = basis.transpose(0, 2, 1) @ dE @ basis
        dq = 2 * np.sum(M * dLocal, axis=(1, 2))
        dS = (
            basis
            @ (scale[:, None, None] * (weights * dLocal + dq[:, None, None] * M))
            @ basis.transpose(0, 2, 1)
        )
        dP = (
            dF @ S
            + F @ dS
            + (kappa * dlogJ)[:, None, None] * A
            + (kappa * logJ)[:, None, None] * dA
        )
    else:
        raise ValueError("Unknown spatial material for tangent action")
    dP += tension * np.einsum("ei,ej->eij", np.einsum("eij,ej->ei", dF, fiber), fiber)
    if not np.isfinite(dP).all():
        raise ValueError("Nonfinite constitutive tangent action")
    return dP


def tangent_action(mesh, u, model, parameters, tension, pressure, vector):
    """Apply the complete material and follower-pressure tangent without global K."""
    vector = np.asarray(vector, dtype=float).reshape(mesh.nodes.shape)
    if not np.isfinite(vector).all():
        raise ValueError("Finite tangent direction required")
    result = np.zeros_like(mesh.nodes)
    for start in range(0, len(mesh.elements), CHUNK):
        stop = start + CHUNK
        cells, gradients = mesh.elements[start:stop], mesh.gradients[start:stop]
        F = np.eye(3) + np.einsum("eai,eaj->eij", u[cells], gradients)
        dF = np.einsum("eai,eaj->eij", vector[cells], gradients)
        dP = stress_action(
            F, dF, model, parameters, mesh.fiber[start:stop], mesh.sheet[start:stop], tension
        )
        local = np.einsum("e,eij,eaj->eai", mesh.volumes[start:stop], dP, gradients)
        np.add.at(result, cells, local)
    x = mesh.nodes + u
    for start in range(0, len(mesh.cavity), CHUNK):
        faces = mesh.cavity[start : start + CHUNK]
        a, b, c = x[faces].transpose(1, 0, 2)
        va, vb, vc = vector[faces].transpose(1, 0, 2)
        darea = (np.cross(vb - va, c - a) + np.cross(b - a, vc - va)) / 2
        np.add.at(result, faces.ravel(), np.repeat(-pressure * darea / 3, 3, axis=0))
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite global tangent action")
    return result.ravel()


def tangent_diagonal(mesh, u, model, parameters, tension):
    """Material diagonal in bounded batches; follower-pressure diagonal is zero."""
    diagonal = np.zeros_like(mesh.nodes)
    for start in range(0, len(mesh.elements), CHUNK):
        stop = start + CHUNK
        cells, g = mesh.elements[start:stop], mesh.gradients[start:stop]
        F = np.eye(3) + np.einsum("eai,eaj->eij", u[cells], g)
        _, _, C, _ = constitutive(
            F, model, parameters, mesh.fiber[start:stop], mesh.sheet[start:stop], tension
        )
        local = np.einsum("e,eaj,eijil,eal->eai", mesh.volumes[start:stop], g, C, g)
        np.add.at(diagonal, cells, local)
    if not np.isfinite(diagonal).all():
        raise ValueError("Nonfinite tangent preconditioner")
    return diagonal.ravel()


def solve_increment(
    mesh,
    u,
    model,
    parameters,
    tension,
    pressure,
    free,
    rhs,
    *,
    relative_tolerance=1e-8,
    max_iterations=1200,
    restart=60,
):
    """Fail closed on nonconvergence, checking the unpreconditioned residual."""
    from scipy.sparse.linalg import LinearOperator, gmres

    size = mesh.nodes.size
    # Freeze the linearization: caller mutation must not alter this linear solve.
    u = np.array(u, dtype=float, copy=True)
    free = np.asarray(free, dtype=int)
    rhs = np.asarray(rhs, dtype=float)
    if rhs.shape != (len(free),) or not np.isfinite(rhs).all() or not len(free):
        raise ValueError("Finite nonempty free-DOF right-hand side required")
    diagonal = np.abs(tangent_diagonal(mesh, u, model, parameters, tension)[free])
    floor = max(float(diagonal.max()), 1.0) * 1e-12
    inverse = 1 / np.maximum(diagonal, floor)

    def action(value):
        full = np.zeros(size)
        full[free] = value
        return tangent_action(mesh, u, model, parameters, tension, pressure, full)[free]

    operator = LinearOperator((len(free), len(free)), matvec=action, dtype=float)
    preconditioner = LinearOperator(operator.shape, matvec=lambda v: inverse * v, dtype=float)
    iterations = 0

    def callback(_):
        nonlocal iterations
        iterations += 1
        if iterations > max_iterations:
            raise ValueError("Matrix-free GMRES exceeded linear iteration limit")

    # SciPy 1.11 uses tol; later versions use rtol. Both use callback_type here.
    tolerance_key = "rtol" if "rtol" in inspect.signature(gmres).parameters else "tol"
    solution, info = gmres(
        operator,
        rhs,
        M=preconditioner,
        restart=min(restart, len(free)),
        maxiter=math.ceil(max_iterations / min(restart, len(free))),
        atol=1e-14,
        callback=callback,
        callback_type="pr_norm",
        **{tolerance_key: relative_tolerance},
    )
    error = float(np.linalg.norm(action(solution) - rhs))
    target = max(relative_tolerance * float(np.linalg.norm(rhs)), 1e-14)
    if info != 0 or not np.isfinite(solution).all() or error > target * (1 + 1e-6):
        raise ValueError(
            f"Matrix-free GMRES failed true residual: info={info}, error={error}, target={target}"
        )
    return solution, {"iterations": iterations, "residual_N": error, "tolerance_N": target}
