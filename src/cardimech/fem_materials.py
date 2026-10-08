"""Consistent stresses and tangents for displacement-based tetrahedral mechanics.

Guccione is explicitly augmented by kappa/2 * log(J)^2 for compressibility.
Active stress is prescribed second-Piola fibre tension: P_a=T*(F f) tensor f.
All material coefficients and stress units are Pa.
"""

from __future__ import annotations

import numpy as np


def constitutive(F, model, parameters, fiber, sheet, tension=0.0):
    F = np.asarray(F, dtype=float)
    if F.ndim != 3 or F.shape[1:] != (3, 3) or not np.isfinite(F).all():
        raise ValueError("Deformation gradients must be finite element-wise 3x3 tensors")
    J = np.linalg.det(F)
    if np.any(J <= 0) or not np.isfinite(J).all():
        raise ValueError("Element inversion/nonfinite deformation determinant")
    A = np.linalg.inv(F).transpose(0, 2, 1)
    eye = np.eye(3)
    logJ = np.log(J)
    kappa = parameters["kappa"]
    if model == "neo_hookean":
        mu = parameters["mu"]
        coefficient = kappa * logJ - mu
        energy = 0.5 * mu * (np.sum(F * F, axis=(1, 2)) - 3) - mu * logJ + 0.5 * kappa * logJ**2
        P = mu * F + coefficient[:, None, None] * A
        tangent = (
            mu * np.einsum("ik,jl->ijkl", eye, eye)[None]
            + kappa * np.einsum("eij,ekl->eijkl", A, A)
            - coefficient[:, None, None, None, None] * np.einsum("eil,ekj->eijkl", A, A)
        )
    elif model == "guccione":
        basis = np.stack((fiber, sheet, np.cross(fiber, sheet)), axis=2)
        E = 0.5 * (F.transpose(0, 2, 1) @ F - eye)
        local = basis.transpose(0, 2, 1) @ E @ basis
        bf, bt, bfs = (parameters[key] for key in ("bf", "bt", "bfs"))
        weights = np.array([[bf, bfs, bfs], [bfs, bt, bt], [bfs, bt, bt]])
        M = weights * local
        q = np.sum(M * local, axis=(1, 2))
        scale = parameters["c"] * np.exp(q)
        S = basis @ (scale[:, None, None] * M) @ basis.transpose(0, 2, 1)
        P = F @ S + (kappa * logJ)[:, None, None] * A
        energy = 0.5 * parameters["c"] * np.expm1(q) + 0.5 * kappa * logJ**2
        tangent = np.empty((len(F), 3, 3, 3, 3))
        for k in range(3):
            for l in range(3):
                dF = np.zeros((3, 3))
                dF[k, l] = 1
                dE = 0.5 * (F.transpose(0, 2, 1) @ dF + dF.T @ F)
                dLocal = basis.transpose(0, 2, 1) @ dE @ basis
                dq = 2 * np.sum(M * dLocal, axis=(1, 2))
                dS = (
                    basis
                    @ (scale[:, None, None] * (weights * dLocal + dq[:, None, None] * M))
                    @ basis.transpose(0, 2, 1)
                )
                tangent[:, :, :, k, l] = dF @ S + F @ dS
        tangent += kappa * np.einsum("eij,ekl->eijkl", A, A) - (kappa * logJ)[
            :, None, None, None, None
        ] * np.einsum("eil,ekj->eijkl", A, A)
    else:
        raise ValueError("Spatial FEM supports only neo_hookean or guccione")
    stretched = np.einsum("eij,ej->ei", F, fiber)
    energy += 0.5 * tension * (np.sum(stretched**2, axis=1) - 1)
    P += tension * np.einsum("ei,ej->eij", stretched, fiber)
    tangent += tension * np.einsum("ik,ej,el->eijkl", eye, fiber, fiber)
    if not all(np.isfinite(x).all() for x in (energy, P, tangent)):
        raise ValueError("Constitutive energy/stress/tangent exceeds finite numerical range")
    return energy, P, tangent, J
