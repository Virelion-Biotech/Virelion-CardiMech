"""Independently linearize the original P1 problem to isolate locking from Newton.

Uses the exact small-strain elasticity tensor and reference pressure force;
no call to constitutive(), nonlinear assembly or Newton/GMRES is made.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from cardimech.artifacts import write_json_artifact
from cardimech.fem_examples import shell_mesh
from cardimech.fem_mesh import load_mesh


def diagnose(output):
    output.mkdir(parents=True, exist_ok=True)
    payload = shell_mesh(3, layers=6)
    ref = write_json_artifact(
        output,
        filename="mesh.json",
        artifact_id="mesh",
        kind="tetra_mechanics_mesh",
        payload=payload,
    )
    mesh = load_mesh(ref, payload["subject_id"])
    dofs = (mesh.elements[:, :, None] * 3 + np.arange(3)).reshape(-1, 12)
    ne = len(dofs)
    rows = np.broadcast_to(dofs[:, :, None], (ne, 12, 12)).ravel()
    cols = np.broadcast_to(dofs[:, None, :], (ne, 12, 12)).ravel()
    # Independent reference surface load with outward-from-cavity orientation.
    force = np.zeros_like(mesh.nodes)
    x = mesh.nodes[mesh.cavity]
    area = np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]) / 2
    pressure, mu, a, b = 0.1, 1000.0, 0.02, 0.03
    np.add.at(force, mesh.cavity.ravel(), np.repeat(pressure * area / 3, 3, axis=0))
    fixed = np.concatenate(
        [3 * np.flatnonzero(abs(mesh.nodes[:, axis]) < 1e-14) + axis for axis in range(3)]
    )
    free = np.setdiff1d(np.arange(mesh.nodes.size), fixed)
    inner = np.unique(mesh.cavity)
    I = np.eye(3)
    results = []
    for lam in (1000.0, 1e5, 1e7):
        C = mu * (
            np.einsum("ik,jl->ijkl", I, I) + np.einsum("il,jk->ijkl", I, I)
        ) + lam * np.einsum("ij,kl->ijkl", I, I)
        ke = np.einsum(
            "e,eaj,ijkl,ebl->eaibk", mesh.volumes, mesh.gradients, C, mesh.gradients
        ).reshape(ne, 12, 12)
        K = coo_matrix((ke.ravel(), (rows, cols)), shape=(mesh.nodes.size, mesh.nodes.size)).tocsr()
        u = np.zeros(mesh.nodes.size)
        u[free] = spsolve(K[free][:, free], force.ravel()[free])
        u = u.reshape(-1, 3)
        B = pressure / (4 * mu * (1 / a**3 - 1 / b**3))
        A = 4 * mu * B / ((3 * lam + 2 * mu) * b**3)
        exact = A * a + B / a**2
        observed = float(np.einsum("ij,ij->i", u[inner], mesh.nodes[inner] / a).mean())
        grad = np.einsum("eai,eaj->eij", u[mesh.elements], mesh.gradients)
        eps = (grad + grad.transpose(0, 2, 1)) / 2
        div = np.trace(grad, axis1=1, axis2=2)
        record = {
            "lambda_Pa": lam,
            "observed_m": observed,
            "exact_m": exact,
            "relative_error": abs(observed - exact) / exact,
            "residual_N": float(np.linalg.norm((K @ u.ravel() - force.ravel())[free])),
            "volumetric_energy_J": float(0.5 * lam * (mesh.volumes @ div**2)),
            "shear_energy_J": float(mu * np.einsum("e,eij,eij->", mesh.volumes, eps, eps)),
            "nodes": len(mesh.nodes),
            "tetrahedra": ne,
        }
        results.append(record)
        print(json.dumps(record), flush=True)
    (output / "linear-diagnosis.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    diagnose(p.parse_args().output)
