from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import exp, log

import numpy as np


@dataclass(frozen=True)
class MaterialDescriptor:
    name: str
    anisotropy: str
    compressibility: str
    parameters: tuple[str, ...]
    role: str


def _vec3(values: Iterable[float], name: str) -> np.ndarray:
    out = np.asarray(tuple(values), dtype=float)
    if out.shape != (3,) or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be a finite 3-vector")
    norm = float(np.linalg.norm(out))
    if norm <= 0.0:
        raise ValueError(f"{name} must be non-zero")
    return out / norm


def _mat3(F: Iterable[Iterable[float]]) -> np.ndarray:
    out = np.asarray(F, dtype=float)
    if out.shape != (3, 3) or not np.all(np.isfinite(out)):
        raise ValueError("F must be a finite 3x3 deformation gradient")
    if np.linalg.det(out) <= 0.0:
        raise ValueError("F must preserve orientation (det(F) > 0)")
    return out


def deformation_invariants(
    F: Iterable[Iterable[float]],
    fiber: Iterable[float] = (1.0, 0.0, 0.0),
    sheet: Iterable[float] = (0.0, 1.0, 0.0),
) -> dict[str, float]:
    Fm = _mat3(F)
    f0, s0 = _vec3(fiber, "fiber"), _vec3(sheet, "sheet")
    C = Fm.T @ Fm
    i1 = float(np.trace(C))
    i2 = float(0.5 * (i1 * i1 - np.trace(C @ C)))
    return {
        "I1": i1,
        "I2": i2,
        "I4f": float(f0 @ C @ f0),
        "I4s": float(s0 @ C @ s0),
        "I8fs": float(f0 @ C @ s0),
        "J": float(np.linalg.det(Fm)),
    }


def green_lagrange_strain(F: Iterable[Iterable[float]]) -> np.ndarray:
    Fm = _mat3(F)
    return 0.5 * (Fm.T @ Fm - np.eye(3))


def principal_green_strains(F: Iterable[Iterable[float]]) -> np.ndarray:
    return np.linalg.eigvalsh(green_lagrange_strain(F))


def neo_hookean_energy(F: Iterable[Iterable[float]], *, mu: float, kappa: float) -> float:
    if mu <= 0 or kappa <= 0:
        raise ValueError("mu and kappa must be positive")
    inv = deformation_invariants(F)
    log_j = log(inv["J"])
    return float(0.5 * mu * (inv["I1"] - 3.0) - mu * log_j + 0.5 * kappa * log_j**2)


def mooney_rivlin_energy(
    F: Iterable[Iterable[float]], *, c10: float, c01: float, kappa: float
) -> float:
    if min(c10, c01, kappa) < 0 or kappa == 0:
        raise ValueError("c10/c01 must be non-negative and kappa positive")
    inv = deformation_invariants(F)
    j = inv["J"]
    j23 = j ** (-2.0 / 3.0)
    i1_bar = j23 * inv["I1"]
    i2_bar = j23**2 * inv["I2"]
    return float(c10 * (i1_bar - 3.0) + c01 * (i2_bar - 3.0) + 0.5 * kappa * (j - 1.0) ** 2)


def guccione_energy(
    F: Iterable[Iterable[float]],
    *,
    c: float,
    bf: float,
    bt: float,
    bfs: float,
    fiber: Iterable[float] = (1.0, 0.0, 0.0),
    sheet: Iterable[float] = (0.0, 1.0, 0.0),
) -> float:
    if min(c, bf, bt, bfs) <= 0:
        raise ValueError("Guccione parameters must be positive")
    f0, s0 = _vec3(fiber, "fiber"), _vec3(sheet, "sheet")
    n0 = np.cross(f0, s0)
    norm = float(np.linalg.norm(n0))
    if norm <= 1e-12:
        raise ValueError("fiber and sheet directions must not be collinear")
    n0 /= norm
    Qm = np.column_stack((f0, s0, n0))
    E = Qm.T @ green_lagrange_strain(F) @ Qm
    q = (
        bf * E[0, 0] ** 2
        + bt * (E[1, 1] ** 2 + E[2, 2] ** 2 + 2.0 * E[1, 2] ** 2)
        + bfs * (2.0 * E[0, 1] ** 2 + 2.0 * E[0, 2] ** 2)
    )
    return float(0.5 * c * (exp(float(q)) - 1.0))


def holzapfel_ogden_energy(
    F: Iterable[Iterable[float]],
    *,
    a: float,
    b: float,
    af: float,
    bf: float,
    as_: float,
    bs: float,
    afs: float,
    bfs: float,
    kappa: float,
    fiber: Iterable[float] = (1.0, 0.0, 0.0),
    sheet: Iterable[float] = (0.0, 1.0, 0.0),
) -> float:
    if min(a, b, af, bf, as_, bs, afs, bfs, kappa) <= 0:
        raise ValueError("Holzapfel-Ogden parameters must be positive")
    inv = deformation_invariants(F, fiber=fiber, sheet=sheet)
    ef = max(inv["I4f"] - 1.0, 0.0)
    es = max(inv["I4s"] - 1.0, 0.0)
    return float(
        a / (2.0 * b) * (exp(b * (inv["I1"] - 3.0)) - 1.0)
        + af / (2.0 * bf) * (exp(bf * ef**2) - 1.0)
        + as_ / (2.0 * bs) * (exp(bs * es**2) - 1.0)
        + afs / (2.0 * bfs) * (exp(bfs * inv["I8fs"] ** 2) - 1.0)
        + 0.5 * kappa * (inv["J"] - 1.0) ** 2
    )


def material_catalog() -> list[MaterialDescriptor]:
    return [
        MaterialDescriptor(
            "neo_hookean",
            "isotropic",
            "penalty-compressible",
            ("mu", "kappa"),
            "baseline hyperelastic verification model",
        ),
        MaterialDescriptor(
            "mooney_rivlin",
            "isotropic",
            "penalty-compressible",
            ("c10", "c01", "kappa"),
            "general soft-tissue verification model",
        ),
        MaterialDescriptor(
            "guccione",
            "orthotropic",
            "backend-dependent",
            ("c", "bf", "bt", "bfs"),
            "myocardial exponential strain-energy family",
        ),
        MaterialDescriptor(
            "holzapfel_ogden",
            "orthotropic",
            "penalty-compressible",
            ("a", "b", "af", "bf", "as", "bs", "afs", "bfs", "kappa"),
            "fiber/sheet-aware myocardial strain-energy family",
        ),
    ]
