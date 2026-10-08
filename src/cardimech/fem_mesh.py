"""Strict tetrahedral reference-mesh contract and canonical cavity orientation."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .handoff import local_path_from_uri
from .serialization import strict_loads


@dataclass
class TetMesh:
    nodes: np.ndarray
    elements: np.ndarray
    cavity: np.ndarray
    gradients: np.ndarray
    volumes: np.ndarray
    fiber: np.ndarray
    sheet: np.ndarray
    sha256: str
    directions_declared: bool


def _indices(raw, width, n, name):
    values = np.asarray(raw)
    if values.ndim != 2 or values.shape[1] != width or values.dtype.kind not in "iu":
        raise ValueError(f"{name} must be integer index rows of width {width}")
    if np.any(values < 0) or np.any(values >= n) or any(len(set(row)) != width for row in values):
        raise ValueError(f"{name} contain invalid/repeated node indices")
    if len({tuple(sorted(row)) for row in values}) != len(values):
        raise ValueError(f"{name} contain duplicate cells/faces")
    return values.astype(int)


def load_mesh(ref, subject):
    if ref.kind != "tetra_mechanics_mesh":
        raise ValueError("Spatial FEM requires tetra_mechanics_mesh anatomy artifact")
    raw = local_path_from_uri(ref.uri).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if ref.sha256 is not None and digest != ref.sha256.lower():
        raise ValueError("Mechanics mesh SHA-256 mismatch")
    payload = strict_loads(raw)
    required = {"subject_id", "units", "nodes", "tetrahedra", "cavity_faces"}
    if (
        not isinstance(payload, dict)
        or not required <= payload.keys()
        or set(payload) - required - {"fiber", "sheet"}
    ):
        raise ValueError(
            "Mesh requires subject_id, units, nodes, tetrahedra and cavity_faces; fiber/sheet optional"
        )
    if payload["subject_id"] != subject or ref.metadata.get("subject_id", subject) != subject:
        raise ValueError("Mechanics mesh belongs to a different subject")
    if payload["units"] != "m":
        raise ValueError("Spatial mechanics mesh coordinates must use m")
    coordinate_values = np.asarray(payload["nodes"])
    if coordinate_values.dtype.kind not in "ifu":
        raise ValueError("Mesh coordinates must be numeric, not strings or booleans")
    nodes = np.asarray(payload["nodes"], dtype=float)
    if (
        nodes.ndim != 2
        or nodes.shape[1] != 3
        or not 4 <= len(nodes) <= 5000
        or not np.isfinite(nodes).all()
    ):
        raise ValueError("Mesh requires 4..5000 finite 3D nodes")
    elements = _indices(payload["tetrahedra"], 4, len(nodes), "tetrahedra")
    if not 1 <= len(elements) <= 20000:
        raise ValueError("Mesh requires 1..20000 tetrahedra")
    if set(elements.ravel()) != set(range(len(nodes))):
        raise ValueError("Every mesh node must belong to a tetrahedron")
    D = (nodes[elements[:, 1:]] - nodes[elements[:, 0, None]]).transpose(0, 2, 1)
    determinants = np.linalg.det(D)
    scale = np.max(np.linalg.norm(D, axis=1), axis=1) ** 3
    if np.any(determinants <= 1e-12 * scale) or not np.isfinite(determinants).all():
        raise ValueError("Tetrahedra must have positive orientation and nondegenerate volume")
    gradients = np.empty((len(elements), 4, 3))
    gradients[:, 1:] = np.linalg.inv(D)
    gradients[:, 0] = -np.sum(gradients[:, 1:], axis=1)
    face_owners = defaultdict(list)
    for e, tet in enumerate(elements):
        for omit in range(4):
            face_owners[tuple(sorted(np.delete(tet, omit)))].append(e)
    if any(len(owners) > 2 for owners in face_owners.values()):
        raise ValueError("Nonmanifold tetrahedral mesh")
    for face, owners in face_owners.items():
        if len(owners) == 2:
            x = nodes[list(face)]
            normal = np.cross(x[1] - x[0], x[2] - x[0])
            side = [np.dot(normal, np.mean(nodes[elements[e]], axis=0) - x[0]) for e in owners]
            if side[0] * side[1] >= 0:
                raise ValueError("Adjacent tetrahedra overlap across a shared face")
    raw_faces = payload["cavity_faces"]
    if not isinstance(raw_faces, list):
        raise TypeError("cavity_faces must be a list")
    cavity = (
        _indices(raw_faces, 3, len(nodes), "cavity_faces")
        if raw_faces
        else np.empty((0, 3), dtype=int)
    )
    neighbors = defaultdict(set)
    for owners in face_owners.values():
        if len(owners) == 2:
            a, b = owners
            neighbors[a].add(b)
            neighbors[b].add(a)
    visited = {0}
    pending = [0]
    while pending:
        for neighbor in neighbors[pending.pop()] - visited:
            visited.add(neighbor)
            pending.append(neighbor)
    if len(visited) != len(elements):
        raise ValueError("Tetrahedral mesh must be face-connected")
    directed = Counter()
    for face in cavity:
        owners = face_owners.get(tuple(sorted(face)), [])
        if len(owners) != 1:
            raise ValueError("Cavity faces must be actual exterior tetrahedral faces")
        x = nodes[face]
        area = np.cross(x[1] - x[0], x[2] - x[0]) / 2
        toward_wall = np.mean(nodes[elements[owners[0]]], axis=0) - np.mean(x, axis=0)
        if np.dot(area, toward_wall) <= 0:
            raise ValueError("Cavity faces must point outward from cavity into myocardium")
        for a, b in zip(face, np.roll(face, -1), strict=True):
            directed[(int(a), int(b))] += 1
    if any(count != 1 or directed[(b, a)] != 1 for (a, b), count in directed.items()):
        raise ValueError("Cavity must be a closed, consistently oriented surface")
    if len(cavity) and cavity_volume(nodes, cavity) <= 0:
        raise ValueError("Cavity enclosed volume must be positive")
    directions = []
    for name, default in [("fiber", [1, 0, 0]), ("sheet", [0, 1, 0])]:
        array = np.asarray(payload.get(name, [default] * len(elements)), dtype=float)
        if array.shape != (len(elements), 3) or not np.isfinite(array).all():
            raise ValueError(f"{name} must have one finite direction per element")
        norms = np.linalg.norm(array, axis=1)
        if np.any(norms <= 0):
            raise ValueError(f"{name} directions must be nonzero")
        directions.append(array / norms[:, None])
    if np.any(abs(np.sum(directions[0] * directions[1], axis=1)) > 1e-10):
        raise ValueError("Fibre and sheet directions must be orthogonal")
    return TetMesh(
        nodes,
        elements,
        cavity,
        gradients,
        determinants / 6,
        *directions,
        digest,
        "fiber" in payload and "sheet" in payload,
    )


def cavity_volume(nodes, faces):
    if not len(faces):
        return 0.0
    x = nodes[faces]
    return float(np.sum(np.einsum("fi,fi->f", x[:, 0], np.cross(x[:, 1], x[:, 2]))) / 6)
