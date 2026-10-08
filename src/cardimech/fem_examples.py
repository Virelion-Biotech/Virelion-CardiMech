"""Synthetic meshes for numerical verification, never patient anatomy."""

from itertools import product

import numpy as np
from scipy.spatial import ConvexHull


def shell_mesh(level=1, layers=3, inner=0.02, outer=0.03):
    directions = [
        np.array(v, dtype=float)
        for v in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    ]
    faces = ConvexHull(directions).simplices.tolist()
    for _ in range(level):
        edges = {}
        refined = []

        def midpoint(a, b, edges=edges):
            key = tuple(sorted((a, b)))
            if key not in edges:
                v = directions[a] + directions[b]
                v /= np.linalg.norm(v)
                edges[key] = len(directions)
                directions.append(v)
            return edges[key]

        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            refined.extend([[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]])
        faces = refined
    for face in faces:
        a, b, c = np.array(directions)[face]
        if np.dot(np.cross(b - a, c - a), a) < 0:
            face[1], face[2] = face[2], face[1]
    count = len(directions)
    nodes = np.concatenate(
        [np.array(directions) * radius for radius in np.linspace(inner, outer, layers)]
    )
    elements = []
    for layer in range(layers - 1):
        for face in faces:
            a, b, c = np.array(sorted(face)) + layer * count
            A, B, C = np.array([a, b, c]) + count
            elements.extend([[a, b, c, C], [a, b, B, C], [a, A, B, C]])
    for tet in elements:
        if np.linalg.det((nodes[tet[1:]] - nodes[tet[0]]).T) < 0:
            tet[1], tet[2] = tet[2], tet[1]
    return {
        "subject_id": "synthetic-shell",
        "units": "m",
        "nodes": nodes.tolist(),
        "tetrahedra": np.array(elements).tolist(),
        "cavity_faces": faces,
    }


def cube_mesh():
    from itertools import permutations

    nodes = np.array(list(product(range(3), repeat=3)), dtype=float) * 0.01
    index = {tuple(x): i for i, x in enumerate(product(range(3), repeat=3))}
    elements = []
    for origin in product(range(2), repeat=3):
        for order in permutations(range(3)):
            point = np.array(origin)
            tet = [index[tuple(point)]]
            for axis in order:
                point = point.copy()
                point[axis] += 1
                tet.append(index[tuple(point)])
            if np.linalg.det((nodes[tet[1:]] - nodes[tet[0]]).T) < 0:
                tet[1], tet[2] = tet[2], tet[1]
            elements.append(tet)
    return {
        "subject_id": "synthetic-cube",
        "units": "m",
        "nodes": nodes.tolist(),
        "tetrahedra": elements,
        "cavity_faces": [],
    }
