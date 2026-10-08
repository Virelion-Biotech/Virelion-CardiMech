"""CPU P1 tetrahedral finite-strain equilibrium with follower cavity pressure."""

from __future__ import annotations

import importlib.util
import warnings

import numpy as np

from .artifacts import write_json_artifact
from .fem_materials import constitutive
from .fem_mesh import cavity_volume, load_mesh
from .models import MechanicsQC, MechanicsSimulationResult
from .serialization import finite_number, integer


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def pressure_load(nodes, faces, pressure):
    force = np.zeros_like(nodes)
    rows = []
    cols = []
    values = []
    for face in faces:
        a, b, c = nodes[face]
        area = np.cross(b - a, c - a) / 2
        derivatives = [skew(c - b) / 2, -skew(c - a) / 2, skew(b - a) / 2]
        for target in face:
            force[target] += pressure * area / 3
            for source, derivative in zip(face, derivatives, strict=True):
                for i in range(3):
                    for j in range(3):
                        rows.append(3 * target + i)
                        cols.append(3 * source + j)
                        values.append(pressure * derivative[i, j] / 3)
    from scipy.sparse import coo_matrix

    return force.ravel(), coo_matrix((values, (rows, cols)), shape=(nodes.size, nodes.size)).tocsr()


def assemble(mesh, u, model, parameters, tension, pressure):
    from scipy.sparse import coo_matrix

    F = np.eye(3) + np.einsum("eai,eaj->eij", u[mesh.elements], mesh.gradients)
    energy, P, C, J = constitutive(F, model, parameters, mesh.fiber, mesh.sheet, tension)
    local = np.einsum("e,eij,eaj->eai", mesh.volumes, P, mesh.gradients)
    residual = np.zeros_like(u)
    np.add.at(residual, mesh.elements, local)
    K = np.einsum("e,eaj,eijkl,ebl->eaibk", mesh.volumes, mesh.gradients, C, mesh.gradients)
    dofs = (mesh.elements[:, :, None] * 3 + np.arange(3)).reshape(len(mesh.elements), 12)
    rows = np.broadcast_to(dofs[:, :, None], (len(dofs), 12, 12)).ravel()
    cols = np.broadcast_to(dofs[:, None, :], (len(dofs), 12, 12)).ravel()
    tangent = coo_matrix((K.reshape(-1), (rows, cols)), shape=(u.size, u.size)).tocsr()
    external, load_tangent = pressure_load(mesh.nodes + u, mesh.cavity, pressure)
    return residual.ravel() - external, tangent - load_tangent, (F, energy, P, J)


class TetrahedralBackend:
    name = "scipy-tetra-v1"

    def available(self):
        return importlib.util.find_spec("scipy") is not None

    def describe(self):
        return {
            "name": self.name,
            "spatial": True,
            "available": self.available(),
            "description": "CPU finite-strain P1 tetrahedral static equilibrium; numerical research backend",
            "patient_validated": False,
        }

    def simulate(self, request):
        from scipy.sparse.linalg import MatrixRankWarning, spsolve

        if (
            request.activation_ref
            or request.observations
            or request.circulation.model != "none"
            or request.circulation.enabled
            or request.circulation.parameters
            or request.circulation.state_ref
        ):
            raise ValueError(
                "Spatial static solver does not consume activation artifacts, observations or circulation"
            )
        mesh = load_mesh(request.anatomy_ref, request.subject_id)
        model = request.parameters.material_model
        keys = {"neo_hookean": {"mu", "kappa"}, "guccione": {"c", "bf", "bt", "bfs", "kappa"}}
        if model not in keys or set(request.parameters.passive) != keys[model]:
            raise ValueError(
                "Spatial material requires exact neo_hookean mu/kappa or guccione c/bf/bt/bfs/kappa parameters"
            )
        parameters = {
            k: finite_number(v, k, strictly_positive=True)
            for k, v in request.parameters.passive.items()
        }
        for key in keys[model] - {"bf", "bt", "bfs"}:
            if request.parameters.units.get(key) != "Pa":
                raise ValueError(f"Spatial material {key} requires explicit Pa units")
        for key in {"bf", "bt", "bfs"} & keys[model]:
            if request.parameters.units.get(key) not in {None, "1"}:
                raise ValueError("Guccione exponential coefficients are dimensionless")
        if request.parameters.active_model != "constant_fiber" or set(request.parameters.active) - {
            "tension"
        }:
            raise ValueError("Spatial active model must be constant_fiber with optional tension")
        tension = finite_number(request.parameters.active.get("tension", 0), "tension", minimum=0)
        if tension and request.parameters.units.get("tension") != "Pa":
            raise ValueError("Active tension requires Pa units")
        if (tension or model == "guccione") and not mesh.directions_declared:
            raise ValueError(
                "Anisotropic/active mechanics requires explicit element fibre and sheet directions"
            )
        allowed = {
            "output_dir",
            "load_steps",
            "max_iterations",
            "absolute_tolerance_N",
            "relative_tolerance",
        }
        if set(request.settings) - allowed or "output_dir" not in request.settings:
            raise ValueError(
                "Spatial settings require output_dir and only supported Newton controls"
            )
        steps = integer(request.settings.get("load_steps", 5), "load_steps", minimum=1)
        iterations = integer(
            request.settings.get("max_iterations", 30), "max_iterations", minimum=1
        )
        if steps > 100 or iterations > 100:
            raise ValueError("Spatial solve limited to 100 steps and 100 iterations per step")
        atol = finite_number(
            request.settings.get("absolute_tolerance_N", 1e-9),
            "absolute_tolerance_N",
            strictly_positive=True,
        )
        rtol = finite_number(
            request.settings.get("relative_tolerance", 1e-8),
            "relative_tolerance",
            strictly_positive=True,
        )
        if rtol > 1e-3 or atol > 1e-3:
            raise ValueError("Equilibrium tolerances must be <=1e-3")
        fixed = {}
        pressure = 0.0
        pressure_seen = False
        for bc in request.boundary_conditions:
            if bc.waveform_ref:
                raise ValueError("Spatial static solver requires scalar boundary conditions")
            if bc.kind == "pressure":
                if (
                    bc.region != "cavity"
                    or bc.unit != "Pa"
                    or bc.metadata
                    or pressure_seen
                    or not len(mesh.cavity)
                ):
                    raise ValueError(
                        "One pressure boundary on closed cavity with Pa units is supported"
                    )
                pressure = finite_number(bc.value, "pressure", minimum=0)
                pressure_seen = True
            elif bc.kind == "fixed":
                if (
                    bc.value is not None
                    or bc.unit != "m"
                    or set(bc.metadata) - {"nodes", "components", "displacement"}
                ):
                    raise ValueError(
                        "Fixed boundaries use explicit nodes/components/displacement metadata and m units"
                    )
                nodes = bc.metadata.get("nodes")
                components = bc.metadata.get("components", [0, 1, 2])
                displacement = bc.metadata.get("displacement", [0.0, 0.0, 0.0])
                if (
                    not isinstance(nodes, list)
                    or not nodes
                    or not isinstance(components, list)
                    or not components
                    or len(displacement) != 3
                ):
                    raise ValueError(
                        "Fixed boundary requires nodes, components and three displacement values"
                    )
                displacement = [finite_number(v, "displacement") for v in displacement]
                for node in nodes:
                    node = integer(node, "node", minimum=0)
                    if node >= len(mesh.nodes):
                        raise ValueError("Fixed node index out of range")
                    for component in components:
                        component = integer(component, "component", minimum=0)
                        if component > 2:
                            raise ValueError("Fixed component out of range")
                        dof = 3 * node + component
                        if dof in fixed and fixed[dof] != displacement[component]:
                            raise ValueError("Conflicting prescribed displacements")
                        fixed[dof] = displacement[component]
            else:
                raise ValueError(
                    "Spatial solver supports only fixed displacement and cavity pressure boundaries"
                )
        rigid = np.zeros((mesh.nodes.size, 6))
        for n, x in enumerate(mesh.nodes - mesh.nodes.mean(axis=0)):
            rigid[3 * n : 3 * n + 3, :3] = np.eye(3)
            rigid[3 * n : 3 * n + 3, 3:] = -skew(x)
        indices = np.array(sorted(fixed), dtype=int)
        if len(indices) < 6 or np.linalg.matrix_rank(rigid[indices]) != 6:
            raise ValueError("Displacement constraints must remove all six rigid body modes")
        free = np.setdiff1d(np.arange(mesh.nodes.size), indices)
        u = np.zeros_like(mesh.nodes)
        history = []
        count = 0
        for step in range(1, steps + 1):
            fraction = step / steps
            u.ravel()[indices] = np.array([fixed[i] for i in indices]) * fraction
            initial = None
            for iteration in range(iterations + 1):
                residual, K, fields = assemble(
                    mesh, u, model, parameters, tension * fraction, pressure * fraction
                )
                norm = float(np.linalg.norm(residual[free]))
                if initial is None:
                    initial = norm
                tolerance = atol + rtol * initial
                if norm <= tolerance:
                    history.append(
                        {
                            "load_fraction": fraction,
                            "iterations": iteration,
                            "residual_N": norm,
                            "tolerance_N": tolerance,
                        }
                    )
                    break
                if iteration == iterations:
                    raise ValueError("Spatial Newton equilibrium did not converge")
                with warnings.catch_warnings():
                    warnings.simplefilter("error", MatrixRankWarning)
                    increment = spsolve(K[free][:, free], -residual[free])
                if not np.isfinite(increment).all():
                    raise ValueError("Nonfinite Newton increment")
                accepted = False
                for power in range(16):
                    trial = u.copy()
                    trial.ravel()[free] += increment * 2.0 ** (-power)
                    try:
                        trial_residual, _, _ = assemble(
                            mesh, trial, model, parameters, tension * fraction, pressure * fraction
                        )
                    except ValueError:
                        continue
                    if np.linalg.norm(trial_residual[free]) < norm * (1 - 1e-4 * 2.0 ** (-power)):
                        u = trial
                        accepted = True
                        break
                if not accepted:
                    raise ValueError("Spatial Newton line search failed")
                count += 1
        F, energy, P, J = fields
        cauchy = np.einsum("eij,ekj->eik", P, F) / J[:, None, None]
        scalars = {
            "cavity_volume_m3": cavity_volume(mesh.nodes + u, mesh.cavity),
            "reference_cavity_volume_m3": cavity_volume(mesh.nodes, mesh.cavity),
            "strain_energy_J": float(mesh.volumes @ energy),
            "residual_N": norm,
            "minimum_J": float(J.min()),
            "maximum_displacement_m": float(np.linalg.norm(u, axis=1).max()),
        }
        notice = "Numerical research backend; no patient myocardial validation. P1 displacement elements can lock near incompressibility."
        payload = {
            "subject_id": request.subject_id,
            "units": {"coordinates": "m", "displacement": "m", "stress": "Pa"},
            "reference_nodes": mesh.nodes.tolist(),
            "tetrahedra": mesh.elements.tolist(),
            "displacement": u.tolist(),
            "deformation_gradient": F.tolist(),
            "green_strain": (0.5 * (F.transpose(0, 2, 1) @ F - np.eye(3))).tolist(),
            "cauchy_stress": cauchy.tolist(),
            "element_J": J.tolist(),
            "reaction_force_N": residual.reshape(-1, 3).tolist(),
            "load_history": history,
            "anatomy_sha256": mesh.sha256,
            "validation_status": "software_checked",
            "warnings": [notice],
        }
        artifact = write_json_artifact(
            request.settings["output_dir"],
            filename="spatial-mechanics.json",
            artifact_id=f"{request.subject_id}:spatial-mechanics",
            kind="spatial_mechanics_fields",
            payload=payload,
        )
        return MechanicsSimulationResult(
            subject_id=request.subject_id,
            backend=self.name,
            parameters=request.parameters,
            outputs=[artifact],
            scalar_outputs=scalars,
            validation_status="software_checked",
            warnings=[notice],
            qc=MechanicsQC(
                passed=True,
                converged=True,
                checks={
                    "equilibrium": norm <= tolerance,
                    "positive_J": bool(np.all(J > 0)),
                    "finite_fields": bool(np.isfinite(cauchy).all()),
                },
                metrics={"residual_N": norm, "newton_iterations": float(count)},
            ),
            provenance={
                "anatomy_sha256": mesh.sha256,
                "solver": "total-Lagrangian P1 tetrahedral Newton",
                "material_model": model,
                "patient_validated": False,
            },
        )
