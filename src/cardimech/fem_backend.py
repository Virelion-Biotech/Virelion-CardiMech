"""CPU P1 tetrahedral finite-strain equilibrium with follower cavity pressure."""

from __future__ import annotations

import importlib.util
import warnings

import numpy as np

from .artifacts import write_json_artifact
from .fem_materials import constitutive, linearized_regime
from .fem_mesh import cavity_volume, load_mesh
from .models import MechanicsQC, MechanicsSimulationResult
from .serialization import finite_number, integer


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


ASSEMBLY_CHUNK_SIZE = 512


def pressure_load(nodes, faces, pressure, *, compute_tangent=True):
    from scipy.sparse import coo_matrix, csr_matrix

    force = np.zeros_like(nodes)
    tangent = csr_matrix((nodes.size, nodes.size)) if compute_tangent else None
    for start in range(0, len(faces), ASSEMBLY_CHUNK_SIZE):
        selected = faces[start : start + ASSEMBLY_CHUNK_SIZE]
        x = nodes[selected]
        a, b, c = x[:, 0], x[:, 1], x[:, 2]
        area = np.cross(b - a, c - a) / 2
        np.add.at(force, selected.ravel(), np.repeat(pressure * area / 3, 3, axis=0))
        if not compute_tangent:
            continue
        derivatives = np.asarray(
            [[skew(q - r) / 2, -skew(q - p) / 2, skew(r - p) / 2] for p, r, q in x]
        )
        # Each of the three target vertices receives one third of pressure*area.
        local = np.broadcast_to(
            pressure * derivatives[:, None] / 3, (len(selected), 3, 3, 3, 3)
        ).transpose(0, 1, 3, 2, 4)
        dofs = (selected[:, :, None] * 3 + np.arange(3)).reshape(len(selected), 9)
        rows = np.broadcast_to(dofs[:, :, None], (len(selected), 9, 9)).ravel()
        cols = np.broadcast_to(dofs[:, None, :], (len(selected), 9, 9)).ravel()
        tangent += coo_matrix(
            (local.reshape(-1), (rows, cols)), shape=(nodes.size, nodes.size)
        ).tocsr()
    return force.ravel(), tangent


def assemble(
    mesh,
    u,
    model,
    parameters,
    tension,
    pressure,
    *,
    chunk_size=ASSEMBLY_CHUNK_SIZE,
    compute_tangent=True,
):
    """Bound constitutive and element-tangent temporaries; preserve all output fields."""
    from scipy.sparse import coo_matrix, csr_matrix

    if type(chunk_size) is not int or not 1 <= chunk_size <= 4096:
        raise ValueError("Assembly chunk_size must be an integer in 1..4096")
    ne = len(mesh.elements)
    F = np.empty((ne, 3, 3))
    energy, J = np.empty(ne), np.empty(ne)
    P = np.empty((ne, 3, 3))
    residual = np.zeros_like(u)
    tangent = csr_matrix((u.size, u.size)) if compute_tangent else None
    for start in range(0, ne, chunk_size):
        stop = min(start + chunk_size, ne)
        selected = mesh.elements[start:stop]
        gradients, volumes = mesh.gradients[start:stop], mesh.volumes[start:stop]
        f = np.eye(3) + np.einsum("eai,eaj->eij", u[selected], gradients)
        e, p, c, j = constitutive(
            f,
            model,
            parameters,
            mesh.fiber[start:stop],
            mesh.sheet[start:stop],
            tension,
            compute_tangent=compute_tangent,
        )
        F[start:stop], energy[start:stop], P[start:stop], J[start:stop] = f, e, p, j
        local = np.einsum("e,eij,eaj->eai", volumes, p, gradients)
        np.add.at(residual, selected, local)
        if compute_tangent:
            local_k = np.einsum("e,eaj,eijkl,ebl->eaibk", volumes, gradients, c, gradients)
            dofs = (selected[:, :, None] * 3 + np.arange(3)).reshape(len(selected), 12)
            rows = np.broadcast_to(dofs[:, :, None], (len(selected), 12, 12)).ravel()
            cols = np.broadcast_to(dofs[:, None, :], (len(selected), 12, 12)).ravel()
            tangent += coo_matrix(
                (local_k.reshape(-1), (rows, cols)), shape=(u.size, u.size)
            ).tocsr()
    external, load_tangent = pressure_load(
        mesh.nodes + u, mesh.cavity, pressure, compute_tangent=compute_tangent
    )
    return (
        residual.ravel() - external,
        tangent - load_tangent if compute_tangent else None,
        (F, energy, P, J),
    )


class TetrahedralBackend:
    name = "scipy-tetra-v1"

    def available(self):
        return importlib.util.find_spec("scipy") is not None

    def describe(self):
        return {
            "name": self.name,
            "spatial": True,
            "available": self.available(),
            "description": "CPU finite-strain tetrahedral static equilibrium; P1 displacement or mixed MINI research formulation",
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
        formulation = request.settings.get("formulation", "auto")
        if formulation not in {"auto", "displacement_p1", "mixed_mini"}:
            raise ValueError("formulation must be auto, displacement_p1 or mixed_mini")
        mixed = formulation == "mixed_mini"
        linear_solver = request.settings.get("linear_solver", "direct")
        profile = request.settings.get("mesh_profile", "standard")
        preconditioner = request.settings.get("linear_preconditioner", "jacobi")
        if preconditioner not in {"jacobi", "block_jacobi"}:
            raise ValueError("linear_preconditioner must be jacobi or block_jacobi")
        if linear_solver not in {"direct", "matrix_free_gmres"}:
            raise ValueError("linear_solver must be direct or matrix_free_gmres")
        if profile == "large_experimental" and linear_solver != "matrix_free_gmres":
            raise ValueError("large_experimental mesh profile requires matrix_free_gmres")
        linear_rtol = finite_number(
            request.settings.get("linear_relative_tolerance", 1e-8),
            "linear_relative_tolerance",
            strictly_positive=True,
        )
        if not 1e-12 <= linear_rtol <= 1e-4:
            raise ValueError("linear_relative_tolerance must be in 1e-12..1e-4")
        linear_limit = integer(
            request.settings.get("linear_max_iterations", 1200), "linear_max_iterations", minimum=1
        )
        linear_restart = integer(
            request.settings.get("linear_restart", 60), "linear_restart", minimum=1
        )
        if linear_limit > 5000 or linear_restart > 100:
            raise ValueError("Linear iteration/restart limits are 5000/100")
        if linear_solver == "direct" and any(
            key.startswith("linear_") and key != "linear_solver" for key in request.settings
        ):
            raise ValueError("GMRES controls require matrix_free_gmres")
        if mixed and (linear_solver != "direct" or profile != "standard"):
            raise ValueError(
                "mixed_mini currently requires direct solver and standard mesh profile"
            )
        mesh = load_mesh(request.anatomy_ref, request.subject_id, profile=profile)
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
        if formulation == "auto":
            if linearized_regime(model, parameters)["p1_locking_risk"]:
                if linear_solver != "direct" or profile != "standard":
                    raise ValueError(
                        "Nearly incompressible default requires mixed_mini with direct solver and standard profile; explicit displacement_p1 is available only for legacy research comparisons"
                    )
                formulation, mixed = "mixed_mini", True
            else:
                formulation, mixed = "displacement_p1", False
        if not mixed and "mixed_quadrature_order" in request.settings:
            raise ValueError("mixed_quadrature_order requires mixed_mini")
        allowed = {
            "output_dir",
            "load_steps",
            "max_iterations",
            "absolute_tolerance_N",
            "relative_tolerance",
            "linear_solver",
            "mesh_profile",
            "linear_relative_tolerance",
            "linear_max_iterations",
            "linear_restart",
            "linear_preconditioner",
            "formulation",
            "mixed_quadrature_order",
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
        linear_history = []
        mixed_extra = {}
        if mixed:
            from .fem_mixed import solve_mini

            u, residual, fields, history, count, mixed_extra = solve_mini(
                mesh,
                model,
                parameters,
                tension,
                pressure,
                indices,
                fixed,
                steps=steps,
                iterations=iterations,
                atol=atol,
                rtol=rtol,
                order=request.settings.get("mixed_quadrature_order", 5),
            )
            norm = history[-1]["residual_N"]
            tolerance = history[-1]["tolerance_N"]
        else:
            for step in range(1, steps + 1):
                fraction = step / steps
                u.ravel()[indices] = np.array([fixed[i] for i in indices]) * fraction
                initial = None
                for iteration in range(iterations + 1):
                    residual, K, fields = assemble(
                        mesh,
                        u,
                        model,
                        parameters,
                        tension * fraction,
                        pressure * fraction,
                        compute_tangent=linear_solver == "direct",
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
                    if linear_solver == "direct":
                        with warnings.catch_warnings():
                            warnings.simplefilter("error", MatrixRankWarning)
                            increment = spsolve(K[free][:, free], -residual[free])
                    else:
                        from .fem_krylov import solve_increment

                        increment, linear_report = solve_increment(
                            mesh,
                            u,
                            model,
                            parameters,
                            tension * fraction,
                            pressure * fraction,
                            free,
                            -residual[free],
                            relative_tolerance=linear_rtol,
                            max_iterations=linear_limit,
                            restart=linear_restart,
                            preconditioner=preconditioner,
                        )
                        linear_history.append(
                            {"load_step": step, "newton_iteration": iteration, **linear_report}
                        )
                    if not np.isfinite(increment).all():
                        raise ValueError("Nonfinite Newton increment")
                    accepted = False
                    for power in range(16):
                        trial = u.copy()
                        trial.ravel()[free] += increment * 2.0 ** (-power)
                        try:
                            trial_residual, _, _ = assemble(
                                mesh,
                                trial,
                                model,
                                parameters,
                                tension * fraction,
                                pressure * fraction,
                                compute_tangent=False,
                            )
                        except ValueError:
                            continue
                        if np.linalg.norm(trial_residual[free]) < norm * (
                            1 - 1e-4 * 2.0 ** (-power)
                        ):
                            u = trial
                            accepted = True
                            break
                    if not accepted:
                        raise ValueError("Spatial Newton line search failed")
                    count += 1
        F, energy, P, J = fields
        cauchy = P if mixed else np.einsum("eij,ekj->eik", P, F) / J[:, None, None]
        scalars = {
            "cavity_volume_m3": cavity_volume(mesh.nodes + u, mesh.cavity),
            "reference_cavity_volume_m3": cavity_volume(mesh.nodes, mesh.cavity),
            "strain_energy_J": float(mesh.volumes @ energy),
            "residual_N": norm,
            "minimum_J": mixed_extra.get("minimum_quadrature_J", float(J.min())),
            "maximum_displacement_m": float(np.linalg.norm(u, axis=1).max()),
        }
        regime = linearized_regime(model, parameters)
        notice = "Numerical research backend; no patient myocardial validation. P1 displacement elements can lock near incompressibility."
        if mixed:
            notice = "Numerical research mixed MINI backend; no patient myocardial validation. Mesh, quadrature and material accuracy require independent qualification."
        notices = [notice]
        if regime["p1_locking_risk"] and not mixed:
            notices.append(
                "Nearly incompressible material: displacement P1 stiffness/accuracy requires separate qualification; equilibrium convergence does not establish accuracy."
            )
        payload = {
            "formulation": formulation,
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
            "linear_history": linear_history,
            "anatomy_sha256": mesh.sha256,
            "validation_status": "software_checked",
            "warnings": notices,
            **mixed_extra,
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
            warnings=notices,
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
                "formulation": formulation,
                "requested_formulation": request.settings.get("formulation", "auto"),
                "mixed_quadrature_order": mixed_extra.get("mixed_quadrature_order"),
                "linearized_material_regime": regime,
                "patient_validated": False,
                "assembly_element_chunk_size": ASSEMBLY_CHUNK_SIZE,
                "line_search_assembles_tangent": False,
                "linear_solver": linear_solver,
                "linear_preconditioner": preconditioner if linear_solver != "direct" else None,
                "mesh_profile": profile,
                "large_mesh_qualification": "experimental synthetic evidence only"
                if profile == "large_experimental"
                else "standard research limits",
                "linear_iterations": sum(x["iterations"] for x in linear_history),
            },
        )
