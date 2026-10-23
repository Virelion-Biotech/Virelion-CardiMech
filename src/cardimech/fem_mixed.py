"""Finite-strain MINI (P1 + cell bubble / continuous P1 pressure) mechanics.

Exact determinant-pressure Legendre transform of the existing log-volume law:
W(F,p) = W(F;kappa=0) + p*(det F-1) - U_star(p), 0<det F<e.
The quartic tetrahedral bubble vanishes on every face. Its three displacement
DOFs are locally condensed from the Newton system, not discarded from residuals.
Pressure is scaled to a length-like algebraic unknown to balance residual units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fem_materials import constitutive


def tetra_rule(order=5):
    """Positive tensor Gauss/Duffy rule, normalized to sum to one.

    Order 5 exactly integrates degree-6 bubble-gradient products (including the
    Duffy Jacobian); finite-strain rational terms still require quadrature checks.
    """
    from numpy.polynomial.legendre import leggauss

    if type(order) is not int or not 5 <= order <= 7:
        raise ValueError("mixed_quadrature_order must be an integer in 5..7")
    x, w = leggauss(order)
    x, w = (x + 1) / 2, w / 2
    r, s, t = np.meshgrid(x, x, x, indexing="ij")
    wr, ws, wt = np.meshgrid(w, w, w, indexing="ij")
    bary = np.stack(((1 - r) * (1 - s) * (1 - t), r, (1 - r) * s, (1 - r) * (1 - s) * t), axis=-1)
    weights = 6 * wr * ws * wt * (1 - r) ** 2 * (1 - s)
    return bary.reshape(-1, 4), weights.ravel()


def volume_dual(pressure, kappa):
    """Convex branch of U(J)=kappa/2*log(J)^2, 0<J<e.

    U'(J)=kappa*log(J)/J; Lambert W inverts this without penalty subtraction.
    The mixed potential is W0 + p*(J-1) - U_star(p).
    """
    from scipy.special import lambertw

    if not np.isfinite(pressure).all() or np.any(pressure >= kappa / np.e):
        raise ValueError("Mixed pressure outside convex volume-energy branch (p<kappa/e)")
    log_j = -lambertw(-np.asarray(pressure) / kappa).real
    j = np.exp(log_j)
    compliance = j * j / (kappa * (1 - log_j))
    dual = pressure * (j - 1) - 0.5 * kappa * log_j**2
    return j, compliance, dual


def mixed_material(F, pressure, model, parameters, fiber, sheet, tension, tangent=True):
    """Stress/tangent at independent volumetric pressure, same physical law."""
    base = {**parameters, "kappa": 0.0}
    energy, P, C, J = constitutive(F, model, base, fiber, sheet, tension, compute_tangent=tangent)
    if np.any(J >= np.e):
        raise ValueError("Mixed MINI requires convex volume-energy branch 0<J<e")
    inverse_t = np.linalg.inv(F).transpose(0, 2, 1)
    cof = J[:, None, None] * inverse_t
    P += pressure[:, None, None] * cof
    if tangent:
        C += (pressure * J)[:, None, None, None, None] * (
            np.einsum("eij,ekl->eijkl", inverse_t, inverse_t)
            - np.einsum("eil,ekj->eijkl", inverse_t, inverse_t)
        )
    _, _, dual = volume_dual(pressure, parameters["kappa"])
    potential = energy + pressure * (J - 1) - dual
    physical = energy + 0.5 * parameters["kappa"] * np.log(J) ** 2
    return potential, physical, P, C, J, cof


def neo_hookean_displacement_block(vw, gradients, cof, J, pressure, mu, fiber, tension):
    """Contract the exact mixed NH tangent without a quadrature 3x3x3x3 tensor.

    With H_ai = F^-T_ij G_aj, C contracted with G_a and G_b is
    mu*delta_ik*(G_a.G_b) + (mu-pJ)*H_bi*H_ak + pJ*H_ai*H_bk.
    Prescribed second-Piola fibre tension contributes
    T*delta_ik*(G_a.f)*(G_b.f). No material or quadrature approximation is used.
    """
    H = np.einsum("eqij,eqaj->eqai", cof / J[:, :, None, None], gradients)
    gram = mu * np.einsum("eqaj,eqbj->eqab", gradients, gradients)
    along_fiber = np.einsum("eqaj,ej->eqa", gradients, fiber)
    gram += tension * np.einsum("eqa,eqb->eqab", along_fiber, along_fiber)
    pJ = pressure * J
    block = np.einsum("eq,eqab,ik->eaibk", vw, gram, np.eye(3), optimize=True)
    block += np.einsum("eq,eqbi,eqak->eaibk", vw * (mu - pJ), H, H, optimize=True)
    block += np.einsum("eq,eqai,eqbk->eaibk", vw * pJ, H, H, optimize=True)
    if not np.isfinite(block).all():
        raise ValueError("Mixed neo-Hookean tangent exceeds finite numerical range")
    return block.reshape(len(vw), 15, 15)


@dataclass
class MixedAssembly:
    residual: np.ndarray
    bubble_residual: np.ndarray
    tangent: object
    condensed_residual: np.ndarray | None
    recovery: list
    fields: tuple
    constraint: np.ndarray
    minimum_J: float

    def bubble_increment(self, increment):
        out = np.empty_like(self.bubble_residual)
        for start, dofs, inv_bb, rb, kbx in self.recovery:
            n = len(dofs)
            out[start : start + n] = -np.einsum(
                "eij,ej->ei", inv_bb, rb + np.einsum("eij,ej->ei", kbx, increment[dofs])
            )
        return out


class MiniAssembler:
    def __init__(self, mesh, model, parameters, order=5):
        self.mesh, self.model, self.parameters = mesh, model, parameters
        self.bary, self.weights = tetra_rule(order)
        self.pressure_scale = parameters.get("mu", parameters.get("c")) / mesh.volumes.sum() ** (
            1 / 3
        )
        self.dofs = np.concatenate(
            (
                (mesh.elements[:, :, None] * 3 + np.arange(3)).reshape(-1, 12),
                mesh.nodes.size + mesh.elements,
            ),
            axis=1,
        )
        # d(256 L0 L1 L2 L3)/dL_a, no division by a near-zero barycentric value.
        self.bubble_derivatives = np.stack(
            [256 * np.prod(np.delete(self.bary, a, axis=1), axis=1) for a in range(4)], axis=1
        )

    def assemble(self, u, q, bubble, tension, pressure, *, tangent=True):
        from scipy.sparse import coo_matrix, csr_matrix

        from .fem_backend import pressure_load

        mesh, bary, weights = self.mesh, self.bary, self.weights
        nq, ne, nn = len(bary), len(mesh.elements), len(mesh.nodes)
        residual = np.zeros(4 * nn)
        rb_all = np.empty((ne, 3))
        constraint = np.zeros(nn)
        condensed = np.zeros(4 * nn) if tangent else None
        K = csr_matrix((4 * nn, 4 * nn)) if tangent else None
        recovery = []
        fields = [np.empty((ne, 3, 3)), np.empty(ne), np.empty((ne, 3, 3)), np.empty(ne)]
        min_j = np.inf
        green = np.empty((ne, 3, 3))
        # Bounded quadrature temporaries, independent of total mesh size.
        for start in range(0, ne, 32):
            stop = min(start + 32, ne)
            el = mesh.elements[start:stop]
            n = len(el)
            grad = mesh.gradients[start:stop]
            bg = np.einsum("qa,eaj->eqj", self.bubble_derivatives, grad)
            G = np.concatenate(
                (np.broadcast_to(grad[:, None], (n, nq, 4, 3)), bg[:, :, None]), axis=2
            )
            F = (
                np.eye(3)
                + np.einsum("eai,eaj->eij", u[el], grad)[:, None]
                + np.einsum("ei,eqj->eqij", bubble[start:stop], bg)
            )
            p = self.pressure_scale * np.einsum("qa,ea->eq", bary, q[el])
            args = (
                F.reshape(-1, 3, 3),
                p.ravel(),
                self.model,
                self.parameters,
                np.repeat(mesh.fiber[start:stop], nq, axis=0),
                np.repeat(mesh.sheet[start:stop], nq, axis=0),
                tension,
            )
            fast_nh = self.model == "neo_hookean"
            energy, _, P, C, J, A = mixed_material(*args, tangent=tangent and not fast_nh)
            P, A, J = P.reshape(n, nq, 3, 3), A.reshape(n, nq, 3, 3), J.reshape(n, nq)
            min_j = min(min_j, float(J.min()))
            vw = mesh.volumes[start:stop, None] * weights
            rd = np.einsum("eq,eqij,eqaj->eai", vw, P, G).reshape(n, 15)
            j_p, compliance, _ = volume_dual(p, self.parameters["kappa"])
            volume_constraint = J - j_p
            rp = self.pressure_scale * np.einsum("eq,qa,eq->ea", vw, bary, volume_constraint)
            np.add.at(constraint, el, rp / self.pressure_scale)
            rx = np.concatenate((rd[:, :12], rp), axis=1)
            rb = rd[:, 12:]
            rb_all[start:stop] = rb
            dofs = self.dofs[start:stop]
            np.add.at(residual, dofs, rx)
            green[start:stop] = 0.5 * (np.einsum("q,eqji,eqjk->eik", weights, F, F) - np.eye(3))
            fields[0][start:stop] = np.einsum("q,eqij->eij", weights, F)
            fields[1][start:stop] = energy.reshape(n, nq) @ weights
            # Store the quadrature-averaged Cauchy stress, not stress at averaged F.
            fields[2][start:stop] = np.einsum("q,eqij,eqkj,eq->eik", weights, P, F, 1 / J)
            fields[3][start:stop] = J @ weights
            if not tangent:
                continue
            if fast_nh:
                kdd = neo_hookean_displacement_block(
                    vw, G, A, J, p, self.parameters["mu"], mesh.fiber[start:stop], tension
                )
            else:
                C = C.reshape(n, nq, 3, 3, 3, 3)
                kdd = np.einsum("eq,eqaj,eqijkl,eqbl->eaibk", vw, G, C, G, optimize=True).reshape(
                    n, 15, 15
                )
            kdp = self.pressure_scale * np.einsum(
                "eq,eqaj,eqij,qb->eaib", vw, G, A, bary, optimize=True
            ).reshape(n, 15, 4)
            kpp = -(self.pressure_scale**2) * np.einsum(
                "eq,eq,qa,qb->eab", vw, compliance, bary, bary
            )
            kxx = np.zeros((n, 16, 16))
            kxx[:, :12, :12] = kdd[:, :12, :12]
            kxx[:, :12, 12:] = kdp[:, :12]
            kxx[:, 12:, :12] = kdp[:, :12].transpose(0, 2, 1)
            kxx[:, 12:, 12:] = kpp
            kbx = np.concatenate((kdd[:, 12:, :12], kdp[:, 12:]), axis=2)
            inv_bb = np.linalg.inv(kdd[:, 12:, 12:])
            kxb_inv = kbx.transpose(0, 2, 1) @ inv_bb
            local_k = kxx - kxb_inv @ kbx
            local_r = rx - np.einsum("eij,ej->ei", kxb_inv, rb)
            np.add.at(condensed, dofs, local_r)
            rows = np.broadcast_to(dofs[:, :, None], (n, 16, 16)).ravel()
            cols = np.broadcast_to(dofs[:, None, :], (n, 16, 16)).ravel()
            K += coo_matrix((local_k.ravel(), (rows, cols)), shape=K.shape).tocsr()
            recovery.append((start, dofs, inv_bb, rb, kbx))
        external, kp = pressure_load(mesh.nodes + u, mesh.cavity, pressure, compute_tangent=tangent)
        residual[: 3 * nn] -= external
        if tangent:
            condensed[: 3 * nn] -= external
            from scipy.sparse import bmat

            K -= bmat([[kp, None], [None, csr_matrix((nn, nn))]], format="csr")
        return MixedAssembly(
            residual, rb_all, K, condensed, recovery, (*fields, green), constraint, min_j
        )


def solve_mini(
    mesh,
    model,
    parameters,
    tension,
    pressure,
    indices,
    fixed,
    *,
    steps,
    iterations,
    atol,
    rtol,
    order=5,
):
    """Assembled sparse direct Newton; accept only the full uncondensed residual."""
    import warnings

    from scipy.sparse.linalg import MatrixRankWarning, spsolve

    assembler = MiniAssembler(mesh, model, parameters, order)
    nn = len(mesh.nodes)
    free_u = np.setdiff1d(np.arange(3 * nn), indices)
    free = np.r_[free_u, np.arange(3 * nn, 4 * nn)]
    u, q, bubble = np.zeros_like(mesh.nodes), np.zeros(nn), np.zeros((len(mesh.elements), 3))
    history = []
    count = 0

    def norm_of(a):
        return float(
            np.sqrt(np.linalg.norm(a.residual[free]) ** 2 + np.linalg.norm(a.bubble_residual) ** 2)
        )

    for step in range(1, steps + 1):
        fraction = step / steps
        u.ravel()[indices] = np.array([fixed[i] for i in indices]) * fraction
        initial = None
        for iteration in range(iterations + 1):
            a = assembler.assemble(u, q, bubble, tension * fraction, pressure * fraction)
            norm = norm_of(a)
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
                        "displacement_residual_N": float(np.linalg.norm(a.residual[free_u])),
                        "bubble_residual_N": float(np.linalg.norm(a.bubble_residual)),
                        "scaled_pressure_residual_N": float(np.linalg.norm(a.residual[3 * nn :])),
                        "weak_volume_constraint_m3": float(np.linalg.norm(a.constraint)),
                    }
                )
                break
            if iteration == iterations:
                raise ValueError("Mixed MINI Newton equilibrium did not converge")
            with warnings.catch_warnings():
                warnings.simplefilter("error", MatrixRankWarning)
                dx = np.zeros(4 * nn)
                dx[free] = spsolve(a.tangent[free][:, free], -a.condensed_residual[free])
            db = a.bubble_increment(dx)
            if not np.isfinite(dx).all() or not np.isfinite(db).all():
                raise ValueError("Nonfinite mixed MINI Newton increment")
            for power in range(16):
                alpha = 2.0 ** (-power)
                ut, qt, bt = (
                    u + alpha * dx[: 3 * nn].reshape(nn, 3),
                    q + alpha * dx[3 * nn :],
                    bubble + alpha * db,
                )
                try:
                    trial = assembler.assemble(
                        ut, qt, bt, tension * fraction, pressure * fraction, tangent=False
                    )
                except ValueError:
                    continue
                if norm_of(trial) < norm * (1 - 1e-4 * alpha):
                    u, q, bubble = ut, qt, bt
                    break
            else:
                raise ValueError(
                    f"Mixed MINI Newton line search failed at step {step}, iteration {iteration}: residual={norm:.6g} N, tolerance={tolerance:.6g} N"
                )
            count += 1
    # a is the converged, fully assembled state, including bubble/pressure residuals.
    extra = {
        "green_strain": a.fields[4].tolist(),
        "mixed_pressure_Pa": (q * assembler.pressure_scale).tolist(),
        "bubble_displacement_m": bubble.tolist(),
        "field_representation": "Element fields are reference-quadrature averages; displacement is P1 plus the stored quartic bubble",
        "mixed_quadrature_order": order,
        "minimum_quadrature_J": a.minimum_J,
        "pressure_scale_Pa_per_m": assembler.pressure_scale,
        "energy_definition": "Integrated mixed potential W0+p*(J-1)-U_star(p), U(J)=kappa/2*log(J)^2 on 0<J<e",
        "pressure_definition": "p=kappa*log(J)/J, continuous P1 dual pressure; weak determinant constraint; not total hydrostatic stress",
    }
    return u, a.residual[: 3 * nn], a.fields[:4], history, count, extra
