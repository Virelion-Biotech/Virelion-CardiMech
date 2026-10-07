from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .activation import periodic_hill_activation
from .artifacts import write_json_artifact
from .circulation import WindkesselParameters, simulate_lv_windkessel
from .handoff import cardiep_activation_delay_s
from .models import MechanicsQC, MechanicsSimulationRequest, MechanicsSimulationResult
from .pv import chamber_pressure_mmHg, pv_metrics, spherical_wall_metrics
from .serialization import finite_number, integer, strict_bool


class ReferenceLumpedBackend:
    """Dependency-light LV mechanics + 0D afterload reference backend.

    This is deliberately not a finite-element solver. It exists for integration,
    parameter plumbing, deterministic regression tests, rapid sweeps, and inverse-loop
    smoke tests before escalation to a spatial mechanics backend.
    """

    name = "numpy-lumped-v1"

    def available(self) -> bool:
        return True

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "available": True,
            "fidelity": "lumped-reference",
            "spatial": False,
            "supports": [
                "periodic activation",
                "nonlinear passive PV relation",
                "active elastance",
                "3-element-like arterial coupling with diode valves",
                "PV/strain/wall-stress summaries",
            ],
            "limitations": [
                "does not consume tetrahedral mechanics mesh fields",
                "does not solve continuum balance equations",
                "spherical wall metrics are geometric surrogates",
            ],
        }

    @staticmethod
    def _parameter(values: dict[str, float], name: str, default: float) -> float:
        return float(values.get(name, default))

    def simulate(self, request: MechanicsSimulationRequest) -> MechanicsSimulationResult:
        request = MechanicsSimulationRequest.model_validate(request.model_dump(mode="json"))
        if request.parameters.active_model != "periodic_hill":
            raise ValueError("Reference backend supports only periodic_hill activation")
        settings = dict(request.settings)
        allowed_settings = {
            "cycle_length_s",
            "dt_s",
            "cycles",
            "max_steps",
            "activation_quantile",
            "wall_thickness_cm",
            "p_atrium_mmHg",
            "p_venous_mmHg",
            "initial_lv_volume_ml",
            "max_volume_step_ml",
            "cycle_volume_tolerance_ml",
            "cycle_pressure_tolerance_mmHg",
            "require_periodic_convergence",
            "inline_series",
            "output_dir",
        }
        if unknown := set(settings) - allowed_settings:
            raise ValueError(f"Unsupported reference settings: {sorted(unknown)}")
        for values, allowed in (
            (request.parameters.passive, {"v0_ml", "a_mmHg", "b"}),
            (request.parameters.active, {"emax_mmHg_per_ml", "rise_s", "decay_s", "onset_s"}),
            (
                request.circulation.parameters,
                {
                    "p_atrium_mmHg",
                    "p_venous_mmHg",
                    "r_mitral",
                    "r_aortic",
                    "r_systemic",
                    "c_arterial",
                    "initial_arterial_pressure_mmHg",
                    "initial_lv_volume_ml",
                },
            ),
        ):
            if unknown := set(values) - allowed:
                raise ValueError(f"Unsupported reference parameters: {sorted(unknown)}")
        if request.boundary_conditions:
            raise ValueError("Reference backend cannot apply spatial boundary conditions")
        if request.circulation.model not in {"none", "windkessel_3e"}:
            raise ValueError("Reference backend supports only its intrinsic/Windkessel afterload")
        if request.circulation.state_ref is not None:
            raise ValueError("Reference backend cannot load circulation state artifacts")
        if not request.circulation.enabled and request.circulation.parameters:
            raise ValueError("Circulation parameters require enabled coupling")
        cycle_length = finite_number(
            settings.get("cycle_length_s", 0.8), "cycle_length_s", strictly_positive=True
        )
        dt = finite_number(settings.get("dt_s", 0.001), "dt_s", strictly_positive=True)
        cycles = integer(settings.get("cycles", 5), "cycles", minimum=2)
        max_steps = integer(settings.get("max_steps", 500_000), "max_steps", minimum=3)
        if cycle_length / dt > (max_steps - 1) / cycles:
            raise ValueError("Reference simulation exceeds max_steps")
        intervals = int(np.ceil(cycle_length / dt))
        if intervals < 2:
            raise ValueError("dt_s must resolve at least two intervals per cycle")
        steps = cycles * intervals + 1
        if steps > max_steps:
            raise ValueError("Reference simulation exceeds max_steps after cycle alignment")
        effective_dt = cycle_length / intervals
        time = np.arange(steps, dtype=float) * effective_dt
        if not np.all(np.isfinite(time)):
            raise OverflowError("Simulation time grid exceeds float64 range")
        require_convergence = strict_bool(
            settings.get("require_periodic_convergence", False), "require_periodic_convergence"
        )
        inline_series = strict_bool(settings.get("inline_series", True), "inline_series")
        volume_tolerance = finite_number(
            settings.get("cycle_volume_tolerance_ml", 15.0),
            "cycle_volume_tolerance_ml",
            strictly_positive=True,
        )
        pressure_tolerance = finite_number(
            settings.get("cycle_pressure_tolerance_mmHg", 15.0),
            "cycle_pressure_tolerance_mmHg",
            strictly_positive=True,
        )
        max_volume_step = finite_number(
            settings.get("max_volume_step_ml", 5.0), "max_volume_step_ml", strictly_positive=True
        )
        passive = request.parameters.passive
        active = request.parameters.active
        v0 = self._parameter(passive, "v0_ml", 10.0)
        a_mmHg = self._parameter(passive, "a_mmHg", 0.08)
        b = self._parameter(passive, "b", 0.055)
        emax = self._parameter(active, "emax_mmHg_per_ml", 2.1)
        rise_s = self._parameter(active, "rise_s", 0.07)
        decay_s = self._parameter(active, "decay_s", 0.24)
        onset_s = self._parameter(active, "onset_s", 0.0)
        activation_handoff: dict[str, object] | None = None
        if request.activation_ref is not None:
            ep_delay_s, activation_handoff = cardiep_activation_delay_s(
                request.activation_ref,
                quantile=float(settings.get("activation_quantile", 0.5)),
            )
            onset_s += ep_delay_s
        wall_thickness_cm = finite_number(
            settings.get("wall_thickness_cm", 1.0), "wall_thickness_cm", strictly_positive=True
        )

        activation = periodic_hill_activation(
            time,
            cycle_length_s=cycle_length,
            onset_s=onset_s,
            rise_s=rise_s,
            decay_s=decay_s,
        )

        cparams = dict(request.circulation.parameters)
        wk = WindkesselParameters(
            p_atrium_mmHg=float(cparams.get("p_atrium_mmHg", settings.get("p_atrium_mmHg", 8.0))),
            p_venous_mmHg=float(cparams.get("p_venous_mmHg", settings.get("p_venous_mmHg", 5.0))),
            r_mitral_mmHg_s_per_ml=float(cparams.get("r_mitral", 0.01)),
            r_aortic_mmHg_s_per_ml=float(cparams.get("r_aortic", 0.015)),
            r_systemic_mmHg_s_per_ml=float(cparams.get("r_systemic", 1.0)),
            c_arterial_ml_per_mmHg=float(cparams.get("c_arterial", 1.5)),
            initial_arterial_pressure_mmHg=float(
                cparams.get("initial_arterial_pressure_mmHg", 75.0)
            ),
            initial_lv_volume_ml=float(
                cparams.get("initial_lv_volume_ml", settings.get("initial_lv_volume_ml", 120.0))
            ),
        )

        def pressure_fn(t_s: float, volume_ml: float) -> float:
            act = float(
                periodic_hill_activation(
                    [t_s],
                    cycle_length_s=cycle_length,
                    onset_s=onset_s,
                    rise_s=rise_s,
                    decay_s=decay_s,
                )[0]
            )
            return float(
                chamber_pressure_mmHg(
                    volume_ml,
                    act,
                    v0_ml=v0,
                    a_mmHg=a_mmHg,
                    b=b,
                    emax_mmHg_per_ml=emax,
                )
            )

        series = simulate_lv_windkessel(time_s=time, pressure_fn=pressure_fn, params=wk)
        series["activation"] = activation
        start = (cycles - 1) * intervals
        final = {name: values[start:] for name, values in series.items()}
        radius, strain, stress = spherical_wall_metrics(
            final["lv_volume_ml"],
            final["lv_pressure_mmHg"],
            reference_volume_ml=float(np.max(final["lv_volume_ml"])),
            wall_thickness_cm=wall_thickness_cm,
        )
        final["radius_cm"] = radius
        final["circumferential_strain"] = strain
        final["wall_stress_kpa"] = stress
        metrics = pv_metrics(final["lv_volume_ml"], final["lv_pressure_mmHg"])
        metrics.update(
            {
                "mean_arterial_pressure_mmHg": float(np.mean(final["arterial_pressure_mmHg"])),
                "peak_wall_stress_kpa": float(np.max(stress)),
                "min_circumferential_strain": float(np.min(strain)),
                "max_circumferential_strain": float(np.max(strain)),
                "wall_thickness_cm": wall_thickness_cm,
                "thin_wall_ratio_max": float(np.max(wall_thickness_cm / radius)),
            }
        )

        prior_start = (cycles - 2) * intervals
        prior_end = start + 1
        prior_v = series["lv_volume_ml"][prior_start:prior_end]
        prior_p = series["arterial_pressure_mmHg"][prior_start:prior_end]
        current_v = final["lv_volume_ml"]
        current_p = final["arterial_pressure_mmHg"]
        n_compare = min(len(prior_v), len(current_v))
        cycle_v_residual = float(np.max(np.abs(prior_v[-n_compare:] - current_v[:n_compare])))
        cycle_p_residual = float(np.max(np.abs(prior_p[-n_compare:] - current_p[:n_compare])))
        max_v_step = float(np.max(np.abs(np.diff(final["lv_volume_ml"]))))

        checks = {
            "finite_outputs": bool(all(np.all(np.isfinite(value)) for value in final.values())),
            "positive_volume": bool(np.min(final["lv_volume_ml"]) > 0.0),
            "bounded_activation": bool(
                np.min(final["activation"]) >= 0.0 and np.max(final["activation"]) <= 1.0
            ),
            "stable_time_step": bool(max_v_step < max_volume_step),
        }
        converged = bool(
            cycle_v_residual < volume_tolerance and cycle_p_residual < pressure_tolerance
        )
        warnings: list[str] = [
            "numpy-lumped-v1 is a non-spatial reference model, not a finite-element mechanics solve.",
            "Reported wall strain/stress use a spherical chamber surrogate.",
        ]
        if not request.circulation.enabled:
            warnings.append(
                "No stack circulation coupling requested; intrinsic reference afterload is used."
            )
        if metrics["thin_wall_ratio_max"] > 0.1:
            warnings.append(
                "Wall thickness/radius exceeds 0.1; thin-wall Laplace stress is only a surrogate."
            )
        if activation_handoff is not None:
            warnings.append(
                "CardiEP spatial activation was reduced to one timing quantile because the reference backend is non-spatial."
            )
        if not converged:
            warnings.append(
                "Periodic cycle residual did not meet the configured reference tolerance."
            )
        if require_convergence:
            checks["periodic_convergence"] = converged
        passed = all(checks.values())
        qc = MechanicsQC(
            passed=passed,
            converged=converged,
            checks=checks,
            metrics={
                "cycle_volume_residual_ml": cycle_v_residual,
                "cycle_pressure_residual_mmHg": cycle_p_residual,
                "max_volume_step_ml": max_v_step,
            },
            warnings=warnings,
            errors=[] if passed else ["Reference solver QC failed"],
        )

        serializable_series = {name: [float(x) for x in values] for name, values in final.items()}
        outputs = []
        output_dir = settings.get("output_dir")
        if output_dir:
            outputs.append(
                write_json_artifact(
                    output_dir,
                    filename="mechanics_timeseries.json",
                    artifact_id=f"{request.subject_id}-mechanics-timeseries",
                    kind="mechanics_timeseries",
                    payload=serializable_series,
                    metadata={"backend": self.name, "cycle": cycles - 1},
                )
            )
            outputs.append(
                write_json_artifact(
                    output_dir,
                    filename="mechanics_summary.json",
                    artifact_id=f"{request.subject_id}-mechanics-summary",
                    kind="mechanics_summary",
                    payload=metrics,
                    metadata={"backend": self.name},
                )
            )

        return MechanicsSimulationResult(
            subject_id=request.subject_id,
            backend=self.name,
            parameters=request.parameters,
            outputs=outputs,
            scalar_outputs=metrics,
            series=serializable_series if inline_series else {},
            qc=qc,
            validation_status="software_checked",
            warnings=warnings,
            provenance={
                "model": "nonlinear LV pressure-volume surrogate + diode-valve arterial Windkessel",
                "backend_description": self.describe(),
                "circulation_parameters": asdict(wk),
                "circulation_mode": "requested_windkessel"
                if request.circulation.enabled
                else "intrinsic_reference_afterload",
                "material_scope": "PV/elastance surrogate; continuum material_model not evaluated",
                "settings": {
                    "cycle_length_s": cycle_length,
                    "dt_s": dt,
                    "effective_dt_s": effective_dt,
                    "intervals_per_cycle": intervals,
                    "cycles": cycles,
                    "wall_thickness_cm": wall_thickness_cm,
                },
                "anatomy_ref": request.anatomy_ref.model_dump(mode="json"),
                "activation_ref": None
                if request.activation_ref is None
                else request.activation_ref.model_dump(mode="json"),
                "activation_handoff": activation_handoff,
            },
        )
