from __future__ import annotations

from math import pi

import numpy as np

from .serialization import finite_number

MMHG_TO_KPA = 0.133322368


def passive_pressure_mmHg(volume_ml: float | np.ndarray, *, v0_ml: float, a_mmHg: float, b: float):
    finite_number(v0_ml, "v0_ml", strictly_positive=True)
    finite_number(a_mmHg, "a_mmHg", minimum=0)
    finite_number(b, "b", strictly_positive=True)
    if v0_ml <= 0 or a_mmHg < 0 or b <= 0:
        raise ValueError("v0_ml and b must be positive; a_mmHg must be non-negative")
    volume = np.asarray(volume_ml, dtype=float)
    if not np.all(np.isfinite(volume)) or np.any(volume <= 0):
        raise ValueError("Volume must be finite and positive")
    with np.errstate(over="ignore", invalid="ignore"):
        exponent = b * (volume / v0_ml - 1.0)
    if not np.all(np.isfinite(exponent)) or np.any(exponent > 50):
        raise ValueError("Passive pressure exponent exceeds supported model range (50)")
    return a_mmHg * np.expm1(exponent)


def active_pressure_mmHg(
    volume_ml: float | np.ndarray,
    activation: float | np.ndarray,
    *,
    v0_ml: float,
    emax_mmHg_per_ml: float,
):
    finite_number(v0_ml, "v0_ml", strictly_positive=True)
    finite_number(emax_mmHg_per_ml, "emax_mmHg_per_ml", minimum=0)
    if v0_ml <= 0 or emax_mmHg_per_ml < 0:
        raise ValueError("v0_ml must be positive and emax must be non-negative")
    volume = np.asarray(volume_ml, dtype=float)
    act = np.asarray(activation, dtype=float)
    if not np.all(np.isfinite(volume)) or np.any(volume <= 0):
        raise ValueError("Volume must be finite and positive")
    if not np.all(np.isfinite(act)) or np.any((act < 0) | (act > 1)):
        raise ValueError("Activation must be finite and within [0, 1]")
    return act * emax_mmHg_per_ml * np.maximum(volume - v0_ml, 0.0)


def chamber_pressure_mmHg(
    volume_ml: float | np.ndarray,
    activation: float | np.ndarray,
    *,
    v0_ml: float,
    a_mmHg: float,
    b: float,
    emax_mmHg_per_ml: float,
):
    return passive_pressure_mmHg(volume_ml, v0_ml=v0_ml, a_mmHg=a_mmHg, b=b) + active_pressure_mmHg(
        volume_ml, activation, v0_ml=v0_ml, emax_mmHg_per_ml=emax_mmHg_per_ml
    )


def stroke_work_mmHg_ml(volume_ml: np.ndarray, pressure_mmHg: np.ndarray) -> float:
    v = np.asarray(volume_ml, dtype=float)
    p = np.asarray(pressure_mmHg, dtype=float)
    if v.shape != p.shape or v.ndim != 1 or len(v) < 2:
        raise ValueError("volume and pressure must be equal-length 1D arrays")
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(p)):
        raise ValueError("PV arrays must be finite")
    v, p = np.append(v, v[0]), np.append(p, p[0])
    integral = np.trapezoid(p, v) if hasattr(np, "trapezoid") else np.trapz(p, v)
    return float(abs(integral))


def pv_metrics(volume_ml: np.ndarray, pressure_mmHg: np.ndarray) -> dict[str, float]:
    v = np.asarray(volume_ml, dtype=float)
    p = np.asarray(pressure_mmHg, dtype=float)
    if v.shape != p.shape or v.ndim != 1 or len(v) < 2:
        raise ValueError("volume and pressure must be equal-length 1D arrays")
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(p)) or np.any(v <= 0):
        raise ValueError("PV arrays must be finite and volumes positive")
    edv = float(np.max(v))
    esv = float(np.min(v))
    sv = edv - esv
    return {
        "edv_ml": edv,
        "esv_ml": esv,
        "stroke_volume_ml": sv,
        "ejection_fraction": sv / edv if edv > 0 else float("nan"),
        "peak_lv_pressure_mmHg": float(np.max(p)),
        "min_lv_pressure_mmHg": float(np.min(p)),
        "stroke_work_mmHg_ml": stroke_work_mmHg_ml(v, p),
        "stroke_work_j": stroke_work_mmHg_ml(v, p) * MMHG_TO_KPA * 1e-3,
    }


def spherical_wall_metrics(
    volume_ml: np.ndarray,
    pressure_mmHg: np.ndarray,
    *,
    reference_volume_ml: float,
    wall_thickness_cm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite_number(reference_volume_ml, "reference_volume_ml", strictly_positive=True)
    finite_number(wall_thickness_cm, "wall_thickness_cm", strictly_positive=True)
    if reference_volume_ml <= 0 or wall_thickness_cm <= 0:
        raise ValueError("reference volume and wall thickness must be positive")
    volume = np.asarray(volume_ml, dtype=float)
    pressure = np.asarray(pressure_mmHg, dtype=float)
    if (
        volume.shape != pressure.shape
        or not np.all(np.isfinite(volume))
        or np.any(volume <= 0)
        or not np.all(np.isfinite(pressure))
    ):
        raise ValueError("Wall metric arrays must be finite, aligned, with positive volume")
    radius_cm = np.cbrt(3.0 * np.maximum(volume, 1e-9) / (4.0 * pi))
    r0_cm = (3.0 * reference_volume_ml / (4.0 * pi)) ** (1.0 / 3.0)
    strain = radius_cm / r0_cm - 1.0
    wall_stress_kpa = pressure * MMHG_TO_KPA * radius_cm / (2.0 * wall_thickness_cm)
    return radius_cm, strain, wall_stress_kpa
