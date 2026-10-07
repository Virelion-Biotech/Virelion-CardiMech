from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindkesselParameters:
    p_atrium_mmHg: float = 8.0
    p_venous_mmHg: float = 5.0
    r_mitral_mmHg_s_per_ml: float = 0.01
    r_aortic_mmHg_s_per_ml: float = 0.015
    r_systemic_mmHg_s_per_ml: float = 1.0
    c_arterial_ml_per_mmHg: float = 1.5
    initial_arterial_pressure_mmHg: float = 75.0
    initial_lv_volume_ml: float = 120.0

    def validate(self) -> None:
        for name, value in vars(self).items():
            if isinstance(value, bool) or not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        positive = {
            "r_mitral": self.r_mitral_mmHg_s_per_ml,
            "r_aortic": self.r_aortic_mmHg_s_per_ml,
            "r_systemic": self.r_systemic_mmHg_s_per_ml,
            "c_arterial": self.c_arterial_ml_per_mmHg,
            "initial_lv_volume": self.initial_lv_volume_ml,
        }
        for name, value in positive.items():
            if value <= 0 or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and positive")


def simulate_lv_windkessel(
    *,
    time_s: np.ndarray,
    pressure_fn: Callable[[float, float], float],
    params: WindkesselParameters,
) -> dict[str, np.ndarray]:
    params.validate()
    t = np.asarray(time_s, dtype=float)
    if t.ndim != 1 or len(t) < 2 or not np.all(np.isfinite(t)) or not np.all(np.diff(t) > 0):
        raise ValueError("time_s must be a strictly increasing 1D array")
    n = len(t)
    volume = np.empty(n, dtype=float)
    p_art = np.empty(n, dtype=float)
    p_lv = np.empty(n, dtype=float)
    q_mitral = np.empty(n, dtype=float)
    q_aortic = np.empty(n, dtype=float)
    q_systemic = np.empty(n, dtype=float)
    volume[0] = params.initial_lv_volume_ml
    p_art[0] = params.initial_arterial_pressure_mmHg

    for i in range(n - 1):
        dt = float(t[i + 1] - t[i])
        p_lv[i] = float(pressure_fn(float(t[i]), float(volume[i])))
        if not np.isfinite(p_lv[i]):
            raise RuntimeError("Chamber pressure must be finite")
        q_mitral[i] = max((params.p_atrium_mmHg - p_lv[i]) / params.r_mitral_mmHg_s_per_ml, 0.0)
        q_aortic[i] = max((p_lv[i] - p_art[i]) / params.r_aortic_mmHg_s_per_ml, 0.0)
        q_systemic[i] = max(
            (p_art[i] - params.p_venous_mmHg) / params.r_systemic_mmHg_s_per_ml, 0.0
        )
        volume[i + 1] = volume[i] + dt * (q_mitral[i] - q_aortic[i])
        p_art[i + 1] = p_art[i] + dt * (q_aortic[i] - q_systemic[i]) / params.c_arterial_ml_per_mmHg
        if not np.isfinite(volume[i + 1]) or volume[i + 1] <= 0:
            raise RuntimeError("0D circulation produced a non-positive or non-finite LV volume")
        if not np.isfinite(p_art[i + 1]):
            raise RuntimeError("0D circulation produced a non-finite arterial pressure")

    p_lv[-1] = float(pressure_fn(float(t[-1]), float(volume[-1])))
    if not np.isfinite(p_lv[-1]):
        raise RuntimeError("Chamber pressure must be finite")
    q_mitral[-1] = max((params.p_atrium_mmHg - p_lv[-1]) / params.r_mitral_mmHg_s_per_ml, 0.0)
    q_aortic[-1] = max((p_lv[-1] - p_art[-1]) / params.r_aortic_mmHg_s_per_ml, 0.0)
    q_systemic[-1] = max((p_art[-1] - params.p_venous_mmHg) / params.r_systemic_mmHg_s_per_ml, 0.0)
    return {
        "time_s": t,
        "lv_volume_ml": volume,
        "lv_pressure_mmHg": p_lv,
        "arterial_pressure_mmHg": p_art,
        "mitral_flow_ml_s": q_mitral,
        "aortic_flow_ml_s": q_aortic,
        "systemic_flow_ml_s": q_systemic,
    }
