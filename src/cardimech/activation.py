from __future__ import annotations

from functools import lru_cache

import numpy as np

from .serialization import finite_number


def _raw(phase, rise, decay, rise_power, decay_power):
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        logarithm = np.log(phase)
        up = rise_power * (logarithm - np.log(rise))
        down = decay_power * (logarithm - np.log(decay))
        return np.exp(-np.logaddexp(0.0, -up)) * np.exp(-np.logaddexp(0.0, down))


@lru_cache(maxsize=128)
def _peak(cycle, rise, decay, rise_power, decay_power):
    grid = np.linspace(0.0, cycle, 4097, endpoint=False)
    peak = float(np.max(_raw(grid, rise, decay, rise_power, decay_power)))
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("Activation time scales cannot be resolved by the normalization grid")
    return peak


def periodic_hill_activation(
    time_s,
    *,
    cycle_length_s,
    onset_s=0.0,
    rise_s=0.08,
    decay_s=0.22,
    rise_power=3.0,
    decay_power=4.0,
):
    """Bounded periodic Hill-like activation; the phase wrap can have a small jump."""
    for name, value in {
        "cycle_length_s": cycle_length_s,
        "rise_s": rise_s,
        "decay_s": decay_s,
        "rise_power": rise_power,
        "decay_power": decay_power,
    }.items():
        finite_number(value, name, strictly_positive=True)
    finite_number(onset_s, "onset_s")
    t = np.asarray(time_s, dtype=float)
    if not np.all(np.isfinite(t)):
        raise ValueError("time_s must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        phase = np.mod(t - onset_s, cycle_length_s)
    if not np.all(np.isfinite(phase)):
        raise OverflowError("Activation phase exceeds float64 range")
    tolerance = 8 * np.spacing(np.maximum(np.abs(t - onset_s), cycle_length_s))
    phase = np.where((cycle_length_s - phase <= tolerance) | (phase <= tolerance), 0.0, phase)
    peak = _peak(cycle_length_s, rise_s, decay_s, rise_power, decay_power)
    return np.clip(_raw(phase, rise_s, decay_s, rise_power, decay_power) / peak, 0.0, 1.0)
