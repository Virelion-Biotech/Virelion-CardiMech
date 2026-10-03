from __future__ import annotations

import numpy as np


def periodic_hill_activation(
    time_s: np.ndarray | list[float],
    *,
    cycle_length_s: float,
    onset_s: float = 0.0,
    rise_s: float = 0.08,
    decay_s: float = 0.22,
    rise_power: float = 3.0,
    decay_power: float = 4.0,
) -> np.ndarray:
    """Smooth periodic activation in [0, 1] using rising/decaying Hill terms."""
    if cycle_length_s <= 0 or rise_s <= 0 or decay_s <= 0:
        raise ValueError("cycle_length_s, rise_s and decay_s must be positive")
    if rise_power <= 0 or decay_power <= 0:
        raise ValueError("Hill powers must be positive")
    t = np.asarray(time_s, dtype=float)
    if not np.all(np.isfinite(t)):
        raise ValueError("time_s must be finite")
    phase = np.mod(t - onset_s, cycle_length_s)
    rise = phase**rise_power / (phase**rise_power + rise_s**rise_power)
    decay = decay_s**decay_power / (phase**decay_power + decay_s**decay_power)
    raw = rise * decay
    # Normalize using a dense deterministic phase grid rather than the caller's sampling.
    grid = np.linspace(0.0, cycle_length_s, 4097, endpoint=False)
    gr = grid**rise_power / (grid**rise_power + rise_s**rise_power)
    gd = decay_s**decay_power / (grid**decay_power + decay_s**decay_power)
    peak = float(np.max(gr * gd))
    return np.clip(raw / peak if peak > 0 else raw, 0.0, 1.0)
