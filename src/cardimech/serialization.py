"""Portable JSON, finite numeric contracts, and atomic file replacement."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path


def finite_number(value, name, *, minimum=None, strictly_positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and numeric")
    if strictly_positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return float(value)


def integer(value, name, *, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def strict_bool(value, name):
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def strict_loads(raw):
    def invalid(value):
        raise ValueError(f"Nonstandard JSON number: {value}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(raw, parse_constant=invalid, object_pairs_hook=unique)


def read_json(path):
    return strict_loads(Path(path).read_text(encoding="utf-8"))


def atomic_write(path, encoded):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".cardimech-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def write_json(path, payload):
    encoded = (json.dumps(payload, allow_nan=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    atomic_write(path, encoded)
