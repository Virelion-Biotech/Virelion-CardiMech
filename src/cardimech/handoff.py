from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np

from .artifacts import sha256_file
from .models import ArtifactRef
from .serialization import strict_loads


class HandoffError(ValueError):
    """Raised when a cross-service artifact cannot be consumed safely."""


def local_path_from_uri(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme not in {"", "file"}:
        raise HandoffError(
            f"Only local/file artifacts are supported by the built-in backend: {uri}"
        )
    if parsed.scheme == "file" and parsed.netloc not in {"", "localhost"}:
        raise HandoffError("Remote file URI authorities are unsupported")
    if parsed.scheme == "file":
        path = Path(unquote(parsed.path))
    else:
        path = Path(uri)
    return path.expanduser().resolve()


def _verify_sha256(path: Path, expected: str | None) -> None:
    if expected is None:
        return
    digest = sha256_file(path)
    if digest.lower() != expected.lower():
        raise HandoffError(f"Artifact SHA-256 mismatch for {path}")


def load_json_artifact(ref: ArtifactRef) -> dict:
    path = local_path_from_uri(ref.uri)
    if not path.is_file():
        raise HandoffError(f"Artifact file does not exist: {path}")
    _verify_sha256(path, ref.sha256)
    try:
        payload = strict_loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise HandoffError(f"Could not read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise HandoffError("Cross-service JSON artifact must contain an object")
    return payload


def cardiep_activation_delay_s(
    ref: ArtifactRef, *, quantile: float = 0.5
) -> tuple[float, dict[str, object]]:
    """Reduce a CardiEP activation map to a declared global activation delay.

    Spatial mechanics plugins should consume the complete activation field. The built-in
    lumped reference backend has no spatial mesh, so it may only reduce that map to one
    scalar timing statistic. The reduction is explicit in provenance.
    """
    if isinstance(quantile, bool) or not 0.0 <= quantile <= 1.0:
        raise HandoffError("activation quantile must lie in [0, 1]")
    payload = load_json_artifact(ref)
    values = payload.get("activation_ms")
    if not isinstance(values, list) or not values:
        raise HandoffError("CardiEP activation artifact requires a non-empty activation_ms array")
    activation_ms = np.asarray(values, dtype=float)
    if (
        activation_ms.ndim != 1
        or not np.all(np.isfinite(activation_ms))
        or np.any(activation_ms < 0)
    ):
        raise HandoffError("CardiEP activation_ms must be a finite 1D array")
    units = str(payload.get("units", "ms"))
    if units != "ms":
        raise HandoffError(f"Unsupported CardiEP activation units: {units!r}; expected 'ms'")
    delay_ms = float(np.quantile(activation_ms, quantile))
    return delay_ms / 1000.0, {
        "source_artifact_id": ref.artifact_id,
        "source_kind": ref.kind,
        "method": "activation-map quantile reduction for non-spatial reference backend",
        "quantile": float(quantile),
        "delay_ms": delay_ms,
        "n_nodes": int(activation_ms.size),
        "schema_version": payload.get("schema_version"),
    }
