from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .models import ArtifactRef


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_artifact(
    output_dir: str | Path,
    *,
    filename: str,
    artifact_id: str,
    kind: str,
    payload: Any,
    metadata: dict[str, Any] | None = None,
) -> ArtifactRef:
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return ArtifactRef(
        artifact_id=artifact_id,
        kind=kind,
        uri=path.as_uri(),
        sha256=sha256_bytes(encoded),
        metadata=metadata or {},
    )
