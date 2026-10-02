import hashlib
import json

import pytest

from cardimech import ArtifactRef
from cardimech.handoff import HandoffError, cardiep_activation_delay_s


def test_cardiep_activation_handoff_reads_and_hashes(tmp_path) -> None:
    path = tmp_path / "activation.json"
    payload = {
        "schema_version": "cardiep-field-v1",
        "units": "ms",
        "activation_ms": [0.0, 10.0, 20.0, 30.0],
    }
    data = (json.dumps(payload) + "\n").encode()
    path.write_bytes(data)
    ref = ArtifactRef(
        artifact_id="ep-activation",
        kind="activation_map",
        uri=path.as_uri(),
        sha256=hashlib.sha256(data).hexdigest(),
    )
    delay, provenance = cardiep_activation_delay_s(ref, quantile=0.5)
    assert delay == pytest.approx(0.015)
    assert provenance["n_nodes"] == 4


def test_cardiep_activation_handoff_rejects_bad_hash(tmp_path) -> None:
    path = tmp_path / "activation.json"
    path.write_text('{"units":"ms","activation_ms":[0,1]}', encoding="utf-8")
    ref = ArtifactRef(
        artifact_id="ep-activation",
        kind="activation_map",
        uri=path.as_uri(),
        sha256="0" * 64,
    )
    with pytest.raises(HandoffError):
        cardiep_activation_delay_s(ref)
