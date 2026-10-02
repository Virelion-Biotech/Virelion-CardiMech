from __future__ import annotations

import json
import os
import sys
from typing import Any

from .api import MechanicsAPI


def _payload() -> dict[str, Any]:
    if os.environ.get("HEARTTWIN_PAYLOAD_STDIN") == "1":
        raw = sys.stdin.read()
    else:
        raw = os.environ.get("HEARTTWIN_PAYLOAD", "{}")
    data = json.loads(raw or "{}")
    if not isinstance(data, dict):
        raise TypeError("HeartTwin payload must contain a JSON object")
    return data


def dispatch(capability: str, payload: dict[str, Any]) -> dict[str, Any]:
    api = MechanicsAPI()
    if capability == "mechanics.health":
        return api.health()
    if capability == "mechanics.backends":
        return api.backends()
    if capability == "mechanics.materials":
        return api.materials()
    if capability == "mechanics.simulate":
        return api.simulate(payload)
    if capability == "mechanics.prepare_calibration":
        return api.prepare_calibration(payload)
    if capability == "mechanics.validate.reference":
        return api.validate_reference()
    if capability == "mechanics.ecosystem":
        return api.ecosystem()
    raise ValueError(f"CardiMech does not support {capability}")


def main() -> int:
    capability = os.environ.get("HEARTTWIN_CAPABILITY", "")
    if not capability:
        print("HEARTTWIN_CAPABILITY is required", file=sys.stderr)
        return 2
    try:
        result = dispatch(capability, _payload())
    except Exception as exc:  # noqa: BLE001 - process boundary normalizes failures
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
