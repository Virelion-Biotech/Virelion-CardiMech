from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from typing import Any

from .api import MechanicsAPI
from .serialization import strict_loads


def _payload() -> dict[str, Any]:
    if os.environ.get("HEARTTWIN_PAYLOAD_STDIN") == "1":
        raw = sys.stdin.read()
    else:
        raw = os.environ.get("HEARTTWIN_PAYLOAD", "{}")
    data = strict_loads(raw or "{}")
    if not isinstance(data, dict):
        raise TypeError("HeartTwin payload must contain a JSON object")
    return data


def _context_request(
    payload: dict[str, Any],
    *,
    context_key: str,
) -> dict[str, Any]:
    """Translate the generic HeartTwin envelope into a CardiMech-native request.

    Direct native requests remain supported. Orchestrated calls provide the full
    specialist request under the requested context key; HeartTwin's entity_id is
    authoritative for subject identity.
    """
    if "subject_id" in payload:
        return payload
    context = payload.get("context")
    if not isinstance(context, dict):
        return payload
    template = context.get(context_key)
    if not isinstance(template, dict):
        return payload
    request = deepcopy(template)
    entity_id = payload.get("entity_id")
    if isinstance(entity_id, str) and entity_id:
        request["subject_id"] = entity_id
    return request


def dispatch(capability: str, payload: dict[str, Any]) -> dict[str, Any]:
    api = MechanicsAPI()
    if capability == "mechanics.health":
        return api.health()
    if capability == "mechanics.backends":
        return api.backends()
    if capability == "mechanics.materials":
        return api.materials()
    if capability == "mechanics.simulate":
        request = _context_request(payload, context_key="cardimech_request")
        return api.simulate(request)
    if capability == "mechanics.prepare_calibration":
        request = _context_request(payload, context_key="cardimech_calibration_request")
        return api.prepare_calibration(request)
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
