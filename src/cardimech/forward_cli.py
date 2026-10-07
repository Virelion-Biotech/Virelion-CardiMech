from __future__ import annotations

import json
import os
import sys

from .api import MechanicsAPI
from .serialization import strict_loads


def main() -> int:
    raw = os.environ.get("HEARTTWIN_PAYLOAD")
    if not raw:
        print("HEARTTWIN_PAYLOAD is required", file=sys.stderr)
        return 2
    try:
        payload = strict_loads(raw)
        if not isinstance(payload, dict):
            raise TypeError("HEARTTWIN_PAYLOAD must contain a JSON object")
        result = MechanicsAPI().simulate(payload)
    except Exception as exc:  # noqa: BLE001 - command boundary normalizes to stderr/exit
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
