from __future__ import annotations

import argparse
import json
import sys

from .api import MechanicsAPI
from .materials import material_point
from .serialization import read_json, write_json


def _read_json(path: str) -> dict:
    data = read_json(path)
    if not isinstance(data, dict):
        raise TypeError("Input JSON must be an object")
    return data


def _main() -> int:
    parser = argparse.ArgumentParser(prog="cardimech")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Report package and backend availability")
    sub.add_parser("backends", help="List mechanics backends")
    sub.add_parser("materials", help="List constitutive material families")
    sub.add_parser("ecosystem", help="Show audited external ecosystem map")
    sub.add_parser("validate-reference", help="Run deterministic software-reference checks")
    simulate = sub.add_parser("simulate", help="Run a mechanics simulation request JSON")
    simulate.add_argument("request")
    calibrate = sub.add_parser(
        "prepare-calibration", help="Create a CardiInfer-ready mechanics inverse problem"
    )
    calibrate.add_argument("request")
    point = sub.add_parser("material-point", help="Evaluate a constitutive point JSON")
    point.add_argument("request")
    for child in (simulate, calibrate, point):
        child.add_argument("--output", help="Write JSON atomically")
    args = parser.parse_args()
    api = MechanicsAPI()

    if args.command == "material-point":
        output = material_point(**_read_json(args.request))
    elif args.command == "doctor":
        output = api.health()
    elif args.command == "backends":
        output = api.backends()
    elif args.command == "materials":
        output = api.materials()
    elif args.command == "ecosystem":
        output = api.ecosystem()
    elif args.command == "validate-reference":
        output = api.validate_reference()
    elif args.command == "simulate":
        output = api.simulate(_read_json(args.request))
    elif args.command == "prepare-calibration":
        output = api.prepare_calibration(_read_json(args.request))
    else:  # pragma: no cover
        return 2
    if getattr(args, "output", None):
        write_json(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


def main() -> int:
    try:
        return _main()
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
