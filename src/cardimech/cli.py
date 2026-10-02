from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import MechanicsAPI


def _read_json(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Input JSON must be an object")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(prog="cardimech")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Report package and backend availability")
    sub.add_parser("backends", help="List mechanics backends")
    sub.add_parser("materials", help="List constitutive material families")
    sub.add_parser("ecosystem", help="Show audited external ecosystem map")
    sub.add_parser("validate-reference", help="Run deterministic software-reference checks")
    simulate = sub.add_parser("simulate", help="Run a mechanics simulation request JSON")
    simulate.add_argument("request")
    calibrate = sub.add_parser("prepare-calibration", help="Create a CardiInfer-ready mechanics inverse problem")
    calibrate.add_argument("request")
    args = parser.parse_args()
    api = MechanicsAPI()

    if args.command == "doctor":
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
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0
