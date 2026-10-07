from __future__ import annotations

import math
from copy import deepcopy
from typing import Any


class ForwardEnvelopeError(ValueError):
    """Raised when a CardiInfer/HeartTwin forward envelope is incomplete or unsafe."""


def _is_native_request(payload: dict[str, Any]) -> bool:
    return bool(payload.get("subject_id") and payload.get("anatomy_ref"))


def _apply_flat_parameter(request: dict[str, Any], name: str, value: float) -> None:
    parts = name.split(".")
    if len(parts) != 2 or not all(parts):
        raise ForwardEnvelopeError(
            f"Unsupported mechanics parameter path {name!r}; expected passive.*, active.*, or circulation.*"
        )
    namespace, parameter = parts
    if namespace in {"passive", "active"}:
        parameters = request.setdefault("parameters", {})
        if not isinstance(parameters, dict):
            raise ForwardEnvelopeError("forward template parameters must be an object")
        group = parameters.setdefault(namespace, {})
        if not isinstance(group, dict):
            raise ForwardEnvelopeError(f"forward template parameters.{namespace} must be an object")
        group[parameter] = float(value)
        parameters["source"] = "prior"
        return
    if namespace == "circulation":
        circulation = request.setdefault("circulation", {"enabled": True, "model": "windkessel_3e"})
        if not isinstance(circulation, dict):
            raise ForwardEnvelopeError("forward template circulation must be an object")
        values = circulation.setdefault("parameters", {})
        if not isinstance(values, dict):
            raise ForwardEnvelopeError("forward template circulation.parameters must be an object")
        values[parameter] = float(value)
        return
    raise ForwardEnvelopeError(
        f"Unsupported mechanics parameter namespace {namespace!r}; allowed: passive, active, circulation"
    )


def normalize_simulation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize either native CardiMech JSON or a CardiInfer forward envelope.

    CardiInfer intentionally sends a solver-neutral envelope with a flat parameter mapping.
    CardiMech calibration bundles put the complete native request under
    ``model_context.cardimech_request``. This adapter merges sampled parameters into that
    template without allowing them to overwrite anatomy, backend, file paths, or arbitrary
    settings.
    """
    if not isinstance(payload, dict):
        raise ForwardEnvelopeError("Mechanics payload must be an object")
    if _is_native_request(payload):
        return dict(payload)

    context = payload.get("context")
    if not isinstance(context, dict):
        raise ForwardEnvelopeError(
            "Mechanics simulation requires a native request or context.cardimech_request"
        )
    template = context.get("cardimech_request")
    if template is None:
        template = context.get("forward_template")  # compatibility with early bundles
    if not isinstance(template, dict):
        raise ForwardEnvelopeError("CardiInfer envelope is missing context.cardimech_request")

    subject_id = payload.get("subject_id", payload.get("entity_id"))
    if not isinstance(subject_id, str) or not subject_id:
        raise ForwardEnvelopeError("CardiInfer envelope requires a non-empty subject_id")
    request = deepcopy(template)
    request["subject_id"] = subject_id

    sampled = payload.get("parameters")
    if not isinstance(sampled, dict) or not sampled:
        raise ForwardEnvelopeError("CardiInfer envelope requires sampled parameters")
    for name, raw_value in sampled.items():
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ForwardEnvelopeError(f"Sampled mechanics parameter {name!r} must be numeric")
        if not math.isfinite(raw_value):
            raise ForwardEnvelopeError("Sampled mechanics parameters must be finite")
        _apply_flat_parameter(request, str(name), float(raw_value))
    return request
