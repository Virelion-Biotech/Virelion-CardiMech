from __future__ import annotations

from typing import Any

from .external import ecosystem_manifest
from .forward import normalize_simulation_payload
from .materials import material_catalog
from .models import MechanicsCalibrationRequest, MechanicsSimulationRequest
from .service import CardiMechService
from .validation import run_reference_validation


class MechanicsAPI:
    capabilities = (
        "mechanics.health",
        "mechanics.backends",
        "mechanics.materials",
        "mechanics.simulate",
        "mechanics.prepare_calibration",
        "mechanics.validate.reference",
        "mechanics.ecosystem",
    )

    def __init__(self, service: CardiMechService | None = None) -> None:
        self.service = service or CardiMechService()

    def health(self) -> dict[str, Any]:
        statuses = self.service.backend_status()
        return {
            "service": "CardiMech",
            "status": "ok" if any(item["available"] for item in statuses) else "degraded",
            "contract_version": "2.0",
            "backends": statuses,
            "capabilities": list(self.capabilities),
            "scientific_status": "research software; no clinical-device claim",
        }

    def backends(self) -> dict[str, Any]:
        return {"backends": self.service.backend_status()}

    def materials(self) -> dict[str, Any]:
        return {"materials": [item.__dict__ for item in material_catalog()]}

    def ecosystem(self) -> dict[str, Any]:
        return {
            "external_projects": ecosystem_manifest(),
            "policy": (
                "CardiMech borrows compatible architecture and integrates heavyweight solvers "
                "through plugins/process boundaries; third-party code is not silently vendored."
            ),
        }

    def validate_reference(self) -> dict[str, Any]:
        return run_reference_validation()

    def simulate(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = MechanicsSimulationRequest.model_validate(normalize_simulation_payload(payload))
        return self.service.simulate(request).model_dump(mode="json")

    def prepare_calibration(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = MechanicsCalibrationRequest.model_validate(payload)
        return self.service.prepare_calibration(request).model_dump(mode="json")
