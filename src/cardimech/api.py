from __future__ import annotations

from typing import Any

from .models import MechanicsSimulationRequest
from .service import CardiMechService


class MechanicsAPI:
    capabilities = ("mechanics.health", "mechanics.simulate")

    def __init__(self, service: CardiMechService | None = None) -> None:
        self.service = service or CardiMechService()

    def health(self) -> dict[str, Any]:
        return {
            "service": "CardiMech",
            "status": "ok",
            "backends": self.service.backends(),
            "capabilities": list(self.capabilities),
        }

    def simulate(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = MechanicsSimulationRequest.model_validate(payload)
        return self.service.simulate(request).model_dump(mode="json")
