from __future__ import annotations

from typing import Protocol

from .models import MechanicsSimulationRequest, MechanicsSimulationResult


class MechanicsBackend(Protocol):
    name: str

    def available(self) -> bool: ...

    def simulate(self, request: MechanicsSimulationRequest) -> MechanicsSimulationResult: ...


class BackendUnavailable(RuntimeError):
    pass
