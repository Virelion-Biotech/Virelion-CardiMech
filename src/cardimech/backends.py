from __future__ import annotations

from importlib.metadata import entry_points
from typing import Protocol

from .models import MechanicsSimulationRequest, MechanicsSimulationResult


class MechanicsBackend(Protocol):
    name: str

    def available(self) -> bool: ...

    def describe(self) -> dict[str, object]: ...

    def simulate(self, request: MechanicsSimulationRequest) -> MechanicsSimulationResult: ...


class BackendUnavailable(RuntimeError):
    pass


class BackendExecutionError(RuntimeError):
    pass


def discover_plugin_backends() -> list[MechanicsBackend]:
    discovered: list[MechanicsBackend] = []
    try:
        candidates = entry_points(group="cardimech.backends")
    except TypeError:  # pragma: no cover - older importlib metadata API
        candidates = entry_points().get("cardimech.backends", [])
    for item in candidates:
        backend = item.load()()
        discovered.append(backend)
    return discovered
