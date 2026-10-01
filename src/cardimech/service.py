from __future__ import annotations

from .backends import BackendUnavailable, MechanicsBackend
from .models import MechanicsSimulationRequest, MechanicsSimulationResult


class ReadinessError(RuntimeError):
    pass


class CardiMechService:
    def __init__(self) -> None:
        self._backends: dict[str, MechanicsBackend] = {}

    def register_backend(self, backend: MechanicsBackend) -> None:
        self._backends[backend.name] = backend

    def backends(self) -> list[str]:
        return sorted(self._backends)

    def _backend(self, name: str) -> MechanicsBackend:
        backend = self._backends.get(name)
        if backend is None or not backend.available():
            raise BackendUnavailable(f"CardiMech backend unavailable: {name}")
        return backend

    def simulate(self, request: MechanicsSimulationRequest) -> MechanicsSimulationResult:
        result = self._backend(request.backend).simulate(request)
        if result.subject_id != request.subject_id:
            raise ReadinessError("Backend returned mechanics for a different subject")
        if result.qc is not None and not result.qc.passed:
            raise ReadinessError("Mechanics result failed QC")
        return result
