from __future__ import annotations

from .backends import BackendUnavailable, MechanicsBackend, discover_plugin_backends
from .calibration import prepare_calibration
from .models import (
    MechanicsCalibrationBundle,
    MechanicsCalibrationRequest,
    MechanicsSimulationRequest,
    MechanicsSimulationResult,
)
from .reference_backend import ReferenceLumpedBackend


class ReadinessError(RuntimeError):
    pass


class CardiMechService:
    def __init__(self, *, load_plugins: bool = True) -> None:
        self._backends: dict[str, MechanicsBackend] = {}
        self.register_backend(ReferenceLumpedBackend())
        if load_plugins:
            for backend in discover_plugin_backends():
                self.register_backend(backend)

    def register_backend(self, backend: MechanicsBackend) -> None:
        if not getattr(backend, "name", ""):
            raise ValueError("Mechanics backends require a non-empty name")
        self._backends[backend.name] = backend

    def backends(self) -> list[str]:
        return sorted(self._backends)

    def backend_status(self) -> list[dict[str, object]]:
        status = []
        for name in self.backends():
            backend = self._backends[name]
            details = dict(backend.describe())
            details["available"] = bool(backend.available())
            details.setdefault("name", name)
            status.append(details)
        return status

    def _backend(self, name: str) -> MechanicsBackend:
        backend = self._backends.get(name)
        if backend is None or not backend.available():
            raise BackendUnavailable(f"CardiMech backend unavailable: {name}")
        return backend

    def simulate(self, request: MechanicsSimulationRequest) -> MechanicsSimulationResult:
        result = self._backend(request.backend).simulate(request)
        if result.subject_id != request.subject_id:
            raise ReadinessError("Backend returned mechanics for a different subject")
        if result.backend != request.backend:
            raise ReadinessError("Backend result identifier does not match request backend")
        if result.qc is not None and not result.qc.passed:
            raise ReadinessError("Mechanics result failed QC")

        result.provenance.setdefault("anatomy_artifact_id", request.anatomy_ref.artifact_id)
        if request.anatomy_ref.sha256 is not None:
            result.provenance.setdefault("anatomy_sha256", request.anatomy_ref.sha256)
        bundle_fingerprint = request.anatomy_ref.metadata.get("bundle_fingerprint")
        if bundle_fingerprint is not None:
            result.provenance.setdefault(
                "anatomy_bundle_fingerprint", str(bundle_fingerprint)
            )
        if request.activation_ref is not None:
            result.provenance.setdefault(
                "activation_artifact_id", request.activation_ref.artifact_id
            )
            if request.activation_ref.sha256 is not None:
                result.provenance.setdefault(
                    "activation_sha256", request.activation_ref.sha256
                )
        return result

    def prepare_calibration(self, request: MechanicsCalibrationRequest) -> MechanicsCalibrationBundle:
        self._backend(request.backend)
        return prepare_calibration(request)
