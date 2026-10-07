from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from .backends import BackendExecutionError
from .models import MechanicsSimulationRequest, MechanicsSimulationResult
from .serialization import finite_number, strict_loads


class SubprocessMechanicsBackend:
    """JSON-file adapter for user-installed spatial mechanics solvers.

    The command is never executed through a shell. Tokens may include the exact
    placeholders ``{request}`` and ``{output}``. The wrapper must return a valid
    ``MechanicsSimulationResult`` JSON document either at ``{output}`` or on stdout.
    """

    def __init__(
        self,
        *,
        name: str,
        command: str | Sequence[str],
        executable: str | None = None,
        timeout_s: float = 3600.0,
        fidelity: str = "external-spatial",
    ) -> None:
        argv = shlex.split(command) if isinstance(command, str) else list(command)
        if not name or not argv:
            raise ValueError("External backend requires a non-empty name and command")
        finite_number(timeout_s, "timeout_s", strictly_positive=True)
        self.name = name
        self.command = argv
        self.executable = executable or argv[0]
        self.timeout_s = float(timeout_s)
        self.fidelity = fidelity

    def available(self) -> bool:
        return bool(shutil.which(self.executable))

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "available": self.available(),
            "fidelity": self.fidelity,
            "spatial": True,
            "transport": "subprocess-json",
            "executable": self.executable,
            "shell": False,
        }

    def simulate(self, request: MechanicsSimulationRequest) -> MechanicsSimulationResult:
        if not self.available():
            raise BackendExecutionError(
                f"External mechanics backend executable is unavailable: {self.executable}"
            )
        with tempfile.TemporaryDirectory(prefix="cardimech-") as temporary:
            root = Path(temporary)
            request_path = root / "request.json"
            output_path = root / "result.json"
            request_path.write_text(
                json.dumps(
                    request.model_dump(mode="json"), indent=2, sort_keys=True, allow_nan=False
                )
                + "\n",
                encoding="utf-8",
            )
            argv = [
                token.replace("{request}", str(request_path)).replace("{output}", str(output_path))
                for token in self.command
            ]
            try:
                process = subprocess.run(
                    argv,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                )
            except subprocess.TimeoutExpired as exc:
                raise BackendExecutionError(
                    f"External mechanics backend timed out after {self.timeout_s:g} s"
                ) from exc
            except OSError as exc:
                raise BackendExecutionError(f"Could not execute mechanics backend: {exc}") from exc

            if process.returncode != 0:
                detail = process.stderr.strip() or process.stdout.strip()
                raise BackendExecutionError(
                    detail or f"External mechanics backend exited with code {process.returncode}"
                )
            try:
                if output_path.is_file():
                    payload = strict_loads(output_path.read_text(encoding="utf-8"))
                elif process.stdout.strip():
                    payload = strict_loads(process.stdout)
                else:
                    raise BackendExecutionError(
                        "External mechanics backend produced no result JSON"
                    )
            except (ValueError, TypeError) as exc:
                raise BackendExecutionError(
                    "External mechanics backend returned invalid JSON"
                ) from exc
            try:
                return MechanicsSimulationResult.model_validate(payload)
            except Exception as exc:
                raise BackendExecutionError(
                    f"External mechanics backend violated the result contract: {exc}"
                ) from exc
