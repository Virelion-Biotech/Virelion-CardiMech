from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExternalProject:
    project: str
    repository: str
    license: str
    role: tuple[str, ...]
    policy: str


EXTERNAL_ECOSYSTEM = (
    ExternalProject(
        "fenicsx-pulse",
        "finsberg/fenicsx-pulse",
        "MIT",
        ("finite-element mechanics", "constitutive laws", "FEniCSx"),
        "preferred optional high-fidelity mechanics adapter",
    ),
    ExternalProject(
        "pulse",
        "finsberg/pulse",
        "LGPL-3.0",
        ("legacy FEniCS mechanics", "continuum mechanics"),
        "reference architecture; prefer fenicsx-pulse for new deployments",
    ),
    ExternalProject(
        "simcardems / simcardems2",
        "ComputationalPhysiology/simcardems2",
        "MIT",
        ("electromechanics", "EP-mechanics coupling"),
        "reference coupling architecture; optional external backend",
    ),
    ExternalProject(
        "Ambit",
        "marchirschvogel/ambit",
        "MIT",
        ("multiphysics", "solid mechanics", "FSI", "0D circulation"),
        "optional external backend; concepts mirrored in solver-neutral contracts",
    ),
    ExternalProject(
        "CardioMechanics",
        "KIT-IBT/CardioMechanics",
        "GPL-3.0",
        ("electromechanics", "unloading", "parameter optimization", "circulation"),
        "external-process/plugin integration; no silent vendoring",
    ),
    ExternalProject(
        "Chaste",
        "Chaste/Chaste",
        "BSD-3-Clause",
        ("electromechanics", "material laws", "verification tests"),
        "architecture and verification reference; optional external integration",
    ),
    ExternalProject(
        "ModularCirc",
        "alan-turing-institute/ModularCirc",
        "MIT",
        ("0D circulation", "chambers", "valves", "vessels"),
        "reference for componentized circulation; optional future adapter",
    ),
    ExternalProject(
        "cardiac_benchmark",
        "finsberg/cardiac_benchmark",
        "MIT",
        ("cardiac elastodynamics", "verification", "benchmark fixtures"),
        "verification target; do not treat agreement as empirical validation",
    ),
    ExternalProject(
        "cardiac-geometriesx",
        "ComputationalPhysiology/cardiac-geometriesx",
        "MIT",
        ("cardiac geometry", "FEniCSx mesh fixtures"),
        "geometry remains owned by CardiAnatomy; useful solver-side adapter",
    ),
    ExternalProject(
        "fenicsx-ldrb",
        "finsberg/fenicsx-ldrb",
        "MIT",
        ("myocardial fibers", "rule-based microstructure"),
        "microstructure remains owned by CardiAnatomy; optional backend dependency",
    ),
)


def ecosystem_manifest() -> list[dict[str, object]]:
    return [asdict(item) for item in EXTERNAL_ECOSYSTEM]
