# Bounded spatial mechanics assembly

The native scipy-tetra-v1 backend now assembles constitutive tensors and element stiffness matrices in batches of at most 512 tetrahedra. Follower-pressure assembly is similarly batched. Newton line searches evaluate residuals without constructing unused constitutive or global tangent matrices. Accepted-state fields and equilibrium tangents retain their existing definitions.

The public constitutive and assembly functions keep their default tangent-producing behavior. Optional compute_tangent=False returns a null tangent while retaining energy, stress, deformation determinants and inversion/nonfinite checks. Assembly chunk_size accepts integers 1..4096 for numerical verification; the service uses a fixed 512-element batch. No new user-facing solver setting or implicit material assumption is introduced.

## Numerical checks
The complete CardiMech suite passed **100 tests** in the user's Edge Colab, including the CardiInfer integration test. Existing material derivative/objectivity, affine patch, follower-load derivative, active equilibrium and shell-refinement checks passed. New tests compare chunk boundaries and residual-only results for neo-Hookean and Guccione materials, preserve inversion rejection and enforce the existing node limit.

A separate synthetic shell stress comparison used 3,078 nodes and 12,288 tetrahedra, fully inside the existing guards. It compared the pre-change assembly from commit 5dc14d886ccbda5a1398b863f9ae7821bbffa18b with the new implementation using the same validated mesh, small deterministic perturbation, passive parameters and prescribed active tension/pressure.

| Material | Legacy peak traced bytes | Bounded peak traced bytes | Largest tangent difference | Residual/field differences |
|---|---:|---:|---:|---:|
| Neo-Hookean | 94,367,579 | 13,953,218 | 2.84217e-14 | 0 |
| Guccione | 94,352,092 | 13,953,448 | 4.26326e-14 | 0 |

Peak traced allocation fell by about **85.2%** in this single Colab comparison. This measures allocations tracked by tracemalloc during assembly, not total process RSS, solver-factorization memory, or a general hardware benchmark. Raw timings and the exact synthetic generator are retained. The reference assembly uses the unchanged default material equations; existing finite-difference tests provide a separate mathematical check.

## Scope and limits
The 5,000-node / 20,000-tetrahedron loader limits remain enforced. This change does **not** qualify the full KIT mesh, provide an unloaded patient reference state, resolve near-incompressible P1 locking, or establish patient accuracy. Large-domain solving still needs a qualified scalable linear-solver strategy and convergence/memory evidence beyond these guards.

Initial Colab setup failures (ensurepip/venv, inherited Ruff binary location and missing CardiInfer) and the passing rerun are retained. A HeartTwin regression initially reported mechanics.simulate unavailable because the isolated bin directory was absent from PATH; correcting the harness restored discovery without weakening any product availability check.

Implementation verified in Colab: 47753cb5141f07c27b6cc64d64aaab11f7630e48. Later documentation/evidence commits do not change the numerical implementation.

## Retained evidence
[synthetic_stack_completion_evidence.zip](../validation/synthetic_completion_2026_10_21/synthetic_stack_completion_evidence.zip) contains raw JUnit/logs, initial failures, final exact-source receipts and the synthetic stress generator/results. SHA-256: `689ed652c865a40e7fd8c4067ccb8f9c96066d43213ffdc51b5568bbc91f8275`. Both repository uploads matched the Colab-computed Git blob `8e7faee23a99e11ef25bb3fb08a243881652ecb4`.
