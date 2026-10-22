# MINI assembly cost follow-up — candidate, not yet qualified

The merged locking correction measured about 193 seconds for the finest
7,680-tetrahedron case in the original Colab campaign. This follow-up targets
the cost of contracting its neo-Hookean displacement tangent. It does not
change the approximation space, quadrature, pressure constraint, condensation,
Newton tolerances or accuracy screen. No speedup has yet been measured.

For `H_ai = F^-T_ij G_aj`, the exact displacement block integrand is

```
mu delta_ik (G_a.G_b) + (mu-pJ) H_bi H_ak + pJ H_ai H_bk
  + T delta_ik (G_a.f) (G_b.f).
```

The candidate contracts these terms directly instead of constructing the
quadrature-wise fourth-order material tensor. The Guccione path retains its
full material tensor. Prescribed active fibre tension is included explicitly.
The public material function retains its full tangent for independent checks.

## Required checks before merge

- Six randomized, nonaffine full-tensor comparisons: quadrature orders 5–7,
  zero/nonzero active tension, signed pressure, rotated fibres.
- Existing independent finite-difference full assembly/Schur recovery test,
  pressure-mode test and both-material finite affine patches.
- Entire CardiMech suite in an isolated Edge Colab environment.
- `scripts/compare_mini_assembly.py`: same-process reference/candidate assembly
  comparison at three shell resolutions, including one active Guccione case;
  tangent, residual, condensed residual, recovered bubbles and every field.
- Unchanged 23-solve `scripts/verify_mixed_locking.py` campaign. All eight frozen
  acceptance gates must pass, with coarse accuracy failures still retained.
- Latest-head CI and stack compatibility review. Do not advance HeartTwin's
  pinned dependency until this candidate is qualified.

## Attempt log, 2026-10-22

Edge Colab connected and cloned frozen reference
`b2dacf79a8d0cdb93fbd34e5d78b19d4ee06d534`. No patient data was used.

1. Standard `python -m venv` failed: the Colab Python 3.13 installation lacks
   `ensurepip`. This is an environment bootstrap failure before solver import.
   `virtualenv` from PyPI successfully seeded pip and installed the package in
   an isolated environment. The replay uses that bootstrap directly.
2. The initial diagnostic was named `profile.py`, shadowing Python's standard
   `profile` module and causing a circular import in `cProfile`. This was a
   diagnostic harness naming error, not a mechanics failure. A rename to
   `assembly_profile.py` was submitted, but browser control timed out before
   its completion could be observed.
3. Two consecutive browser requests timed out. No profile, candidate numerical
   check or timing result was retrieved. The candidate remains unvalidated;
   the merged solver and HeartTwin pin are unchanged.
4. A subsequent browser retry was explicitly blocked because the originating
   Edge browser context was unavailable. Static replay review found an evidence
   export defect: required setup errors aborted before packaging, and the final
   failure assertion prevented offering the archive after a failed regression.
   The replay now captures setup exceptions, packages partial logs and the
   traceback, and offers the archive before raising its final failure gate.
   This is a diagnostic/reproducibility fix; no new solver result is implied.

The original [qualified results](MIXED_MINI_RESULTS.md) still apply only to the
merged implementation. Coarse mesh errors, direct-solver capacity and patient
accuracy limitations remain unresolved by this candidate.
