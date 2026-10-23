# Exact MINI assembly contraction — executed Colab results

On 2026-10-23 the pinned replay completed in Edge Colab: **152 tests passed,
zero failures/errors/skips, 89.43% coverage; all 23 equilibrium solves converged
and all eight unchanged locking-qualification gates passed**.

Executed code: `584785c32caadd3c6bc42fef712b3762789add85`.
Executed public notebook: `7a3ae5c11331c1238f023b0e51e11e147158f2fb`.
The comparison reference is the merged locking correction,
`b2dacf79a8d0cdb93fbd34e5d78b19d4ee06d534`.

## Paired assembly comparison

Three repeats per implementation, alternating execution order in the same
process/runtime; active fibre tension, nonzero signed nodal pressure, nonzero
interior bubbles and follower pressure. These are assembly probes, not paired
whole-solver timing measurements. No universal speedup or peak-memory claim is
made. The unchanged Guccione path is a control.

| Material | Tetrahedra | Reference median (s) | Candidate median (s) | Speedup | Relative tangent difference |
|---|---:|---:|---:|---:|---:|
| Neo-Hookean | 288 | 2.153 | 0.265 | 8.13x | 3.17e-16 |
| Neo-Hookean | 1,536 | 12.232 | 2.330 | 5.25x | 2.93e-16 |
| Neo-Hookean | 7,680 | 62.465 | 8.381 | 7.45x | 2.79e-16 |
| Guccione | 288 | 2.851 | 2.844 | 1.00x | 0 |

All global tangent, full residual, bubble residual, condensed residual, pressure
constraint, recovered bubble increment, minimum J and every output-field
comparison passed. Six additional randomized full-tensor tests cover quadrature
orders 5–7, active/inactive fibres, rotated fibres and signed pressure. Existing
independent finite-difference condensation/recovery, pressure-mode and both-material
finite patch tests also passed.

The implementation contracts the exact neo-Hookean tangent directly, avoiding
the quadrature-wise fourth-order tensor. It preserves prescribed active fibre
tension and the original pressure dual, approximation space, quadrature,
condensation, convergence residual and tolerances.
[Derivation and attempt log](MIXED_MINI_ASSEMBLY_COST.md).

## Accuracy retained, failures retained

The unchanged 23-solve campaign passed all eight frozen gates. All three finest
MINI regimes remain within the original 15% research screen, and error decreases
under refinement. Finest displacement errors are 1.9766% (lambda=1,000 Pa),
2.5162% (1e5 Pa) and **2.5347% (1e7 Pa)**. The original P1 near-incompressible
case still has **80.1939%** error. Three coarse MINI cases still fail the screen;
they remain in the qualification JSON and raw evidence.

Quadrature, lower-load, higher-bulk and finite-load checks pass. The 100 Pa
finite-load pair establishes equilibrium/quadrature consistency, not nonlinear
continuum accuracy against an exact reference: the listed analytical comparison
is a small-strain solution. The finest MINI solve at lambda=1e7 took 37.53 s in
this replay. The earlier 193 s whole-solve observation used another runtime, so
their ratio is not a controlled whole-solver speedup measurement.

## Evidence

[Evidence directory](../validation/mini_assembly_2026_10_23) contains the compact
archive, full manifest, all qualification/comparison records, JUnit, receipts,
executed notebook. The executed notebook explicitly identifies main output
reconstructed from retained stage logs; transient transfer/display-only cells
are omitted. Numerical code and criteria are unchanged.

Compact ZIP: **2,351,471 bytes**, SHA-256
`70939b72de4271c92728f85633a4a0952276ccf4e3d1425d06bc634500fdb844`.
Its 118 entries include all **23 raw states**, complete logs, dependency versions
and source copies. Source SHA-256 values match the published candidate. ZIP
integrity, SHA-256 and Git blob identity were verified after transfer.

Full mesh/element-field JSON is omitted from the compact archive, with hashes
retained and raw coordinates, connectivity, nodal displacement, pressure/bubble
unknowns where present and element J included for regeneration. A full-field
archive was also generated and offered as a browser download: 21,730,402 bytes,
SHA-256 `954ab046fbfadbb2e7ca22b07194feb92caacecbb2adb61730b1ec16267832cb`.
Browser download completion was not independently verified; that larger archive
is not committed here.

After successful numerical execution and archive transfer, Colab reported that
JavaScript files needed for a revised HTML output could not load, suggesting
expired account access or third-party-cookie restrictions. Those possible causes
were not independently established. Subsequent browser reads stalled; a useful
final screenshot could not be retained. This is an unresolved display/connection
limitation, not a failed mechanics solve. No cookie or security settings were
changed. Earlier setup/import/browser failures remain documented. A truncated local copy
of the prior 2026-10-22 compact archive was recovered from the intact committed
Git object, verified against its original SHA-256, with the partial copy retained
separately; no published prior result was changed.

## Remaining limits

This qualifies the tested synthetic regression and measured assembly probes.
It does not add scalable mixed preconditioning, a matrix-free mixed solver,
exact incompressibility, broad distorted/interface-mesh qualification, whole-heart
physiology or patient/clinical accuracy. Standard/direct mixed controls and mesh
guards remain. The three coarse accuracy failures remain real failures.
