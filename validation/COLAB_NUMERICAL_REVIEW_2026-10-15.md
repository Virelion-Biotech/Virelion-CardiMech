# CardiMech Colab numerical evidence — 2026-10-15

The notebook completed with return code zero in 2.272 seconds and status `NUMERICAL_ONLY`. All 14 unique archive members are retained in [`colab_numerical_2026_10_15/original`](colab_numerical_2026_10_15/original/); all 13 original artifact hashes match. Launcher hashes and the canonical protocol identity also match. Supplied launchers are archived as inert `.py.txt` files and were not executed during this review. The notebook calls `validate_spatial.verify()` directly through its numerical handler; an empty generic `jobs` list does not mean the shell verification was omitted.

The 64-file snapshot matches pinned source `69c02886a24f9faa989b4c3c56c0ea26eea05d69`; 29 current implementation/harness files match the uploaded run. The inputs are manufactured in code, consistent with empty download receipts. No independent patient dataset was used.

## Reproduction and interpretation

The small-pressure thick-shell solve converges on the tested refinements: inner-displacement relative errors are 49.19%, 20.12% and 6.22%. Force residuals are below 1e-15 N. The analytic inner displacement is 8.78947e-7 m at pressure 0.1 Pa, mu=1000 Pa and lambda=1000 Pa.

The repository harness was rerun locally on CPU. Numerical result fields reproduce within relative tolerance 1e-7 and absolute tolerance 1e-10; environment-version fields differ and are retained explicitly in the fresh reports. Pass/fail flags and source hashes agree. Original and reproduced reports are separate. `REVIEW.json` identifies each pair, and `PAYLOAD_SHA256.json` covers all retained evidence.

89 local tests passed, with 87.55% statement coverage. The local CardiMech suite skips its optional CardiInfer integration module because CardiInfer is not installed in this review environment; the hosted workflow has a dedicated pinned integration job. These checks do not establish empirical or clinical validation.

## Decision and remaining scientific work

No demonstrated algorithmic defect was identified in this export. Algorithms, parameters, solver settings and acceptance thresholds remain unchanged. There is no notebook execution failure to repair.

- A small residual establishes equilibrium of the discrete equations; it does not remove the 6.22% remaining discretization/model-limit error.
- The report acceptance gate is monotone reduction plus finest relative error below 15%. Passing this gate does not mean high accuracy at every mesh resolution.
- This is a compressible, small-strain synthetic sphere comparison with slip constraints. It is not near-incompressible myocardial qualification, the full Land benchmark, or a patient deformation/pressure comparison.
- Finer independently checked meshes, near-incompressibility/locking tests, finite-strain material benchmarks and measured strain/pressure data are needed for broader mechanics claims. No constitutive parameters or tolerances were tuned against this uploaded run.

Reproduce from an installed checkout with the corresponding optional numerical dependencies:

```bash
python scripts/validate_spatial.py
```

The existing pinned notebook remains a valid record of its source revision. A passing numerical benchmark does not establish model adequacy or patient accuracy.
