# CardiMech architecture

CardiMech is the mechanics/electromechanics layer between personalized anatomy, electrophysiology, circulation, inference, and HeartTwin.

```text
CardiAnatomy
 mesh + fibers + regions
          |
          +---- CardiEP activation / timing
          |              |
          v              v
      mechanics domain + parameters
                    |
          boundary/loading conditions
                    |
          registered mechanics backend
                    |
        displacement / strain / stress
                    |
       pressure-volume / chamber metrics
                    |
          optional 0D circulation
                    |
             CardiInfer / HeartTwin
```

## Responsibility boundary

CardiMech owns myocardial mechanics execution and mechanics-specific contracts. CardiAnatomy owns geometry and coordinate frames. CardiEP owns electrical activation. CardiInfer owns posterior inference and uncertainty. CardiFlow owns detailed flow/CFD when introduced.

The first circulation target should be a lightweight 0D coupling contract. Full chamber/vascular CFD belongs in CardiFlow rather than CardiMech.

## Backend roadmap

Potential backend families include finite-element passive mechanics, active-tension models, electromechanical solvers, and FEniCS/FEniCSx-based implementations. Heavy dependencies should stay optional and solver-specific.

## Validation ladder

1. Contract/software checks.
2. Mesh/material/boundary-condition consistency.
3. Numerical convergence and benchmark problems.
4. Synthetic parameter recovery through CardiInfer.
5. Held-out imaging/hemodynamic agreement.
6. External patient/cohort validation.

Numerical convergence alone does not establish physiological correctness.
