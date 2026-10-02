# HeartTwin integration

CardiMech advertises:

- `mechanics.health`
- `mechanics.backends`
- `mechanics.materials`
- `mechanics.simulate`
- `mechanics.prepare_calibration`
- `mechanics.validate.reference`
- `mechanics.ecosystem`

## Data flow

```text
CardiAnatomy -- mechanics-ready mesh/fibres --> CardiMech
CardiEP ----- activation artifact ----------> CardiMech
CMR/Echo/MyoTrace/hemodynamics ------------> MechanicsObservation[]
CardiMech --- predicted mechanics ----------> HeartTwin / CardiEval
CardiMech --- calibration bundle -----------> CardiInfer
CardiInfer -- calibrated parameter set -----> CardiMech
CardiTrace <- provenance -------------------- HeartTwin
```

HeartTwin should preserve backend name, parameter source, anatomy and activation references, QC, validation status, output hashes, and solver provenance.

The built-in reduced-order backend may be used in clean-stack CI without spatial geometry. For scientific spatial mechanics, HeartTwin should first call CardiAnatomy readiness validation with `target="mechanics"` and then select an installed spatial backend.

## Native registry entry

```yaml
- name: CardiMech
  repository: Virelion-Biotech/Virelion-CardiMech
  capabilities: [mechanics.health, mechanics.backends, mechanics.materials, mechanics.simulate, mechanics.prepare_calibration, mechanics.validate.reference, mechanics.ecosystem]
  builtin: cardimech
  endpoint: ${CARDIMECH_URL}
```

## Inference boundary

CardiMech's calibration endpoint creates a problem definition; it does not fit a posterior. CardiInfer is the single inverse-problem layer for ABC-SMC, MCMC, MAP, sensitivity, identifiability, and uncertainty propagation.
