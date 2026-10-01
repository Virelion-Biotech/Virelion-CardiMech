# HeartTwin integration

Initial capabilities:

- `mechanics.health`
- `mechanics.simulate`

HeartTwin should pass:
- a ready CardiAnatomy geometry artifact;
- explicit mechanics parameters or CardiInfer posterior-derived parameters;
- boundary/loading conditions;
- optional CardiEP activation artifact;
- optional 0D circulation state.

## Proposed registry entry

```yaml
- name: CardiMech
  repository: Virelion-Biotech/Virelion-CardiMech
  capabilities: [mechanics.health, mechanics.simulate]
  builtin: cardimech
  endpoint: ${CARDIMECH_URL}
```

HeartTwin should retain `MechanicsSimulationResult` as a typed artifact and preserve backend, parameter source, anatomy/activation references, QC, validation status, and provenance.

Mechanics calibration belongs in CardiInfer rather than a second statistical framework inside CardiMech.
