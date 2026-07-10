# FST-Nash Validation-Evidence Ledger

**Run date:** 2026-07-10
**Decision:** diagnostic classification only; no claim upgrade.

This ledger records evidence status rather than treating an existing result file as validation.
A strong pass requires quantitative, independent evidence on all applicable external axes.

| Case | Experimental dynamics | AlphaFold baseline | Energetic-frustration baseline | Independent cycle anchor | Strong pass |
|---|---|---|---|---|---|
| XCL1 salt shift | quantitative_but_partially_circular | not_computed | not_computed | not_applicable_no_chaperone_cycle | False |
| GroEL/GroES cycle | qualitative_partial_2_of_3 | not_computed | not_computed | partial_qualitative | False |
| Hsp90/Cdc37 | parameter_partial_delta_s_assumed | not_computed | not_computed | source_parameter_only | False |
| DnaJ hold-out | convention_consistency_only | not_computed | not_computed | not_present | False |
| Hsp26 hold-out | convention_consistency_only | not_computed | not_computed | not_present | False |
| Hsp33 hold-out | convention_consistency_only | not_computed | not_computed | not_present | False |
| Hsc70 hold-out | convention_consistency_only | not_computed | not_computed | not_present | False |
| Thermosome dual model | model_choice_demonstration | not_computed | not_computed | not_independent | False |

## Summary

- Rows: 8
- Construction-conditioned rows: 8
- Computed AlphaFold baselines: 0
- Computed energetic-frustration baselines: 0
- Quantitative independent cycle anchors: 0
- Full validation passes: 0
- Claim upgrade allowed: false

## Waterline

Diagnostic classification only; one partially circular quantitative XCL1 check and one partial GroEL cycle anchor do not establish a validated mechanistic model.

The next valid progress unit is a preregistered case with a computed AlphaFold conformational baseline, a matched energetic-frustration baseline, and an independent dynamic/cycle observable not reused to set the game payoffs.
