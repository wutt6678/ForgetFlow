# Multi-Target Smoke Study Summary

- **Status**: DIAGNOSTIC
- **Commit**: 86bfb521f1b6ffc39079b7078c3d9fa0497d401d-dirty
- **Mode**: diagnostic
- **Run mode**: test
- **Fixtures**: 1
- **Seeds**: 3
- **Conditions**: 5
- **Total runs**: 15
- **Audit valid**: True
- **Manifest valid**: True

## Assertions

- [PASS] **F001_exposure_independent_of_F002**: F001-only=27, F002-only=24, both=6
- [PASS] **positive_F001_only_exposure**: F001-only exposure turns: 27
- [PASS] **positive_F002_only_exposure**: F002-only exposure turns: 24
- [PASS] **positive_combined_exposure**: Combined F001+F002 exposure turns: 6
- [PASS] **tracker_state_per_agent_record_pair**: Validated 15 agent-record state pairs
- [PASS] **protected_unprotected_outcome_symmetry**: Compared 6 identical-text turn pairs
- [PASS] **reintroduced_subset_of_exposed**: All reintroduced IDs are subsets of exposed IDs
- [PASS] **positive_F001_reconstruction**: F001-only reconstruction turns: 12
- [PASS] **positive_F002_reconstruction**: F002 reconstruction turns: 12 (F002-only=0, both=12)
- [PASS] **reconstructed_ids_record_specific**: All reconstructed IDs are F001 or F002
- [PASS] **rr_denominator_positive**: RR denominator=30, numerator=12
- [PASS] **rr_numerator_le_denominator**: RR=12/30
- [PASS] **crr_numerator_le_denominator**: CRR=24/60
- [PASS] **multi_target_has_multiple_items**: Scenario has 2 sensitive items
- [PASS] **multi_target_has_recontamination_steps**: Recontamination steps: 3
- [PASS] **multi_target_audit_valid**: Audit errors: 0
- [PASS] **disk_metrics_match_in_memory**: Verified 15 results match across disk round-trip

## Metrics

| Metric | Value | Numerator | Denominator |
|--------|------:|----------:|------------:|
| PU-RER | 0.31666666666666665 | 57 | 180 |
| CRR | 0.4 | 24 | 60 |
| RR | 0.4 | 12 | 30 |
| FBR | 0.0 | 0 | 30 |
