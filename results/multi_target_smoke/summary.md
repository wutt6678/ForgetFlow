# Multi-Target Smoke Study Summary

- **Status**: DIAGNOSTIC
- **Commit**: 79e18d6356f9247e68280e06ee28f16757eafe8a
- **Mode**: diagnostic
- **Run mode**: test
- **Fixtures**: 2
- **Seeds**: 3
- **Conditions**: 5
- **Total runs**: 30
- **Audit valid**: True
- **Manifest valid**: True
- **Utility valid**: True
- **Conditions valid**: True

## Assertions

- [PASS] **F001_exposure_independent_of_F002**: F001-only=27, F002-only=30, both=12
- [PASS] **positive_F001_only_exposure**: F001-only exposure turns: 27
- [PASS] **positive_F002_only_exposure**: F002-only exposure turns: 30
- [PASS] **positive_combined_exposure**: Combined F001+F002 exposure turns: 12
- [PASS] **tracker_state_per_agent_record_pair**: Validated 30 agent-record state pairs
- [PASS] **expected_tracker_pairs_present**: Expected (CK,F001) and (CK,F002) pairs found in all results
- [PASS] **state_isolation**: F001-only=18, F002-only=21, combined=6, mismatches=0
- [PASS] **protected_unprotected_outcome_symmetry**: compared_turn_count=63, mismatch_count=0
- [PASS] **reintroduced_subset_of_exposed**: All reintroduced IDs are subsets of exposed and targeted IDs
- [PASS] **positive_F001_reconstruction**: F001-only reconstruction turns: 12
- [PASS] **positive_F002_reconstruction**: F002-only reconstruction turns: 12 (both=24)
- [PASS] **reconstructed_ids_record_specific**: All reconstructed IDs are F001 or F002
- [PASS] **rr_denominator_positive**: RR denominator=39, numerator=12
- [PASS] **rr_numerator_le_denominator**: RR=12/39
- [PASS] **crr_numerator_le_denominator**: CRR=48/240
- [PASS] **rr_clean_denominator_positive**: RR_clean denominator=39, numerator=12
- [PASS] **rr_clean_numerator_le_denominator**: RR_clean=12/39
- [PASS] **multi_target_has_multiple_items**: Scenario has 2 sensitive items
- [PASS] **multi_target_has_recontamination_steps**: Recontamination steps: 3
- [PASS] **multi_target_audit_valid**: Audit errors: 0
- [FAIL] **protected_unexpected_recontamination_zero**: Unexpected by condition: {'no_firewall': 0, 'monitoring_0': 0, 'exact_only': 6, 'full_mvp': 0, 'binary_policy': 3}
- [FAIL] **disk_metrics_match_in_memory**: Mismatched aggregate metrics: ['pu_rer']
- [PASS] **disk_record_level_fields**: All record-level fields match across 30 results

## Metrics

| Metric | Value | Numerator | Denominator |
|--------|------:|----------:|------------:|
| PU-RER | 0.0 | 0 | 270 |
| CRR | 0.2 | 48 | 240 |
| RR | 0.3076923076923077 | 12 | 39 |
| RR_clean | 0.3076923076923077 | 12 | 39 |
| RR_at_risk | 0.5714285714285714 | 12 | 21 |
| FBR | 0.0 | 0 | 60 |
