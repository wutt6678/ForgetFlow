# Multi-Target Smoke Study Summary

- **Status**: DIAGNOSTIC
- **Commit**: 27313bd23b2ee8edcd5b237074f667cf990beb6a-dirty
- **Mode**: diagnostic
- **Run mode**: test
- **Fixtures**: 2
- **Seeds**: 3
- **Conditions**: 5
- **Total runs**: 30
- **Audit valid**: False
- **Manifest valid**: False
- **Utility valid**: True
- **Conditions valid**: False

## Assertions

- [PASS] **F001_exposure_independent_of_F002**: F001-only=39, F002-only=18, both=0
- [PASS] **positive_F001_only_exposure**: F001-only exposure turns: 39
- [PASS] **positive_F002_only_exposure**: F002-only exposure turns: 18
- [FAIL] **positive_combined_exposure**: Combined F001+F002 exposure turns: 0
- [PASS] **tracker_state_per_agent_record_pair**: Validated 30 agent-record state pairs
- [PASS] **expected_tracker_pairs_present**: Expected (CK,F001) and (CK,F002) pairs found in all results
- [PASS] **state_isolation**: F001-only=21, F002-only=12, combined=0, mismatches=0
- [PASS] **protected_unprotected_outcome_symmetry**: compared_turn_count=63, mismatch_count=0
- [PASS] **reintroduced_subset_of_exposed**: All reintroduced IDs are subsets of exposed and targeted IDs
- [FAIL] **positive_F001_reconstruction**: F001-only reconstruction turns: 0
- [FAIL] **positive_F002_reconstruction**: F002-only reconstruction turns: 0 (both=0)
- [PASS] **reconstructed_ids_record_specific**: All reconstructed IDs are F001 or F002
- [PASS] **rr_denominator_positive**: RR denominator=39, numerator=3
- [PASS] **rr_numerator_le_denominator**: RR=3/39
- [PASS] **crr_numerator_le_denominator**: CRR=0/60
- [PASS] **rr_clean_denominator_positive**: RR_clean denominator=39, numerator=3
- [PASS] **rr_clean_numerator_le_denominator**: RR_clean=3/39
- [PASS] **multi_target_has_multiple_items**: Scenario has 2 sensitive items
- [PASS] **multi_target_has_recontamination_steps**: Recontamination steps: 3
- [FAIL] **multi_target_audit_valid**: Audit errors: 48
- [FAIL] **protected_unexpected_recontamination_zero**: Unexpected by condition: {'binary_policy': 3, 'no_firewall': 0, 'exact_only': 6, 'monitoring_0': 3, 'full_mvp': 0}
- [PASS] **disk_metrics_match_in_memory**: Verified 30 results match across disk round-trip
- [PASS] **disk_record_level_fields**: All record-level fields match across 30 results

## Metrics

| Metric | Value | Numerator | Denominator |
|--------|------:|----------:|------------:|
| PU-RER | 0.0 | 0 | 270 |
| CRR | 0.0 | 0 | 60 |
| RR | 0.07692307692307693 | 3 | 39 |
| RR_clean | 0.07692307692307693 | 3 | 39 |
| RR_at_risk | 0.42857142857142855 | 9 | 21 |
| FBR | 0.0 | 0 | 60 |
