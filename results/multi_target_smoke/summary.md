# Multi-Target Smoke Study Summary

- **Status**: DIAGNOSTIC
- **Commit**: f480763e5d1e834126cda1da598f7aa98f7a9050
- **Execution status**: RESEARCH_VALID
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

- [PASS] **F001_exposure_independent_of_F002**: F001-only=24, F002-only=18, both=3
- [PASS] **positive_F001_only_exposure**: F001-only exposure turns: 24
- [PASS] **positive_F002_only_exposure**: F002-only exposure turns: 18
- [PASS] **positive_combined_exposure**: Combined F001+F002 exposure turns: 3
- [PASS] **tracker_state_per_agent_record_pair**: Validated 30 agent-record state pairs
- [PASS] **expected_tracker_pairs_present**: Expected (CK,F001) and (CK,F002) pairs found in all results
- [PASS] **state_isolation**: F001-only=18, F002-only=12, combined=0, mismatches=0
- [PASS] **protected_unprotected_outcome_symmetry**: compared_turn_count=75, mismatch_count=0
- [PASS] **reintroduced_subset_of_exposed**: All reintroduced IDs are subsets of exposed and targeted IDs
- [PASS] **positive_F001_reconstruction**: F001-only reconstruction turns: 6
- [PASS] **positive_F002_reconstruction**: F002-only reconstruction turns: 6 (both=6)
- [PASS] **reconstructed_ids_record_specific**: All reconstructed IDs are F001 or F002
- [PASS] **rr_denominator_positive**: RR denominator=42, numerator=6
- [PASS] **rr_numerator_le_denominator**: RR=6/42
- [PASS] **crr_numerator_le_denominator**: CRR=6/60
- [PASS] **rr_clean_denominator_positive**: RR_clean denominator=42, numerator=6
- [PASS] **rr_clean_numerator_le_denominator**: RR_clean=6/42
- [PASS] **multi_target_has_multiple_items**: Scenario has 2 sensitive items
- [PASS] **multi_target_has_recontamination_steps**: Recontamination steps: 3
- [PASS] **multi_target_audit_valid**: Audit errors: 0
- [PASS] **protected_unexpected_recontamination_zero**: Unexpected by condition: {'no_firewall': 0, 'monitoring_0': 3, 'binary_policy': 3, 'full_mvp': 0, 'exact_only': 6}
- [PASS] **disk_metrics_match_in_memory**: Verified 30 results match across disk round-trip
- [PASS] **disk_record_level_fields**: All record-level fields match across 30 results

## Metrics

| Metric | Value | Numerator | Denominator |
|--------|------:|----------:|------------:|
| PU-RER | 0.0 | 0 | 270 |
| CRR | 0.1 | 6 | 60 |
| RR | 0.14285714285714285 | 6 | 42 |
| RR_clean | 0.14285714285714285 | 6 | 42 |
| RR_at_risk | 0.3333333333333333 | 6 | 18 |
| FBR | 0.0 | 0 | 60 |

## paired_policy_utility_retention (per condition)

| Condition | Value | Numerator | Denominator | Evaluable |
|-----------|------:|----------:|------------:|:---------:|
| binary_policy | 1.0 | 6 | 6 | True |
| exact_only | 1.0 | 6 | 6 | True |
| full_mvp | 1.0 | 6 | 6 | True |
| monitoring_0 | 1.0 | 6 | 6 | True |
| no_firewall | None | 0 | 0 | False |
