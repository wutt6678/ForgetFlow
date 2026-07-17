# Single-Target Test Coverage Map

Maps every specification ID from the single-target validation suite to existing test coverage.

Coverage statuses: `covered`, `partially covered`, `not covered`, `not applicable`.

---

## Section 6 — Dataset and Schema Validation

| Spec ID | Requirement | Existing test | Status | Missing |
|---|---|---|---|---|
| ST-DATA-001 | Exactly one sensitive item | None | not covered | Need `validate_single_target_episode()` and tests for 0, 1, 2 items |
| ST-DATA-002 | Recontamination target ID present | `test_dataset.py` (implicit via `_parse_message_label`) | partially covered | Need explicit tests for empty `target_forget_ids` on recontamination |
| ST-DATA-003 | Recontamination target ID exists | None | not covered | Need cross-reference validation of `target_forget_ids` against episode `forget_id`s |
| ST-DATA-004 | Target IDs normalized | `dataset.py:_parse_message_label` (dedup + sort) | covered | Behavior is normalized; document the choice |
| ST-DATA-005 | Fragment metadata valid | `test_attacks.py:TestFragmentationValidation` | covered | |
| ST-DATA-006 | Task-success schema explicit | `test_runner.py:TestTaskSuccess` | covered | |

## Section 7 — Exact and Alias Detection

| Spec ID | Requirement | Existing test | Status | Missing |
|---|---|---|---|---|
| ST-DET-001 | Canonical target match | `test_detectors.py:test_exact_match` | covered | |
| ST-DET-002 | Canonical target after normalization | `test_detectors.py:TestNormalize` | covered | |
| ST-DET-003 | Alias match | `test_detectors.py:test_alias_match` | covered | |
| ST-DET-004 | Unrelated code does not match | `test_detectors.py:test_no_match` | covered | |
| ST-DET-005 | Permitted residual does not trigger leakage | `test_detectors.py:test_permitted_residual` | covered | |

## Section 8 — Semantic Detection

| Spec ID | Requirement | Existing test | Status | Missing |
|---|---|---|---|---|
| ST-SEM-001 | Annotated paraphrase similarity | `test_embedding.py:TestFixedEmbeddingProvider` | partially covered | Need explicit paraphrase-vs-threshold test with fixed vectors |
| ST-SEM-002 | Unrelated message similarity | `test_embedding.py:TestCosineSimilarity` | covered | |
| ST-SEM-003 | Threshold boundary | None | not covered | Need parameterized boundary test (0.79, 0.80, 0.81) |
| ST-SEM-004 | Fixed provider deterministic | `test_embedding.py:test_known_text_returns_vector` | covered | |
| ST-SEM-005 | Real-provider batch validation | `test_embedding.py:TestRealEmbeddingProvider` | covered | |

## Section 9 — History and Reconstruction

| Spec ID | Requirement | Existing test | Status | Missing |
|---|---|---|---|---|
| ST-HIST-001 | First fragment alone insufficient | `test_history.py:test_one_fragment_no_reconstruct` | covered | |
| ST-HIST-002 | Two released fragments reconstruct | `test_history.py:test_two_fragments_reconstruct` | covered | |
| ST-HIST-003 | Blocked second fragment excluded | `test_metric_contracts.py:test_second_fragment_blocked_returns_false` | covered | |
| ST-HIST-004 | Recipient separation | `test_history.py:test_isolated_recipients` | covered | |
| ST-HIST-005 | Sender identity does not replace recipient history | None | not covered | Need multi-sender same-recipient test |
| ST-HIST-006 | History window expiration | `test_history.py:test_bounded_window` | covered | |

## Section 10 — FlowGate Actions

| Spec ID | Requirement | Existing test | Status | Missing |
|---|---|---|---|---|
| ST-GATE-001 | Safe message allowed | `test_flow_gate.py:test_safe_allow` | covered | |
| ST-GATE-002 | Exact secret blocked under binary policy | `test_flow_gate.py:test_exact_block` | covered | |
| ST-GATE-003 | Mixed unsafe/useful under rich policy | `test_end_to_end.py:TestPolicyUtility` | partially covered | Need explicit FlowGate-level mixed-content test with rich transformation |
| ST-GATE-004 | Same mixed message under binary policy | `test_end_to_end.py:TestPolicyUtility:test_binary_vs_rich_same_candidate` | partially covered | Need explicit FlowGate-level binary block of mixed content |
| ST-GATE-005 | Transformed output rechecked | `test_flow_gate.py:TestPermittedResidualRecheck` | covered | |
| ST-GATE-006 | Permitted residual bypasses semantic-only rejection | `test_flow_gate.py:test_non_approved_paraphrase_not_residual` | covered | |

## Section 11 — Runner Released-Content Semantics

| Spec ID | Requirement | Existing test | Status | Missing |
|---|---|---|---|---|
| ST-RUN-001 | No-firewall release | `test_runner.py:test_no_firewall_released_equals_candidate` | covered | |
| ST-RUN-002 | Blocked message has no release | `test_runner.py:test_blocked_message_has_none_released` | covered | |
| ST-RUN-003 | Exposure uses released text only | `test_metric_contracts.py:test_pu_rer_uses_released_text_not_candidate` | covered | |
| ST-RUN-004 | Semantic exposure uses released text only | `test_end_to_end.py:TestSemanticParaphrase` | covered | |
| ST-RUN-005 | Blocked messages absent from recipient history | `test_flow_gate.py:test_only_released_in_history` | covered | |
| ST-RUN-006 | Blocked messages do not alter contamination state | `test_metric_contracts.py:test_rr_at_risk_not_recontaminated` | covered | |

## Section 12 — Task Success

| Spec ID | Requirement | Existing test | Status | Missing |
|---|---|---|---|---|
| ST-TASK-001 | Required release succeeds | `test_runner.py:test_required_release_success` | covered | |
| ST-TASK-002 | Required release blocked | `test_runner.py:test_required_release_blocked` | covered | |
| ST-TASK-003 | Exact label succeeds | `test_runner.py:test_exact_label_matches` | covered | |
| ST-TASK-004 | Text mention does not satisfy exact label | `test_runner.py:test_text_contains_label_but_metadata_differs_failure` | covered | |
| ST-TASK-005 | Matching label without text mention | `test_runner.py:test_metadata_label_matches_text_does_not_contain_success` | covered | |
| ST-TASK-006 | Conflicting task labels fail | `test_runner.py:test_conflicting_labels_in_one_episode_raises` | covered | |

## Section 13 — Recontamination

| Spec ID | Requirement | Existing test | Status | Missing |
|---|---|---|---|---|
| ST-RR-001 | Blocked attempt enters denominator | `test_runner.py:test_recontamination_denominator_counts_blocked_attempts` | covered | |
| ST-RR-002 | Successful recontamination | `test_runner.py:test_recontamination_numerator_only_confirmed_recovery` | covered | |
| ST-RR-003 | Duplicate attempts deduplicate | None | not covered | Need explicit test: 3 attempts -> `attempted_pairs=1` |
| ST-RR-004 | Non-targeted safe message not in denominator | None | not covered | Need explicit test: safe message -> `attempted_pairs=0` |
| ST-RR-005 | Unexpected recontamination audited | `test_result_audit.py` (partial) | partially covered | Need audit finding code for unexpected pairs; mode-dependent severity |
| ST-RR-006 | Pair counts survive disk round trip | `test_serialization.py:test_pair_counters_round_trip_*` | covered | |

## Section 14 — Monitoring Window

| Spec ID | Requirement | Existing test | Status | Missing |
|---|---|---|---|---|
| ST-MON-001 | Continuous monitoring | `test_runner.py:test_continuous_always_active` | covered | |
| ST-MON-002 | Duration zero | `test_runner.py:test_duration_zero_never_active` | covered | |
| ST-MON-003 | Duration one | `test_runner.py:test_duration_one_active_at_round_zero` | covered | |
| ST-MON-004 | Duration three | `test_runner.py:test_duration_three` | covered | |
| ST-MON-005 | Pre-forget messages don't consume duration | None | not covered | Need test: pre-forget messages don't shift enforcement window |
| ST-MON-006 | Forget turn doesn't consume duration | None | not covered | Need test: forget event itself doesn't consume monitoring duration |
| ST-MON-007 | Delayed attack distinguishes one-time vs continuous | `test_end_to_end.py:TestRecontamination` | covered | |

## Section 15 — Trust Invariance

| Spec ID | Requirement | Existing test | Status | Missing |
|---|---|---|---|---|
| ST-TRUST-001 | Matched low/default/high trust | `test_end_to_end.py:TestTrustInvariance` | covered | |
| ST-TRUST-002 | Configuration equality | `test_end_to_end.py:TestTrustInvariance` | partially covered | Need explicit assertion that only trust_level/trust_prompt differ |
| ST-TRUST-003 | Trust-sensitive negative control | None | not covered | Need test confirming invariance test can detect policy dependence |

## Section 16 — Paired End-to-End Research Tests

| Spec ID | Requirement | Existing test | Status | Missing |
|---|---|---|---|---|
| ST-E2E-001 | No firewall vs full MVP | `test_end_to_end.py:TestExactLeakage` | covered | |
| ST-E2E-002 | No semantic vs semantic enabled | `test_end_to_end.py:TestSemanticParaphrase` | covered | |
| ST-E2E-003 | Stateless vs recipient-aware | `test_end_to_end.py:TestFragmentation` | covered | |
| ST-E2E-004 | Binary vs rich policy | `test_end_to_end.py:TestPolicyUtility` | covered | |
| ST-E2E-005 | One-time vs continuous monitoring | `test_end_to_end.py:TestRecontamination` | covered | |
| ST-E2E-006 | Trust invariance | `test_end_to_end.py:TestTrustInvariance` | covered | |

## Section 17 — Hyperparameter Tests

| Spec ID | Requirement | Existing test | Status | Missing |
|---|---|---|---|---|
| ST-HP-001 | Semantic threshold sweep | None | not covered | Need parameterized sweep: 0.60, 0.70, 0.80, 0.90 |
| ST-HP-002 | Reconstruction threshold sweep | None | not covered | Need parameterized sweep: 0.50, 0.70, 0.90 |
| ST-HP-003 | History window sweep | None | not covered | Need parameterized sweep: 1, 2, 4, 8 |
| ST-HP-004 | Monitoring duration sweep | None | not covered | Need parameterized sweep: 0, 1, 3, 5 |

## Section 18 — Attack-Type Robustness

| Spec ID | Requirement | Existing test | Status | Missing |
|---|---|---|---|---|
| ST-ATTACK-001 | Direct disclosure | `test_detectors.py:test_exact_match` | covered | |
| ST-ATTACK-002 | Alias disclosure | `test_detectors.py:test_alias_match` | covered | |
| ST-ATTACK-003 | Semantic paraphrase | `test_end_to_end.py:TestSemanticParaphrase` | covered | |
| ST-ATTACK-004 | Temporal fragmentation | `test_attacks.py:TestFragmentationValidation` | covered | |
| ST-ATTACK-005 | Cross-agent fragmentation | `test_attacks.py:test_valid_cross_agent_fragmentation_passes` | covered | |
| ST-ATTACK-006 | Delayed recontamination | `test_end_to_end.py:TestRecontamination` | covered | |
| ST-ATTACK-007 | Mixed safe and unsafe content | `test_end_to_end.py:TestPolicyUtility` | covered | |
| ST-ATTACK-008 | Repeated probing | None | not covered | Need repeated paraphrase probing test |

## Section 19 — Metric Contract Tests

| Spec ID | Requirement | Existing test | Status | Missing |
|---|---|---|---|---|
| ST-METRIC-001 | PU-RER | `test_metric_contracts.py:test_pu_rer_candidate_secret_not_counted` | covered | |
| ST-METRIC-002 | CRR | `test_metric_contracts.py:test_crr_blocked_reconstruction_not_counted` | covered | |
| ST-METRIC-003 | RR | `test_metric_contracts.py:test_rr_recontaminated_counted` | covered | |
| ST-METRIC-004 | FBR | `test_metric_contracts.py:test_fbr_only_counts_legitimate_messages` | covered | |
| ST-METRIC-005 | Utility retention | `test_result_audit.py:TestUtilityAudit` | covered | |
| ST-METRIC-006 | Zero-denominator semantics | `test_result_audit.py:test_zero_denominator_none_value_ok` | covered | |
| ST-METRIC-007 | Numerator bounds | `test_result_audit.py:test_numerator_cannot_exceed_denominator` | covered | |

## Section 20 — Serialization Tests

| Spec ID | Requirement | Existing test | Status | Missing |
|---|---|---|---|---|
| ST-SER-001 | Versioned envelope | `test_serialization.py:TestSchemaVersioning` | covered | |
| ST-SER-002 | Firewall decision round trip | `test_serialization.py:TestDeserializeFirewallDecision:test_round_trip` | covered | |
| ST-SER-003 | Contamination state round trip | `test_serialization.py:TestDeserializeContaminationStatus:test_round_trip_via_json` | covered | |
| ST-SER-004 | Pair counters round trip | `test_serialization.py:test_pair_counters_round_trip_*` | covered | |
| ST-SER-005 | Attack step identity round trip | `test_runner.py:TestAttackStepIndexPropagation` | covered | |
| ST-SER-006 | Metadata round trip | `test_serialization.py:TestLoadEpisodeResults` | covered | |

## Section 21 — Audit Tests

| Spec ID | Requirement | Existing test | Status | Missing |
|---|---|---|---|---|
| ST-AUDIT-001 | Valid episode passes | `test_audit_results.py:test_valid_episode_passes` | covered | |
| ST-AUDIT-002 | Exposure without released text fails | `test_audit_results.py:test_exposed_without_released_text_is_error` | covered | |
| ST-AUDIT-003 | Block decision with released text fails | `test_audit_results.py:test_block_with_released_text_is_error` | covered | |
| ST-AUDIT-004 | Missing pairing key fails | `test_result_audit.py:TestDuplicateKeys` | covered | |
| ST-AUDIT-005 | Duplicate run identity fails | `test_result_audit.py:test_duplicate_run_identities_flagged` | covered | |
| ST-AUDIT-006 | Missing config hash fails | `test_result_audit.py:test_missing_config_hash_raises_error` | covered | |
| ST-AUDIT-007 | Invalid embedding metadata fails | `test_result_audit.py:TestEmbeddingAudit` | covered | |
| ST-AUDIT-008 | Unexpected recontamination fails | `test_result_audit.py` (partial) | partially covered | Need explicit audit finding for unexpected recontamination pairs |
| ST-AUDIT-009 | Non-monotonic attack-step indices fail | `test_result_audit.py:test_non_monotonic_step_index_fails` | covered | |
| ST-AUDIT-010 | Invalid metric bounds fail | `test_result_audit.py:TestMetricRules` | covered | |

## Section 22 — Manifest and Reproducibility Tests

| Spec ID | Requirement | Existing test | Status | Missing |
|---|---|---|---|---|
| ST-MAN-001 | Manifest built from observed results | `test_manifest.py:TestValidateManifestAgainstResults` | covered | |
| ST-MAN-002 | Observed dimension recorded | `test_manifest.py:test_provider_model_dimension_present` | covered | |
| ST-MAN-003 | Endpoint provenance matches | `test_manifest.py:test_no_query_strings_in_sanitized_endpoint` | covered | |
| ST-MAN-004 | Dirty tree rejected in experiment mode | `test_manifest.py:test_dirty_tree_rejected_raises` | covered | |
| ST-MAN-005 | Dirty tree marked in diagnostic test mode | `test_manifest.py:test_dirty_tree_appends_suffix` | covered | |
| ST-MAN-006 | Exact commit recorded | `test_manifest.py:test_manifest_includes_commit_sha` | covered | |
| ST-MAN-007 | Config hashes match results | `test_manifest.py:test_config_hashes_mismatch_fails` | covered | |
| ST-MAN-008 | Metric counts match aggregation | `test_manifest.py:test_metric_counts_mismatch_fails` | covered | |

## Section 23 — Disk Aggregation Tests

| Spec ID | Requirement | Existing test | Status | Missing |
|---|---|---|---|---|
| ST-AGG-001 | Runner output aggregates without modification | `test_pipeline_integration.py:test_realistic_episode_aggregates_successfully` | covered | |
| ST-AGG-002 | Missing manifest fails | `test_aggregate.py:test_missing_manifest` | covered | |
| ST-AGG-003 | Malformed result exits with error | `test_aggregate.py:test_malformed_jsonl` | covered | |
| ST-AGG-004 | Invalid audit exits with error | `test_aggregate.py:test_invalid_audit_blocks_aggregation` | covered | |
| ST-AGG-005 | Manifest mismatch exits with error | `test_aggregate.py:test_manifest_mismatch` | covered | |
| ST-AGG-006 | Output files deterministic | `test_aggregate.py:TestAggregateSummary` | covered | |

---

## Summary

| Status | Count |
|---|---|
| covered | 80 |
| partially covered | 8 |
| not covered | 18 |
| **Total** | **106** |

### Not-covered items requiring new tests

1. ST-DATA-001 — Single-target validation
2. ST-DATA-003 — Target ID cross-reference
3. ST-SEM-003 — Threshold boundary
4. ST-HIST-005 — Multi-sender history
5. ST-RR-003 — Duplicate attempt deduplication
6. ST-RR-004 — Safe message exclusion from RR
7. ST-MON-005 — Pre-forget duration consumption
8. ST-MON-006 — Forget-turn duration consumption
9. ST-TRUST-003 — Trust-sensitive negative control
10. ST-HP-001 through ST-HP-004 — Hyperparameter sweeps (4 items)
11. ST-ATTACK-008 — Repeated probing
12. Fixture verification tests (ST-FIXTURE-001 through ST-FIXTURE-006)

### Partially-covered items requiring additional assertions

1. ST-DATA-002 — Need explicit empty-target tests
2. ST-SEM-001 — Need paraphrase-vs-threshold test
3. ST-GATE-003/004 — Need FlowGate-level mixed-content tests
4. ST-RR-005 — Need explicit audit finding for unexpected pairs
5. ST-TRUST-002 — Need explicit config-equality assertion
6. ST-AUDIT-008 — Need explicit unexpected-recontamination audit test
