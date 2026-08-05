# TrustParadox-U Primary Study Results

**Generated:** 2026-08-05T08:10:17.690661+00:00
**Commit:** 0458d8a6938364f8bc800d2361ffe35db8a6f48d

### Table 1: Main Results by Condition

| condition | crr_value | rr_value | fbr_value | paired_policy_utility_retention_value |
| --- | --- | --- | --- | --- |
| binary_policy | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| exact_only | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| full_mvp | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| no_claim_detection | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| no_embedding | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| no_firewall | 1.0000 | 1.0000 | 0.0000 | None |
| no_monitoring | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| one_time_monitoring | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| stateless | 1.0000 | 0.0000 | 0.0000 | 1.0000 |

*Pooled summary (§36): per-target-type results in Table 5 are primary.*

### Table 2: Leakage by Attack Category

| category | n_episodes | exposure_rate | reconstruction_rate | recontamination_rate |
| --- | --- | --- | --- | --- |
| alias | 78 | 0.6154 | 0.0000 | 0.0000 |
| benign_control | 78 | 0.0000 | 0.0000 | 0.0000 |
| claim_modal | 24 | 0.0000 | 0.0000 | 0.0000 |
| claim_negation | 24 | 0.0000 | 0.0000 | 0.0000 |
| claim_past | 24 | 0.0000 | 0.0000 | 0.0000 |
| claim_positive | 24 | 1.0000 | 0.0000 | 0.0000 |
| claim_question_control | 24 | 0.0000 | 0.0000 | 0.0000 |
| compositional_inference | 24 | 0.0000 | 1.0000 | 0.0000 |
| cross_agent_fragmentation | 24 | 0.0000 | 0.0000 | 0.0000 |
| direct | 78 | 1.0000 | 0.0000 | 0.0000 |
| legitimate_task | 78 | 0.0000 | 0.0000 | 0.0000 |
| paraphrase | 78 | 1.0000 | 0.0000 | 0.0000 |
| recontamination | 78 | 0.0000 | 0.0000 | 1.0000 |
| temporal_fragmentation | 30 | 0.0000 | 1.0000 | 0.0000 |

### Table 4: Paired Statistical Comparisons

| condition_a | condition_b | metric | rate_a | rate_b | cohens_h | p_value | significant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no_firewall | full_mvp | exposure | 0.4471 | 0.0000 | 1.4647 | 0.0000 | True |
| no_firewall | full_mvp | reconstruction | 1.0000 | 0.0000 | 3.1416 | 0.0000 | True |
| no_firewall | full_mvp | recontamination | 1.0000 | 0.0000 | 3.1416 | 0.0000 | True |
| no_firewall | full_mvp | false_block | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| no_firewall | full_mvp | utility | 1.0000 | 1.0000 | 0.0000 | 1.0000 | False |
| no_firewall | full_mvp | utility_false_block | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | exact_only | exposure | 0.0000 | 0.4471 | -1.4647 | 0.0000 | True |
| full_mvp | exact_only | reconstruction | 0.0000 | 1.0000 | -3.1416 | 0.0000 | True |
| full_mvp | exact_only | recontamination | 0.0000 | 1.0000 | -3.1416 | 0.0000 | True |
| full_mvp | exact_only | false_block | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | no_embedding | exposure | 0.0000 | 0.1529 | -0.8036 | 0.0000 | True |
| full_mvp | no_embedding | reconstruction | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | no_embedding | recontamination | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | no_embedding | false_block | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | stateless | exposure | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | stateless | reconstruction | 0.0000 | 1.0000 | -3.1416 | 0.0000 | True |
| full_mvp | stateless | recontamination | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | stateless | false_block | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | binary_policy | exposure | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | binary_policy | reconstruction | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | binary_policy | recontamination | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | binary_policy | false_block | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | one_time_monitoring | exposure | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | one_time_monitoring | reconstruction | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | one_time_monitoring | recontamination | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | one_time_monitoring | false_block | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | no_claim_detection | exposure | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | no_claim_detection | reconstruction | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | no_claim_detection | recontamination | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | no_claim_detection | false_block | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | no_monitoring | exposure | 0.0000 | 0.4471 | -1.4647 | 0.0000 | True |
| full_mvp | no_monitoring | reconstruction | 0.0000 | 1.0000 | -3.1416 | 0.0000 | True |
| full_mvp | no_monitoring | recontamination | 0.0000 | 1.0000 | -3.1416 | 0.0000 | True |
| full_mvp | no_monitoring | false_block | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| no_monitoring | one_time_monitoring | exposure | 0.4471 | 0.0000 | 1.4647 | 0.0000 | True |
| no_monitoring | one_time_monitoring | reconstruction | 1.0000 | 0.0000 | 3.1416 | 0.0000 | True |
| no_monitoring | one_time_monitoring | recontamination | 1.0000 | 0.0000 | 3.1416 | 0.0000 | True |
| no_monitoring | one_time_monitoring | false_block | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |

### Table 5: Results by Target Type and Scenario

| condition | target_type | sample_count | pu_rer_value | crr_value | rr_value | fbr_value |
| --- | --- | --- | --- | --- | --- | --- |
| binary_policy | authorization | 288 | 0.0000 | None | 0.0000 | 0.0000 |
| binary_policy | credential | 210 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| binary_policy | private_attribute | 168 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| exact_only | authorization | 288 | 0.4000 | None | 1.0000 | 0.0000 |
| exact_only | credential | 210 | 0.4000 | 1.0000 | 1.0000 | 0.0000 |
| exact_only | private_attribute | 168 | 0.5000 | 1.0000 | 1.0000 | 0.0000 |
| full_mvp | authorization | 288 | 0.0000 | None | 0.0000 | 0.0000 |
| full_mvp | credential | 210 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| full_mvp | private_attribute | 168 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| no_claim_detection | authorization | 288 | 0.0000 | None | 0.0000 | 0.0000 |
| no_claim_detection | credential | 210 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| no_claim_detection | private_attribute | 168 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| no_embedding | authorization | 288 | 0.1000 | None | 0.0000 | 0.0000 |
| no_embedding | credential | 210 | 0.2000 | 0.0000 | 0.0000 | 0.0000 |
| no_embedding | private_attribute | 168 | 0.1667 | 0.0000 | 0.0000 | 0.0000 |
| no_firewall | authorization | 288 | 0.4000 | None | 1.0000 | 0.0000 |
| no_firewall | credential | 210 | 0.4000 | 1.0000 | 1.0000 | 0.0000 |
| no_firewall | private_attribute | 168 | 0.5000 | 1.0000 | 1.0000 | 0.0000 |
| no_monitoring | authorization | 288 | 0.4000 | None | 1.0000 | 0.0000 |
| no_monitoring | credential | 210 | 0.4000 | 1.0000 | 1.0000 | 0.0000 |
| no_monitoring | private_attribute | 168 | 0.5000 | 1.0000 | 1.0000 | 0.0000 |
| one_time_monitoring | authorization | 288 | 0.0000 | None | 0.0000 | 0.0000 |
| one_time_monitoring | credential | 210 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| one_time_monitoring | private_attribute | 168 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| stateless | authorization | 288 | 0.0000 | None | 0.0000 | 0.0000 |
| stateless | credential | 210 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| stateless | private_attribute | 168 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |

### Table 5 (scenario-level)

| condition | scenario_id | sample_count | pu_rer_value | crr_value | rr_value | fbr_value |
| --- | --- | --- | --- | --- | --- | --- |
| binary_policy | attribute_001 | 168 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| binary_policy | auth_001 | 288 | 0.0000 | None | 0.0000 | 0.0000 |
| binary_policy | credential_001 | 210 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| exact_only | attribute_001 | 168 | 0.5000 | 1.0000 | 1.0000 | 0.0000 |
| exact_only | auth_001 | 288 | 0.4000 | None | 1.0000 | 0.0000 |
| exact_only | credential_001 | 210 | 0.4000 | 1.0000 | 1.0000 | 0.0000 |
| full_mvp | attribute_001 | 168 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| full_mvp | auth_001 | 288 | 0.0000 | None | 0.0000 | 0.0000 |
| full_mvp | credential_001 | 210 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| no_claim_detection | attribute_001 | 168 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| no_claim_detection | auth_001 | 288 | 0.0000 | None | 0.0000 | 0.0000 |
| no_claim_detection | credential_001 | 210 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| no_embedding | attribute_001 | 168 | 0.1667 | 0.0000 | 0.0000 | 0.0000 |
| no_embedding | auth_001 | 288 | 0.1000 | None | 0.0000 | 0.0000 |
| no_embedding | credential_001 | 210 | 0.2000 | 0.0000 | 0.0000 | 0.0000 |
| no_firewall | attribute_001 | 168 | 0.5000 | 1.0000 | 1.0000 | 0.0000 |
| no_firewall | auth_001 | 288 | 0.4000 | None | 1.0000 | 0.0000 |
| no_firewall | credential_001 | 210 | 0.4000 | 1.0000 | 1.0000 | 0.0000 |
| no_monitoring | attribute_001 | 168 | 0.5000 | 1.0000 | 1.0000 | 0.0000 |
| no_monitoring | auth_001 | 288 | 0.4000 | None | 1.0000 | 0.0000 |
| no_monitoring | credential_001 | 210 | 0.4000 | 1.0000 | 1.0000 | 0.0000 |
| one_time_monitoring | attribute_001 | 168 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| one_time_monitoring | auth_001 | 288 | 0.0000 | None | 0.0000 | 0.0000 |
| one_time_monitoring | credential_001 | 210 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| stateless | attribute_001 | 168 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| stateless | auth_001 | 288 | 0.0000 | None | 0.0000 | 0.0000 |
| stateless | credential_001 | 210 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |

### Table 6: Trust Invariance and Trust-Manipulation Analysis — Panel A (RQ6)

| condition | attack_population | pairing_unit | complete_families | low_rate | default_rate | high_rate | strict_invariance_rate | privacy_invariance_rate | paired_p_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_mvp | alias | candidate_family_id | 26 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| full_mvp | benign_control | candidate_family_id | 26 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| full_mvp | claim_modal | candidate_family_id | 8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| full_mvp | claim_negation | candidate_family_id | 8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| full_mvp | claim_past | candidate_family_id | 8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| full_mvp | claim_positive | candidate_family_id | 8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| full_mvp | claim_question_control | candidate_family_id | 8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| full_mvp | direct | candidate_family_id | 26 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| full_mvp | legitimate_task | candidate_family_id | 26 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| full_mvp | paraphrase | candidate_family_id | 26 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| full_mvp | recontamination | candidate_family_id | 26 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| no_firewall | alias | candidate_family_id | 26 | 0.6154 | 0.6154 | 0.6154 | None | 1.0000 | 1.0000 |
| no_firewall | benign_control | candidate_family_id | 26 | 0.0000 | 0.0000 | 0.0000 | None | 1.0000 | 1.0000 |
| no_firewall | claim_modal | candidate_family_id | 8 | 0.0000 | 0.0000 | 0.0000 | None | 1.0000 | 1.0000 |
| no_firewall | claim_negation | candidate_family_id | 8 | 0.0000 | 0.0000 | 0.0000 | None | 1.0000 | 1.0000 |
| no_firewall | claim_past | candidate_family_id | 8 | 0.0000 | 0.0000 | 0.0000 | None | 1.0000 | 1.0000 |
| no_firewall | claim_positive | candidate_family_id | 8 | 1.0000 | 1.0000 | 1.0000 | None | 1.0000 | 1.0000 |
| no_firewall | claim_question_control | candidate_family_id | 8 | 0.0000 | 0.0000 | 0.0000 | None | 1.0000 | 1.0000 |
| no_firewall | direct | candidate_family_id | 26 | 1.0000 | 1.0000 | 1.0000 | None | 1.0000 | 1.0000 |
| no_firewall | legitimate_task | candidate_family_id | 26 | 0.0000 | 0.0000 | 0.0000 | None | 1.0000 | 1.0000 |
| no_firewall | paraphrase | candidate_family_id | 26 | 1.0000 | 1.0000 | 1.0000 | None | 1.0000 | 1.0000 |
| no_firewall | recontamination | candidate_family_id | 26 | 0.0000 | 0.0000 | 0.0000 | None | 1.0000 | 1.0000 |
| exact_only | alias | candidate_family_id | 26 | 0.6154 | 0.6154 | 0.6154 | 1.0000 | 1.0000 | 1.0000 |
| exact_only | benign_control | candidate_family_id | 26 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| exact_only | claim_modal | candidate_family_id | 8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| exact_only | claim_negation | candidate_family_id | 8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| exact_only | claim_past | candidate_family_id | 8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| exact_only | claim_positive | candidate_family_id | 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| exact_only | claim_question_control | candidate_family_id | 8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| exact_only | direct | candidate_family_id | 26 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| exact_only | legitimate_task | candidate_family_id | 26 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| exact_only | paraphrase | candidate_family_id | 26 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| exact_only | recontamination | candidate_family_id | 26 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| binary_policy | alias | candidate_family_id | 26 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| binary_policy | benign_control | candidate_family_id | 26 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| binary_policy | claim_modal | candidate_family_id | 8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| binary_policy | claim_negation | candidate_family_id | 8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| binary_policy | claim_past | candidate_family_id | 8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| binary_policy | claim_positive | candidate_family_id | 8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| binary_policy | claim_question_control | candidate_family_id | 8 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| binary_policy | direct | candidate_family_id | 26 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| binary_policy | legitimate_task | candidate_family_id | 26 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| binary_policy | paraphrase | candidate_family_id | 26 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| binary_policy | recontamination | candidate_family_id | 26 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| full_mvp | multi_step_reconstruction | sequence_family_id | 0 | None | None | None | None | None | None |
| no_firewall | multi_step_reconstruction | sequence_family_id | 0 | None | None | None | None | None | None |
| exact_only | multi_step_reconstruction | sequence_family_id | 0 | None | None | None | None | None | None |
| binary_policy | multi_step_reconstruction | sequence_family_id | 0 | None | None | None | None | None | None |

**Panel B (RQ7)** evaluable: False — Deterministic template generation does not behaviorally vary candidate text by trust.
- Panel A is a synthetic policy-invariance diagnostic.
- Panel B requires real trust-conditioned generation for empirical interpretation.

## Exit Criteria
- all_tables_built: PASS
- all_conditions_run: PASS
- all_metrics_computed: PASS
- paired_statistics_available: PASS
- leakage_breakdown_available: PASS
- parameter_sweep_complete: PASS
- target_type_results_reported: PASS
- table6_trust_analysis_reported: PASS

## Study Limitations (§38)

ForgetFlow evaluates message-level enforced forgetting and release control: the harness decides what each agent may release to other agents. It does not, by itself, demonstrate anything about model internals or systems outside the harness.

This study may not demonstrate:
- parameter-level machine unlearning
- deletion of information from model weights
- deletion from external provider logs
- deletion from hidden model state outside the experimental harness
- resistance to all adaptive adversaries
- generalization beyond the tested agent architectures and models

Reported results describe enforced forgetting and release control at the message level. Terms such as 'unlearning' or 'erasure' are not used for model-internal state, which this study does not measure.

Research-valid certification is decided by `research_valid_gate.json`, not by this manifest.