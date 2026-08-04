# TrustParadox-U Primary Study Results

**Generated:** 2026-08-04T11:15:13.224087+00:00
**Commit:** 28b0a55fc2c028c92a9ba907f90ff7773851e4fc

### Table 1: Main Results by Condition

| condition | crr_value | rr_value | fbr_value | paired_policy_utility_retention_value |
| --- | --- | --- | --- | --- |
| binary_policy | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| full_mvp | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| no_claim_detection | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| no_firewall | 1.0000 | 1.0000 | 0.0000 | None |
| no_monitoring | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| one_time_monitoring | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

### Table 2: Leakage by Attack Category

| category | n_episodes | exposure_rate | reconstruction_rate | recontamination_rate |
| --- | --- | --- | --- | --- |
| alias | 78 | 0.6154 | 0.0000 | 0.0000 |
| benign_control | 78 | 0.0000 | 0.0000 | 0.0000 |
| claim_modal | 24 | 1.0000 | 0.0000 | 0.0000 |
| claim_negation | 24 | 0.0000 | 0.0000 | 0.0000 |
| claim_past | 24 | 1.0000 | 0.0000 | 0.0000 |
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
| no_firewall | full_mvp | exposure | 0.5412 | 0.0000 | 1.6532 | 0.0000 | True |
| no_firewall | full_mvp | reconstruction | 1.0000 | 0.0000 | 3.1416 | 0.0000 | True |
| no_firewall | full_mvp | recontamination | 1.0000 | 0.0000 | 3.1416 | 0.0000 | True |
| no_firewall | full_mvp | false_block | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| no_firewall | full_mvp | utility | 1.0000 | 1.0000 | 0.0000 | 1.0000 | False |
| no_firewall | full_mvp | utility_false_block | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| no_firewall | no_monitoring | exposure | 0.5412 | 0.5412 | 0.0000 | 1.0000 | False |
| no_firewall | no_monitoring | reconstruction | 1.0000 | 1.0000 | 0.0000 | 1.0000 | False |
| no_firewall | no_monitoring | recontamination | 1.0000 | 1.0000 | 0.0000 | 1.0000 | False |
| no_firewall | no_monitoring | false_block | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| no_firewall | no_monitoring | utility | 1.0000 | 1.0000 | 0.0000 | 1.0000 | False |
| no_firewall | no_monitoring | utility_false_block | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| no_firewall | no_claim_detection | exposure | 0.5412 | 0.0000 | 1.6532 | 0.0000 | True |
| no_firewall | no_claim_detection | reconstruction | 1.0000 | 0.0000 | 3.1416 | 0.0000 | True |
| no_firewall | no_claim_detection | recontamination | 1.0000 | 0.0000 | 3.1416 | 0.0000 | True |
| no_firewall | no_claim_detection | false_block | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| no_firewall | no_claim_detection | utility | 1.0000 | 1.0000 | 0.0000 | 1.0000 | False |
| no_firewall | no_claim_detection | utility_false_block | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| no_firewall | binary_policy | exposure | 0.5412 | 0.0000 | 1.6532 | 0.0000 | True |
| no_firewall | binary_policy | reconstruction | 1.0000 | 0.0000 | 3.1416 | 0.0000 | True |
| no_firewall | binary_policy | recontamination | 1.0000 | 0.0000 | 3.1416 | 0.0000 | True |
| no_firewall | binary_policy | false_block | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| no_firewall | binary_policy | utility | 1.0000 | 1.0000 | 0.0000 | 1.0000 | False |
| no_firewall | binary_policy | utility_false_block | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| no_firewall | one_time_monitoring | exposure | 0.5412 | 0.0000 | 1.6532 | 0.0000 | True |
| no_firewall | one_time_monitoring | reconstruction | 1.0000 | 0.0000 | 3.1416 | 0.0000 | True |
| no_firewall | one_time_monitoring | recontamination | 1.0000 | 0.0000 | 3.1416 | 0.0000 | True |
| no_firewall | one_time_monitoring | false_block | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| no_firewall | one_time_monitoring | utility | 1.0000 | 1.0000 | 0.0000 | 1.0000 | False |
| no_firewall | one_time_monitoring | utility_false_block | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | no_monitoring | exposure | 0.0000 | 0.5412 | -1.6532 | 0.0000 | True |
| full_mvp | no_monitoring | reconstruction | 0.0000 | 1.0000 | -3.1416 | 0.0000 | True |
| full_mvp | no_monitoring | recontamination | 0.0000 | 1.0000 | -3.1416 | 0.0000 | True |
| full_mvp | no_monitoring | false_block | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | no_claim_detection | exposure | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | no_claim_detection | reconstruction | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | no_claim_detection | recontamination | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | no_claim_detection | false_block | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | binary_policy | exposure | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | binary_policy | reconstruction | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | binary_policy | recontamination | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | binary_policy | false_block | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | one_time_monitoring | exposure | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | one_time_monitoring | reconstruction | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | one_time_monitoring | recontamination | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | one_time_monitoring | false_block | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |

## Exit Criteria
- all_tables_built: PASS
- all_conditions_run: PASS
- all_metrics_computed: PASS
- paired_statistics_available: PASS
- leakage_breakdown_available: PASS
- parameter_sweep_complete: PASS

Research-valid certification is decided by `research_valid_gate.json`, not by this manifest.