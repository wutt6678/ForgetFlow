# TrustParadox-U Primary Study Results

**Generated:** 2026-08-03T09:09:19.445602+00:00
**Commit:** 0c8d516ebe2934113dff732f656c30580a00dc0d

### Table 1: Main Results by Condition

| condition | crr_value | rr_value | fbr_value | paired_policy_utility_retention_value |
| --- | --- | --- | --- | --- |
| binary_policy | 0.3333 | 0.0000 | 0.0137 | None |
| full_mvp | 0.3333 | 0.0000 | 0.0000 | None |
| no_claim_detection | 0.3333 | 0.0000 | 0.0000 | None |
| no_monitoring | 1.0000 | 1.0000 | 0.0000 | None |
| one_time_monitoring | 0.3333 | 0.0000 | 0.0000 | None |

### Table 2: Leakage by Attack Category

| category | n_episodes | exposure_rate | reconstruction_rate | recontamination_rate |
| --- | --- | --- | --- | --- |
| claim_control | 600 | 1.0000 | 0.0000 | 0.2000 |
| control | 780 | 0.3077 | 0.0000 | 0.0615 |
| direct_disclosure | 1170 | 0.3077 | 0.0000 | 0.0615 |
| recontamination | 390 | 0.3077 | 0.0000 | 0.0615 |
| sequential_reconstruction | 390 | 0.3077 | 0.0000 | 0.0615 |

### Table 4: Paired Statistical Comparisons

| condition_a | condition_b | metric | rate_a | rate_b | cohens_h | p_value | significant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_mvp | no_monitoring | exposure | 0.4324 | 0.4324 | 0.0000 | 1.0000 | False |
| full_mvp | no_monitoring | recontamination | 0.0000 | 0.4324 | -1.4352 | 0.0000 | True |
| full_mvp | no_monitoring | task_success | 0.3018 | 0.2342 | 0.1528 | 0.0000 | True |
| full_mvp | no_claim_detection | exposure | 0.4324 | 0.4324 | 0.0000 | 1.0000 | False |
| full_mvp | no_claim_detection | recontamination | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | no_claim_detection | task_success | 0.3018 | 0.2703 | 0.0698 | 0.0000 | True |
| full_mvp | binary_policy | exposure | 0.4324 | 0.4324 | 0.0000 | 1.0000 | False |
| full_mvp | binary_policy | recontamination | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | binary_policy | task_success | 0.3018 | 0.2342 | 0.1528 | 0.0000 | True |
| full_mvp | one_time_monitoring | exposure | 0.4324 | 0.4324 | 0.0000 | 1.0000 | False |
| full_mvp | one_time_monitoring | recontamination | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| full_mvp | one_time_monitoring | task_success | 0.3018 | 0.3018 | 0.0000 | 1.0000 | False |
| no_monitoring | no_claim_detection | exposure | 0.4324 | 0.4324 | 0.0000 | 1.0000 | False |
| no_monitoring | no_claim_detection | recontamination | 0.4324 | 0.0000 | 1.4352 | 0.0000 | True |
| no_monitoring | no_claim_detection | task_success | 0.2342 | 0.2703 | -0.0830 | 0.0000 | True |
| binary_policy | one_time_monitoring | exposure | 0.4324 | 0.4324 | 0.0000 | 1.0000 | False |
| binary_policy | one_time_monitoring | recontamination | 0.0000 | 0.0000 | 0.0000 | 1.0000 | False |
| binary_policy | one_time_monitoring | task_success | 0.2342 | 0.3018 | -0.1528 | 0.0000 | True |

## Exit Criteria
- all_conditions_run: PASS
- all_metrics_computed: PASS
- paired_statistics_available: PASS
- leakage_breakdown_available: PASS
- parameter_sweep_complete: PASS