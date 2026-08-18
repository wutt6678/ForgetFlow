# Single-Target Smoke Study Summary

- **Status**: DIAGNOSTIC ONLY
- **Execution status**: DIAGNOSTIC_VALID
- **Commit**: 006247c0f479c4fd2264a5e9afd85a5c6b768754-dirty
- **Mode**: diagnostic
- **Fixtures**: 3
- **Seeds**: 3
- **Conditions**: 14
- **Total runs**: 126
- **Audit valid**: True
- **Audit errors**: 0
- **Duplicate identities**: 0
- **paired_policy_utility_retention**: 1.0000 (3/3, evaluable=True)

## Metrics

| Metric | Value | Numerator | Denominator |
|--------|------:|----------:|------------:|
| PU-RER | 0.0966 | 69 | 714 |
| CRR | 0.1905 | 24 | 126 |
| RR | 0.5000 | 21 | 42 |
| FBR | 0.0000 | 0 | 168 |

## Directional Checks

- [PASS] **privacy_mvp_better**: full_mvp PU-RER < no_firewall PU-RER
- [PASS] **semantic_protection**: full_mvp PU-RER < no_embedding PU-RER
- [PASS] **stateful_reconstruction_safer**: stateful CRR < stateless CRR
- [PASS] **rich_utility_ge_binary**: full_mvp task_success >= binary task_success
- [PASS] **firewall_reduces_rr**: no_firewall RR > full_mvp RR
- [PASS] **binary_policy_reconstruction_increases**: binary_policy CRR >= full_mvp CRR
- [PASS] **one_time_monitoring_recontamination_increases**: one_time_monitoring RR >= full_mvp RR
