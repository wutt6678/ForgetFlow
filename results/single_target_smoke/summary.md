# Single-Target Smoke Study Summary

- **Status**: DIAGNOSTIC ONLY
- **Execution status**: RESEARCH_VALID
- **Commit**: fa80b57912ab715c73cd5a8dcc6e9d346f23fdcc
- **Mode**: diagnostic
- **Fixtures**: 3
- **Seeds**: 3
- **Conditions**: 13
- **Total runs**: 117
- **Audit valid**: True
- **Audit errors**: 0
- **Duplicate identities**: 0
- **paired_policy_utility_retention**: 1.0000 (3/3, evaluable=True)

## Metrics

| Metric | Value | Numerator | Denominator |
|--------|------:|----------:|------------:|
| PU-RER | 0.0724 | 48 | 663 |
| CRR | 0.0769 | 18 | 234 |
| RR | 0.2308 | 9 | 39 |
| FBR | 0.0000 | 0 | 156 |

## Directional Checks

- [PASS] **privacy_mvp_better**: full_mvp PU-RER < no_firewall PU-RER
- [PASS] **semantic_protection**: full_mvp PU-RER < no_embedding PU-RER
- [PASS] **stateful_reconstruction_safer**: stateful CRR < stateless CRR
- [PASS] **rich_utility_ge_binary**: full_mvp task_success >= binary task_success
- [PASS] **firewall_reduces_rr**: no_firewall RR > full_mvp RR
