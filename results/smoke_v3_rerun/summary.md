# Single-Target Smoke Study Summary

- **Status**: DIAGNOSTIC ONLY
- **Commit**: 9e1370a5cdbdffc9625130da94d78ab86d16254b
- **Mode**: diagnostic
- **Fixtures**: 3
- **Seeds**: 3
- **Conditions**: 11
- **Total runs**: 99
- **Audit valid**: True
- **Audit errors**: 0
- **Duplicate identities**: 0
- **Utility retention**: 1.0000

## Metrics

| Metric | Value | Numerator | Denominator |
|--------|------:|----------:|------------:|
| PU-RER | 0.0909 | 36 | 396 |
| CRR | 0.0636 | 21 | 330 |
| RR | 0.0909 | 3 | 33 |
| FBR | 0.0000 | 0 | 132 |

## Directional Checks

- [PASS] **privacy_mvp_better**: full_mvp PU-RER < no_firewall PU-RER
- [PASS] **semantic_protection**: full_mvp PU-RER < no_embedding PU-RER
- [PASS] **stateful_reconstruction_safer**: stateful CRR < stateless CRR
- [PASS] **rich_utility_ge_binary**: full_mvp task_success >= binary task_success
- [FAIL] **continuous_rr_lt_finite**: continuous RR < monitoring_0 RR
