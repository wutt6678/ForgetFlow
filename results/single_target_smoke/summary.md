# Single-Target Smoke Study Summary

- **Status**: DIAGNOSTIC ONLY
- **Commit**: 62100b4c35bd69b37785a82af580663ef3ea3d2e-dirty
- **Mode**: diagnostic
- **Fixtures**: 3
- **Seeds**: 3
- **Conditions**: 9
- **Total runs**: 81
- **Audit valid**: True
- **Audit errors**: 0
- **Duplicate identities**: 0
- **Utility retention**: 1.0000

## Metrics

| Metric | Value | Numerator | Denominator |
|--------|------:|----------:|------------:|
| PU-RER | 0.2130 | 69 | 324 |
| CRR | 0.0889 | 12 | 135 |
| RR | 0.3333 | 9 | 27 |
| FBR | 0.0000 | 0 | 108 |

## Directional Checks

- [PASS] **privacy_mvp_better**: full_mvp PU-RER < no_firewall PU-RER
- [PASS] **semantic_protection**: full_mvp PU-RER < no_semantic PU-RER
- [PASS] **stateful_reconstruction_safer**: stateful CRR < stateless CRR
- [PASS] **rich_utility_ge_binary**: full_mvp task_success >= binary task_success
- [PASS] **continuous_rr_lt_finite**: continuous RR < monitoring_0 RR
