# Single-Target Smoke Study Summary

- **Status**: DIAGNOSTIC ONLY
- **Commit**: 1ecace5e913b90f9d120a57ad0b67b2d53b2e882
- **Mode**: diagnostic
- **Fixtures**: 3
- **Seeds**: 3
- **Conditions**: 13
- **Total runs**: 117
- **Audit valid**: True
- **Audit errors**: 0
- **Duplicate identities**: 0
- **Utility retention**: 1.0000

## Metrics

| Metric | Value | Numerator | Denominator |
|--------|------:|----------:|------------:|
| PU-RER | 0.0947 | 48 | 507 |
| CRR | 0.2000 | 1 | 5 |
| RR | 0.2308 | 9 | 39 |
| FBR | 0.0000 | 0 | 156 |

## Directional Checks

- [PASS] **privacy_mvp_better**: full_mvp PU-RER < no_firewall PU-RER
- [PASS] **semantic_protection**: full_mvp PU-RER < no_embedding PU-RER
- [PASS] **stateful_reconstruction_safer**: stateful CRR < stateless CRR
- [PASS] **rich_utility_ge_binary**: full_mvp task_success >= binary task_success
- [PASS] **firewall_reduces_rr**: no_firewall RR > full_mvp RR
