# Single-Target Smoke Study Summary

- **Status**: DIAGNOSTIC ONLY
- **Commit**: 27313bd23b2ee8edcd5b237074f667cf990beb6a
- **Mode**: diagnostic
- **Fixtures**: 3
- **Seeds**: 3
- **Conditions**: 13
- **Total runs**: 117
- **Audit valid**: False
- **Audit errors**: 105
- **Duplicate identities**: 0
- **Utility retention**: 1.0000

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
