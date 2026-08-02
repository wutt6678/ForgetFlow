# Single-Target Validation Report: DIAGNOSTIC ONLY

## Run Identity

- **Repository commit**: 3699ecb527f23da5aa344099bfad05e1e455cef4-dirty
- **Repository clean**: False
- **Generated at**: 2026-08-02T15:07:54.305931+00:00
- **Mode**: diagnostic

## Fixture Matrix

- **Fixtures**: 3
- **Seeds**: 3
- **Conditions**: 13
- **Total runs**: 117

## Audit Status

- **Audit valid**: True
- **Audit errors**: 0
- **Duplicate identities**: 0

## Manifest Status

- **Manifest valid**: True

## Aggregate Metrics

| Metric | Value | Numerator | Denominator |
|--------|------:|----------:|------------:|
| PU_RER | 0.0724 | 48 | 663 |
| CRR | 0.0769 | 18 | 234 |
| RR | 0.2308 | 9 | 39 |
| FBR | 0.0000 | 0 | 156 |

## Directional Checks

- [PASS] **privacy_mvp_better** (strict_improvement): full_mvp PU-RER < no_firewall PU-RER
  - LHS: 0.0, RHS: 0.17647058823529413
- [PASS] **semantic_protection** (strict_improvement): full_mvp PU-RER < no_embedding PU-RER
  - LHS: 0.0, RHS: 0.058823529411764705
- [PASS] **stateful_reconstruction_safer** (strict_improvement): stateful CRR < stateless CRR
  - LHS: 0.0, RHS: 0.16666666666666666
- [PASS] **rich_utility_ge_binary** (non_inferiority): full_mvp task_success >= binary task_success
  - LHS: 3, RHS: 0
- [PASS] **firewall_reduces_rr** (strict_improvement): no_firewall RR > full_mvp RR
  - LHS: 1.0, RHS: 0.0

## Utility Pairing

- **Expected pairs**: 9
- **Matched pairs**: 9
- **Unmatched pairs**: 0
- **Baseline successful pairs**: 3
- **Utility retention**: 1.0000

## GO/NO-GO Decision

**DIAGNOSTIC ONLY**

This run was in diagnostic mode and is not release-valid.