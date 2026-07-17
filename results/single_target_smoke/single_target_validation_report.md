# Single-Target Validation Report: GO

## Run Identity

- **Repository commit**: f87986b29f54771d9b1fdcd33b1fcaf5d1d5e28b
- **Repository clean**: True
- **Generated at**: 2026-07-17T09:18:07.892803+00:00
- **Mode**: release

## Fixture Matrix

- **Fixtures**: 3
- **Seeds**: 3
- **Conditions**: 9
- **Total runs**: 81

## Audit Status

- **Audit valid**: True
- **Audit errors**: 0
- **Duplicate identities**: 0

## Manifest Status

- **Manifest valid**: True

## Aggregate Metrics

| Metric | Value | Numerator | Denominator |
|--------|------:|----------:|------------:|
| PU_RER | 0.2130 | 69 | 324 |
| CRR | 0.0889 | 12 | 135 |
| RR | 0.3333 | 9 | 27 |
| FBR | 0.0000 | 0 | 108 |

## Directional Checks

- [PASS] **privacy_mvp_better** (strict_improvement): full_mvp PU-RER < no_firewall PU-RER
  - LHS: 0.0, RHS: 0.3333333333333333
- [PASS] **semantic_protection** (strict_improvement): full_mvp PU-RER < no_semantic PU-RER
  - LHS: 0.0, RHS: 0.16666666666666666
- [PASS] **stateful_reconstruction_safer** (strict_improvement): stateful CRR < stateless CRR
  - LHS: 0.0, RHS: 0.2
- [PASS] **rich_utility_ge_binary** (non_inferiority): full_mvp task_success >= binary task_success
  - LHS: 3, RHS: 3
- [PASS] **continuous_rr_lt_finite** (strict_improvement): continuous RR < monitoring_0 RR
  - LHS: 0.0, RHS: 1.0

## Utility Pairing

- **Expected pairs**: 9
- **Matched pairs**: 9
- **Unmatched pairs**: 0
- **Baseline successful pairs**: 3
- **Utility retention**: 1.0000

## GO/NO-GO Decision

**GO**
