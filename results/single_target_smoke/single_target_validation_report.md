# Single-Target Validation Report: DIAGNOSTIC ONLY

## Run Identity

- **Repository commit**: 85eb04cb6835ad33ea5d3287582dad163b9c96b7
- **Repository clean**: True
- **Generated at**: 2026-08-04T02:05:56.158394+00:00
- **Mode**: diagnostic

## Fixture Matrix

- **Fixtures**: 3
- **Seeds**: 3
- **Conditions**: 14
- **Total runs**: 126

## Audit Status

- **Audit valid**: True
- **Audit errors**: 0
- **Duplicate identities**: 0

## Manifest Status

- **Manifest valid**: True

## Aggregate Metrics

| Metric | Value | Numerator | Denominator |
|--------|------:|----------:|------------:|
| PU_RER | 0.0966 | 69 | 714 |
| CRR | 0.5714 | 24 | 42 |
| RR | 0.5000 | 21 | 42 |
| FBR | 0.0000 | 0 | 168 |

## Directional Checks

- [PASS] **privacy_mvp_better** (strict_improvement): full_mvp PU-RER < no_firewall PU-RER
  - LHS: 0.0, RHS: 0.17647058823529413
- [PASS] **semantic_protection** (strict_improvement): full_mvp PU-RER < no_embedding PU-RER
  - LHS: 0.0, RHS: 0.058823529411764705
- [PASS] **stateful_reconstruction_safer** (strict_improvement): stateful CRR < stateless CRR
  - LHS: 0.0, RHS: 1.0
- [PASS] **rich_utility_ge_binary** (non_inferiority): full_mvp task_success >= binary task_success
  - LHS: 3, RHS: 0
- [PASS] **firewall_reduces_rr** (strict_improvement): no_firewall RR > full_mvp RR
  - LHS: 1.0, RHS: 0.0
- [PASS] **binary_policy_reconstruction_increases** (non_inferiority): binary_policy CRR >= full_mvp CRR
  - LHS: 0.0, RHS: 0.0
- [PASS] **one_time_monitoring_recontamination_increases** (non_inferiority): one_time_monitoring RR >= full_mvp RR
  - LHS: 0.0, RHS: 0.0

## Utility Pairing

- **Expected pairs**: 9
- **Matched pairs**: 9
- **Unmatched pairs**: 0
- **Baseline successful pairs**: 3
- **paired_policy_utility_retention**: 1.0000

## GO/NO-GO Decision

**DIAGNOSTIC ONLY**

This run was in diagnostic mode and is not release-valid.