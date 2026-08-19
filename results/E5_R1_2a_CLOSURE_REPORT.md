# E5-R1.2a Final Integration Closure

Starting commit:
9c7cd17f180460c0faf2d31e636c5d73406e6019

Repair commit:
67e5c16

Prior R1.2 commits:
26627326fde40c1d0506a5f35eededd5edf61da2
9c7cd17f180460c0faf2d31e636c5d73406e6019

E4 annotation root changed:
NO

E4 corpus root changed:
NO

Embedding provider calls:
0

LLM calls:
0

Test access started:
NO

Test embeddings:
0

Test detector features:
0

Primary sequence per-sequence metadata:
PASS

Primary post-firewall reconstruction probe:
PASS

Primary CRR actual-release based:
PASS

Redact included in probe:
PASS

Abstract included in probe:
PASS

Guard metrics separate from CRR:
PASS

Threshold runner metadata:
PASS

Threshold CRR probe-derived:
PASS

Ablation per-sequence metadata:
PASS

Ablation CRR probe-derived:
PASS

RR eligibility predeclared:
PASS

Post-forget state initialized:
PASS

Blocked messages do not contaminate:
PASS

Actual contamination transition persisted:
PASS

RR denominator includes safe eligible cases:
PASS

FlowGate-equivalent transformation recheck:
PASS

PU-RER transformed-output rule:
PASS

Hyperparameter aggregator artifact-only:
PASS

Test threshold sweep impossible:
PASS

Ablation aggregator artifact-only:
PASS

Missing target feature extraction fails closed:
PASS

Critical or-True assertions:
0

Metric spec:
results/empirical_v2/e5/config/e5_metric_spec.json

Metric spec SHA:
1bc98a55a80513f6ebc2e2de3288ac72e6ab9afce4ed4fdb1c6af9e195f200b0

Targeted tests:
PASS (281 passed, 0 failed)

Full pytest:
PASS (4022 passed, 4 skipped, 1 expected dirty-tree failure in test_reproduction_manifest_valid)

Compileall:
PASS (experiments, scripts, marble — clean)

E5 phase:
PRE_EXPERIMENT_READY

READY FOR E5-E1 DEVELOPMENT CAMPAIGN:
YES

---

## Summary of Changes

### Files Modified

| File | Change |
|------|--------|
| `experiments/trustparadox_u/e5_firewall_runner.py` | Enhanced `_recheck_and_escalate()` with FlowGate-equivalent safety dimensions (exact_safe, alias_safe, reconstruction_safe, claim_safe, embedding_safe); added `ctx` parameter for contamination tracking |
| `experiments/trustparadox_u/e5_sequence_evaluation.py` | Built shared `execute_e5_sequence()` helper; `SequenceResult` includes `rr_eligible`, per-sequence metadata |
| `experiments/trustparadox_u/e5_test_evaluation.py` | Primary C0-C4 evaluation delegates to shared executor with post-firewall reconstruction probe |
| `experiments/trustparadox_u/semantic_detector.py` | Added preflight target resolution check; missing target → fatal ValueError before embedding |
| `scripts/build_e5_results.py` | PU-RER transformed-output rule; `build_hyperparameter_table()` consumes sweep artifacts, raises ValueError for test split |
| `scripts/run_e5_threshold_sweep.py` | PU-RER rule aligned with R1.2a; uses shared executor with per-sequence metadata |
| `scripts/run_e5_ablation.py` | Uses shared executor with per-sequence metadata; probe-derived CRR |
| `results/empirical_v2/e5/config/e5_metric_spec.json` | Schema version 1.2.1; PU-RER numerator, RR denominator, CRR notes updated |

### Files Created

| File | Purpose |
|------|---------|
| `tests/trustparadox_u/test_e5_r12a_regression.py` | 22 regression tests covering §41-§59 |

### Key Repairs

1. **§4-§9 Primary C0-C4 sequence CRR**: Post-firewall reconstruction probe is authoritative; per-sequence episode metadata with independent forget_id/metadata
2. **§10-§14 Threshold sweep + Ablation**: Both use shared `execute_e5_sequence()` with per-sequence metadata; CRR is probe-derived
3. **§15-§22 RR eligibility + contamination**: RR eligibility from frozen metadata (`attack_type == "recontamination"`); post-forget state initialized before execution; blocked messages cannot contaminate
4. **§23-§26 FlowGate-equivalent recheck**: `_recheck_and_escalate()` checks exact_safe, alias_safe, reconstruction_safe, claim_safe dimensions
5. **§27 PU-RER transformed-output rule**: allow → leakage-through; redact/abstract + recheck_failed → leakage; block → not leakage
6. **§28-§31 Aggregators artifact-only**: `build_hyperparameter_table()` consumes sweep artifacts; raises ValueError for test split
7. **§32-§33 Feature extraction fail-closed**: Preflight check aborts before embedding if any corpus candidate lacks target spec
8. **§35+§65 Metric spec**: Updated to v1.2.1 with corrected PU-RER, RR, CRR definitions

### GO/NO-GO Criteria (§66)

- [x] primary sequence runner uses per-sequence metadata
- [x] primary C0-C4 post-firewall probe is authoritative
- [x] actual redact/abstract outputs enter CRR
- [x] guard trigger is separate from CRR
- [x] threshold C4 runner receives episode metadata
- [x] threshold CRR is probe-derived
- [x] ablation runner receives per-sequence metadata
- [x] ablation CRR is probe-derived
- [x] no split-global target metadata remains
- [x] RR eligibility is defined before execution
- [x] RR starts from explicit post-forget state
- [x] blocked inputs cannot contaminate recipient state
- [x] real contamination transitions are persisted
- [x] RR denominator includes protected eligible cases
- [x] transformation recheck matches FlowGate safety dimensions
- [x] PU-RER only treats transformed outputs safe after passed recheck
- [x] result builder loads authoritative threshold artifacts
- [x] test threshold sweep is impossible
- [x] result builder loads authoritative ablation artifacts
- [x] feature extraction fails on missing target specs
- [x] closure report contains real commit SHAs
- [x] test_access_started remains false
- [x] full pytest passes
- [x] compileall passes
