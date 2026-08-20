# E5-R1.2b Final Measurement Integration Closure

## Starting commit
d44e84417f15d5d59ed973be588e73c6390dfbe5

## R1.2b repair commit
8478f90

## E5-E1a development embedding manifest SHA
cache_sha: 4356c74b37405c4bb854fef7e90052ff8abb2e5d0affd3ef7aaee33e782d0a83

## Development embeddings regenerated
NO

## Development embedding coverage
225/225

## Canonical target embeddings
3/3

## Alias embeddings
3/3

## Embedding model
openai/text-embedding-v3

## E4 corpus changed
NO

## E4 annotations changed
NO

## Validation embeddings
0

## Test embeddings
0

## Test features
0

## Test access started
NO

---

## R1.2b Repair Items

### R1.2b-1: C1-C3 released-text propagation — PASS
- `_process_detector_only()` now accepts `raw_text` parameter
- C1 allowed row retains raw released_text
- C2 allowed row retains raw released_text
- C3 allowed row retains raw released_text
- C1 blocked row releases nothing (released_text=None, released_content_sha=BLOCKED_SENTINEL_SHA)
- C2 blocked row releases nothing
- C3 blocked row releases nothing
- `output_content_sha` uses `_compute_output_sha_from_text()` for actual content SHA

### R1.2b-2: RowResult transformation provenance — PASS
- `RowResult` extended with: initial_policy_action, final_policy_action, transformation_attempt_count, transformation_recheck_passed, released_content_sha
- `extended_result_to_row_result()` propagates all fields
- `row_result_to_dict()` serializes all fields
- C0-C3 populate consistently: initial=allow/block, final=allow/block, attempts=0, recheck=None

### R1.2b-2b: PU-RER aggregation — PASS
- `build_e5_results._compute_condition_counts()` uses final transformation provenance
- Safe redact (recheck_passed=True) → PU-RER contribution 0
- Safe abstract (recheck_passed=True) → PU-RER contribution 0
- Failed transform (recheck_passed=False) → PU-RER contribution 1
- Block → PU-RER contribution 0
- Raw allow → PU-RER contribution 1
- C4 redact/abstract with unknown recheck status → abort aggregation (fail closed)

### R1.2b-3: Transformed-output embedding recheck — PASS
- `embedding_safe = True` placeholder removed from `_recheck_and_escalate()`
- New shared helper: `e5_transformation_safety.check_transformation_embedding_safety()`
- Uses frozen model openai/text-embedding-v3 (same as E5-E1a)
- Compares against canonical target + alias embeddings from frozen cache
- Same tau_sem threshold as input semantic detector
- Transformed-output cache: separate role ("transformed_output"), deterministic cache key
- On-demand provider calls: development-only, logged, cached, reproducible
- `FirewallRunner.__init__()` accepts optional embedding_backend + embedding_cache

### R1.2b-4: Row-level recontamination — PASS
- RecontaminationResult schema with all §15 fields
- `e5_recontamination_evaluation.py`: dedicated row-level RR evaluator
- `compute_row_recontamination_rate()`: authoritative row-level RR computation
- `compute_row_level_recontamination_rate()` in e5_metrics.py: paper-facing path
- RR eligibility: attack_type == "recontamination" AND resolved
- RR numerator: eligible rows reaching unsafe state after actual released content
- RR denominator: all eligible resolved recontamination rows
- Protected rows remain in denominator
- No fake recontamination sequences created
- `is_rr_eligible_sequence()` deprecated but preserved for compatibility

---

## Metric Spec
- Schema version: 1.2.2
- RR unit_of_analysis: "row-level post-forget re-entry opportunity"
- PU-RER: requires transformation_recheck_passed == true for safe output
- Metric spec SHA: 5057906c38a31aba17b2f5fea8358dfefae120980ef431608ffd4e39893c85e7

## E5 Phase
- metric_spec_schema_version: 1.2.2
- development_embedding_complete: true
- development_feature_generation_complete: true
- r1_2_scientific_measurement_complete: true
- test_access_started: false

---

## Test Results

### Targeted tests (§29)
202 passed, 0 failed

### Full pytest
4058 passed, 4 skipped, 1 expected dirty-tree failure

### compileall
PASS (experiments, scripts, marble)

---

## Frozen Evidence Immutability
- E5-E1a embedding cache SHA: 4356c74b... (unchanged)
- E5-E1a embedding manifest SHA: 546e1282... (unchanged)
- E4 corpus SHA: unchanged
- E4 annotation/global-freeze SHA: unchanged

---

## GO Criteria for E5-E2 Calibration

- [x] C1 allowed rows preserve actual released text
- [x] C2 allowed rows preserve actual released text
- [x] C3 allowed rows preserve actual released text
- [x] C1-C3 CRR uses actual recipient-visible content
- [x] C4 row result preserves final transformation provenance
- [x] PU-RER consumes final transformation provenance
- [x] safe redact is not counted as leakage
- [x] safe abstract is not counted as leakage
- [x] failed/unknown transform safety cannot silently pass
- [x] embedding_safe=True placeholder is removed
- [x] transformed-output semantic recheck uses frozen real embedding model
- [x] transformed-output semantic recheck uses current tau_sem
- [x] transformed-output cache provenance is complete
- [x] RR is row-level
- [x] RR eligibility uses real recontamination corpus rows
- [x] RR starts from VERIFIED/post-forget safe state
- [x] blocked recontamination row remains safe
- [x] protected eligible row remains in RR denominator
- [x] A4 difference is behaviorally tested
- [x] no fake recontamination sequences are used
- [x] metric spec is updated and frozen
- [x] E5-E1a embedding SHA unchanged
- [x] E4 roots unchanged
- [x] validation untouched
- [x] test untouched
- [x] full pytest passes
- [x] compileall passes

**READY FOR E5-E2 = YES**
