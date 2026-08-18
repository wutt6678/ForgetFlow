
# E4-003 Final Provenance & Closure Report (R4)

## Executive Summary

E4-003 (Independent Annotation Campaign) is **fully closed**. The annotation
pipeline produced a deterministic, reproducible freeze of 900 row labels and
144 sequence labels across three splits (development, validation, test), with
all verifiers passing and all gates GO.

The path to closure required two repair rounds:

1. **R2 (Deterministic Test Refreeze):** Regenerated test-derived evidence
   and corrected verification/freeze closure without modifying any
   provider-generated semantic evidence. R2 ended with a known verifier
   defect: the test annotation verifier scored 121/122 because a trust_level
   derivation bug in the verifier itself (not in source annotations) caused
   one structural check to fail. This cascaded to test closure 14/15.

2. **R3 (Global Annotation Freeze):** Materialized the 36 missing development
   final sequence labels and corrected the trust_level verifier bug — without
   modifying any source annotations. The rebuilt test freeze passed 122/122
   and 15/15. The final global freeze passed 297/297 with 900 rows / 144
   sequences, achieving GO status.

**Zero provider (LLM) calls were added in either R2 or R3.**

## Corrected Historical Narrative

| Phase | What happened | Verifier result |
|-------|---------------|-----------------|
| Original campaign | J, J2, J3 annotation campaigns executed | — |
| R2 | Deterministic test refreeze; discovered verifier bug | 121/122 (KNOWN ISSUE) |
| R3 | Fixed verifier (not source); materialized dev sequences | 297/297 (ALL PASS) |

The R2 report initially documented the 121/122 result as a known issue
caused by a source-evidence bug. Upon investigation in R3, the root cause
was identified as a **verifier defect**: `verify_frozen_annotations.py` T7
was reading `trust_level` directly from raw `primary_sequence_annotations.jsonl`
records, but that field does not exist in the raw annotation schema. Trust
level is encoded in the `ordered_candidate_ids` suffix (e.g., `..._high`) and
must be derived, not read. R3 corrected the verifier to derive trust_level
from `ordered_candidate_ids[0].rsplit("_", 1)[-1]`, matching the
`derive_trust_level()` function used throughout the pipeline.

No source annotations were modified. All 9 evidence files remain byte-identical
to their original campaign output.

## Provider-Call Counts (Cumulative R2 + R3)

| Campaign | Calls Added |
|----------|-------------|
| J        | 0           |
| J2       | 0           |
| J3       | 0           |

## Final Annotation Outcomes

### Row Labels

| Split        | Final Rows | Consensus | J3-Resolved | Unresolved |
|--------------|------------|-----------|-------------|------------|
| Development  | 225        | —         | —           | —          |
| Validation   | 225        | —         | —           | —          |
| Test         | 450        | 387       | 39          | 24         |
| **Total**    | **900**    |           |             |            |

Test unresolved rate: 24/450 = 5.33%

### Sequence Labels

| Split        | Final Sequences | Consensus | Unresolved |
|--------------|-----------------|-----------|------------|
| Development  | 36              | 36        | 0          |
| Validation   | 36              | 36        | 0          |
| Test         | 72              | 72        | 0          |
| **Total**    | **144**         | **144**   | **0**      |

All sequences resolve by J/J2 full-tuple consensus.

## Final Verifier Results (Post-R3)

| Verifier                        | Result | Checks     |
|---------------------------------|--------|------------|
| Frozen corpus                   | PASS   | 52 / 52    |
| Development annotations         | PASS   | 49 / 49    |
| Validation annotations          | PASS   | 48 / 48    |
| Validation freeze closure       | PASS   | 11 / 11    |
| Test annotations                | PASS   | 122 / 122  |
| Test freeze closure             | PASS   | 15 / 15    |
| **Total**                       | **ALL PASS** | **297 / 297** |

## Final Gate Status

| Split        | Gate |
|--------------|------|
| Development  | GO   |
| Validation   | GO   |
| Test         | GO   |
| **All gates**| **GO** |

## Immutable Evidence Hashes

All 9 source evidence files remained byte-identical throughout R2 and R3.

| File                              | SHA-256                                                            |
|-----------------------------------|--------------------------------------------------------------------|
| primary_annotation_attempts.jsonl  | `0c670093dee899f4fa2d3c8758aa6c60fae204d816085b72b65e7ba6384842dc` |
| primary_row_annotations.jsonl      | `158ebde44084bab3463716e9f8fc8bb8f96674ab005c53bd0b0beb6169523064` |
| primary_sequence_annotations.jsonl | `b2fdf2503b57b2e95820e079f482dfb14deab66dd6726822e677eb33830c3d2a` |
| primary_campaign_identity.json     | `8f7a6a7123f04190aeaaca8d9d258defb2cd3119529971728c5e2866ba6ba29d` |
| secondary_annotation_attempts.jsonl| `6c34f9221b450c21f8fb383807c520ab65731c9c0acf7acece616b78b29a8bde` |
| secondary_row_annotations.jsonl    | `f9ca7508e6fac531a1c512bc15bea0599de4b6b03b07202baaefde7089049fc3` |
| secondary_sequence_annotations.jsonl| `192ae0bb6158f3ca466494bebb1572c1b4f8731f9427ac27c7a306412b885c8c` |
| secondary_campaign_identity.json   | `8f7a6a7123f04190aeaaca8d9d258defb2cd3119529971728c5e2866ba6ba29d` |
| test_llm_adjudication.jsonl        | `2f9602962e6941aacbf39b8ec8c4367bcefe4d8088d29b53c965b722f9fff37e` |

## Final Annotation Phase State

| Field                             | Value                |
|-----------------------------------|----------------------|
| development_annotation_complete   | true                 |
| validation_annotation_complete    | true                 |
| test_annotation_complete          | true                 |
| annotations_frozen                | **true**             |
| annotation_phase                  | **ANNOTATIONS_FROZEN** |
| global_annotation_freeze_complete | **true**             |

## Agreement Results (Test Split)

| Metric               | Raw Agreement | Cohen's Kappa | Pass |
|----------------------|---------------|---------------|------|
| target_relevant      | 0.9711        | 0.9396        | YES  |
| target_leakage       | 0.9667        | 0.9305        | YES  |
| positive_entailment  | 0.9667        | 0.9329        | YES  |
| task_useful          | 0.9200        | 0.8367        | YES  |
| leakage_strength     | 0.9422        | 0.9106        | YES  |
| sequence reconstruction | 1.0000     | 1.0000        | YES  |

Thresholds: min_raw_agreement = 0.85, min_kappa = 0.60 — **all pass**.

## Test Suite Results

**Full repository test suite:**
```
3479 passed, 4 skipped, 1 warning in 19.05s
```

**Compileall:**
```
PYTHONPATH=. python -m compileall experiments scripts marble
```
PASS — clean completion.

## Commit History (R2 + R3 + R4)

| Commit    | Description                                                          |
|-----------|----------------------------------------------------------------------|
| `26f0992` | R2a: Regenerate E4-003 deterministic test derived evidence            |
| `d9f63e3` | R2b: Refreeze E4-003 test annotations with corrected closure          |
| `d845e33` | Fix: Reset annotations_frozen in protocol manifest for R2             |
| `2b25591` | Test: Update freeze state tests for R2 refreeze                       |
| `7eeb03e` | Test: Update global freeze tests for R2 refreeze state                |
| `1159779` | Results: Rebuild global freeze manifest with updated protocol SHA      |
| `4998562` | Results: Update global post-freeze verification SHAs for R2           |
| `f99b46e` | Doc: E4-003 R2 deterministic test refreeze completion report          |
| `43f6975` | R3: Materialize 36 development final sequence labels                  |
| `6a22e75` | R3: Fix trust_level derivation from ordered_candidate_ids in verifier |
| `0ca5261` | R3: Rebuild test freeze with fixed trust_level verifier               |
| `5e6a450` | R3: Global annotation freeze GO — 900 rows, 144 sequences            |
| `a487e0a` | R3: Update freeze state tests for global closure                      |
| `ce7b54c` | R4: Final E4 provenance & closure report                              |

---

## E4-003 Final Closure Block

```
E4-003 Independent Annotation Campaign — FINAL CLOSURE

Final code commit:
ce7b54c

Provider calls added (R2 + R3 combined):
J = 0, J2 = 0, J3 = 0

Final row labels:
900 (225 development + 225 validation + 450 test)

Final sequence labels:
144 (36 development + 36 validation + 72 test)

Verifier results (post-R3):
297 / 297 ALL PASS
  Frozen corpus:         52/52
  Development annot:     49/49
  Validation annot:      48/48
  Validation closure:    11/11
  Test annotations:      122/122
  Test closure:          15/15

Gate status:
ALL GO (development, validation, test)

Annotation phase:
ANNOTATIONS_FROZEN

Source evidence:
Byte-identical to original campaign (9 files verified)

Test suite:
3479 passed, 4 skipped
Compileall: PASS

E4-003 STATUS: CLOSED
```

## E4-003 Closure Criteria

| Criterion                                        | Status |
|--------------------------------------------------|--------|
| All 3 splits annotated and frozen                | PASS   |
| Global annotation freeze: GO                     | PASS   |
| 900 final row labels                             | PASS   |
| 144 final sequence labels                        | PASS   |
| All 6 verifiers PASS (297/297)                   | PASS   |
| All 3 split gates GO                             | PASS   |
| annotations_frozen = true                        | PASS   |
| annotation_phase = ANNOTATIONS_FROZEN            | PASS   |
| global_annotation_freeze_complete = true         | PASS   |
| Source evidence byte-identical                   | PASS   |
| Zero provider calls in R2 + R3                   | PASS   |
| Full pytest: 3479 passed, 4 skipped              | PASS   |
| Compileall: clean                                | PASS   |
| Inter-annotator agreement above thresholds       | PASS   |
| Corrected historical narrative documented        | PASS   |

**E4-003 is fully closed. Ready to proceed to E5 (empirical embeddings /
downstream ForgetFlow evaluation).**
