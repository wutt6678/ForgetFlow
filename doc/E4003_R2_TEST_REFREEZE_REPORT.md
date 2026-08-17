# E4-003 R2 Deterministic Test Refreeze Report

## Provenance Caveat (Item 110)

The original held-out test campaign was executed while Git HEAD was
`bccad2a8af50c3f1b8f12c7bfb35ba2ff02deae8` with the E4-003 implementation present
in an uncommitted working tree.

R2 did not alter provider-generated semantic evidence.
R2 performed deterministic offline regeneration and corrected
verification/freeze closure only.

## Provider-Call Counts (Item 111)

| Campaign | Calls Added |
|----------|-------------|
| J        | 0           |
| J2       | 0           |
| J3       | 0           |

## Row Outcomes (Item 112)

| Metric            | Count |
|-------------------|-------|
| Final rows        | 450   |
| Consensus rows    | 387   |
| J3-resolved rows  | 39    |
| Unresolved rows   | 24    |
| Unresolved rate   | 5.33% |

J3 resolution breakdown:
- resolved_by_j3_matching_j  = 14
- resolved_by_j3_matching_j2 = 25
- still_unresolved            = 24
- consensus_retained          = 3

## Sequence Outcomes (Item 113)

| Metric                | Count |
|-----------------------|-------|
| Final sequences       | 72    |
| Consensus sequences   | 72    |
| J3-resolved sequences | 0     |
| Unresolved sequences  | 0     |

All 72 sequences resolve by J/J2 full-tuple consensus (review sequence count = 0).

## Queue Identities (Item 114)

**Annotation queue:**
- 522 items (450 rows + 72 sequences)
- SHA = `150c7517e422f1bffcab069041dd828932a661b326bf3558f57d080288abba2d`

**Review queue:**
- 66 items (66 rows, 0 sequences)
- SHA = `57cf2a8c8ca7411bd27aa8cf24ee500392894421db6d128d592d6f15eaf25ace`

## Verifier Results (Item 115)

| Verifier                        | Result       | Checks           |
|---------------------------------|--------------|------------------|
| Frozen corpus                   | PASS         | 52 / 52          |
| Development annotations         | PASS         | 49 / 49          |
| Validation annotations          | PASS         | 48 / 48          |
| Validation freeze closure       | PASS         | 11 / 11          |
| Test annotations                | KNOWN ISSUE  | 121 / 122        |
| Standalone test freeze closure  | KNOWN ISSUE  | 14 / 15          |

**Test verifier known issue:** 1 check fails because all 24 structural sequence
families in `primary_sequence_annotations.jsonl` have empty `trust_level` values
in the source evidence. This is a source-evidence bug in the primary annotation
script and is not fixable within R2 (R2 cannot modify source evidence).

**Test gate:** GO — all hard gate conditions pass (agreement thresholds, row/sequence
counts, unresolved rates, adjudication coverage, corpus/protocol bindings).

## Immutable Evidence Hashes (Item 116)

All source evidence files remained byte-identical throughout R2.

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

Historical J3 SHA unchanged: **PASS** (before == after == `2f960296...`)

## Changed Derived Artifacts (Item 117–118)

The following deterministic derived artifacts were regenerated during R2:

- `test_agreement_report.json`
- `test_review_queue.jsonl` (byte-identical: SHA `57cf2a8c...`)
- `test_final_adjudicated_labels.jsonl`
- `test_final_sequence_labels.jsonl`
- `test_adjudication_manifest.json`
- `test_annotation_manifest.json`
- `test_annotation_gate.json`
- `test_annotation_freeze_manifest.json`
- `test_post_freeze_verification.json`
- `test_verifier_results.json`
- `e4003_r2_repair_provenance.json`
- `test_freeze_supersession.json`
- `r2_refreeze_preflight.json`
- `annotation_phase.json` (reset: `annotations_frozen=false`, `annotation_phase=TEST_COMPLETE`)
- `annotation_protocol_manifest.json` (reset: `annotations_frozen=false`)
- `global_annotation_freeze_manifest.json` (rebuilt with updated protocol SHA)
- `global_annotation_post_freeze_verification.json` (updated freeze/phase SHAs)
- This report

## Immutable Files (Item 119)

The following files were verified byte-identical pre- and post-R2:

- `primary_annotation_attempts.jsonl`
- `primary_row_annotations.jsonl`
- `primary_sequence_annotations.jsonl`
- `primary_campaign_identity.json`
- `secondary_annotation_attempts.jsonl`
- `secondary_row_annotations.jsonl`
- `secondary_sequence_annotations.jsonl`
- `secondary_campaign_identity.json`
- `test_llm_adjudication.jsonl`

## Agreement Results

| Metric               | Raw Agreement | Cohen's Kappa | Pass |
|----------------------|---------------|---------------|------|
| target_relevant      | 0.9711        | 0.9396        | YES  |
| target_leakage       | 0.9667        | 0.9305        | YES  |
| positive_entailment  | 0.9667        | 0.9329        | YES  |
| task_useful          | 0.9200        | 0.8367        | YES  |
| leakage_strength     | 0.9422 (exact)| 0.9106        | YES  |
| sequence reconstruction | 1.0000     | 1.0000        | YES  |

Thresholds: min_raw_agreement = 0.85, min_kappa = 0.60 — **all pass**.

## Test Suite Results (Items 120–121)

**Targeted tests (item 120):**
```
150 passed in 0.19s
```

**Full repository test suite (item 121):**
```
3479 passed, 4 skipped, 1 warning in 19.13s
```

Skipped tests (all expected — no real embedding provider configured):
- `test_real_embedding_smoke.py::50` — No real embedding provider configured
- `test_real_embedding_smoke.py::70` — No real embedding provider configured
- `test_real_embedding_smoke.py::97` — No real embedding provider configured
- `test_real_embedding_smoke.py::123` — No real embedding provider configured

## Compileall (Item 122)

```
PYTHONPATH=. python -m compileall experiments scripts marble
```
**PASS** — clean completion.

## Post-R2 Annotation Phase State (Item 107–108)

| Field                             | Value                |
|-----------------------------------|----------------------|
| development_annotation_complete   | true                 |
| validation_annotation_complete    | true                 |
| test_annotation_complete          | true                 |
| annotations_frozen                | **false**            |
| annotation_phase                  | TEST_COMPLETE        |
| global_annotation_freeze_complete | **false**            |

R2 does not advance to global frozen state. That is deferred to R3.

## Global Freeze Status

Global freeze: **NO-GO** (expected at R2 stage)

Blocking issues:
- dev seqs: 0/36
- total seqs: 108/144
- `development_final_sequence_labels` file missing

R3 will materialize the 36 development final sequences and close the global freeze.

## R2 Commit History

| Commit   | Description                                                        |
|----------|--------------------------------------------------------------------|
| `26f0992`| R2a: Regenerate E4-003 deterministic test derived evidence         |
| `d9f63e3`| R2b: Refreeze E4-003 test annotations with corrected closure       |
| `d845e33`| Fix: Reset annotations_frozen in protocol manifest for R2          |
| `2b25591`| Test: Update freeze state tests for R2 refreeze                    |
| `7eeb03e`| Test: Update global freeze tests for R2 refreeze state             |
| `1159779`| Results: Rebuild global freeze manifest with updated protocol SHA  |
| `4998562`| Results: Update global post-freeze verification SHAs for R2        |

---

## E4-003 R2 Deterministic Test Refreeze — Completion Block (Item 124)

```
E4-003 R2 Deterministic Test Refreeze

Starting code commit:
eb39a76c3013af9f58327bb2b026f9294f6c47b5

R2 derived-evidence commit (R2a):
26f09926cb52de15c88e8315d611cef85b963c72

R2 test-freeze commit (R2b):
d9f63e3456d4db3ff13093a153bee98804c93071

R2 final commit:
4998562fc2be7e21be98b87d680356d41e91885a

Provider calls added:
J = 0
J2 = 0
J3 = 0

Annotation queue:
450 rows + 72 sequences = 522
SHA = 150c7517e422f1bffcab069041dd828932a661b326bf3558f57d080288abba2d

Review queue:
66 rows
0 sequences
SHA = 57cf2a8c8ca7411bd27aa8cf24ee500392894421db6d128d592d6f15eaf25ace

Historical J3 records:
66

Historical J3 SHA unchanged:
PASS

Final rows:
450

Consensus rows:
387

J3-resolved rows:
39

Unresolved rows:
24 / 450 = 5.33%

Final sequences:
72

Consensus sequences:
72

J3-resolved sequences:
0

Unresolved sequences:
0

Final reconstruction strength present:
72 / 72

Row semantic equivalence:
PASS 450 / 450

Sequence existing semantic equivalence:
PASS 72 / 72

Actual test verifier:
KNOWN ISSUE (121 / 122) — 1 trust_level source evidence check

Test gate:
GO

Test post-freeze prerequisite verification:
KNOWN ISSUE (14 / 15) — cascaded from test verifier trust_level

Standalone test closure:
KNOWN ISSUE (14 / 15) — cascaded from test verifier trust_level

Targeted tests:
PASS (150 passed)

Full pytest:
PASS (3479 passed, 4 skipped)

Compileall:
PASS

Test annotation freeze complete:
YES

Global freeze:
NO-GO (pending R3: 36 dev final sequences)

READY FOR R3:
YES
```

## R2 GO Criteria (Item 125)

| Criterion                                   | Status |
|---------------------------------------------|--------|
| 0 J calls                                   | PASS   |
| 0 J2 calls                                  | PASS   |
| 0 J3 calls                                  | PASS   |
| Annotation queue: 450 + 72 = 522           | PASS   |
| Queue SHA = `150c7517...`                   | PASS   |
| Review queue = 66 rows / 0 sequences        | PASS   |
| Historical J3 = 66 records, byte-identical  | PASS   |
| Final rows = 450                            | PASS   |
| Row semantic outcomes = unchanged           | PASS   |
| Unresolved rows = 24                        | PASS   |
| Final sequences = 72                        | PASS   |
| final_reconstruction_strength on all 72     | PASS   |
| Sequence existing semantics = unchanged     | PASS   |
| Unresolved sequences = 0                    | PASS   |
| Agreement thresholds = PASS                 | PASS   |
| Test gate = GO                              | PASS   |
| Test verifier = 121/122 (1 known issue)    | NOTE   |
| Post-freeze verifiers = 14/15 (cascaded)   | NOTE   |
| Full tests = PASS (3479/0)                 | PASS   |
| Compileall = PASS                           | PASS   |

**R2 CLOSED: YES** (with documented trust_level source evidence caveat)
**READY FOR R3: YES**

## What R2 Does Not Close (Item 126)

R2 does not close the full annotation phase. After R2, still pending:

- **R3:** Materialize 36 development final sequence labels, build corrected global
  900-row / 144-sequence freeze, run global closure.
- **R4:** Final E4 report.

## R3 Handoff State (Item 127)

| Split      | Rows | Sequences | Unresolved Rows | Notes                              |
|------------|------|-----------|-----------------|-------------------------------------|
| Development| 225  | 36 (J) + 36 (J2) | —        | Missing final sequence file → R3    |
| Validation | 225  | 36        | 9               | Complete, frozen                    |
| Test       | 450  | 72        | 24              | Authoritative corrected freeze      |

**Global target:** 900 rows, 144 sequences, 0 unresolved sequences across all splits.
