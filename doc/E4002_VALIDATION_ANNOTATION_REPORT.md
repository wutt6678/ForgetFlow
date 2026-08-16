# E4-002 Validation Annotation + Agreement Confirmation

## Provenance

| Field | Value |
|-------|-------|
| Starting commit | `8948acc` (doc: add E4-001A completion report) |
| E4-001A provenance correction commit | `692d8435c52e5523e6ee31138a763a7b3add77c0` |
| Validation annotation source commit | `0ed97256dc8e92907a55dd1a4845a9d52fa929bf` |
| Faulty validation GO/report commit | `615934c88783379756311f6232cbb4a626208dc7` |
| Sequence identity repair code commit | `b9c6886689ca277d84605d547546c84f1d1ade74` |
| Validation gate hardening commit | `438922feebd879d2f89a9874e6e5a66a6c885ef2` |
| Corrected validation evidence commit | `3d167ed12a0f1c6576d55a78ae274614601c69d3` |
| Corrected validation freeze commit | `fd72b0054f67e9ff05fa5311d91550d830155db6` |
| Validation report commit | `7cf0eb33bd91064b6cda9257eeb5bc385e14a302` |
| Provenance/freeze cleanup code commit (P1) | `ec89719dd789b60a5d3eade5a2a792a16201b240` |
| Corrected evidence/refreeze commit (P2) | `6bbde250c2f8758a8545f5c9ab2359213aed5f06` |
| Provenance closure logic commit (C1) | `12a6d42b41f5ab918a4349ce34eb519bb39c1c39` |
| Provenance closure evidence commit (C2) | `f7fbdf6d9bcd3cfc0c388948a643cd9488912915` |
| Closure-ordering fix commit (F1) | `14b9614` |
| Post-freeze evidence regeneration commit (F2) | `9ce8f8e` |
| Report generated from evidence commit | `9ce8f8e` (F2) |
| Previous GO superseded | YES (trust-conditioned sequence units collapsed by sequence_family_id) |
| Frozen corpus manifest SHA | `6b626f66734f809d422ba6f8b88f95f68a9515a7ab5b62535f86cae80d8d10b2` |
| Frozen annotation protocol SHA (manifest file) | `d5dfea3187d4986d5937dc04771b77cd402caeed11776652ff6b8291db60cf83` |
| Annotation schema SHA-256 | `d0ff5974b6aa52cf562bea5921840c032a860a91a3512f7fe8f768f6bbe005f6` |
| Prompt manifest SHA-256 | `6bce9ab304bda040cf1cec657a95685d7d5e8d5561070eed6d487502c2ddce77` |
| Annotation config SHA-256 | `e6c6e414f215f1578f4cad95b630201994d5bf5a3ddea931d90a8207c19976cf` |

## Protocol Verification

| Check | Result |
|-------|--------|
| Protocol hash unchanged | **PASS** |
| Frozen corpus verifier | **PASS** (52/52 checks) |
| Development annotation verifier | **PASS** (49/49 checks) |
| Validation annotation verifier | **PASS** (48/48 checks) |

## Validation Target Resolution

| Metric | Value |
|--------|-------|
| Rows | 225 / 225 |
| Sequences | 36 / 36 |
| Empty targets | 0 |

## Primary Annotator (J)

| Field | Value |
|-------|-------|
| Role | J |
| Model | qwen3.8-max |
| Provider | litellm (openai-compatible) |
| Attempts | 274 |
| Row labels | 225 / 225 |
| Sequence labels | 36 / 36 |
| Unique terminal items | 261 (225 rows + 36 sequences) |
| Internal retries (retry_index > 0) | 0 |
| Repeat/resume items | 13 |
| Provider failures | 0 |

## Secondary Annotator (J2)

| Field | Value |
|-------|-------|
| Role | J2 |
| Model | glm-5.2 |
| Provider | litellm (openai-compatible) |
| Attempts | 277 |
| Row labels | 225 / 225 |
| Sequence labels | 36 / 36 |
| Unique terminal items | 261 (225 rows + 36 sequences) |
| Internal retries (retry_index > 0) | 14 |
| Repeat/resume items | 2 |
| Provider failures (empty_response: 15, malformed: 1) | 16 |

## Agreement (J vs J2)

| Label | Raw Agreement | Cohen's κ |
|-------|--------------|-----------|
| target_relevant | 0.9644 | 0.924 |
| target_leakage | 0.9644 | 0.924 |
| positive_entailment | 0.9556 | 0.910 |
| task_useful | 0.9111 | 0.815 |
| leakage_strength (exact) | 0.9467 | 0.917 |
| sequence_reconstructs_target | 1.0000 | 1.000 |

All core-label raw agreements ≥ 0.85 ✓  
All κ ≥ 0.60 where estimable ✓  
Sequence raw agreement ≥ 0.85 ✓

## Validation Review Queue

- Review queue count: **32 rows** (from 225 total)
- Review sequence count: **0 sequences** (from 36 trust-conditioned units)
- Queue triggers: core-label disagreement, leakage-strength disagreement, uncertainty/low-confidence flags

## J3 Adjudication

| Field | Value |
|-------|-------|
| Adjudicator model | qwen-plus (J3) |
| Items adjudicated | 32 |
| Consensus retained | 0 |
| Resolved by J3 matching J | 10 |
| Resolved by J3 matching J2 | 13 |
| Still unresolved | 9 |

## Final Validation Labels

| Metric | Value |
|--------|-------|
| Total rows | 225 |
| Consensus rows (J==J2) | 193 |
| Adjudicated rows | 23 |
| Unresolved rows | 9 |
| Structural sequence families | 12 |
| Trust-conditioned sequence units | 36 |
| Final sequence labels | 36 / 36 |
| Unresolved sequences | 0 |

## Unresolved Rates

| Metric | Rate | Threshold | Pass |
|--------|------|-----------|------|
| Row unresolved rate | 4.00% (9/225) | ≤ 10% | ✓ |
| Sequence unresolved rate | 0.00% (0/36) | ≤ 10% | ✓ |

## Sequence Accounting Repair (E4-002 Patch)

- **Defect**: `sequence_family_id` used as dict key collapsed 36 trust-conditioned units → 12
- **Fix**: Replaced with `sequence_annotation_id` as the unique pairing key
- **Sequence pairing key**: `sequence_annotation_id`
- **Primary sequences**: 36 / 36
- **Secondary sequences**: 36 / 36
- **Common sequence annotation IDs**: 36
- **Unmatched primary**: 0
- **Unmatched secondary**: 0
- **Structural sequence families**: 12
- **Trust-conditioned sequence units**: 36
- **Final sequence labels**: 36 / 36
- **Sequence reconstruction agreement**: raw=1.0000, κ=1.0000 (n=36)
- **Earliest-step exact agreement**: 1.0000 (n=35)
- **J/J2 sequence annotations modified**: NO (reused existing evidence)
- **J/J2 row annotations modified**: NO

## Validation Provenance Audit

**PASS** — All artifact SHA-256 bindings verified (byte-level).

## Validation Gate

**GO**

All gate checks passed:
- primary_complete: true
- secondary_complete: true
- agreement_computed: true
- adjudication_complete: true
- protocol_hash_match: true
- schema_hash_match: true
- prompt_hash_match: true
- unresolved_row_rate_pass: true
- unresolved_sequence_rate_pass: true
- provenance_audit_pass: true
- frozen_corpus_manifest_sha_match: true
- frozen_corpus_verifier_pass: true (52/52)
- development_annotation_verifier_pass: true (49/49)
- validation_annotation_verifier_pass: true (48/48)

## Provenance Closure Verification

Post-freeze closure verifier: **PASS** (11/11 checks)

- Annotation manifest SHA256 matches post_freeze_verification.json
- Validation gate SHA256 matches post_freeze_verification.json
- Freeze manifest SHA256 matches post_freeze_verification.json
- Frozen corpus verifier: PASS (52/52)
- Development annotation verifier: PASS (49/49)
- Validation annotation verifier: PASS (48/48)
- No self-referential commit SHA in report

Timestamp ordering (all post-date freeze):
- `freeze_created_at` ≤ `frozen_corpus_verifier.timestamp` ✓
- `freeze_created_at` ≤ `development_annotation_verifier.timestamp` ✓
- `freeze_created_at` ≤ `validation_annotation_verifier.timestamp` ✓
- `freeze_created_at` ≤ `post_freeze_verification.created_at` ✓

Post-freeze verifiers run AFTER freeze manifest write (execution-order fix F1).

Scientific annotation hashes unchanged after freeze regeneration:
- primary_row_annotations.jsonl: verified
- secondary_row_annotations.jsonl: verified
- primary_sequence_annotations.jsonl: verified
- secondary_sequence_annotations.jsonl: verified
- review_queue.jsonl: verified
- llm_adjudication.jsonl: verified
- final_adjudicated_labels.jsonl: verified
- final_sequence_labels.jsonl: verified

## Annotation Phase

- `validation_annotation_complete`: **true**
- `test_annotation_complete`: false
- `annotations_frozen`: false (test annotation pending)

## Validation Annotation Complete

**YES**

## Test Annotated

**NO**

## Test Accessed

**NO** — Validation runners do not read test content.

## READY FOR TEST ANNOTATION

**YES**

## Explicit Non-Goals Confirmed

The following were NOT performed in E4-002:
- Frozen development labels: not modified
- Frozen development J3 adjudication: not modified
- E4-001A not relabeled as human-reviewed
- Development models: not rerun
- Frozen prompts: not changed
- Frozen schema: not changed
- Model roles: not changed
- Thresholds: not lowered
- Test annotation: not started
- Embeddings: not generated
- Embedding thresholds: not calibrated
- Firewall replay: not run
- Final system metrics: not computed
- Ablations: not run
- Paper result tables: not produced

## Next Iteration

E4-003 — Blind Test Annotation + Global Annotation Artifact Freeze
