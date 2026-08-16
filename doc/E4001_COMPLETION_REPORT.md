# E4-001 Repair + Development Pilot v3 Completion Report

## Sec 88: Required coding-agent completion report

### Pilot identity

- **Specification**: E4-001 Repair Checklist
- **Starting commit**: `05c498db16384ee587f7e6fd5a35989ae6f6972c`
- **Annotation source commit**: `e0c379b4b1713dab34ac7345d2c3ab8a08338fd0`
- **v3 evidence commit**: `4667a68c53a7b70e0543b5dd64019fbf4de79f11`
- **Completion report commit**: `05c498db16384ee587f7e6fd5a35989ae6f6972c`
- **Old pilot superseded**: YES (v1 → `pilot_supersession.json`)

### Frozen corpus

- **Frozen corpus verifier before**: PASS
- **Frozen corpus verifier after**: PASS (52/52 checks)
- **Frozen corpus manifest SHA256**: `6b626f66734f809d422ba6f8b88f95f68a9515a7ab5b62535f86cae80d8d10b2`

### Target registry

- **Registry**: `data/trustparadox_u/empirical_v2/target_specs.jsonl` (12 targets)

### Development target resolution

- Rows resolved: **225 / 225**
- Sequences resolved: **36 / 36**
- Empty targets: **0**
- Failures: **0**

### Queue

- Row items: **225 / 225**
- Sequence items: **36 / 36**
- Row IDs unique: **PASS**
- Sequence IDs unique: **PASS**
- Queue SHA256: `b8c290051b99daff7fed413c0aea52f3b241bc777053f30a54d659af7ad0b189`

### Primary annotator (J)

- **Provider**: openai (LiteLLM transport)
- **Requested model**: qwen3.8-max
- **Returned model**: qwen3.8-max
- **Model revision**: not_exposed_by_provider
- **Transport**: openai-compatible API via LiteLLM
- **Provider attempts**: 261
- **Row labels**: 225 / 225
- **Sequence labels**: 36 / 36
- **Retries**: 0
- **Malformed**: 0
- **Timeouts**: 0
- **Provider errors**: 0

### Secondary annotator (J2)

- **Provider**: openai (LiteLLM transport)
- **Requested model**: glm-5.2
- **Returned model**: glm-5.2
- **Model revision**: not_exposed_by_provider
- **Transport**: openai-compatible API via LiteLLM
- **Provider attempts**: 277 (261 first run + 16 resume retries)
- **Row labels**: 225 / 225
- **Sequence labels**: 36 / 36
- **Retries**: 16 (2 failed in first run, resolved via resume)
- **Malformed**: 0
- **Timeouts**: 0
- **Provider errors**: 0

### Provider evidence

- **Raw provider responses retained**: YES (261/261 primary, 262/277 secondary — 15 failed attempts had no response body)
- **Provider request IDs retained**: YES (261/261 primary, 277/277 secondary)
- **Corpus binding present in all labels**: YES

### Agreement (v3 — after Sec 63 bounded revision)

| Label | Raw Agreement | Cohen's Kappa |
|---|---|---|
| target_relevant | 0.9600 | 0.9147 |
| target_leakage | 0.9600 | 0.9147 |
| positive_entailment | 0.9422 | 0.8834 |
| task_useful | 0.9378 | 0.8699 |
| leakage_strength (exact) | 0.9333 | 0.8978 |
| sequence_reconstructs_target | 1.0000 | 1.0000 |

### Comparison: v2 → v3

| Metric | v2 | v3 | Threshold |
|---|---|---|---|
| Core-label min raw agreement | 0.8311 | 0.9378 | ≥ 0.85 |
| Core-label min kappa | 0.3036 | 0.8699 | ≥ 0.60 |
| Sequence raw agreement | 1.0000 | 1.0000 | ≥ 0.85 |
| Unresolved row rate | 0.4400 | 0.1600 | ≤ 0.10 |
| Unresolved sequence rate | 0.0000 | 0.0000 | ≤ 0.10 |

### Review queue

- Row items: **38**
- Sequence items: **0**

### Unresolved

- Rows: **36** (16.0%)
- Sequences: **0** (0.0%)

### Freeze status

- Annotation schema frozen: **NO** (NO-GO)
- Annotation prompts frozen: **NO** (NO-GO)
- Annotation protocol frozen: **NO** (NO-GO)

### Gate assessment (v3)

| Gate | Status | Detail |
|---|---|---|
| full_coverage | PASS | unmatched=0 |
| model_role_separation | PASS | violations=0 |
| row_agreement_acceptable | PASS | min_kappa=0.8699 |
| sequence_agreement_estimable | PASS | raw=1.0, kappa=1.0 |
| unresolved_row_rate | **FAIL** | 0.16 > 0.10 |

### Sec 63 bounded revision

The v2 pilot failed due to systematic semantic ambiguity in `target_relevant` and `task_useful` label definitions. Per Sec 63, one bounded development-only revision was applied:

1. **Diagnosed**: J (qwen3.8-max) marked generic concept mentions as target_relevant; J2 (glm-5.2) did not. J marked leaked target info as task_useful; J2 did not.
2. **Revised**: Added IMPORTANT clarification paragraphs to both label definitions in `ROW_USER_PROMPT_TEMPLATE`.
3. **Re-ran**: Full v3 annotation from scratch in new `development_v3/` directory.
4. **Result**: Agreement improved dramatically (kappa 0.30→0.87 for task_useful, 0.47→0.91 for target_relevant), but 36 rows remain unresolved due to leakage_strength and edge-case disagreements.

The bounded revision has been exhausted. Per Sec 90, thresholds must not be lowered to obtain GO. This is a legitimate NO-GO per Sec 64.

### Validation/Test

- Validation annotated: **NO**
- Test annotated: **NO**

### READY FOR VALIDATION ANNOTATION: **NO**

The pilot is in a legitimate NO-GO state. All repairs (A–K) are implemented and tested. The annotation pipeline is functionally correct. The remaining 36 unresolved rows require either human adjudication or a future protocol decision beyond the scope of this repair checklist.
