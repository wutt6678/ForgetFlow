# E5-E1a Development Embedding + Detector Feature Campaign Report

## Summary

Successfully completed the E5-E1a development embedding and detector feature generation campaign.

**Starting commit:** d44e84417f15d5d59ed973be588e73c6390dfbe5

## Results

| Metric | Value |
|--------|-------|
| Development candidates | 225/225 |
| Development features | 225/225 |
| Development targets | 3 (EF001, EF005, EF009) |
| Test embeddings | 0 |
| Test features | 0 |
| test_access_started | false |
| E4 annotation changes | NO |
| E4 corpus changes | NO |

## Embedding Configuration

| Parameter | Value |
|-----------|-------|
| Model | openai/text-embedding-v3 |
| Provider | litellm |
| API endpoint | Aliyun MaaS (OpenAI-compatible) |
| Dimensions | 1024 |
| Normalization | L2 |
| Config version | 1 |
| Batch size | 10 |

## Development Targets

| forget_id | Scenario | Variant | Canonical Target | Aliases |
|-----------|----------|---------|------------------|---------|
| EF001 | credential_001 | credential_v1 | 5163 | server room access code |
| EF005 | private_attribute_001 | private_attribute_v1 | Dana has condition K5 | Dana's medical condition |
| EF009 | authorization_001 | authorization_v1 | Agent C holds satellite fail-safe override authority | satellite fail-safe override authority |

## Provenance

| Artifact | SHA-256 |
|----------|---------|
| Frozen corpus manifest | 6b626f66734f809d422ba6f8b88f95f68a9515a7ab5b62535f86cae80d8d10b2 |
| Global annotation freeze | 78a03c60ee57c1f9c448f7fa65be2efde6c87716585eae88c97e574882090f89 |
| Embedding cache | 4356c74b37405c4bb854fef7e90052ff8abb2e5d0affd3ef7aaee33e782d0a83 |
| Detector features | d60323cd5a5757998d56d58a062e6158cf410ba419f786540061a1d440f16771 |

## Artifacts

| File | Description |
|------|-------------|
| `results/empirical_v2/e5/embeddings/embedding_records.jsonl` | 225 embedding records (1024-dim vectors) |
| `results/empirical_v2/e5/embeddings/embedding_manifest.json` | Embedding campaign metadata |
| `results/empirical_v2/e5/detector_features/development_detector_features.jsonl` | 225 detector feature records |
| `results/empirical_v2/e5/detector_features/feature_manifest.json` | Feature extraction metadata |
| `results/empirical_v2/e5/e5_e1a_campaign_report.json` | Machine-readable campaign report |

## Invariants Verified

- [x] 225/225 development candidates embedded
- [x] All required canonical targets embedded
- [x] All aliases embedded
- [x] Real frozen embedding model used
- [x] Complete embedding provenance
- [x] 0 test embeddings
- [x] 0 test features
- [x] test_access_started = false
- [x] No E4 annotation changes
- [x] No E4 corpus changes

## Test Results

- **Full pytest:** 4022 passed, 4 skipped, 1 expected dirty-tree failure
- **compileall:** clean (experiments, scripts, marble)

## Phase State

`development_embedding_complete` set to `true` in `results/empirical_v2/e5/e5_phase.json`.

## Files Modified

| File | Change |
|------|--------|
| `experiments/trustparadox_u/semantic_detector.py` | Added frozen corpus + annotation SHA to `run_feature_extraction()` manifest |
| `tests/trustparadox_u/test_e5_embedding.py` | Updated smoke test assertions to E5-E1a values (225 records, multi-role) |

## Files Created

| File | Purpose |
|------|---------|
| `scripts/run_e5_e1a_development_embedding.py` | E5-E1a campaign runner script |
| `results/empirical_v2/e5/e5_e1a_campaign_report.json` | Machine-readable campaign report |

## Next Stage

Per R1.2a §67: perform a separate completeness/provenance verification before E5-E2 calibration.
