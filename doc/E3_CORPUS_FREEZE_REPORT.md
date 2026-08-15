# E3 Corpus Freeze Report

## Summary

The E3 empirical corpus has been frozen and immutable artifact provenance has been established. All 900 accepted candidates across 3 splits are bound to their source commit, endpoint configuration, and generation parameters via a chain of SHA256 hashes.

## Freeze Provenance

| Field | Value |
|---|---|
| **Source generation commit** | `f72e6f4a5f426911fd98ac2822e4695211d61ca0` |
| **Artifact freeze commit** | `cc471f3543c3355b76f195e32f7a180490bcca12` |
| **Frozen at** | `2026-08-15T01:37:14.043843+00:00` |
| **Corpus frozen** | `true` |

## Corpus Composition

| Split | Plan Items | Accepted | Scientific Units | Missing |
|---|---|---|---|---|
| Development | 225 | 225 | 165 | 0 |
| Validation | 225 | 225 | 165 | 0 |
| Test | 450 | 450 | 330 | 0 |
| **Total** | **900** | **900** | **660** | **0** |

## Endpoint Provenance

| Field | Value |
|---|---|
| Host | `llm-jhxtd03gjg0gd2o2.ap-southeast-1.maas.aliyuncs.com` |
| SHA256 | `3d1699591685ab6c385ef26f30f469653ca8f506b33f1ded71e497a133d3d2c6` |
| Protocol | `openai_compatible` |

## Artifact Hashes

### Top-Level Freeze Artifacts

| Artifact | SHA256 |
|---|---|
| Frozen corpus manifest | `6b626f66734f809d422ba6f8b88f95f68a9515a7ab5b62535f86cae80d8d10b2` |
| Freeze artifact inventory | `e91ba459179249304fc25e8d92d3eaee255778f72c0f4a75acb6a7173ff9c5f0` |
| Combined audit report | `c10113ea8746abef0a7dc54933e5f5e92b4a7f37fa2d5a5836768e8109613e3f` |

### Source Data Bindings

| Artifact | SHA256 |
|---|---|
| Generation config | `fdb2d97b941a24b826760cc0bd543b8ea90a84736850569934e31080781e8efc` |
| Full generation plan | `91168d04f767bb51537a0dc7743c2100cd9955f1811ed8bede9a91f6e18db9fb` |
| Target registry | `f5bac18dadda613e5f2a71d94d2de8a072777071ffb5529a0165bcc4e1806380` |
| Prompt manifest | `a29588f5e16fe2cd0034cb1944d3d1870a36ff1955eef5b2ba24734ddb49d53c` |

## Audit Results

### Pre-Freeze Verifications (Sections 4-14)

- **Split gates**: 3/3 PASS (development, validation, test)
- **Campaign identities**: 3/3 source commit verified
- **Endpoint consistency**: 3/3 host + SHA256 + protocol verified
- **Generation counts**: 900/900 accepted candidates, 0 missing
- **Combined audit**: PASSED, 0 blocking findings, 17/17 audit sections clean

### Post-Freeze Verifier (Section 42)

- **49/49 checks PASSED**, 0 failed
- All 29 inventoried files re-hashed and verified
- Split gates, campaign identities, endpoint consistency all confirmed
- Combined audit hash matches frozen manifest
- Corpus frozen flag confirmed in phase manifest

### Mutation Detection Tests (Section 36)

- **11/11 tests PASSED**
- 7 mutation-detection scenarios verified (modify accepted candidate, raw attempt, audit report, source commit, endpoint SHA, count, combined audit)
- 2 integration tests (verifier passes on clean corpus, detects mutation)
- 2 corpus-frozen guard tests (generation rejected when frozen, guard reads phase file)

## Artifact Inventory

29 entries across 3 splits:

| Per-Split Artifacts (×3) | Top-Level Artifacts |
|---|---|
| campaign_identity.json | frozen_corpus_manifest.json |
| raw_generation_attempts.jsonl | freeze_artifact_inventory.json |
| accepted_candidates.jsonl | freeze_completion_report.json |
| corpus_manifest.json | corpus_freeze_gate.json |
| prompt_manifest.json | full_corpus_validation_report.json |
| sequence_generation_report.json | {split}_generation_gate.json (×3) |
| validation_report.json | |
| audit_report.json | |
| {split}_generation_gate.json | |

Plus: `data/trustparadox_u/empirical_v2/manifests/full_generation_config.json`

## Immutability Guarantees

1. **Corpus frozen guard**: `scripts/run_full_corpus_generation.py` rejects generation/resume when `corpus_frozen=true`
2. **Freeze verifier**: `scripts/verify_frozen_empirical_corpus.py` provides read-only verification of all freeze artifacts
3. **Mutation detection**: `tests/trustparadox_u/test_corpus_freeze_mutation.py` provides regression tests ensuring any modification is detectable
4. **Git-provenanced**: All freeze artifacts committed at `cc471f3`

## Circular Dependency Avoidance (Section 31)

The `empirical_phase.json` is excluded from the artifact inventory to avoid a circular hash dependency: the freeze process modifies this file to record freeze provenance hashes. Its hash is recorded separately in the frozen corpus manifest and phase manifest rather than in the inventory root hash.

## Files Committed

| File | Operation |
|---|---|
| `.gitignore` | Modified: added freeze artifact exceptions |
| `data/.../empirical_phase.json` | Modified: added freeze provenance fields |
| `scripts/run_full_corpus_generation.py` | Modified: added corpus_frozen guard |
| `scripts/build_corpus_freeze.py` | New: freeze artifact builder |
| `scripts/verify_frozen_empirical_corpus.py` | New: read-only freeze verifier |
| `tests/.../test_corpus_freeze_mutation.py` | New: mutation detection tests |
| `results/.../frozen_corpus_manifest.json` | New: top-level provenance binding |
| `results/.../freeze_artifact_inventory.json` | New: 29-entry artifact inventory |
| `results/.../freeze_completion_report.json` | New: freeze summary |
| `results/.../corpus_freeze_gate.json` | New: freeze_ready=true gate |
| `results/.../full_corpus_validation_report.json` | New: combined audit report |
| `results/.../{dev,val,test}_generation_gate.json` | New: split gates |

## Conclusion

The E3 empirical corpus freeze is **COMPLETE**. All 900 accepted candidates are immutably bound to their source commit `f72e6f4`, endpoint configuration, and generation parameters. The freeze verifier passes 49/49 checks, all 11 mutation-detection tests pass, and the corpus-frozen guard prevents any future generation from modifying the frozen corpus.
