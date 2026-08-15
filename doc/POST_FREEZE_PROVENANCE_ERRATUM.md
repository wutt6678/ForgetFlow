# Post-Freeze Provenance Erratum

**Date**: 2026-08-15  
**References**: Corpus Freeze and Immutable Artifact Provenance Plan  

---

## Correction 1 — Git materialization snapshot is 8f7d8ec, not cc471f3

### Issue

The freeze report (`doc/E3_CORPUS_FREEZE_REPORT.md`) and earlier freeze commit
(`cc471f3`) claimed that all freeze artifacts were committed at `cc471f3`.
This was incomplete:

| Commit | SHA | What it added |
|---|---|---|
| **Source generation** | `f72e6f4` | Source code patches (requirement enforcement) |
| **Freeze metadata** | `cc471f3` | Top-level freeze manifest, inventory, gates, reports, scripts |
| **Freeze report** | `344c812` | `doc/E3_CORPUS_FREEZE_REPORT.md` |
| **Gitignore fix** | `1e8f56f` | Un-ignored per-split directories in `.gitignore` |
| **Complete artifact snapshot** | `8f7d8ec` | **All 30 per-split corpus files** (~3 MB) |

At `cc471f3`, the per-split corpus subdirectories (`development/`,
`validation/`, `test/`) were still gitignored. The later `1e8f56f` corrected
`.gitignore`, and `8f7d8ec` actually committed the per-split corpus data
(campaign identities, raw attempts, accepted candidates, corpus manifests, audit
reports, prompt manifests, etc.).

**Result**: No change to the frozen manifest. The `frozen_corpus_manifest.json`
itself is hash-bound and remains unchanged — it deliberately has
`"artifact_freeze_commit": null` under the two-level design. Instead, this
erratum documents the correct provenance chain as supplementary metadata.

### Corrected provenance hierarchy

```
source_generation_commit     = f72e6f4a5f426911fd98ac2822e4695211d61ca0
freeze_metadata_commit       = cc471f3543c3355b76f195e32f7a180490bcca12
freeze_report_commit         = 344c8129643c5bca01fd8287c285e32c05ec43ba
complete_snapshot_commit     = 8f7d8ec8a5805696e4e0c0582a78563f825e9004
```

**Frozen corpus identity** → existing `frozen_corpus_manifest` SHA
(**6b626f66734f809d422ba6f8b88f95f68a9515a7ab5b62535f86cae80d8d10b2**)

**Git materialization snapshot** → `8f7d8ec`

This preserves immutability: the frozen manifest is not rewritten, but a
machine-readable provenance record exists for downstream tooling that needs to
know where the full corpus bytes live in Git history.

---

## Correction 2 — `acceptance_rate` field is invalid due to mixed units

### Issue

The combined audit recorded `acceptance_rate` values of **1.3636...** (136.36%)
for every split:

| Split | Numerator | Denominator | Value |
|---|---|---|---|
| Development | 225 accepted candidates | 165 scientific units | 1.3636 |
| Validation | 225 accepted candidates | 165 scientific units | 1.3636 |
| Test | 450 accepted candidates | 330 scientific units | 1.3636 |

The numerator is **candidate/sequence-step rows**. The denominator is
**scientific units**. These are different units of analysis — sequence attacks
contribute multiple candidate rows per scientific sequence, so 225 candidates
per 165 scientific units is structurally expected, not a rate > 100%.

This is not an anomaly; it is a **labeling error**: `acceptance_rate` implies
"a fraction ≤ 1" but computes "candidates per scientific unit."

### Root cause

In `experiments/trustparadox_u/audit_empirical_corpus.py::compute_coverage_stats()`:

```python
"acceptance_rate": (
    len(split_candidates) / len(unit_keys) if unit_keys else 0.0
)
```

This divides candidate-row count by scientific-unit set size, producing a
mixed-unit ratio rather than a true rate.

### Fix applied

**File modified**: `experiments/trustparadox_u/audit_empirical_corpus.py`

Two new properly-defined metrics replace `acceptance_rate`:

| Metric | Formula | Result (all splits) |
|---|---|---|
| `candidate_row_acceptance_rate` | `accepted_candidates / eligible_candidate_rows` | 225 / 225 = **1.0** (100%) |
| `scientific_unit_acceptance_rate` | `(unit_keys − rejected_units) / len(unit_keys)` | 165 / 165 = **1.0** (100%) |

Both express valid rates in [0, 1]. All candidate rows and all scientific units
were accepted (zero rejected), hence both rates are 100%.

### Frozen artifact integrity

**No change to the frozen combined audit report.**
`results/empirical_v2/corpus_generation/full_corpus_validation_report.json`
remains part of the immutable 29-file inventory. The `acceptance_rate` field is
marked invalid/mixed-unit in subsequent analysis.

Subsequent audits and analyses should use `candidate_row_acceptance_rate` or
`scientific_unit_acceptance_rate`.

---

## Summary

These corrections document the actual Git provenance and metric definitions
without modifying any hash-bound frozen artifacts:

1. **Git snapshot**: `8f7d8ec` materializes all per-split corpus files; the complete snapshot includes ~38 tracked freeze artifacts across the metadata and corpus layers.
2. **Coverage metrics**: `acceptance_rate` replaced with `candidate_row_acceptance_rate` and `scientific_unit_acceptance_rate`; both yield 100% because zero candidates were rejected.
