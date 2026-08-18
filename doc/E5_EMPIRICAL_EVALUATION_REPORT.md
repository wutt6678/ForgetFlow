# E5 Real-Embedding and Downstream Evaluation Report

## 1. Objective

E5 evaluates the ForgetFlow leakage-prevention framework using real
embedding representations and downstream analyses. The evaluation
follows a strict protocol: frozen E4 annotations, real embeddings,
development calibration, validation confirmation, test freeze, and
held-out test evaluation with zero post-test tuning.

This report documents the E5 evaluation pipeline scaffold, from
embedding provenance through the planned closure workflow.
**Empirical experiments have not yet been run.**

## 2. Frozen E4 Inputs

All E5 evaluations operate on the E4 global annotation freeze:

- **900 row annotations** (candidates across 5 scenarios)
- **144 sequence annotations** (ordered candidate sequences)
- **24 structural families** with trust-variant expansions
- **Annotation root SHA**: recorded in E4-003 R3 freeze manifest
- **Resolution status**: 24 row annotations remain unresolved and
  are excluded from eligible denominators where required

No E4 labels were modified, reopened, or re-annotated during E5.

## 3. Embedding Model and Provenance

The real embedding backend (`experiments/trustparadox_u/embedding.py`)
produces deterministic semantic similarity scores for each candidate
pair. Embeddings are computed once and cached to ensure reproducibility.

- **Embedding backend**: sentence-transformers-compatible adapter
- **Cache**: deterministic, seed-controlled
- **Provenance**: embedding model identity and version recorded in
  the preflight manifest (`e5_preflight.json`)

## 4. Calibration Protocol (Development Split)

The development split will be used to:

1. Calibrate the semantic similarity threshold τ_sem
2. Verify detector feature extraction (exact, alias, semantic)
3. Confirm confusion matrix computation aligns with E4 labels

Threshold selection rule (plan §11.2):
- Require leakage recall ≥ 0.90
- Among qualifying thresholds, choose lowest FBR
- Tie-break: highest utility retention

## 5. Validation Confirmation

The validation split will confirm that the development-calibrated
configuration generalises without modification.

## 6. Held-Out Test Lock

Before any test evaluation, the experimental configuration will be frozen:

- **τ_sem**: to be determined by calibration
- **Random seed**: 42 (for bootstrap and statistical tests)
- **Test split**: `data/trustparadox_u/splits/test.jsonl`
- **Conditions**: C0 (no firewall), C1 (exact only), C2 (+alias),
  C3 (+semantic), C4 (full ForgetFlow)

The test freeze manifest will record file hashes and configuration to
detect any post-hoc changes.

## 7. Experimental Conditions

| Condition | Description |
|-----------|-------------|
| C0 | No firewall (baseline — all messages allowed) |
| C1 | Exact-match detector only |
| C2 | C1 + alias/coreference detector |
| C3 | C2 + semantic similarity detector |
| C4 | Full ForgetFlow (C3 + reconstruction guard + purge) |

## 8. Metrics

### Primary Metrics

- **PU-RER** (Protected-User Recall / Leakage Prevention):
  Among truly leaking rows, fraction blocked.
  Wilson 95% CI reported.

- **FBR** (False Blocking Rate):
  Among eligible non-leaking rows, fraction incorrectly blocked.
  Wilson 95% CI reported.

- **Utility Retention**:
  Among task-useful eligible rows, fraction preserved (allowed).
  Wilson 95% CI reported.

### Secondary Metrics

- **Leakage Precision**: TP / (TP + FP)
- **Leakage F1**: Harmonic mean of precision and recall
- **Sequence Reconstruction Recall**: Among reconstructing sequences,
  fraction where at least one candidate is detected
- **Compositional Reconstruction Resistance (CRR)**: 1 − reconstruction rate

## 9. Primary Test Results

Primary comparisons: C4 vs C0 (full system vs no firewall) and
C4 vs C1 (full system vs exact-only).

Results are computed by `scripts/build_e5_results.py` and include
Wilson confidence intervals for all primary proportions.

Statistical significance is assessed via McNemar's test (paired
binary outcomes) and bootstrap CI for metric differences
(`experiments/trustparadox_u/e5_statistics.py`).

## 10. Attack Robustness

Attack-type breakdown (`experiments/trustparadox_u/e5_attack_analysis.py`):

| Attack Type | Description |
|-------------|-------------|
| direct_disclosure | Explicit statement of private information |
| semantic_paraphrase | Rephrased leakage preserving semantics |
| alias_or_coreference | Indirect reference via aliases |
| recontamination | Re-introducing forgotten information |
| fragmentation_sequence | Split across multiple messages |
| compositional_sequence | Multi-step reconstruction |
| hard_negative_control | Topically similar but non-leaking |
| legitimate_task | Genuinely useful task messages |

For each attack type: baseline leakage (C0), ForgetFlow leakage (C4),
relative reduction, utility retention, and FBR are reported.

## 11. Trust-Conditioned Analysis

Metrics broken down by trust level (low, default, high) to assess
whether the policy behaves consistently across trust contexts.

Trust drift = max(trust_levels) − min(trust_levels) for each metric.

## 12. Utility Analysis

### Hard-Negative Analysis

Quantifies whether semantic protection over-blocks topically similar
but non-leaking content.

### Legitimate-Task Analysis

Measures practical utility preservation for genuinely useful task
messages.

## 13. Sequence / Reconstruction Analysis

Sequence-level metrics (`experiments/trustparadox_u/e5_sequence_evaluation.py`):

- **Sequence reconstruction recall**: fraction of reconstructing
  sequences caught by detecting at least one candidate
- **Sequence leakage rate**: fraction of candidates in reconstructing
  sequences that are detected
- **Earliest-step detection**: position of first detected candidate
  in reconstructing sequences

## 14. Component Ablations

Ablation study (`experiments/trustparadox_u/e5_ablation_study.py`):

| Ablation | Disabled Component |
|----------|-------------------|
| A0 | Full ForgetFlow (baseline) |
| A1 | − Semantic detector |
| A2 | − Recipient/history-aware state (ForgetGraph) |
| A3 | − ReconstructGuard |
| A4 | − Purge/recontamination handling |

For each ablation: leakage prevention, FBR, utility retention,
and relative impact vs baseline.

## 15. Hyperparameter Sensitivity

Threshold sweep (`experiments/trustparadox_u/e5_hyperparameter_study.py`)
across τ_sem ∈ {0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90}:

- Per-threshold: leakage recall, FBR, utility retention, CRR
- Tradeoff figure data: leakage prevention vs utility retention
- Optimal threshold recommendation with rationale

## 16. Statistical Uncertainty

Statistical analysis (`experiments/trustparadox_u/e5_statistics.py`):

- **Wilson 95% CI** for all primary proportions (PU-RER, FBR,
  utility retention)
- **McNemar's test** for paired binary outcomes (C4 vs C0, C4 vs C1)
  with continuity correction
- **Bootstrap** (seed=42, 10000 resamples) for metric difference CIs
- **Metric eligibility manifest**: for each metric, records
  numerator, denominator, excluded unresolved count, and split/condition

## 17. Limitations

The following limitations must be acknowledged:

1. **Annotations are LLM-generated**: All row and sequence annotations
   were produced by independent LLM annotators with third-LLM
   adjudication, not human-validated ground truth. Annotation quality
   depends on the LLMs' capability and the prompt design.

2. **Unresolved annotations excluded**: 24 test row annotations remain
   unresolved (no definitive leakage/useful determination) and are
   excluded from eligible denominators. This may introduce selection
   bias if unresolved rows are systematically different.

3. **Embedding similarity is operational, not definitive**: Semantic
   similarity scores from the embedding model serve as an operational
   detector, not a proof of semantic equivalence. Adversarial attacks
   that evade embedding similarity are possible.

4. **Information-theoretic claims are proxy-based**: The
   information-theoretic discussion of forgetting is supported by
   empirical reconstruction proxies (sequence reconstruction recall)
   rather than a direct proof of zero mutual information between
   model outputs and forgotten data.

5. **No overclaiming**: This evaluation does NOT claim:
   - Perfect forgetting
   - Zero information leakage
   - Formal unlearning guarantees
   - Human-ground-truth validation
   - Universal attack robustness

## 18. Reproducibility Manifest

All evidence is deterministic given:

- E4 annotation freeze (immutable)
- Embedding cache (deterministic, cached)
- Frozen configuration: τ_sem = 0.75, seed = 42
- Test split: `data/trustparadox_u/splits/test.jsonl`

Reproduction steps:

```bash
# 1. Verify E4 freeze
python scripts/check_source_integrity.py

# 2. Run full test suite
python -m pytest tests/ -x

# 3. Build aggregated results
python scripts/build_e5_results.py \
  --input-dir results/ --labels <labels.json> \
  --corpus <corpus.json> --output results/e5_results.json

# 4. Verify test freeze
python scripts/verify_e5_test_freeze.py \
  --manifest results/freeze_manifest.json
```

Key file hashes are recorded in the test freeze manifest and
checksums file.

## 19. Final E5 Closure Status

> **E5 IMPLEMENTATION SCAFFOLD COMPLETE**
> **EMPIRICAL E5 EXPERIMENTS NOT YET RUN**

The current status reflects completed implementation scaffold only.
No held-out test evaluation, calibration, or validation has been executed.

| Criterion | Status |
|-----------|--------|
| E4 annotation root unchanged | PASS |
| Real embedding provenance (smoke) | PASS (10-item smoke) |
| Full development embeddings | NOT RUN |
| Development calibration | NOT RUN |
| Validation confirmation | NOT RUN |
| Test configuration frozen | NOT CREATED |
| Held-out test evaluation | NOT RUN |
| Zero post-test tuning | NOT VERIFIED |
| Primary metrics | NOT COMPUTED |
| Attack analysis | NOT COMPUTED |
| Trust analysis | NOT COMPUTED |
| Utility analysis | NOT COMPUTED |
| Sequence analysis | NOT COMPUTED |
| Core ablations | NOT COMPUTED |
| Core hyperparameter study | NOT COMPUTED |
| Statistical uncertainty | NOT COMPUTED |
| Test freeze verifier | NOT RUN |
| Final report | NOT COMPLETE |

**E5 CLOSED: NO**

### Required truth

- E5 preflight: PASS
- Real embedding smoke: PASS (10 texts, 1 API call)
- Full development embeddings: NOT RUN
- Development calibration: NOT RUN
- Validation: NOT RUN
- Held-out test access: NOT STARTED
- Held-out test evaluation: NOT RUN
- E5 test freeze: NOT CREATED
- E5 CLOSED: NO

### Implementation Summary

| Iteration | Module | Commit |
|-----------|--------|--------|
| E5-000 | Frozen-input preflight | `bb8bbaa` |
| E5-001 | Real embedding backend | `bb8bbaa` |
| E5-002 | Semantic detector features | `bb8bbaa` |
| E5-003 | Row/sequence metrics | `bb8bbaa` |
| E5-004 | Validation confirmation | `28c3395` |
| E5-005 | Experimental-condition freeze | `6ffe9c2` |
| E5-006 | Sequence evaluation | `f02081c` |
| E5-007 | Held-out test evaluation | `8d7dbf3` |
| E5-008 | Attack/trust analysis | `b8f8df2` |
| E5-009 | Component ablations | `69f58f0` |
| E5-010 | Hyperparameter sensitivity | `45fe719` |
| E5-011 | Statistical analysis | `e7a2cc1` |
| E5-012 | Result aggregation & report | `c771165` |
