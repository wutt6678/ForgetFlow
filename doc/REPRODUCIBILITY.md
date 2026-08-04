# Reproducibility Guide

This document describes how to reproduce ForgetFlow experiments and the requirements for each run mode.

---

## Installation

### Development and offline tests

```bash
poetry install
```

This installs all dependencies needed for unit tests, fixed-embedding smoke runs, and CI validation.

### Real embedding experiments

```bash
poetry install -E experiment
```

This additionally installs `litellm`, which is required for real semantic embedding experiments. **Without this extra, experiment-mode runs will fail at import time.**

---

## Run Modes

### Test mode (fixed embeddings)

```yaml
run:
  mode: test

models:
  embedding_provider: fixed
  embedding_model: null
  embedding_dimension: 3
```

- Uses deterministic `FixedEmbeddingProvider` with predefined vectors.
- Suitable for CI, unit tests, and reproducible smoke validation.
- **Not suitable for real semantic claims.**

### Experiment mode (real embeddings)

```yaml
run:
  mode: experiment

models:
  embedding_provider: litellm
  embedding_model: openai/text-embedding-v3
  embedding_dimension: 1024
  api_base: https://your-endpoint.example.com/compatible-mode/v1
```

- Uses `RealEmbeddingProvider` via LiteLLM to call a real embedding API.
- Requires `litellm` installed (`pip install litellm` or `poetry install -E experiment`).
- Requires a valid API key (e.g., `OPENAI_API_KEY`) in the environment.
- `api_base` is optional — set it for custom OpenAI-compatible endpoints (e.g., Alibaba Cloud MaaS).
- `embedding_model` uses LiteLLM provider prefix format (e.g., `openai/text-embedding-v3`).
- Provider, model, and dimension are recorded in result metadata.

---

## Preflight Checks

Before launching a real experiment, run the preflight module:

```bash
poetry run python -m experiments.trustparadox_u.preflight \
  --config experiments/trustparadox_u/configs/full_mvp.yaml
```

This verifies:
- Configuration loads and validates.
- LiteLLM is importable (experiment mode).
- Output directory is writable.

To also probe the embedding provider (makes a real API call):

```bash
poetry run python -m experiments.trustparadox_u.preflight \
  --config experiments/trustparadox_u/configs/full_mvp.yaml \
  --probe-provider
```

---

## Result Metadata

Every episode result records:

| Field | Description |
|-------|-------------|
| `run_mode` | `test` or `experiment` |
| `semantic_enabled` | Whether semantic detection is active |
| `embedding_provider` | `fixed` or `litellm` |
| `embedding_model` | Model name or `null` |
| `embedding_dimension` | Expected or observed vector dimension |
| `semantic_threshold` | Cosine similarity threshold |
| `monitoring_continuous` | Whether monitoring is continuous |
| `monitoring_duration_rounds` | Monitoring duration in rounds |
| `post_forget_round_count` | Final post-forget round count |
| `fragment_count` | Maximum fragment count across sensitive items |
| `pairing_key` | Structured dict identifying the experiment pairing |
| `config_hash` | SHA-256 of the resolved configuration |

---

## Frozen Configuration Manifest (remediation §29/§30)

Every model, threshold, prompt, annotation, and policy decision that feeds
the primary analysis is frozen **before** the final test evaluation and
recorded in one committed manifest:

```
results/frozen_config/frozen_threshold_manifest.json
```

Generate and validate it with:

```bash
python -m experiments.trustparadox_u.frozen_thresholds
```

The manifest anchors:

- **Swept thresholds** with their selection sweep (split, selection rule,
  tie-breaking rule, selected value) — sourced from
  `results/parameter_sweep/sweep_summary.json`.
- **Unswept behavioral parameters** as fixed defaults with a rationale.
- **Scenario definitions** (SHA-256 per committed scenario YAML).
- **Candidate-generation prompts** (corpus manifest prompt template hashes)
  and **annotation instructions** (annotation manifest hash).
- **The primary hypotheses** via the versioned research protocol
  (`PROTOCOL_VERSION`) and the frozen config/condition hashes.

### Freeze discipline (§29)

- The manifest is committed **before** the test results it governs.
- The final test split is evaluated exactly once for the primary analysis
  (the frozen-config evaluation at the end of the parameter sweep).
- Any rerun after code or protocol changes bumps `STUDY_VERSION` and keeps
  the previous manifest committed alongside the results it governed.
- Post-test fixes invalidate or version the previous result rather than
  silently replacing it.

### Sweep purpose labels (§30)

Every parameter sweep is labelled:

- `selection` — chooses the frozen value, using development/validation
  splits only; a selection sweep never touches the test split.
- `sensitivity` — post hoc exploratory analysis that must never choose the
  main reported result; carries an explanatory note and states the
  evaluation split it used.

The label travels in `sweep_summary.json` (`sweep_purpose`), in Table 3 of
the final artifacts, and is enforced by `build_sweep_validation` and the
`frozen_threshold_manifest` gate.

---

## Important Notes

- **Fixed embeddings are only for deterministic tests.** Real semantic claims require experiment mode.
- **Provider/model/dimension must appear in result metadata** for reproducibility.
- **Credentials must not be committed.** API keys should be set via environment variables.
- **CI does not call the real provider.** All CI tests use fixed embeddings or mock the provider.
