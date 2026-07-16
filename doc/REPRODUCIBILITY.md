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
  embedding_model: text-embedding-3-small
  embedding_dimension: 1536
```

- Uses `RealEmbeddingProvider` via LiteLLM to call a real embedding API.
- Requires `poetry install -E experiment`.
- Requires a valid API key (e.g., `OPENAI_API_KEY`) in the environment.
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
| `config_hash` | SHA-256 of the resolved configuration |

---

## Important Notes

- **Fixed embeddings are only for deterministic tests.** Real semantic claims require experiment mode.
- **Provider/model/dimension must appear in result metadata** for reproducibility.
- **Credentials must not be committed.** API keys should be set via environment variables.
- **CI does not call the real provider.** All CI tests use fixed embeddings or mock the provider.
