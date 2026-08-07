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

## Three-Way Provenance (remediation §32)

Commit names alone cannot identify a result: the pipeline executes code at
one commit and *stores* the generated artifacts in a later commit.  Every
certifying manifest therefore records three identities:

| Identity | Field | Meaning |
| --- | --- | --- |
| Tested code | `tested_code_commit` / `artifact_generation_commit` | the commit whose code was executed (must be equal) |
| Workspace | `artifact_generation_tree` | SHA-256 over every source path (`marble`, `experiments`, `scripts`, `tests`, `data`, `doc`) plus `pyproject.toml`, `poetry.lock`, `environment.yml` |
| Environment | `environment_lock_hash` | SHA-256 of the dependency lock file |

The storage commit (`artifact_storage_commit`) is unknowable at generation
time — it is the commit that later *stores* the artifact.  Scientific
artifacts therefore never embed one; the authoritative storage record is
the release's `STORAGE_PROVENANCE.json` sidecar (see the release
provenance model below).  Certification rules:

- `tested_code_commit == artifact_generation_commit`;
- `repository_clean == true` (clean code tree at generation);
- the tested commit is an ancestor of the artifact's storage commit — the
  commit-first flow legitimately stores artifacts after running the code.

`validate_three_way_provenance` enforces the base rules plus the workspace
and environment anchors; findings are never silent.

## Single-Command Reproduction (remediation §33)

```
python -m experiments.trustparadox_u.reproduce
```

is the one documented path from frozen inputs to final tables.  It:

1. validates the environment (interpreter, commit, workspace and lock hashes);
2. validates — **never regenerates** — the frozen corpus, annotations and
   the frozen threshold manifest;
3. runs the condition matrix, leakage analysis, paired statistics and final
   tables as recorded subprocess steps;
4. recomputes metrics from the regenerated trial artifacts;
5. writes `results/reproduction/reproduction_manifest.json` with the §32
   three-way provenance, input hashes, step records and a SHA-256 for every
   regenerated artifact.

Any missing or mismatched input, failed step, or metric mismatch aborts
with a non-zero exit code.  The reproduction manifest is a provenance
artifact: the `repository_provenance` gate and the `reproduction_manifest`
gate both verify it.

## Immutable Release Bundles (remediation §34)

```
python -m experiments.trustparadox_u.release_bundle
```

freezes the current study state as `results/releases/<release_id>/`.  The
`release_id` is `trustparadox_u-v{study_version}-{digest}` where the digest
hashes the canonical component manifest (study version, corpus/annotation
hashes, and a SHA-256 per component) — identical content can never receive
two identifiers.  Each bundle preserves:

- protocol version and three-way provenance (from the reproduction manifest);
- corpus and annotations, bound by content hash;
- frozen condition configurations and resolved conditions;
- raw replay/reconstruction/recontamination/utility trials;
- statistical outputs, leakage analysis, sweep summary;
- the final tables and study manifest.

Building a new release supersedes the previous active bundle: it moves to
`results/archive/<release_id>/` with an `INVALIDATION_MARKER.json`
(`status: superseded`), so superseded studies remain auditable but can
never be mistaken for the current release.  Papers and reports must cite
the exact `release_id`.  The `release_bundles` gate requires exactly one
active release whose component hashes all verify.

## Release Provenance Model (FP-001..FP-014)

Provenance is split into two records with distinct owners:

> Scientific artifacts record **generation provenance**.  Storage
> provenance is recorded authoritatively in the release storage sidecar
> (`results/releases/<release_id>/STORAGE_PROVENANCE.json`).

### Generation identity

Every scientific manifest (study, reproduction, run, frozen-threshold,
bundle, gate snapshot) carries the same generation record:

| Field | Meaning |
| --- | --- |
| `tested_code_commit` | the commit whose code was executed |
| `artifact_generation_commit` | the commit that generated the artifact (equal to the tested commit) |
| `artifact_generation_tree` | SHA-256 workspace hash over every source path |
| `environment_lock_hash` | SHA-256 of the dependency lock file |
| `protocol_version` / `study_version` | the protocol and study versions the run followed |

Generation records carry `artifact_storage_commit: null` plus a
`storage_provenance` pointer (`{"source": "STORAGE_PROVENANCE.json",
"authoritative": true}`) — an embedded storage commit (empty string or
otherwise) is a validation finding, because it cannot be known when the
artifact is generated.  The `release_storage_provenance` gate requires all
generation fields to be synchronized across every manifest.

### Storage identity

The sidecar is the sole authoritative storage record (schema 1.2):

| Field | Meaning |
| --- | --- |
| `artifact_storage_commit` | the commit that stored the exact bundle bytes |
| `gate_evidence_commit` | the commit that stored the certifying gate result |
| `gate_evidence_sha256` | SHA-256 of the gate result's historical bytes |
| `gate_snapshot_commit` | deprecated alias of `gate_evidence_commit`; when both are present they must be equal |
| `storage_metadata_digest` | digest of the sidecar's own lineage content |
| `scientific_release_digest` | the scientific content digest of the release |

The gate-evidence digest is computed over the exact bytes of
`results/final_artifacts/research_valid_gate.json` at the
`gate_evidence_commit`, as read from git history
(`git show <gate_evidence_commit>:results/final_artifacts/research_valid_gate.json`).

Git ancestry is verified from history, never from timestamps:
generation → storage → gate evidence → review commit, and the exact
bundle must exist byte-identically at `artifact_storage_commit`.  The
bundle manifest echoes the storage identity at its top level; both copies
must agree with the sidecar.

### Digest model

Three digests bind the release together:

- `scientific_release_digest` hashes study version, corpus/annotation
  hashes, and every component's SHA-256.  Storage metadata is never a
  component, so sidecar or certification updates can never change it.
- `storage_metadata_digest` hashes the sidecar lineage fields (excluding
  `verified_at` bookkeeping and the digest field itself).
- `gate_evidence_sha256` pins the historical gate result byte-for-byte:
  the sidecar's digest must equal the SHA-256 of the gate result bytes
  stored at the `gate_evidence_commit`.

### Gate-evidence validity (GE-001..GE-017)

A gate-evidence commit is valid only when the historical gate result
itself contains the required passing research status and supporting gate
evidence.  The certification therefore loads the gate result from git
history and checks it semantically — file existence alone never
certifies.  A commit whose historical gate result reports a failed
research status or failing tests can never certify a release, even when
it is a real ancestor of HEAD.

### Local versus CI certification

Local certification (`workflow_run_id = workflow_attempt =
certification_source = "local"`) verifies deterministic research
artifacts on the local machine; it is **not** GitHub Actions evidence.
CI certification requires numeric `workflow_run_id` / `workflow_attempt`
identity.  A local run is never labeled as CI evidence, and vice versa.

### Certification sequence

The full certification chain follows eight steps, always in order:

1. Generate the scientific artifacts on a clean tree (commit-first flow).
2. Build the immutable scientific release bundle from those artifacts.
3. Commit the scientific release (`artifact_storage_commit`).
4. Run the gate and obtain passing gate evidence for that release.
5. Commit the gate evidence (`gate_evidence_commit`).
6. Finalize the storage sidecar with the gate-evidence commit and its
   digest, then re-audit (`storage_metadata_digest` updates;
   `scientific_release_digest` does not).
7. Run the storage certification, which writes
   `FINAL_STORAGE_CERTIFICATION.json` into the bundle — a durable,
   non-self-referential record of the full lineage.
8. Do not rewrite the scientific artifacts or the gate evidence; any
   change requires a new release through steps 1–7.

The certification record itself is storage metadata: it is never a
bundle component and can never alter the scientific digest or release
identity.  A researcher can reconstruct the chain from the sidecar and
`FINAL_STORAGE_CERTIFICATION.json` alone, without reading source code.

### Release immutability

- Scientific files inside a release are immutable; changing any of them
  changes the scientific digest and therefore the release identity.
- Storage metadata may be finalized in the sidecar after the storage
  commit (e.g. recording `gate_evidence_commit` and
  `gate_evidence_sha256`) without forking the release — that is exactly
  what the three-digest model permits.
- Archived releases (`results/archive/`) are never modified; their
  `INVALIDATION_MARKER.json` stays intact.
- Supersession produces a **new release ID** only when scientific
  content changes, and formally archives the superseded release with
  documented reasons — an existing release is never silently altered.

---

## Important Notes

- **Fixed embeddings are only for deterministic tests.** Real semantic claims require experiment mode.
- **Provider/model/dimension must appear in result metadata** for reproducibility.
- **Credentials must not be committed.** API keys should be set via environment variables.
- **CI does not call the real provider.** All CI tests use fixed embeddings or mock the provider.
