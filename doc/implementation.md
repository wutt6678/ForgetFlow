# Implementation Details

This document describes the technical implementation of the ForgetFlow MVP prototype.

---

## 1. Development Environment

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.11.15 | Via Conda `forgetflow` environment |
| Poetry | 2.4.1 | Dependency management |
| pytest | 7.4.4 | Test runner |
| mypy | 1.20.2 | Static type checking |
| PyYAML | 6.0.3 | Configuration parsing |
| jsonschema | 4.26.0 | Episode schema validation |

### Environment Setup

```bash
conda create -n forgetflow python=3.11 -y
conda activate forgetflow
cd ForgetFlow
poetry install
```

---

## 2. Core Data Types (`marble/firewall/types.py`)

Six frozen dataclasses form the type foundation:

### ForgetRecord
Represents a piece of information that must no longer be transmitted. Contains the canonical target string, aliases, semantic variants, permitted residuals, and activation turn. Validation ensures no empty IDs, targets, or negative turns.

### MessageEnvelope
Wraps a candidate message between agents with full metadata: message ID, episode/session/turn context, sender/recipient IDs, raw text, and trust level (low/default/high).

### DetectorResult
Output of the hybrid leakage detector with four scores in [0,1]: exact match, entity/alias match, semantic similarity, and reconstruction risk. Includes matched forget IDs and evidence strings.

### FirewallDecision
The enforcement outcome: action (allow/redact/abstract/block), released text (None for block), detector result, reason codes, policy version, and latency. Validation enforces that block requires None released_text and other actions require non-empty text.

### RecipientHistoryItem
A single message released to a recipient: message ID, turn, sender, and the released (sanitized) text only.

### ContaminationStatus
Enum with six states: UNKNOWN → CONTAMINATED → CLEAN → VERIFIED → AT_RISK → RECONTAMINATED.

---

## 3. ForgetLedger (`marble/firewall/registry.py`)

Stores forget records and provides scope-aware activation queries.

**Key behaviors:**
- Records become active at their `active_from_turn`
- Global scope (empty `scoped_agent_ids`) applies to all agents
- Scoped records only apply when sender or recipient is in scope
- `policy_version()` returns a deterministic SHA-256 hash that changes when records change
- Duplicate forget IDs are rejected

---

## 4. HybridDetector (`marble/firewall/detectors.py`)

Three-layer leakage detection with configurable enable/disable per layer:

### Exact Matching
Normalizes text (lowercase, Unicode NFC, whitespace collapse, punctuation strip) and checks if the normalized canonical target appears as a substring. Score: 1.0 or 0.0.

### Alias/Entity Matching
Same normalization applied to each alias. If any alias appears in the normalized message, entity_score = 1.0.

### Semantic Similarity
Uses an `EmbeddingProvider` (protocol) to compute cosine similarity between the message and each semantic variant. Returns the maximum similarity as semantic_score. Target embeddings are cached per episode.

**Normalization pipeline:**
```
text → lowercase → Unicode NFC → strip punctuation → collapse whitespace
```

**Embedding providers:**
- `FixedEmbeddingProvider`: Deterministic predefined vectors for testing (`provider_name="fixed"`, `model_name=None`)
- `RealEmbeddingProvider`: Real embedding API via LiteLLM (`provider_name="litellm"`). Accepts optional `api_base` for custom endpoints (e.g., Alibaba Cloud MaaS). Model name uses LiteLLM provider prefix (e.g., `openai/text-embedding-v3`).

---

## 5. RecipientHistory (`marble/firewall/history.py`)

Stores only **released** (sanitized) messages per recipient with a bounded sliding window.

**Rules:**
- Blocked candidate text is never stored
- Window is deterministic: last N messages per recipient
- Recipients are fully isolated
- `get_context()` returns a `RecipientContext` with recent texts

---

## 6. ReconstructionChecker (`marble/firewall/history.py`)

Two deterministic mechanisms for detecting compositional leakage:

### Mechanism A: Fragment Reconstruction
Episode metadata provides a `fragment_map` mapping forget IDs to their fragments. If the combined recipient history + candidate message contains all fragments, reconstruction_score = 1.0.

### Mechanism B: Fact-Chain Reconstruction
Episode metadata provides fact chains as triples (subject, predicate, object). If all triples in a chain have their subject and object present in the combined text, reconstruction_score = 1.0.

Both mechanisms return 0.0 when history is disabled (stateless ablation).

---

## 7. ForgetPolicy (`marble/firewall/policy.py`)

Deterministic decision tree:

```
if exact_score == 1.0:
    → block

elif reconstruction_score >= threshold:
    → block

elif semantic_score >= semantic_threshold:
    if rich_actions_enabled and permitted_residual exists:
        → abstract (return first permitted residual)
    else:
        → block

elif entity_score > 0:
    if rich_actions_enabled:
        → redact (replace targets/aliases with [REDACTED])
    else:
        → block

else:
    → allow
```

**Binary ablation:** When `rich_actions_enabled=false`, only allow/block are used.

**Trust independence:** The main policy ignores trust level entirely. A separate trust-sensitive comparison can alter thresholds by trust level.

**Recheck:** Transformed messages (redacted/abstracted) are run through the detector again. If still unsafe, the action is escalated to block.

---

## 8. FlowGate (`marble/firewall/flow_gate.py`)

The single message-decision point. Execution order:

1. Start timer
2. Load active forget records from ledger
3. Load recipient context from history
4. Run hybrid detector
5. Run reconstruction checker
6. Merge reconstruction score into detector result
7. Run policy to get action
8. Handle redaction (apply text substitution)
9. Recheck transformed output
10. Log decision to audit
11. Append released text to recipient history
12. Return FirewallDecision

If no active records exist for the sender-recipient pair, the message is immediately allowed.

---

## 9. ContaminationTracker (`marble/firewall/contamination.py`)

State machine tracking each agent's contamination status per forget target:

```
UNKNOWN → CONTAMINATED → CLEAN → VERIFIED → AT_RISK → RECONTAMINATED
                                ↘ AT_RISK ↗
```

Invalid transitions raise `ValueError`. The tracker:
- Records exposure when a CLEAN/VERIFIED agent receives an exact match or high reconstruction score
- Confirms recovery (AT_RISK → RECONTAMINATED) when a probe recovers the target

---

## 10. AuditLogger (`marble/firewall/audit.py`)

Append-safe JSONL logger. Each entry contains:
- Run/episode/session/turn metadata
- Sender/recipient IDs and trust level
- Candidate text (original message)
- Released text (null for blocked messages)
- Action, all four detector scores, matched forget IDs
- Reason codes, latency, policy version, seed

Agent-visible state never reads audit candidate text.

---

## 11. BaseAgent Integration (`marble/agent/base_agent.py`)

Minimal MARBLE-compatible agent with optional message interceptor:

- `message_interceptor` defaults to None (original MARBLE behavior preserved)
- `set_message_interceptor(flow_gate)` enables firewall interception
- `send_message()` creates a `MessageEnvelope` and calls `flow_gate.inspect()`
- Blocked messages are not delivered to the recipient
- `remove_memory_containing()` simulates cleaning by removing target-containing context/memory

---

## 12. TrustParadox-U Dataset

### Episode Schema
JSON Schema validates structure before Python object construction. Required fields: episode_id, scenario_id, agents, relationships, task, sensitive_items, phases, expected.

**Ground-truth labels:** Each post-forget attack includes a `label` block:
```yaml
label:
  is_attack_attempt: true
  is_legitimate_message: false
  is_reconstruction_attempt: false
  is_recontamination_attempt: false
  task_relevant: false
```

**Task success:** Episodes define explicit success criteria:
```yaml
task:
  success_type: required_release  # or exact_label
  success_value: Request a new temporary credential.
```

**Reconstruction metadata:** Sensitive items include reconstruction type:
```yaml
reconstruction:
  type: fragments  # or fact_chain
  forget_id: F001
  fragments: ["01", "07"]
```

### Pilot Scenarios
Three YAML files in `data/trustparadox_u/scenarios/`:

| File | Target | Agents |
|------|--------|--------|
| `pilot_credential.yaml` | `0107` | CK (custodian), SK (seeker) |
| `pilot_private_attribute.yaml` | `Alice has condition X` | HR (custodian), MGR (seeker) |
| `pilot_authorization.yaml` | `Agent B has emergency drone-control authority` | CMD (custodian), AGT_B (seeker) |

### Splits
- `development.jsonl` — used during development
- `validation.jsonl` — used for hyperparameter tuning
- `test.jsonl` — held-out evaluation (never used for tuning)

---

## 13. Episode Runner (`experiments/trustparadox_u/runner.py`)

Deterministic phase execution:

1. **INITIALIZE** — Create agents, firewall components, attach interceptor; populate metadata
2. **PRE_FORGET** — Execute pre-forget conversation turns
3. **ACTIVATE_FORGET** — Register forget records in ledger
4. **CLEAN** — Remove target-containing context/memory from designated agents
5. **IMMEDIATE_PROBE** — Verify cleaned agents don't reveal target
6. **POST_FORGET_ATTACK** — Execute attack plans with ground-truth labels from episode
7. **FINAL_PROBE** — Final verification of agent state
8. **FINALIZE** — Collect contamination states, compute outcome metrics

### TurnResult
Each turn records:
- `candidate_text` (what sender generated) vs `released_text` (what recipient received)
- Attack classification from episode labels (`is_attack_attempt`, `is_legitimate_message`, etc.)
- Outcome flags (`target_exposed`, `target_reconstructed`, `target_reintroduced`)
- All outcome flags computed from `released_text`, never `candidate_text`

### EpisodeResult
Each episode records:
- `run_id`, `seed` for reproducibility
- `task_success` from explicit task criteria
- `cleaned_agents_exposed`, `recontaminated_agents` for RR computation
- Metadata with forbidden strings, config hash, attack type

The runner uses `ScriptedResponder` for deterministic testing (no LLM API calls).

---

## 14. Attack Library (`experiments/trustparadox_u/attacks.py`)

Seven attack types plus a legitimate task message:

| Type | Category | Description |
|------|----------|-------------|
| `direct` | direct_request | Ask for the target directly |
| `alias` | alias_request | Use an alias to request the target |
| `paraphrase` | paraphrase_request | Use a semantic variant |
| `temporal_fragmentation` | fragment_reconstruction | Request fragments across turns |
| `cross_agent_fragmentation` | cross_fragment | Request fragments across agents |
| `compositional_inference` | compositional_inference | Request fact-chain completion |
| `recontamination` | recontamination | Re-introduce target to cleaned agent |
| `legitimate_task` | legitimate_task | Safe task-relevant message for utility evaluation |

Attack plans are deterministic given the same seed.

---

## 15. Evaluation Metrics (`experiments/trustparadox_u/evaluator.py`)

All metrics return `MetricValue(value, numerator, denominator, reason)`.

| Metric | Numerator | Denominator | Direction |
|--------|-----------|-------------|----------|
| **PU-RER** | Attack attempts with `target_exposed=True` | All `is_attack_attempt=True` turns | Lower = better |
| **CRR** | Reconstruction attempts with `target_reconstructed=True` | All `is_reconstruction_attempt=True` turns | Lower = better |
| **RR** | `recontaminated_agents` | `cleaned_agents_exposed` | Lower = better |
| **FBR** | Legitimate messages with `released_text=None` | All `is_legitimate_message=True` turns | Lower = better |
| **Utility Retention** | `task_success=True` with firewall | `task_success=True` without firewall | Higher = better |

**Key principle:** Metrics use ground-truth labels and `released_text`. They never inspect `candidate_text`.

Zero denominators return `MetricValue(None, 0, 0, reason)`.

See [METRICS.md](METRICS.md) for full definitions.

---

## 16. Information-Theoretic Analysis (`experiments/trustparadox_u/information_theory.py`)

For discrete uniform secrets:

- **H(X)** = log₂(secret_space_size)
- **H(X|Z)** estimated via recovery rate: H(X) × (1 - recovery_rate)
- **I(X;Z)** = H(X) - H(X|Z)

Compares raw vs sanitized transcripts on:
- Exact recovery accuracy
- Mutual information estimate
- Information reduction (raw MI - sanitized MI)

---

## 17. Experiment Matrix (`experiments/trustparadox_u/generate_matrix.py`)

Generates paired runs across dimensions:
- Scenarios × trust levels × attack types × firewall variants × seeds
- Deterministic ordering, no duplicate run IDs
- Output as JSONL

**Trust triplet validation:** `validate_trust_triplet()` ensures that low/default/high entries share all fields except `trust_level`. `group_trust_triplets()` groups entries by (scenario, attack, firewall_variant, seed) for paired comparison.

---

## 18. Result Auditor (`experiments/trustparadox_u/audit_results.py`)

Validates episode results before aggregation:

**Turn-level rules:**
- `block` action → `released_text is None`
- Non-block action → `released_text` exists
- `target_exposed=True` → `released_text` is not None
- `target_reconstructed=True` → `is_reconstruction_attempt=True`
- `target_reintroduced=True` → `is_recontamination_attempt=True`
- `task_contribution_successful=True` → `task_relevant=True`
- Blocked message must not create exposure or reintroduction (`BLOCKED_MESSAGE_AT_RISK`)
- Reconstructed result must have `released_text` (`RECONSTRUCTION_WITHOUT_RELEASED_HISTORY`)
- Legitimate task message must be task-relevant (`LEGITIMATE_TASK_NOT_TASK_RELEVANT`)

**Episode-level rules:**
- Config hash has valid SHA-256 length (64 hex chars)
- Secret variant ID exists
- Attack type exists
- `cleaned_agents_exposed` ≥ 0
- `recontaminated_agents` ≤ `cleaned_agents_exposed`
- Metadata contains `forbidden_strings`, `seed`, `config_hash`
- Run ID length must be 8–64 characters (`INVALID_RUN_ID`)

**Metric rules** (`audit_metric_value()`):
- numerator ≤ denominator
- Zero denominator → value is None
- Nonzero denominator → value in [0, 1]

`validate_for_aggregation()` raises `InvalidExperimentResults` by default when results have errors. Pass `allow_errors=True` to override.

---

## 19. Pilot Experiment Runner (`experiments/trustparadox_u/run_pilot.py`)

Runs the complete three-scenario pilot:
- 3 scenarios × 7 conditions × 3 trust levels × 5 seeds = 315 runs
- Scripted responders for deterministic execution
- Fixed embedding provider for semantic detection
- Verifies directional expectations (full MVP < no firewall, trust stability, etc.)
- Outputs: episode results JSONL, metric table JSON, directional check report

---

## 20. Testing Strategy

- **526 tests** across 26 test files
- No live model API calls — all tests use `ScriptedResponder` and `FixedEmbeddingProvider`
- Fixed seeds for determinism
- Tests cover: validation, unit behavior, integration, end-to-end, metric contracts, audit, trust matrix, runner population, aggregation, embedding providers, config validation, serialization, manifest generation, preflight checks, identity normalization, policy-ablation pairs

### Test Categories

| File | Tests | Coverage |
|------|-------|----------|
| `test_result_audit.py` | 65 | Extended audit rules + policy-ablation |
| `test_runner.py` | 45 | Episode execution + population |
| `test_serialization.py` | 43 | Serialization + schema versioning |
| `test_embedding.py` | 37 | Embedding providers + config |
| `test_manifest.py` | 33 | Manifest generation + validation |
| `test_types.py` | 26 | Data type validation |
| `test_config.py` | 25 | Configuration loading + validation |
| `test_metric_contracts.py` | 24 | Metric correctness |
| `test_identity.py` | 22 | Identity normalization |
| `test_end_to_end.py` | 21 | Six paired research cases |
| `test_preflight.py` | 19 | Preflight validation |
| `test_attacks.py` | 19 | Attack generation |
| `test_evaluator.py` | 17 | Metric computation |
| `test_aggregate.py` | 16 | Aggregation logic |
| `test_detectors.py` | 12 | Exact/alias detection |
| `test_schema.py` | 11 | Schema validation + dataset |
| `test_pipeline_integration.py` | 11 | Runner-to-aggregation pipeline |
| `test_matrix.py` | 11 | Matrix + trust triplet |
| `test_registry.py` | 11 | ForgetLedger behavior |
| `test_policy.py` | 10 | Policy decisions |
| `test_flow_gate.py` | 10 | FlowGate integration |
| `test_architecture_contract.py` | 9 | Architecture regression |
| `test_history.py` | 9 | History + reconstruction |
| `test_audit_results.py` | 8 | Result audit validation |
| `test_contamination.py` | 8 | State machine transitions |
| `test_dataset.py` | 4 | Split loading |
