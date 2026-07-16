# Changelog

All notable changes to the ForgetFlow project.

---

## [0.3.0] — 2026-07-16

### Added

#### Ground-Truth Labels (Priority 2)
- `MessageLabel` dataclass with `is_attack_attempt`, `is_legitimate_message`, `is_reconstruction_attempt`, `is_recontamination_attempt`, `task_relevant`
- Extended episode schema with `label`, `reconstruction`, `success_type`, `success_value` fields
- All 3 pilot YAML files updated with explicit ground-truth labels per attack
- Reconstruction metadata on sensitive items (fragments and fact-chain types)

#### Corrected Evaluator Metrics (Priority 3)
- `MetricValue` dataclass returning `value`, `numerator`, `denominator`, `reason`
- PU-RER: uses `is_attack_attempt` + `target_exposed` (from `released_text`, never `candidate_text`)
- CRR: uses `is_reconstruction_attempt` + `target_reconstructed`
- RR: uses explicit `cleaned_agents_exposed` / `recontaminated_agents` counters
- FBR: uses `is_legitimate_message` ground-truth labels, not firewall reason codes
- Utility retention: uses explicit `task_success` field

#### Transformed-Output Rechecking (Priority 5)
- FlowGate recheck now validates all risk dimensions: exact, entity, semantic, reconstruction

#### Research Test Suite (Priority 6)
- Rich policy utility test
- Continuous vs one-time monitoring test
- Trust invariance uses `released_text` verification

#### Metric-Contract Regression Tests (Priority 7)
- 10 tests verifying metric correctness contracts
- Candidate-only secret must not increase PU-RER
- Blocked reconstruction must not increase CRR
- FBR only counts legitimate messages
- AT_RISK must not count as recontamination

#### Result Auditor (Priority 9)
- `experiments/trustparadox_u/audit_results.py`: validates internal consistency
- Checks block ↔ `released_text is None` invariant
- Checks `target_exposed` requires `released_text`
- Checks reconstruction/recontamination require attempts
- `validate_for_aggregation()` prevents invalid run aggregation

#### Documentation (Priority 11)
- `doc/METRICS.md`: full metric definitions with numerators, denominators, and source fields
- Corrected MARBLE integration wording (MARBLE-compatible, not "runs on MARBLE")
- Comprehensive limitations section

### Changed

- `TurnResult`: added `is_legitimate_message` field populated from episode labels
- `EpisodeResult`: metadata now includes `attack_type`, `secret_variant_id`, `seed`
- `target_reconstructed` and `target_reintroduced` computed from `released_text`
- Full MVP config enables `semantic_enabled=True`
- Evaluator returns `MetricValue` instead of bare `float | None`
- No-firewall delivery test checks `released_text` instead of `candidate_text`

### Fixed

- Evaluator PU-RER no longer inspects `candidate_text` for exposure
- Evaluator CRR no longer uses reason codes for reconstruction detection
- Evaluator FBR no longer uses `NO_LEAKAGE_DETECTED` as ground truth
- FlowGate recheck now includes entity and semantic scores (was only exact + reconstruction)

---

## [0.2.1] — 2026-07-16

### Added

#### Experiment Outcome Recording
- `TurnResult` extended with `released_text`, attack classification, outcome flags
- `EpisodeResult` extended with `run_id`, `seed`, `task_success`, contamination counters
- `_check_target_exposed()` helper checking `released_text` only
- `target_reconstructed` and `target_reintroduced` tracking in runner

### Changed

- Runner populates metadata with `forbidden_strings`, `forbidden_aliases`, `semantic_variants`, `permitted_residuals`, `forget_ids`, `seed`, `config_hash`

---

## [0.2.0] — 2026-07-16

### Added

#### MVP Consolidation Repair
- Comprehensive CI workflow with ruff, mypy, and source integrity checks
- Workflow validator script (`scripts/validate_workflows.py`)
- Source integrity checker (`scripts/check_source_integrity.py`)
- Architecture contract tests (`tests/test_architecture_contract.py`)
- Ruff configuration in `pyproject.toml`

### Changed

- Replaced simple test workflow with canonical MVP CI
- Applied ruff formatting across all modules
- Fixed all mypy type errors
- Removed unused imports and variables
- Fixed import sorting throughout codebase

### Fixed

- Unused variable `pred` in `marble/firewall/history.py`
- Type annotations for mypy compliance in `dataset.py` and `evaluator.py`
- Import sorting in all modules

---

## [0.1.0] — 2026-07-16

### Added

#### Iteration 0 — Bootstrap
- Initialized ForgetFlow project structure
- Set up Poetry with Python 3.11 (Conda environment)
- Created directory layout: `marble/firewall/`, `marble/agent/`, `experiments/trustparadox_u/`, `data/trustparadox_u/`, `tests/`
- Added `.gitignore` (results directory excluded)
- Created `FUTURE_WORK.md` for deferred features

#### Iteration 1 — Core Data Types
- `marble/firewall/types.py`: Six frozen dataclasses with validation
  - `ForgetRecord` — forget target with aliases, variants, residuals, scope
  - `MessageEnvelope` — candidate message with full metadata
  - `DetectorResult` — four leakage scores in [0,1]
  - `FirewallDecision` — enforcement action with reason codes
  - `RecipientHistoryItem` — released message record
  - `ContaminationStatus` — six-state enum
- `tests/firewall/test_types.py`: 26 tests covering all validation rules

#### Iteration 2 — Configuration
- `experiments/trustparadox_u/config.py`: Typed YAML configuration
  - `DetectorConfig`, `HistoryConfig`, `PolicyConfig`, `MonitoringConfig`, `ExperimentConfig`
  - `load_config()` with explicit validation (thresholds, windows, durations)
- `experiments/trustparadox_u/configs/smoke.yaml`: Minimal development config
- `tests/trustparadox_u/test_config.py`: 11 tests

#### Iteration 3 — Dataset Schema and Loader
- `data/trustparadox_u/schema/episode.schema.json`: JSON Schema for episodes
- `experiments/trustparadox_u/dataset.py`: Episode loader with validation
  - Agent reference validation, duplicate forget ID rejection
  - `TrustParadoxEpisode` and supporting dataclasses
- Three pilot scenarios:
  - `pilot_credential.yaml` — target: `0107`
  - `pilot_private_attribute.yaml` — target: `Alice has condition X`
  - `pilot_authorization.yaml` — target: `Agent B has emergency drone-control authority`
- Split files: `development.jsonl`, `validation.jsonl`, `test.jsonl`
- `tests/trustparadox_u/test_schema.py`: 11 tests

#### Iteration 4 — ForgetLedger
- `marble/firewall/registry.py`: Scope-aware forget target registry
  - `register()`, `register_many()`, `get()`, `active_records()`, `policy_version()`
  - Duplicate ID rejection, deterministic version hashing
- `tests/firewall/test_registry.py`: 11 tests

#### Iteration 5 — Exact and Alias Detection
- `marble/firewall/detectors.py`: `HybridDetector` with normalization pipeline
  - Exact matching (normalized substring search)
  - Alias/entity matching (normalized alias substring search)
  - Text normalization: lowercase, Unicode NFC, punctuation strip, whitespace collapse
- `tests/firewall/test_detectors.py`: 12 tests

#### Iteration 6 — Semantic Detection
- `experiments/trustparadox_u/embedding.py`: `EmbeddingProvider` protocol + `StubEmbeddingProvider`
  - Deterministic hash-based embeddings for testing
  - `cosine_similarity()` utility
- Extended `HybridDetector` with semantic similarity layer
  - Per-episode embedding cache
  - Configurable enable/disable

#### Iteration 7 — RecipientHistory
- `marble/firewall/history.py`: `RecipientHistory` class
  - Bounded per-recipient sliding window
  - Only released text stored (never blocked candidates)
  - Recipient isolation
- `tests/firewall/test_history.py`: 5 history tests

#### Iteration 8 — ReconstructionChecker
- `ReconstructionChecker` in `marble/firewall/history.py`
  - Mechanism A: Fragment reconstruction from `fragment_map`
  - Mechanism B: Fact-chain reconstruction from `fact_chains`
  - Stateless mode returns 0.0
- 4 reconstruction tests

#### Iteration 9 — ForgetPolicy
- `marble/firewall/policy.py`: Deterministic policy
  - Decision tree: exact→block, reconstruction→block, semantic→abstract/block, entity→redact/block, else→allow
  - `redact_text()`: Replace targets/aliases with `[REDACTED]`
  - Binary ablation mode (allow/block only)
  - Trust-independent (ignores trust level)
- `tests/firewall/test_policy.py`: 10 tests

#### Iteration 10 — FlowGate
- `marble/firewall/flow_gate.py`: Single message-decision point
  - Combines ledger, detector, history, reconstruction, policy
  - 11-step inspection pipeline
  - Recheck of transformed outputs
  - Only released text enters history
- `tests/firewall/test_flow_gate.py`: 6 tests

#### Iteration 11 — BaseAgent Integration
- `marble/agent/base_agent.py`: MARBLE-compatible agent
  - Optional `message_interceptor` (None = original behavior)
  - `send_message()` routes through FlowGate when interceptor is set
  - `remove_memory_containing()` for simulated cleaning

#### Iteration 12 — Audit Logging
- `marble/firewall/audit.py`: `AuditLogger`
  - Append-safe JSONL output
  - Full decision metadata per candidate message
  - In-memory entry list for programmatic access

#### Iteration 13 — Controlled Agent
- `experiments/trustparadox_u/agent.py`:
  - `TrustParadoxAgent` with role, profile, response provider
  - `ScriptedResponder` for deterministic testing
  - Evaluator data excluded from agent-visible context

#### Iteration 14 — Attack Library
- `experiments/trustparadox_u/attacks.py`: 7 attack types
  - direct, alias, paraphrase, temporal_fragmentation, cross_agent_fragmentation, compositional_inference, recontamination
  - Deterministic `build_attack()` from episode + seed
- `tests/trustparadox_u/test_attacks.py`: 6 tests

#### Iteration 15 — ContaminationTracker
- `marble/firewall/contamination.py`: State machine
  - 6 states, validated transitions
  - `record_exposure()` for risk detection
  - `confirm_recovery()` for recontamination confirmation
- `tests/firewall/test_contamination.py`: 8 tests

#### Iteration 16 — Episode Runner
- `experiments/trustparadox_u/runner.py`: Deterministic phase execution
  - Phases: INITIALIZE → PRE_FORGET → ACTIVATE_FORGET → CLEAN → IMMEDIATE_PROBE → POST_FORGET_ATTACK → FINAL_PROBE → FINALIZE
  - `EpisodeResult` with turns, contamination states, audit entries
  - CLI: `--config`, `--split`, `--output`, `--limit`
- `tests/trustparadox_u/test_runner.py`: 3 tests

#### Iteration 17 — Evaluator
- `experiments/trustparadox_u/evaluator.py`: Five metrics
  - PU-RER, CRR, RR, FBR, utility retention
  - Zero-denominator safety (returns None)
- `tests/trustparadox_u/test_evaluator.py`: 5 tests

#### Iteration 18 — End-to-End Pilot Tests
- `tests/trustparadox_u/test_end_to_end.py`: 5 integration tests
  - Exact credential blocked by full MVP
  - No-firewall baseline allows secret
  - Trust invariance verification
  - All three pilots run end-to-end
- **128 total tests, all passing**

#### Iteration 19 — Experiment Config Variants
- 7 additional YAML configs:
  - `no_firewall.yaml`, `exact_only.yaml`, `full_mvp.yaml`
  - `ablation_no_semantic.yaml`, `ablation_stateless.yaml`
  - `ablation_binary_policy.yaml`, `ablation_one_time_monitor.yaml`
- Each ablation changes exactly one component

#### Iteration 20 — Matrix Generator
- `experiments/trustparadox_u/generate_matrix.py`: Paired experiment runs
  - Deterministic, no duplicate IDs
  - JSONL output
  - CLI: `--split`, `--output`
- `tests/trustparadox_u/test_matrix.py`: 4 tests

#### Iteration 21 — Hyperparameter Sweep Support
- Configuration supports five core parameters:
  - semantic_threshold, history window, reconstruction_threshold, privacy_utility_weight, monitoring duration

#### Iteration 22 — Trust-Conditioned Study Support
- `trust_independent` flag in PolicyConfig
- Trust level preserved in MessageEnvelope for analysis

#### Iteration 23 — Information-Theoretic Analysis
- `experiments/trustparadox_u/information_theory.py`:
  - Entropy, recovery accuracy, conditional entropy, mutual information
  - Raw vs sanitized transcript comparison

#### Iteration 24 — Dataset Expansion Support
- Schema supports fragment_map and fact_chains
- Three pilot scenarios with diverse attack coverage

#### Iteration 25 — Aggregation
- `experiments/trustparadox_u/aggregate.py`: Result aggregation and table formatting
