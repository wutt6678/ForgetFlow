# ForgetFlow

**A communication firewall for enforcing machine forgetting in LLM-based multi-agent systems.**

ForgetFlow is an academic research prototype that intercepts messages between agents in a multi-agent system and prevents information that has been marked for forgetting from leaking through — whether directly, via aliases, paraphrases, fragmented reconstruction, or recontamination.

The system runs on top of the [MARBLE](https://github.com/ulab-uiuc/MARBLE) multi-agent framework and is evaluated using the **TrustParadox-U** benchmark, an extension of scenarios from the paper *"The Trust Paradox in LLM-Based Multi-Agent Systems: When Collaboration Becomes a Security Vulnerability"*.

---

## Overview

ForgetFlow addresses a critical gap in LLM-based multi-agent systems: when information must be forgotten (e.g., a revoked credential, a retracted private attribute, a cancelled authorization), standard message-passing architectures have no mechanism to prevent that information from continuing to flow between agents.

ForgetFlow sits as a **firewall** between agents, inspecting every message before delivery and applying one of four enforcement actions:

| Action | Description |
|--------|-------------|
| **allow** | Message is safe to deliver as-is |
| **redact** | Replace leaked entities with `[REDACTED]` |
| **abstract** | Replace with a permitted residual statement |
| **block** | Prevent delivery entirely |

---

## Architecture

```
Agent A creates candidate message
            ↓
         FlowGate
            ↓
   Load active forget targets (ForgetLedger)
            ↓
   Exact / alias / semantic checks (HybridDetector)
            ↓
   Check recipient history (RecipientHistory)
            ↓
   Estimate reconstruction risk (ReconstructionChecker)
            ↓
   Policy decides: allow / redact / abstract / block (ForgetPolicy)
            ↓
   Recheck transformed message
            ↓
   Deliver only sanitized output
            ↓
   Store only released output in history
```

### Core Components

| Component | File | Responsibility |
|-----------|------|----------------|
| **ForgetLedger** | `marble/firewall/registry.py` | Stores and retrieves active forget targets with scope-aware activation |
| **HybridDetector** | `marble/firewall/detectors.py` | Exact, alias/entity, and semantic similarity leakage detection |
| **RecipientHistory** | `marble/firewall/history.py` | Bounded per-recipient message history (released text only) |
| **ReconstructionChecker** | `marble/firewall/history.py` | Fragment and fact-chain reconstruction detection |
| **ForgetPolicy** | `marble/firewall/policy.py` | Deterministic policy: allow / redact / abstract / block |
| **FlowGate** | `marble/firewall/flow_gate.py` | Single message-decision point combining all components |
| **ContaminationTracker** | `marble/firewall/contamination.py` | Agent contamination state machine (unknown → contaminated → clean → verified → at_risk → recontaminated) |
| **AuditLogger** | `marble/firewall/audit.py` | Structured JSONL audit of every firewall decision |

---

## TrustParadox-U Benchmark

Three pilot scenarios exercise different forgetting challenges:

| Scenario | Target | Attack Types |
|----------|--------|--------------|
| **Credential** | `0107` (warehouse access code) | direct, alias, paraphrase, temporal fragmentation |
| **Private Attribute** | `Alice has condition X` | direct, paraphrase, compositional inference |
| **Authorization** | `Agent B has emergency drone-control authority` | high-trust request, recontamination, cross-agent fragmentation |

---

## Quick Start

### Prerequisites

- Python 3.11+ (via Conda)
- Poetry 1.x or 2.x

### Setup

```bash
# Create and activate the Conda environment
conda create -n forgetflow python=3.11 -y
conda activate forgetflow

# Install dependencies
cd ForgetFlow
poetry install
```

### Run Tests

```bash
conda activate forgetflow
poetry run pytest tests/ -v
```

### Run a Smoke Experiment

```bash
conda activate forgetflow
poetry run python -m experiments.trustparadox_u.runner \
  --config experiments/trustparadox_u/configs/smoke.yaml \
  --limit 3
```

### Generate Experiment Matrix

```bash
poetry run python -m experiments.trustparadox_u.generate_matrix \
  --output results/trustparadox_u/matrix.jsonl
```

### Run Full MVP Experiment

```bash
poetry run python -m experiments.trustparadox_u.runner \
  --config experiments/trustparadox_u/configs/full_mvp.yaml \
  --split validation \
  --output results/trustparadox_u/full_mvp
```

---

## Experiment Configurations

| Config | Description |
|--------|-------------|
| `smoke.yaml` | Minimal single-repetition run for development |
| `no_firewall.yaml` | Baseline with all firewall components disabled |
| `exact_only.yaml` | Only exact string matching, no semantic/history |
| `full_mvp.yaml` | Full system: exact + alias + semantic + history + rich policy + continuous monitoring |
| `ablation_no_semantic.yaml` | Full MVP minus semantic detector |
| `ablation_stateless.yaml` | Full MVP minus recipient history |
| `ablation_binary_policy.yaml` | Full MVP with allow/block only (no redact/abstract) |
| `ablation_one_time_monitor.yaml` | Full MVP with one-time instead of continuous monitoring |

---

## Research Questions

The system is designed to answer:

1. **RQ1**: Does semantic detection reduce leakage compared with exact/alias matching only?
2. **RQ2**: Does recipient-aware filtering reduce fragmented reconstruction?
3. **RQ3**: Does a rich policy (allow/redact/abstract/block) preserve more utility than binary (allow/block)?
4. **RQ4**: Does continuous monitoring reduce recontamination?
5. **RQ5**: Does the firewall remain effective under low, default, and high trust?
6. **RQ6**: How sensitive is performance to core hyperparameters?
7. **RQ7**: Do sanitized transcripts reveal less information than raw transcripts?

---

## Project Structure

```
ForgetFlow/
├── marble/
│   ├── firewall/           # Core firewall components
│   │   ├── types.py        # Core data types
│   │   ├── registry.py     # ForgetLedger
│   │   ├── detectors.py    # HybridDetector
│   │   ├── history.py      # RecipientHistory + ReconstructionChecker
│   │   ├── policy.py       # ForgetPolicy
│   │   ├── flow_gate.py    # FlowGate
│   │   ├── contamination.py# ContaminationTracker
│   │   └── audit.py        # AuditLogger
│   └── agent/
│       └── base_agent.py   # BaseAgent with interceptor support
├── experiments/
│   └── trustparadox_u/     # Experiment framework
│       ├── config.py       # Configuration loading
│       ├── dataset.py      # Episode loader
│       ├── embedding.py    # Embedding providers
│       ├── agent.py        # TrustParadoxAgent
│       ├── attacks.py      # Attack library
│       ├── runner.py       # Episode runner
│       ├── evaluator.py    # Metrics computation
│       ├── generate_matrix.py  # Experiment matrix
│       ├── aggregate.py    # Result aggregation
│       ├── information_theory.py # Information-theoretic analysis
│       └── configs/        # YAML experiment configs
├── data/
│   └── trustparadox_u/     # Benchmark data
│       ├── schema/         # JSON Schema
│       ├── scenarios/      # Pilot YAML scenarios
│       └── splits/         # Train/val/test splits
├── tests/                  # 128 unit + integration tests
├── doc/                    # Documentation
└── results/                # Experiment outputs (gitignored)
```

---

## Documentation

- [Implementation Details](doc/implementation.md)
- [Changelog](doc/changelog.md)
- [Key Results](doc/results.md)
- [Future Work](FUTURE_WORK.md)

---

## Important Limitations

This MVP does **not** implement:
- Model-weight unlearning / parameter erasure
- Fine-tuning or retraining
- Production deployment or distributed systems
- Formal zero-leakage guarantees
- Legal compliance claims

ForgetFlow evaluates whether a **communication firewall** can enforce forgetting after an unlearning event. It makes no claims about erasing information from model parameters.

---

## License

Academic research prototype. See [FUTURE_WORK.md](FUTURE_WORK.md) for planned extensions.
