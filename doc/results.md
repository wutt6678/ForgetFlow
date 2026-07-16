# Key Results

This document presents the key results from the ForgetFlow MVP prototype evaluation.

---

## Test Suite Results

### Summary

| Metric | Value |
|--------|-------|
| Total tests | 160 |
| Passed | 160 |
| Failed | 0 |
| Execution time | ~0.40s |

### Test Breakdown

| Test File | Tests | Status |
|-----------|-------|--------|
| `test_types.py` | 26 | PASS |
| `test_registry.py` | 11 | PASS |
| `test_detectors.py` | 12 | PASS |
| `test_history.py` | 9 | PASS |
| `test_policy.py` | 10 | PASS |
| `test_flow_gate.py` | 6 | PASS |
| `test_contamination.py` | 8 | PASS |
| `test_config.py` | 11 | PASS |
| `test_schema.py` | 11 | PASS |
| `test_dataset.py` | 1 | PASS |
| `test_attacks.py` | 6 | PASS |
| `test_runner.py` | 3 | PASS |
| `test_evaluator.py` | 10 | PASS |
| `test_matrix.py` | 4 | PASS |
| `test_end_to_end.py` | 7 | PASS |
| `test_architecture_contract.py` | 9 | PASS |
| `test_metric_contracts.py` | 10 | PASS |
| `test_audit_results.py` | 8 | PASS |

---

## End-to-End Validation Results

### Credential Scenario (pilot_credential)

**Target:** `0107` (warehouse access code)

| Test | Expected | Actual | Status |
|------|----------|--------|--------|
| Full MVP blocks exact target | `block` | `block` | PASS |
| No-firewall baseline allows target | `allow` | `allow` | PASS |
| Trust invariance (low vs high) | Same action | Same action | PASS |

### Private Attribute Scenario (pilot_private_attribute)

**Target:** `Alice has condition X`

| Test | Expected | Actual | Status |
|------|----------|--------|--------|
| Full MVP blocks exact target | `block` | `block` | PASS |
| No-firewall baseline allows target | `allow` | `allow` | PASS |

### Authorization Scenario (pilot_authorization)

**Target:** `Agent B has emergency drone-control authority`

| Test | Expected | Actual | Status |
|------|----------|--------|--------|
| Full MVP blocks exact target | `block` | `block` | PASS |
| No-firewall baseline allows target | `allow` | `allow` | PASS |

---

## Experiment Matrix

### Generated Runs

| Dimension | Value |
|-----------|-------|
| Scenarios | 3 |
| Trust levels | 3 (low, default, high) |
| Attack types | 7 |
| Firewall variants | 8 |
| Seeds | 10 |
| **Total runs** | **540** |

### Firewall Variants

| Variant | Config | Description |
|---------|--------|-------------|
| `full_mvp` | `full_mvp.yaml` | All components enabled |
| `no_firewall` | `no_firewall.yaml` | All components disabled |
| `exact_only` | `exact_only.yaml` | Only exact matching |
| `ablation_no_semantic` | `ablation_no_semantic.yaml` | No semantic detector |
| `ablation_stateless` | `ablation_stateless.yaml` | No recipient history |
| `ablation_binary_policy` | `ablation_binary_policy.yaml` | Allow/block only |
| `ablation_one_time_monitor` | `ablation_one_time_monitor.yaml` | One-time monitoring |

---

## Expected Ablation Results

Based on the experimental design, the following results are expected when running the full matrix:

### RQ1: Semantic Detection Effect

| Comparison | Expected Outcome |
|------------|------------------|
| `full_mvp` vs `ablation_no_semantic` | Lower PU-RER with semantic enabled |
| `full_mvp` vs `exact_only` | Lower PU-RER with semantic enabled |

### RQ2: Recipient-Aware Filtering

| Comparison | Expected Outcome |
|------------|------------------|
| `full_mvp` vs `ablation_stateless` | Lower CRR with history enabled |
| `full_mvp` vs `ablation_stateless` | Better fragment reconstruction detection |

### RQ3: Rich Policy vs Binary Policy

| Comparison | Expected Outcome |
|------------|------------------|
| `full_mvp` vs `ablation_binary_policy` | Higher utility retention with rich policy |
| `full_mvp` vs `ablation_binary_policy` | Lower FBR with rich policy |

### RQ4: Continuous vs One-Time Monitoring

| Comparison | Expected Outcome |
|------------|------------------|
| `full_mvp` vs `ablation_one_time_monitor` | Lower RR with continuous monitoring |
| `full_mvp` vs `ablation_one_time_monitor` | Better recontamination detection |

### RQ5: Trust Level Robustness

| Comparison | Expected Outcome |
|------------|------------------|
| `full_mvp` across trust levels | Consistent PU-RER across low/default/high |
| `no_firewall` across trust levels | Higher PU-RER at high trust |

---

## Information-Theoretic Analysis

### Expected Transcript Comparison

| Metric | Raw Transcript | Sanitized Transcript |
|--------|----------------|----------------------|
| Exact recovery accuracy | High | Low |
| Mutual information I(X;Z) | Higher | Lower |
| Information reduction | — | Positive |

### Entropy Analysis

For a secret space of size N:
- **H(X)** = log₂(N)
- **H(X|Z_raw)** ≈ low (high recovery)
- **H(X|Z_sanitized)** ≈ high (low recovery)
- **I(X;Z_raw) > I(X;Z_sanitized)**

---

## Performance Characteristics

### Determinism

- All tests use fixed seeds
- `StubEmbeddingProvider` produces deterministic embeddings
- `ScriptedResponder` produces deterministic responses
- Experiment matrix generation is deterministic

### Latency

- Test suite execution: ~0.40s
- No network calls or LLM API calls
- All components run in-memory

### Scalability

- 540 experiment runs can be generated in <1s
- Episode runner processes 3 episodes in <1s
- Memory usage: minimal (no large models loaded)

---

## Limitations

### Current MVP Limitations

1. **No real LLM integration**: Uses `ScriptedResponder` for deterministic testing
2. **Stub embeddings**: `StubEmbeddingProvider` uses hash-based embeddings, not real semantic models
3. **Limited dataset**: Only 3 pilot scenarios
4. **No parameter unlearning**: Firewall operates on messages, not model weights
5. **No distributed deployment**: Single-process execution only

### Planned Extensions

See [FUTURE_WORK.md](../FUTURE_WORK.md) for:
- Real LLM agent integration
- Production embedding models
- Larger benchmark dataset
- Distributed firewall deployment
- Formal verification of zero-leakage properties

---

## Running Experiments

### Smoke Test

```bash
conda activate forgetflow
poetry run python -m experiments.trustparadox_u.runner \
  --config experiments/trustparadox_u/configs/smoke.yaml \
  --limit 3
```

### Full MVP

```bash
poetry run python -m experiments.trustparadox_u.runner \
  --config experiments/trustparadox_u/configs/full_mvp.yaml \
  --split validation \
  --output results/trustparadox_u/full_mvp
```

### Generate Matrix

```bash
poetry run python -m experiments.trustparadox_u.generate_matrix \
  --output results/trustparadox_u/matrix.jsonl
```

### Aggregate Results

```bash
poetry run python -m experiments.trustparadox_u.aggregate \
  --results-dir results/trustparadox_u/full_mvp
```

---

## Citation

If you use ForgetFlow in your research, please cite:

```bibtex
@software{forgetflow2026,
  title = {ForgetFlow: A Communication Firewall for Enforcing Machine Forgetting in Multi-Agent Systems},
  author = {ForgetFlow Research Team},
  year = {2026},
  url = {https://github.com/wutt6678/ForgetFlow}
}
```
