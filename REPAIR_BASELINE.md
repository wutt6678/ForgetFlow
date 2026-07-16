# Repair Baseline

Recorded before any source changes.

## Commit SHA

```
3c0ec8a3caccacc41ba5d69c75045a87ca5af000
```

## Environment

- Python: 3.11.15 (via Conda)
- Poetry: 2.4.1

## Baseline Status

- `python -m compileall marble experiments`: PASS
- `poetry check`: PASS (deprecation warnings only)
- `poetry run pytest --collect-only -q`: 128 tests collected
- `poetry run pytest -q`: 128 passed

## Current Architecture

The repository already follows the canonical MVP architecture:

```
marble/firewall/
├── types.py          # Core data types
├── registry.py       # ForgetLedger
├── detectors.py      # HybridDetector
├── history.py        # RecipientHistory + ReconstructionChecker
├── policy.py         # ForgetPolicy
├── flow_gate.py      # FlowGate
├── contamination.py  # ContaminationTracker
└── audit.py          # AuditLogger
```

No duplicate implementations (flowgate.py, ledger.py, etc.) exist.

## Current CI Workflow

`.github/workflows/test.yml` uses:
- Miniconda setup
- Python 3.11
- Poetry 2.4.1
- pytest

## Improvements Needed

1. Replace with comprehensive CI (ruff, mypy, source integrity)
2. Add workflow validator script
3. Add source integrity checker
4. Add architecture contract tests
5. Add import contract verification
