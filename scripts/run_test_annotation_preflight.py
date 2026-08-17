#!/usr/bin/env python3
"""E4-003: Test annotation preflight runner.

Runs test_input_preflight() and reports results.

Usage:
  PYTHONPATH=. python scripts/run_test_annotation_preflight.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TEST_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "annotations" / "test"
_PREFLIGHT_REPORT_PATH = _TEST_DIR / "test_annotation_preflight.json"


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    from experiments.trustparadox_u.empirical_annotation import test_input_preflight

    print("=" * 60)
    print("E4-003 TEST ANNOTATION PREFLIGHT")
    print("=" * 60)

    report = test_input_preflight()

    # Print summary
    print(f"rows: {report['row_queue_count']} / 450")
    print(f"targets resolved: {report['resolved_target_count']} / {report['test_candidate_count']}")
    print(f"sequence units: {report['sequence_queue_count']} / 72")
    print(f"structural families: {report['structural_family_count']} / 24")
    print(f"trust variants per family: {report['trust_variants_per_family']} / 3")
    print(f"protocol hashes: {'PASS' if report['protocol_hash_match'] else 'FAIL'}")
    print(f"validation closure: {'PASS' if report['validation_gate_go'] else 'FAIL'}")

    if report["findings"]:
        print(f"\nFindings ({len(report['findings'])}):")
        for f in report["findings"]:
            print(f"  - {f}")

    status = "PASS" if report["passed"] else "FAIL"
    print(f"\nTEST PREFLIGHT: {status}")

    # Write report
    report["preflight_timestamp"] = datetime.now(timezone.utc).isoformat()
    _write_json(_PREFLIGHT_REPORT_PATH, report)
    print(f"Wrote preflight report to {_PREFLIGHT_REPORT_PATH.name}")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
