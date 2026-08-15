#!/usr/bin/env python3
"""E4-001 Sec 51-52: Annotation input preflight.

Runs ordered preflight checks before any provider call:
1. Frozen corpus verifier
2. Target-resolution preflight
3. Queue completeness preflight
4. Provenance preflight

Creates:
  results/empirical_v2/annotations/preflight/annotation_input_preflight_report.json

Usage:
    PYTHONPATH=. python scripts/run_annotation_input_preflight.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.trustparadox_u.empirical_annotation import (  # noqa: E402
    annotation_input_preflight,
)


def main() -> int:
    """Run annotation input preflight and write report."""
    output_dir = PROJECT_ROOT / "results" / "empirical_v2" / "annotations" / "preflight"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ANNOTATION INPUT PREFLIGHT (Sec 51-52)")
    print("=" * 60)

    try:
        report = annotation_input_preflight()
    except Exception as exc:
        print(f"Preflight failed with error: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        # Write a failure report
        report = {
            "frozen_corpus_manifest_sha256": "",
            "development_candidate_count": 0,
            "resolved_target_count": 0,
            "row_queue_count": 0,
            "sequence_queue_count": 0,
            "target_resolution_failures": [],
            "sequence_structure_failures": [],
            "empty_target_count": 0,
            "queue_hash": "",
            "passed": False,
            "findings": [f"preflight_exception: {exc}"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # Add timestamp
    report["timestamp"] = datetime.now(timezone.utc).isoformat()

    # Write report
    report_path = output_dir / "annotation_input_preflight_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    # Print summary
    print(f"\nFrozen corpus manifest SHA256: {report['frozen_corpus_manifest_sha256'][:32]}...")
    print(f"Development candidates: {report['development_candidate_count']}")
    print(f"Resolved targets: {report['resolved_target_count']}")
    print(f"Row queue: {report['row_queue_count']}")
    print(f"Sequence queue: {report['sequence_queue_count']}")
    print(f"Empty targets: {report['empty_target_count']}")
    print(f"Target failures: {len(report['target_resolution_failures'])}")
    print(f"Sequence structure failures: {len(report['sequence_structure_failures'])}")
    print(f"Queue hash: {report['queue_hash'][:32]}...")
    print(f"\nBlocking findings: {len(report['findings'])}")
    for finding in report["findings"]:
        print(f"  - {finding}")
    print(f"\nPreflight PASSED: {report['passed']}")
    print(f"Report written: {report_path.relative_to(PROJECT_ROOT)}")
    print("=" * 60)

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
