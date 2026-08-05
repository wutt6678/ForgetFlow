#!/usr/bin/env python3
"""Assemble complete_results_manifest.json from CI job outcomes (Section 17/18).

The certification step (scripts/generate_certification.py) derives the final
status from a complete_results_manifest.json describing the outcome of each
pipeline phase. In CI the phases run as separate jobs, so this helper rebuilds
an equivalent manifest from the observed job/step outcomes.

Usage:
    poetry run python scripts/write_ci_manifest.py \
        --output results/complete_results_manifest.json \
        --tested-commit "$GITHUB_SHA" \
        --phase static_checks=PASS \
        --phase test_suite=PASS \
        --phase single_target_smoke=PASS \
        --phase multi_target_smoke=PASS \
        --phase independent_verification=PASS \
        --phase consistency_validation=PASS

The assertion suite phase is retired (SC-010, Option B) and no longer needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_PASS_LIKE = ("PASS", "SKIP")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Write complete_results_manifest.json")
    parser.add_argument("--output", type=Path, required=True, help="Manifest output path")
    parser.add_argument("--tested-commit", required=True, help="Tested commit SHA")
    parser.add_argument(
        "--phase",
        action="append",
        default=[],
        metavar="NAME=STATUS",
        help="Phase outcome (repeatable), e.g. --phase test_suite=PASS",
    )
    args = parser.parse_args()

    phases: list[dict[str, object]] = []
    failure_reasons: list[str] = []
    for spec in args.phase:
        name, sep, status = spec.partition("=")
        if not sep or not name:
            print(f"Error: invalid --phase value (expected NAME=STATUS): {spec!r}")
            return 2
        status = status.strip().upper()
        phases.append(
            {
                "phase_name": name.strip(),
                "status": status,
                "start_time": "",
                "end_time": "",
                "errors": [],
                "commands": [],
            }
        )
        if status not in _PASS_LIKE:
            failure_reasons.append(f"{name.strip()} did not pass ({status})")

    manifest = {
        "schema_version": "1.0.0",
        "tested_commit": args.tested_commit,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "ci",
        "phases": phases,
        "failure_reasons": failure_reasons,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2))

    print(f"CI manifest written to {args.output} ({len(phases)} phases)")
    for reason in failure_reasons:
        print(f"  - {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
