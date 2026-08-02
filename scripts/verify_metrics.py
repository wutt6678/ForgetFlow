#!/usr/bin/env python3
"""Independent metric verification.

Reads only persisted artifacts and recomputes all metrics independently,
then compares against persisted aggregates with documented tolerance.

The recomputation deserializes the persisted ``episodes.jsonl`` and applies the
canonical evaluator functions, then compares the result against the persisted
``metrics.json`` aggregate.  Any disagreement indicates the persisted aggregate
is inconsistent with the persisted raw episodes.

Usage:
    poetry run python scripts/verify_metrics.py --results-dir results/<sha>
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.trustparadox_u.evaluator import (  # noqa: E402
    compute_crr,
    compute_fbr,
    compute_pu_rer,
    compute_rr_clean,
)
from experiments.trustparadox_u.serialization import load_episode_results  # noqa: E402

# Comparison tolerance for floating-point values
TOLERANCE = 1e-9

# Metrics independently recomputed and compared against metrics.json.
# NOTE: the canonical top-level ``rr`` is the clean/verified-population rate
# (s4), i.e. compute_rr_clean — the same function evaluate_all() reports.
_RECOMPUTERS = {
    "pu_rer": compute_pu_rer,
    "crr": compute_crr,
    "rr": compute_rr_clean,
    "fbr": compute_fbr,
}


@dataclass
class VerificationResult:
    """Result of a single metric verification."""

    metric_name: str
    persisted_value: float | None
    recomputed_value: float | None
    persisted_numerator: int
    recomputed_numerator: int
    persisted_denominator: int
    recomputed_denominator: int
    match: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "persisted_value": self.persisted_value,
            "recomputed_value": self.recomputed_value,
            "persisted_numerator": self.persisted_numerator,
            "recomputed_numerator": self.recomputed_numerator,
            "persisted_denominator": self.persisted_denominator,
            "recomputed_denominator": self.recomputed_denominator,
            "match": self.match,
            "detail": self.detail,
        }


def verify_metrics(results_dir: Path, subdir: str) -> list[VerificationResult]:
    """Verify metrics for a specific result subdirectory."""
    results: list[VerificationResult] = []

    # Load persisted metrics aggregate.
    metrics_path = results_dir / subdir / "metrics.json"
    if not metrics_path.exists():
        return results

    persisted = json.loads(metrics_path.read_text())

    # Handle provenance wrapper.
    if "artifact_provenance" in persisted:
        persisted = {k: v for k, v in persisted.items() if k != "artifact_provenance"}

    # Load persisted raw episodes.
    episodes_path = results_dir / subdir / "episodes.jsonl"
    if not episodes_path.exists():
        return results

    try:
        episode_results = load_episode_results(episodes_path)
    except (ValueError, FileNotFoundError) as exc:
        print(f"  [WARN] Could not deserialize {episodes_path}: {exc}")
        return results

    if not episode_results:
        return results

    # Independently recompute each metric from the persisted episodes and
    # compare against the persisted aggregate.
    for name, recompute in _RECOMPUTERS.items():
        recomputed = recompute(episode_results).to_dict()
        persisted_metric = persisted.get(name, {})

        match = persisted_metric.get("numerator", -1) == recomputed.get(
            "numerator"
        ) and persisted_metric.get("denominator", -1) == recomputed.get("denominator")

        persisted_value = persisted_metric.get("value")
        recomputed_value = recomputed.get("value")
        if persisted_value is not None and recomputed_value is not None:
            match = match and abs(persisted_value - recomputed_value) < TOLERANCE

        results.append(
            VerificationResult(
                metric_name=name,
                persisted_value=persisted_value,
                recomputed_value=recomputed_value,
                persisted_numerator=persisted_metric.get("numerator", 0),
                recomputed_numerator=recomputed.get("numerator", 0),
                persisted_denominator=persisted_metric.get("denominator", 0),
                recomputed_denominator=recomputed.get("denominator", 0),
                match=match,
                detail=f"{name}: {recomputed.get('numerator')}/{recomputed.get('denominator')}",
            )
        )

    return results


def main() -> int:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Independent metric verification")
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Results directory containing subdirectories",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1

    print("Independent Metric Verification")
    print("=" * 50)
    print(f"Tolerance: {TOLERANCE}")
    print()

    all_results: list[VerificationResult] = []
    subdirs = ["single_target_smoke", "multi_target_smoke", "trustparadox_u"]

    for subdir in subdirs:
        subdir_path = results_dir / subdir
        if not subdir_path.exists():
            print(f"[SKIP] {subdir}: not found")
            continue

        print(f"Verifying {subdir}...")
        results = verify_metrics(results_dir, subdir)

        if not results:
            print("  [SKIP] No metrics to verify")
            continue

        all_results.extend(results)

        for r in results:
            icon = "PASS" if r.match else "FAIL"
            print(f"  [{icon}] {r.metric_name}: {r.detail}")

    # Write verification report
    report = {
        "tolerance": TOLERANCE,
        "total_checks": len(all_results),
        "passed": sum(1 for r in all_results if r.match),
        "failed": sum(1 for r in all_results if not r.match),
        "results": [r.to_dict() for r in all_results],
    }

    report_path = results_dir / "ci" / "metric_verification.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    print()
    print(f"Total checks: {report['total_checks']}")
    print(f"Passed: {report['passed']}")
    print(f"Failed: {report['failed']}")
    print(f"Report: {report_path}")

    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
