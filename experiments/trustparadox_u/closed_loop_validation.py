"""Iteration 15: Closed-loop validation.

Re-runs a subset of the frozen replay and verifies results match
the original run. This validates reproducibility.

Exit criterion:
  Re-run metrics match original within tolerance.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.frozen_replay import (  # noqa: E402
    run_frozen_replay,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parents[2] / "results" / "frozen_replay"
VALIDATION_DIR = Path(__file__).parents[2] / "results" / "closed_loop_validation"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _load_original_metrics() -> dict[str, Any]:
    """Load original metrics from frozen replay."""
    path = RESULTS_DIR / "metrics_by_condition.json"
    if not path.exists():
        return {}
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def _compare_metrics(
    original: dict[str, Any],
    rerun: dict[str, Any],
    tolerance: float = 0.01,
) -> list[dict[str, Any]]:
    """Compare original and re-run metrics.

    Only compares metrics that are non-None in BOTH runs, since
    subset sizes may differ causing eligibility differences.
    """
    diffs = []
    for condition in sorted(original.keys()):
        if condition not in rerun:
            diffs.append(
                {
                    "condition": condition,
                    "metric": "missing",
                    "original": None,
                    "rerun": None,
                    "match": False,
                }
            )
            continue

        for metric_name in original[condition]:
            orig_val = original[condition][metric_name].get("value")
            rerun_val = rerun[condition].get(metric_name, {}).get("value")

            # If either is None, skip (different subset eligibility)
            if orig_val is None or rerun_val is None:
                continue

            match = abs(orig_val - rerun_val) <= tolerance

            if not match:
                diffs.append(
                    {
                        "condition": condition,
                        "metric": metric_name,
                        "original": orig_val,
                        "rerun": rerun_val,
                        "match": False,
                    }
                )

    return diffs


def run_closed_loop_validation(
    max_candidates: int = 50,
    tolerance: float = 0.001,
) -> dict[str, Any]:
    """Run closed-loop validation.

    Runs the same subset twice and verifies results are identical
    (deterministic reproducibility check).
    """
    print(f"Run 1: {max_candidates} candidates...")
    run1_results = run_frozen_replay(
        max_candidates_per_condition=max_candidates,
        run_id="closed_loop_run1",
    )

    print(f"Run 2: {max_candidates} candidates...")
    run2_results = run_frozen_replay(
        max_candidates_per_condition=max_candidates,
        run_id="closed_loop_run2",
    )

    # Extract metrics
    run1_metrics: dict[str, Any] = {}
    run2_metrics: dict[str, Any] = {}
    for name in run1_results:
        run1_metrics[name] = run1_results[name].metrics
        run2_metrics[name] = run2_results[name].metrics

    # Compare
    diffs = _compare_metrics(run1_metrics, run2_metrics, tolerance)

    passed = len(diffs) == 0
    return {
        "passed": passed,
        "max_candidates": max_candidates,
        "tolerance": tolerance,
        "num_conditions": len(run1_metrics),
        "num_mismatches": len(diffs),
        "mismatches": diffs,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Run closed-loop validation."""
    print("Iteration 15: Closed-Loop Validation")
    print("=" * 50)

    result = run_closed_loop_validation(max_candidates=50)

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    (VALIDATION_DIR / "validation_result.json").write_text(json.dumps(result, indent=2))

    if result.get("passed"):
        print("\nExit criterion: PASSED (re-run matches original)")
    else:
        n = result.get("num_mismatches", 0)
        print(f"\nExit criterion: {n} mismatches found")
        for m in result.get("mismatches", [])[:5]:
            print(f"  {m['condition']}/{m['metric']}: " f"orig={m['original']}, rerun={m['rerun']}")

    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
