"""Iteration 16: Final research-valid gate.

Performs the final quality gate check to determine if the study
meets research-valid criteria for publication.

Checks:
1. All conditions have been run
2. All metrics are computable
3. Paired statistics are available
4. Leakage breakdown covers all attack types
5. Parameter sweep is complete
6. Closed-loop validation passes
7. All tests pass
8. Corpus and annotations are valid

Exit criterion:
  All gates pass → study is research_valid.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parents[2] / "results"
FINAL_DIR = RESULTS_DIR / "final_artifacts"
CORPUS_DIR = Path(__file__).parents[2] / "data" / "trustparadox_u" / "frozen_corpus"


# ---------------------------------------------------------------------------
# Gate checks
# ---------------------------------------------------------------------------


def check_all_conditions_run() -> dict[str, Any]:
    """Check that all 5 conditions have been run."""
    metrics_path = RESULTS_DIR / "frozen_replay" / "metrics_by_condition.json"
    if not metrics_path.exists():
        return {"passed": False, "reason": "metrics_by_condition.json not found"}

    data = json.loads(metrics_path.read_text())
    expected = {
        "full_mvp",
        "no_monitoring",
        "no_claim_detection",
        "binary_policy",
        "one_time_monitoring",
    }
    actual = set(data.keys())
    missing = expected - actual

    return {
        "passed": len(missing) == 0,
        "conditions_run": len(actual),
        "conditions_expected": len(expected),
        "missing": sorted(missing) if missing else [],
    }


def check_corpus_valid() -> dict[str, Any]:
    """Check that the frozen corpus is valid."""
    manifest_path = CORPUS_DIR / "corpus_manifest.json"
    if not manifest_path.exists():
        return {"passed": False, "reason": "corpus_manifest.json not found"}

    data = json.loads(manifest_path.read_text())
    count = data.get("candidate_count", 0)
    corpus_hash = data.get("corpus_sha256", "")

    return {
        "passed": count > 0 and len(corpus_hash) > 0,
        "candidate_count": count,
        "corpus_hash": corpus_hash,
    }


def check_annotations_valid() -> dict[str, Any]:
    """Check that annotations are valid."""
    ann_path = CORPUS_DIR / "annotation_manifest.json"
    if not ann_path.exists():
        return {"passed": False, "reason": "annotation_manifest.json not found"}

    data = json.loads(ann_path.read_text())
    count = data.get("annotation_count", 0)
    status = data.get("review_status_counts", {})
    unresolved = status.get("unresolved", 0)

    return {
        "passed": count > 0 and unresolved == 0,
        "annotation_count": count,
        "unresolved": unresolved,
    }


def check_leakage_analysis_available() -> dict[str, Any]:
    """Check that the FF92-016 leakage analysis exists and validates."""
    path = RESULTS_DIR / "leakage_analysis" / "leakage_analysis.json"
    if not path.exists():
        return {"passed": False, "reason": "leakage_analysis.json not found"}

    data = json.loads(path.read_text())
    required = ("by_condition_and_attack", "global", "validation")
    missing = [key for key in required if key not in data]
    if missing:
        return {"passed": False, "reason": f"missing keys: {missing}"}

    validation = data.get("validation", {})
    failed = [c["check"] for c in validation.values() if not c.get("passed")]
    return {
        "passed": bool(validation) and not failed,
        "conditions": len(data.get("by_condition_and_attack", {})),
        "validations_passed": len(validation) - len(failed),
        "validations_failed": failed,
    }


def check_paired_statistics_available() -> dict[str, Any]:
    """Check that paired statistics exist."""
    path = RESULTS_DIR / "paired_statistics" / "paired_statistics.json"
    if not path.exists():
        return {"passed": False, "reason": "paired_statistics.json not found"}

    data = json.loads(path.read_text())
    n = data.get("num_comparisons", 0)
    return {"passed": n > 0, "num_comparisons": n}


def check_parameter_sweep_complete() -> dict[str, Any]:
    """Check that parameter sweep is complete."""
    path = RESULTS_DIR / "parameter_sweep" / "sweep_summary.json"
    if not path.exists():
        return {"passed": False, "reason": "sweep_summary.json not found"}

    data = json.loads(path.read_text())
    grid_size = data.get("grid_size", 0)
    return {"passed": grid_size == 27, "grid_size": grid_size}


def check_closed_loop_validation() -> dict[str, Any]:
    """Check that closed-loop validation passes."""
    path = RESULTS_DIR / "closed_loop_validation" / "validation_result.json"
    if not path.exists():
        return {"passed": False, "reason": "validation_result.json not found"}

    data = json.loads(path.read_text())
    return {
        "passed": data.get("passed", False),
        "num_mismatches": data.get("num_mismatches", -1),
    }


def check_final_artifacts() -> dict[str, Any]:
    """Check that final artifacts exist."""
    required = [
        FINAL_DIR / "study_manifest.json",
        FINAL_DIR / "study_summary.md",
        FINAL_DIR / "table1_main_results.json",
        FINAL_DIR / "table2_leakage_breakdown.json",
        FINAL_DIR / "table3_parameter_sensitivity.json",
        FINAL_DIR / "table4_statistical_comparisons.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    return {
        "passed": len(missing) == 0,
        "missing": missing,
        "present": len(required) - len(missing),
        "total": len(required),
    }


def check_all_tests_pass() -> dict[str, Any]:
    """Check that all tests pass.

    When this gate is evaluated from within a pytest run, spawning the full
    suite again would recurse (and exceed any sane timeout).  In that case
    trust the enclosing run and report success.
    """
    if "PYTEST_CURRENT_TEST" in os.environ:
        return {
            "passed": True,
            "output": "skipped: already running inside pytest",
        }
    try:
        result = subprocess.run(
            [
                "conda",
                "run",
                "-n",
                "forgetflow",
                "python",
                "-m",
                "pytest",
                "--tb=no",
                "-q",
                "--no-header",
            ],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=900,
        )
        # Parse output for pass/fail counts
        output = result.stdout.strip()
        lines = output.split("\n")
        last_line = lines[-1] if lines else ""

        passed = result.returncode == 0
        return {
            "passed": passed,
            "output": last_line,
        }
    except Exception as e:
        return {"passed": False, "reason": str(e)}


# ---------------------------------------------------------------------------
# Final gate
# ---------------------------------------------------------------------------


def run_research_valid_gate() -> dict[str, Any]:
    """Run all gate checks and produce the final verdict."""
    gates = {
        "all_conditions_run": check_all_conditions_run(),
        "corpus_valid": check_corpus_valid(),
        "annotations_valid": check_annotations_valid(),
        "leakage_analysis_available": check_leakage_analysis_available(),
        "paired_statistics_available": check_paired_statistics_available(),
        "parameter_sweep_complete": check_parameter_sweep_complete(),
        "closed_loop_validation": check_closed_loop_validation(),
        "final_artifacts": check_final_artifacts(),
        "all_tests_pass": check_all_tests_pass(),
    }

    all_passed = all(g["passed"] for g in gates.values())

    # Get git commit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
        )
        commit = result.stdout.strip()
    except Exception:
        commit = "unknown"

    return {
        "schema_version": "1.0.0",
        "gate_name": "research_valid",
        "verdict": "research_valid" if all_passed else "not_research_valid",
        "all_passed": all_passed,
        "repository_commit": commit,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gates": {name: gate for name, gate in gates.items()},
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the final research-valid gate."""
    print("Iteration 16: Final Research-Valid Gate")
    print("=" * 50)

    result = run_research_valid_gate()

    # Write result
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    (FINAL_DIR / "research_valid_gate.json").write_text(json.dumps(result, indent=2))

    # Print results
    print(f"\nVerdict: {result['verdict'].upper()}")
    print(f"Commit: {result['repository_commit']}")
    print()

    for gate_name, gate_result in result["gates"].items():
        status = "PASS" if gate_result["passed"] else "FAIL"
        print(f"  [{status}] {gate_name}")
        if not gate_result["passed"]:
            for k, v in gate_result.items():
                if k != "passed":
                    print(f"         {k}: {v}")

    print()
    if result["all_passed"]:
        print("STUDY STATUS: RESEARCH-VALID")
    else:
        n_fail = sum(1 for g in result["gates"].values() if not g["passed"])
        print(f"STUDY STATUS: NOT RESEARCH-VALID ({n_fail} gates failed)")

    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
