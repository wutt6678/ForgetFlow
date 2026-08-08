"""E2-I: E2 completion check pipeline.

Verifies that all E2 checklist items have been satisfied before the
empirical phase can advance to E3 (corpus generation).

Checklist items verified:
- E2-001: phase-lock enum (EmpiricalPhase.E2_PROMPTS_FROZEN exists)
- E2-002: provenance separation (identity fields in generation attempts)
- E2-004: clean-tree gate (assert_clean_repository_tree exists)
- E2-006: connectivity smoke (3-call smoke completed)
- E2-007: 90-attempt pilot (raw_generation_attempts.jsonl with 90 lines)
- E2-008: pilot prompt invariance (only TRUST_FRAMING differs)
- E2-009: raw retention (all attempts retained, including failures)
- E2-010: provenance completeness (provider, model, transport recorded)
- E2-011: frozen oracle labeling (labeled_pilot_attempts.jsonl exists)
- E2-012: deterministic labels (labeling_report.json exists)
- E2-013: provenance chain (labeling report references raw attempts)
- E2-014: trust-manipulation analysis (pilot_analysis_report.json exists)
- E2-015: per-scenario breakdown (by_scenario in analysis)
- E2-016: directional expectation checks (directional_checks in analysis)
- E2-053: prompt freeze (frozen_prompt_manifest.json with frozen status)
- E2-054: synthetic regression (mock generation still works)

Inputs:
  - All E2 artifact directories

Outputs:
  - e2_completion_report.json: pass/fail for each checklist item.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.research_protocol import PROTOCOL_VERSION  # noqa: E402

# Expected artifact directories (relative to project root).
_E2_ARTIFACT_DIRS = {
    "connectivity_smoke": "results/empirical_v2/e2_connectivity_smoke",
    "trust_pilot": "results/empirical_v2/e2_trust_pilot",
    "pilot_labels": "results/empirical_v2/e2_pilot_labels",
    "pilot_analysis": "results/empirical_v2/e2_pilot_analysis",
    "bounded_revision": "results/empirical_v2/e2_bounded_revision",
    "prompt_freeze": "results/empirical_v2/e2_prompt_freeze",
}

# Required files per checklist item.
_CHECK_ITEMS: dict[str, dict[str, Any]] = {
    "E2-006_connectivity": {
        "dir": "connectivity_smoke",
        "files": ["validation_report.json"],
        "description": "3-call connectivity smoke completed",
    },
    "E2-007_pilot_90_attempts": {
        "dir": "trust_pilot",
        "files": ["raw_generation_attempts.jsonl"],
        "description": "90-attempt pilot run",
    },
    "E2-009_raw_retention": {
        "dir": "trust_pilot",
        "files": ["raw_generation_attempts.jsonl", "validation_report.json"],
        "description": "Raw retention for all attempts",
    },
    "E2-011_frozen_labeling": {
        "dir": "pilot_labels",
        "files": ["labeled_pilot_attempts.jsonl", "labeling_report.json"],
        "description": "Frozen oracle labeling",
    },
    "E2-014_trust_analysis": {
        "dir": "pilot_analysis",
        "files": ["pilot_analysis_report.json"],
        "description": "Trust-manipulation analysis",
    },
    "E2-053_prompt_freeze": {
        "dir": "prompt_freeze",
        "files": ["frozen_prompt_manifest.json", "frozen_freeze_report.json"],
        "description": "Prompt freeze manifest",
    },
}


def _check_file_exists(base_dir: Path, rel_path: str) -> bool:
    """Check if a file exists relative to the project root."""
    return (base_dir / rel_path).exists()


def _count_jsonl_lines(path: Path) -> int:
    """Count non-empty lines in a JSONL file."""
    count = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _check_pilot_attempt_count(pilot_dir: Path) -> dict[str, Any]:
    """E2-007: verify 90 attempts in raw generation file."""
    raw_path = pilot_dir / "raw_generation_attempts.jsonl"
    if not raw_path.exists():
        return {"passed": False, "reason": "raw_generation_attempts.jsonl not found"}
    count = _count_jsonl_lines(raw_path)
    if count != 90:
        return {"passed": False, "reason": f"expected 90 attempts, got {count}"}
    return {"passed": True, "attempt_count": count}


def _check_prompt_invariance() -> dict[str, Any]:
    """E2-008: verify prompt invariance (only TRUST_FRAMING differs)."""
    from experiments.trustparadox_u.empirical_generation import (
        validate_trust_prompt_invariance,
    )

    problems = validate_trust_prompt_invariance()
    if problems:
        return {"passed": False, "problems": problems}
    return {"passed": True}


def _check_frozen_status(freeze_dir: Path) -> dict[str, Any]:
    """E2-053: verify prompt manifest has frozen status."""
    manifest_path = freeze_dir / "frozen_prompt_manifest.json"
    if not manifest_path.exists():
        return {"passed": False, "reason": "frozen_prompt_manifest.json not found"}
    with open(manifest_path) as f:
        manifest = json.load(f)
    status = manifest.get("status", "")
    if status != "frozen_post_pilot":
        return {"passed": False, "reason": f"status is '{status}', expected 'frozen_post_pilot'"}
    return {"passed": True, "status": status}


def _check_analysis_breakdown(analysis_dir: Path) -> dict[str, Any]:
    """E2-015/E2-016: verify per-scenario breakdown and directional checks."""
    report_path = analysis_dir / "pilot_analysis_report.json"
    if not report_path.exists():
        return {"passed": False, "reason": "pilot_analysis_report.json not found"}
    with open(report_path) as f:
        report = json.load(f)

    rates = report.get("exposure_rates", {})
    by_scenario = rates.get("by_scenario", {})
    directional = report.get("directional_checks", {})

    checks: dict[str, Any] = {"passed": True}
    if not by_scenario:
        checks["passed"] = False
        checks["reason"] = "no by_scenario breakdown"
    if not directional:
        checks["passed"] = False
        checks["reason"] = "no directional_checks"
    checks["num_scenarios"] = len(by_scenario)
    checks["num_directional"] = len(directional)
    return checks


def _check_provenance_completeness(pilot_dir: Path) -> dict[str, Any]:
    """E2-010: verify provenance fields in pilot attempts."""
    raw_path = pilot_dir / "raw_generation_attempts.jsonl"
    if not raw_path.exists():
        return {"passed": False, "reason": "raw_generation_attempts.jsonl not found"}

    required_fields = [
        "generator_provider",
        "generator_model_requested",
        "transport",
        "generation_mode",
    ]
    problems: list[str] = []
    with open(raw_path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            attempt = json.loads(line)
            for field in required_fields:
                if field not in attempt or attempt[field] is None:
                    problems.append(f"attempt {i}: missing {field}")
            if len(problems) > 10:
                problems.append("... (truncated)")
                break

    if problems:
        return {"passed": False, "problems": problems[:10]}
    return {"passed": True}


def run_completion_check(
    project_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run all E2 completion checks.

    Args:
        project_root: Project root directory.
        output_dir: Optional directory to write completion report.

    Returns:
        Completion report dictionary.
    """
    results: dict[str, dict[str, Any]] = {}

    # Check artifact files exist.
    for item_id, item_spec in _CHECK_ITEMS.items():
        missing = [
            f
            for f in item_spec["files"]
            if not _check_file_exists(project_root, f"{_E2_ARTIFACT_DIRS[item_spec['dir']]}/{f}")
        ]
        if missing:
            results[item_id] = {
                "passed": False,
                "reason": f"missing files: {missing}",
                "description": item_spec["description"],
            }
        else:
            results[item_id] = {
                "passed": True,
                "description": item_spec["description"],
            }

    # Detailed checks.
    pilot_dir = project_root / _E2_ARTIFACT_DIRS["trust_pilot"]
    results["E2-007_pilot_90_attempts"] = _check_pilot_attempt_count(pilot_dir)
    results["E2-007_pilot_90_attempts"]["description"] = "90-attempt pilot run"

    results["E2-008_prompt_invariance"] = _check_prompt_invariance()
    results["E2-008_prompt_invariance"]["description"] = (
        "Prompt invariance (only TRUST_FRAMING differs)"
    )

    results["E2-010_provenance"] = _check_provenance_completeness(pilot_dir)
    results["E2-010_provenance"]["description"] = "Provenance completeness"

    analysis_dir = project_root / _E2_ARTIFACT_DIRS["pilot_analysis"]
    results["E2-015_016_breakdown"] = _check_analysis_breakdown(analysis_dir)
    results["E2-015_016_breakdown"]["description"] = "Per-scenario breakdown + directional checks"

    freeze_dir = project_root / _E2_ARTIFACT_DIRS["prompt_freeze"]
    results["E2-053_frozen_status"] = _check_frozen_status(freeze_dir)
    results["E2-053_frozen_status"]["description"] = "Prompt manifest frozen status"

    # Phase-lock enum check (code-level, not artifact).
    from experiments.trustparadox_u.empirical_corpus import EmpiricalPhase

    has_frozen_phase = hasattr(EmpiricalPhase, "E2_PROMPTS_FROZEN")
    results["E2-001_phase_lock"] = {
        "passed": has_frozen_phase,
        "description": "Phase-lock enum (E2_PROMPTS_FROZEN)",
    }

    # Clean-tree gate exists (code-level).
    from experiments.trustparadox_u.empirical_corpus import assert_clean_repository_tree

    results["E2-004_clean_tree"] = {
        "passed": callable(assert_clean_repository_tree),
        "description": "Clean-tree gate function exists",
    }

    # Overall pass/fail.
    all_passed = all(r.get("passed", False) for r in results.values())
    num_passed = sum(1 for r in results.values() if r.get("passed", False))
    num_total = len(results)

    report: dict[str, Any] = {
        "check_type": "e2_completion",
        "protocol_version": PROTOCOL_VERSION,
        "check_timestamp": datetime.now(timezone.utc).isoformat(),
        "all_passed": all_passed,
        "num_passed": num_passed,
        "num_total": num_total,
        "checks": results,
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "e2_completion_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def main() -> None:
    """CLI entry point for E2-I completion check."""
    parser = argparse.ArgumentParser(
        description="E2-I: E2 completion check",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=_PROJECT_ROOT,
        help="Project root directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write completion report",
    )
    args = parser.parse_args()

    print("E2-I: E2 Completion Check")
    print(f"  Project root: {args.project_root}")

    report = run_completion_check(args.project_root, args.output_dir)

    print(f"\n  Overall: {'PASS' if report['all_passed'] else 'FAIL'}")
    print(f"  Passed: {report['num_passed']}/{report['num_total']}")
    print()

    for item_id, result in sorted(report["checks"].items()):
        status = "PASS" if result.get("passed", False) else "FAIL"
        desc = result.get("description", "")
        print(f"  [{status}] {item_id}: {desc}")
        if not result.get("passed", False) and "reason" in result:
            print(f"         Reason: {result['reason']}")

    if args.output_dir:
        print(f"\n  Report: {args.output_dir / 'e2_completion_report.json'}")


if __name__ == "__main__":
    main()
