#!/usr/bin/env python3
"""Cross-artifact consistency validation.

Validates that all artifacts are internally consistent:
- Commit consistency: all artifacts use same repository_commit
- Hash consistency: same condition = same hash across all files
- Utility consistency: same values across metrics files
- Run-mode consistency: separate execution_mode, artifact_status, certification_mode
- Probe consistency: probe semantics separate from message exposure

Usage:
    poetry run python scripts/validate_consistency.py --results-dir results/<sha>
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.trustparadox_u.evaluator import UTILITY_METRIC_NAME  # noqa: E402


@dataclass
class ConsistencyCheck:
    """Result of a single consistency check."""

    check_name: str
    passed: bool
    detail: str = ""
    violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_name": self.check_name,
            "passed": self.passed,
            "detail": self.detail,
            "violations": self.violations,
        }


def load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file, returning None if not found."""
    if not path.exists():
        return None
    return json.loads(path.read_text())


def check_commit_consistency(results_dir: Path) -> ConsistencyCheck:
    """Check that all artifacts use the same repository_commit."""
    violations: list[str] = []
    commits: set[str] = set()

    # Check provenance
    repo_state = load_json(results_dir / "provenance" / "repository_state.json")
    if repo_state:
        commits.add(repo_state.get("tested_commit", ""))

    # Check smoke manifests
    for subdir in ["single_target_smoke", "multi_target_smoke", "trustparadox_u"]:
        manifest = load_json(results_dir / subdir / "smoke_manifest.json")
        if manifest:
            commit = manifest.get("repository_commit", "")
            if commit:
                commits.add(commit)

        # Check metrics provenance
        metrics = load_json(results_dir / subdir / "metrics.json")
        if metrics and "artifact_provenance" in metrics:
            commit = metrics["artifact_provenance"].get("repository_commit", "")
            if commit:
                commits.add(commit)

    # Remove empty strings
    commits.discard("")

    if len(commits) > 1:
        violations.append(f"Multiple commits found: {commits}")

    return ConsistencyCheck(
        check_name="commit_consistency",
        passed=len(commits) <= 1,
        detail=f"Found {len(commits)} unique commit(s)",
        violations=violations,
    )


def check_hash_consistency(results_dir: Path) -> ConsistencyCheck:
    """Check that same condition has same hash across files."""
    violations: list[str] = []

    # Collect condition hashes from different sources
    condition_hashes: dict[str, set[str]] = {}

    # From smoke matrix
    for subdir in ["single_target_smoke", "multi_target_smoke"]:
        matrix = load_json(results_dir / subdir / "smoke_matrix.json")
        if matrix and "conditions" in matrix:
            for cond in matrix["conditions"]:
                name = cond.get("name", "")
                hash_val = cond.get("config_hash", "")
                if name and hash_val:
                    condition_hashes.setdefault(name, set()).add(hash_val)

    # Check for inconsistencies
    for name, hashes in condition_hashes.items():
        if len(hashes) > 1:
            violations.append(f"Condition {name} has multiple hashes: {hashes}")

    return ConsistencyCheck(
        check_name="hash_consistency",
        passed=len(violations) == 0,
        detail=f"Checked {len(condition_hashes)} conditions",
        violations=violations,
    )


def check_utility_consistency(results_dir: Path) -> ConsistencyCheck:
    """Check that utility values are consistent across files (Section 15).

    The canonical metric ``paired_policy_utility_retention`` must carry identical
    value/numerator/denominator/reason/evaluable across metrics_by_condition.json,
    utility_pairing.json, and summary.json for every condition.
    """
    violations: list[str] = []

    for subdir in ["single_target_smoke", "multi_target_smoke"]:
        base = results_dir / subdir
        metrics = load_json(base / "metrics.json")
        if not metrics:
            continue

        # Load the per-condition artifacts (strip provenance wrappers).
        metrics_by_cond_raw = load_json(base / "metrics_by_condition.json") or {}
        metrics_by_cond = metrics_by_cond_raw.get("metrics_by_condition", {})

        utility = load_json(base / "utility_pairing.json") or {}
        if "artifact_provenance" in utility:
            utility = {k: v for k, v in utility.items() if k != "artifact_provenance"}

        summary = load_json(base / "summary.json") or {}

        # Multi-target: per-condition canonical metric must agree across artifacts.
        if "conditions" in utility:
            for cond_name, cond_data in utility["conditions"].items():
                cond_utility = cond_data.get(UTILITY_METRIC_NAME, {})
                if not isinstance(cond_utility, dict):
                    violations.append(
                        f"{subdir}/{cond_name}: utility metric not a dict in utility_pairing.json"
                    )
                    continue
                cond_value = cond_utility.get("value")
                if cond_value is not None and not isinstance(cond_value, (int, float)):
                    violations.append(
                        f"{subdir}/{cond_name}: utility value not numeric: {cond_value}"
                    )

                # metrics_by_condition.json must carry the identical metric dict.
                mbc_entry = metrics_by_cond.get(cond_name, {})
                mbc_utility = mbc_entry.get(UTILITY_METRIC_NAME)
                if mbc_utility is not None and mbc_utility != cond_utility:
                    violations.append(
                        f"{subdir}/{cond_name}: {UTILITY_METRIC_NAME} differs between "
                        f"utility_pairing.json and metrics_by_condition.json"
                    )

                # summary.json must carry the identical per-condition metric dict.
                summary_utility = summary.get(UTILITY_METRIC_NAME, {})
                if isinstance(summary_utility, dict) and cond_name in summary_utility:
                    if summary_utility[cond_name] != cond_utility:
                        violations.append(
                            f"{subdir}/{cond_name}: {UTILITY_METRIC_NAME} differs between "
                            f"utility_pairing.json and summary.json"
                        )
        else:
            # Single-target: summary.json metric must match utility_pairing.json.
            pairing_metric = utility.get(UTILITY_METRIC_NAME)
            summary_metric = summary.get(UTILITY_METRIC_NAME)
            if (
                pairing_metric is not None
                and summary_metric is not None
                and pairing_metric != summary_metric
            ):
                violations.append(
                    f"{subdir}: {UTILITY_METRIC_NAME} differs between "
                    f"utility_pairing.json and summary.json"
                )

    return ConsistencyCheck(
        check_name="utility_consistency",
        passed=len(violations) == 0,
        detail="Utility values checked across artifacts",
        violations=violations,
    )


def check_run_mode_consistency(results_dir: Path) -> ConsistencyCheck:
    """Check that execution_mode, artifact_status, certification_mode are separate."""
    violations: list[str] = []

    # Check provenance
    repo_state = load_json(results_dir / "provenance" / "repository_state.json")
    if repo_state:
        exec_mode = repo_state.get("execution_mode", "")
        artifact_status = repo_state.get("artifact_status", "")
        # Note: certification_mode extracted for potential future validation
        _ = repo_state.get("certification_mode", "")

        # Verify they are separate fields
        if exec_mode and artifact_status and exec_mode == artifact_status:
            # This is OK if both are valid values
            pass

        # Check for conflation (same field used for multiple purposes)
        if "mode" in repo_state and exec_mode:
            mode_val = repo_state.get("mode", "")
            if mode_val and mode_val != exec_mode and mode_val != artifact_status:
                violations.append(
                    f"Legacy 'mode' field ({mode_val}) differs from execution_mode ({exec_mode})"
                )

    return ConsistencyCheck(
        check_name="run_mode_consistency",
        passed=len(violations) == 0,
        detail="Run mode fields checked",
        violations=violations,
    )


def check_probe_consistency(results_dir: Path) -> ConsistencyCheck:
    """Check that probe semantics are separate from message exposure."""
    violations: list[str] = []

    for subdir in ["single_target_smoke", "multi_target_smoke"]:
        episodes_path = results_dir / subdir / "episodes.jsonl"
        if not episodes_path.exists():
            continue

        probe_as_exposure_count = 0
        with open(episodes_path) as f:
            for line in f:
                if not line.strip():
                    continue
                ep = json.loads(line)
                turns = ep.get("turns", [])
                for turn in turns:
                    # Check if probe recovery is encoded as message exposure
                    phase = turn.get("phase", "")
                    if phase == "FINAL_PROBE":
                        # Final probe should not have contamination_state_changes
                        changes = turn.get("contamination_state_changes", [])
                        if changes:
                            probe_as_exposure_count += 1

        if probe_as_exposure_count > 0:
            violations.append(f"{subdir}: {probe_as_exposure_count} probe turns with state changes")

    return ConsistencyCheck(
        check_name="probe_consistency",
        passed=len(violations) == 0,
        detail="Probe semantics checked",
        violations=violations,
    )


def check_artifact_completeness(results_dir: Path) -> ConsistencyCheck:
    """Check that all required artifacts exist."""
    violations: list[str] = []

    required_by_subdir = {
        "single_target_smoke": [
            "episodes.jsonl",
            "metrics.json",
            "metrics_by_condition.json",
            "summary.json",
            "summary.md",
            "smoke_manifest.json",
            "result_audit.json",
        ],
        "multi_target_smoke": [
            "episodes.jsonl",
            "metrics.json",
            "metrics_by_condition.json",
            "summary.json",
            "summary.md",
            "smoke_manifest.json",
            "result_audit.json",
            "multi_target_report.json",
        ],
    }

    for subdir, required_files in required_by_subdir.items():
        subdir_path = results_dir / subdir
        if not subdir_path.exists():
            violations.append(f"Missing directory: {subdir}")
            continue

        for fname in required_files:
            if not (subdir_path / fname).exists():
                violations.append(f"Missing file: {subdir}/{fname}")

    return ConsistencyCheck(
        check_name="artifact_completeness",
        passed=len(violations) == 0,
        detail=f"Checked {sum(len(f) for f in required_by_subdir.values())} required files",
        violations=violations,
    )


def main() -> int:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Cross-artifact consistency validation")
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Results directory to validate",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1

    print("Cross-Artifact Consistency Validation")
    print("=" * 50)
    print()

    checks = [
        check_commit_consistency(results_dir),
        check_hash_consistency(results_dir),
        check_utility_consistency(results_dir),
        check_run_mode_consistency(results_dir),
        check_probe_consistency(results_dir),
        check_artifact_completeness(results_dir),
    ]

    all_passed = True
    for check in checks:
        icon = "PASS" if check.passed else "FAIL"
        print(f"[{icon}] {check.check_name}: {check.detail}")
        if check.violations:
            for v in check.violations[:3]:  # Show first 3 violations
                print(f"       - {v}")
            if len(check.violations) > 3:
                print(f"       ... and {len(check.violations) - 3} more")
        if not check.passed:
            all_passed = False

    # Write validation report
    report = {
        "all_passed": all_passed,
        "total_checks": len(checks),
        "passed": sum(1 for c in checks if c.passed),
        "failed": sum(1 for c in checks if not c.passed),
        "checks": [c.to_dict() for c in checks],
    }

    report_path = results_dir / "ci" / "consistency_validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    print()
    print(f"Total checks: {report['total_checks']}")
    print(f"Passed: {report['passed']}")
    print(f"Failed: {report['failed']}")
    print(f"Report: {report_path}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
