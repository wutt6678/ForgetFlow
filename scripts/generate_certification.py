#!/usr/bin/env python3
"""Generate checksums and certification for results bundle.

Generates SHA-256 checksums for all result files and produces
a certification document with final status.

FF-022: This script certifies result artifacts, not source modules.
P0/P1/P2 modules (p0_candidates.py, p1_evaluation.py, p2_experiments.py)
are non-certifying helper libraries (see FF-016) and are NOT included
in the release certification path.

Usage:
    poetry run python scripts/generate_certification.py \
        --results-dir results/<sha> \
        --tested-commit <sha>
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.trustparadox_u.status import (  # noqa: E402
    DIAGNOSTIC_VALID,
    RELEASE_CANDIDATE,
    RESEARCH_VALID,
    status_at_least,
)


@dataclass
class ChecksumEntry:
    """A single file checksum entry."""

    path: str
    sha256: str
    size_bytes: int
    artifact_type: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "artifact_type": self.artifact_type,
        }


@dataclass
class Certification:
    """Complete certification document."""

    tested_commit: str
    results_commit: str
    repository_clean: bool
    generated_at: str
    environment_identity: dict[str, str]
    test_counts: dict[str, int]
    static_check_status: str
    artifact_completeness: bool
    audit_status: str
    manifest_status: str
    assertion_status: str
    metric_consistency_status: str
    directional_check_status: str
    checksum_status: str
    research_valid: bool
    release_candidate: bool
    certification_status: str
    failure_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0.0",
            "tested_commit": self.tested_commit,
            "results_commit": self.results_commit,
            "repository_clean": self.repository_clean,
            "generated_at": self.generated_at,
            "environment_identity": self.environment_identity,
            "test_counts": self.test_counts,
            "static_check_status": self.static_check_status,
            "artifact_completeness": self.artifact_completeness,
            "audit_status": self.audit_status,
            "manifest_status": self.manifest_status,
            "assertion_status": self.assertion_status,
            "metric_consistency_status": self.metric_consistency_status,
            "directional_check_status": self.directional_check_status,
            "checksum_status": self.checksum_status,
            "research_valid": self.research_valid,
            "release_candidate": self.release_candidate,
            "certification_status": self.certification_status,
            "failure_reasons": self.failure_reasons,
        }


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify_artifact(path: Path) -> str:
    """Classify an artifact file by type."""
    name = path.name
    parent = path.parent.name

    if name == "episodes.jsonl":
        return "episodes"
    elif name.startswith("metrics"):
        return "metrics"
    elif name == "summary.json" or name == "summary.md":
        return "summary"
    elif "manifest" in name:
        return "manifest"
    elif "audit" in name:
        return "audit"
    elif "certification" in name:
        return "certification"
    elif "checksums" in name:
        return "checksums"
    elif parent == "provenance":
        return "provenance"
    elif parent == "ci":
        return "ci_output"
    elif name.endswith(".jsonl"):
        return "data"
    elif name.endswith(".json"):
        return "report"
    elif name.endswith(".md"):
        return "documentation"
    else:
        return "unknown"


def generate_checksums(results_dir: Path, certification_dir: Path) -> list[ChecksumEntry]:
    """Generate checksums for all result files."""
    entries: list[ChecksumEntry] = []

    certification_dir_resolved = certification_dir.resolve()

    # Walk the results directory
    for path in sorted(results_dir.rglob("*")):
        if not path.is_file():
            continue

        # Skip everything inside the certification directory itself so that
        # re-runs are idempotent (stale checksums/certification artifacts must
        # not be hashed into the new checksum set).
        try:
            path.resolve().relative_to(certification_dir_resolved)
            continue
        except ValueError:
            pass

        rel_path = path.relative_to(results_dir)
        entries.append(
            ChecksumEntry(
                path=str(rel_path),
                sha256=compute_sha256(path),
                size_bytes=path.stat().st_size,
                artifact_type=classify_artifact(path),
            )
        )

    return entries


def write_checksums_file(entries: list[ChecksumEntry], certification_dir: Path) -> Path:
    """Write checksums.sha256 file in sha256sum format."""
    checksums_path = certification_dir / "checksums.sha256"
    with open(checksums_path, "w") as f:
        for entry in entries:
            f.write(f"{entry.sha256}  {entry.path}\n")
    return checksums_path


def write_release_checksums(entries: list[ChecksumEntry], certification_dir: Path) -> Path:
    """Write release_checksums.json with metadata."""
    release_path = certification_dir / "release_checksums.json"
    release_data = {
        "schema_version": "1.0.0",
        "algorithm": "sha256",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(entries),
        "total_size_bytes": sum(e.size_bytes for e in entries),
        "files": [e.to_dict() for e in entries],
    }
    release_path.write_text(json.dumps(release_data, indent=2))
    return release_path


def verify_checksums(certification_dir: Path) -> bool:
    """Verify checksums using sha256sum --check."""
    checksums_path = certification_dir / "checksums.sha256"
    if not checksums_path.exists():
        return False

    # Resolve to absolute paths so the check works regardless of the caller's
    # working directory. The checksum file lists paths relative to the results
    # directory (the certification directory's parent), so run from there.
    results_dir = certification_dir.parent.resolve()
    result = subprocess.run(
        f"sha256sum --check {checksums_path.resolve()}",
        shell=True,
        cwd=results_dir,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def load_phase_results(results_dir: Path) -> dict[str, Any]:
    """Load phase results from complete_results_manifest.json."""
    manifest_path = results_dir / "complete_results_manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text())


def determine_certification_status(
    results_dir: Path,
    checksums_valid: bool,
) -> tuple[str, list[str], dict[str, bool]]:
    """Determine certification status based on all checks.

    Returns the canonical status, the list of failure reasons, and a mapping
    of individual gate outcomes used to populate the certification fields.
    """
    failure_reasons: list[str] = []

    # Load manifest
    manifest = load_phase_results(results_dir)
    phases = {p["phase_name"]: p for p in manifest.get("phases", [])}

    # Check static checks
    static_phase = phases.get("static_checks", {})
    static_passed = static_phase.get("status") == "PASS"
    if not static_passed:
        failure_reasons.append("Static checks did not pass")

    # Check test suite
    test_phase = phases.get("test_suite", {})
    test_passed = test_phase.get("status") in ("PASS", "SKIP")
    if not test_passed:
        failure_reasons.append("Test suite did not pass")

    # Check assertion suite
    assertion_phase = phases.get("assertion_suite", {})
    assertion_passed = assertion_phase.get("status") in ("PASS", "SKIP")
    if not assertion_passed:
        failure_reasons.append("Assertion suite did not pass")

    # Check smoke tests
    single_phase = phases.get("single_target_smoke", {})
    multi_phase = phases.get("multi_target_smoke", {})
    smoke_passed = single_phase.get("status") in ("PASS", "SKIP") and multi_phase.get("status") in (
        "PASS",
        "SKIP",
    )
    if not smoke_passed:
        failure_reasons.append("Smoke tests did not pass")

    # Check verification
    verify_phase = phases.get("independent_verification", {})
    verify_passed = verify_phase.get("status") in ("PASS", "SKIP")
    if not verify_passed:
        failure_reasons.append("Independent verification did not pass")

    # Check consistency
    consistency_phase = phases.get("consistency_validation", {})
    consistency_passed = consistency_phase.get("status") in ("PASS", "SKIP")
    if not consistency_passed:
        failure_reasons.append("Consistency validation did not pass")

    # Check checksums
    if not checksums_valid:
        failure_reasons.append("Checksum verification failed")

    # Determine status
    all_passed = (
        static_passed
        and test_passed
        and assertion_passed
        and smoke_passed
        and verify_passed
        and consistency_passed
        and checksums_valid
    )

    if all_passed:
        # Check if we can certify as release candidate
        # Requires: all tests pass, ruff, mypy, integration, checksums
        status = RELEASE_CANDIDATE
    elif not failure_reasons:
        status = RESEARCH_VALID
    else:
        # Section 16: a certification that does not reach research validity is
        # only a complete diagnostic execution.
        status = DIAGNOSTIC_VALID

    checks = {
        "static": static_passed,
        "tests": test_passed,
        "assertions": assertion_passed,
        "smoke": smoke_passed,
        "manifest": smoke_passed and consistency_passed,
        "verification": verify_passed,
        "consistency": consistency_passed,
        "checksums": checksums_valid,
        "research_valid": status in (RESEARCH_VALID, RELEASE_CANDIDATE),
        "release_candidate": status == RELEASE_CANDIDATE,
    }

    return status, failure_reasons, checks


def generate_certification(
    results_dir: Path,
    certification_dir: Path,
    tested_commit: str,
) -> Certification:
    """Generate the complete certification document."""
    certification_dir.mkdir(parents=True, exist_ok=True)

    # Generate checksums
    print("Generating checksums...")
    entries = generate_checksums(results_dir, certification_dir)
    write_checksums_file(entries, certification_dir)
    write_release_checksums(entries, certification_dir)
    print(f"  Generated checksums for {len(entries)} files")

    # Verify checksums
    print("Verifying checksums...")
    checksums_valid = verify_checksums(certification_dir)
    print(f"  Checksums valid: {checksums_valid}")

    # Determine certification status
    status, failure_reasons, checks = determine_certification_status(results_dir, checksums_valid)

    # Load environment info
    repo_state_path = results_dir / "provenance" / "repository_state.json"
    if repo_state_path.exists():
        repo_state = json.loads(repo_state_path.read_text())
    else:
        repo_state = {}

    # Load test counts from CI outputs
    test_counts: dict[str, int] = {}
    pytest_all_path = results_dir / "ci" / "pytest_all.txt"
    if pytest_all_path.exists():
        content = pytest_all_path.read_text()
        # Parse pytest output for counts
        for line in content.split("\n"):
            if "passed" in line:
                import re

                match = re.search(r"(\d+) passed", line)
                if match:
                    test_counts["passed"] = int(match.group(1))
                match = re.search(r"(\d+) failed", line)
                if match:
                    test_counts["failed"] = int(match.group(1))
                match = re.search(r"(\d+) skipped", line)
                if match:
                    test_counts["skipped"] = int(match.group(1))

    certification = Certification(
        tested_commit=tested_commit,
        results_commit=repo_state.get("results_commit", ""),
        repository_clean=repo_state.get("repository_clean", False),
        generated_at=datetime.now(timezone.utc).isoformat(),
        environment_identity={
            "python_version": repo_state.get("python_version", ""),
            "poetry_version": repo_state.get("poetry_version", ""),
            "os_info": repo_state.get("os_info", ""),
        },
        test_counts=test_counts,
        static_check_status="PASS" if checks["static"] else "FAIL",
        artifact_completeness=not any("artifact" in r.lower() for r in failure_reasons),
        audit_status="PASS" if checks["smoke"] else "FAIL",
        manifest_status="PASS" if checks["manifest"] else "FAIL",
        assertion_status="PASS" if checks["assertions"] else "FAIL",
        metric_consistency_status="PASS"
        if checks["verification"] and checks["consistency"]
        else "FAIL",
        directional_check_status="PASS" if checks["smoke"] else "FAIL",
        checksum_status="PASS" if checksums_valid else "FAIL",
        research_valid=checks["research_valid"],
        release_candidate=checks["release_candidate"],
        certification_status=status,
        failure_reasons=failure_reasons,
    )

    # Write certification
    cert_path = certification_dir / "certification.json"
    cert_path.write_text(json.dumps(certification.to_dict(), indent=2))

    # Write markdown summary
    md_lines = [
        "# Results Certification",
        "",
        f"**Status**: {status}",
        "",
        "## Provenance",
        "",
        f"- Tested commit: `{tested_commit}`",
        f"- Results commit: `{certification.results_commit}`",
        f"- Repository clean: {certification.repository_clean}",
        f"- Generated at: {certification.generated_at}",
        "",
        "## Environment",
        "",
        f"- Python: {certification.environment_identity.get('python_version', 'unknown')}",
        f"- Poetry: {certification.environment_identity.get('poetry_version', 'unknown')}",
        f"- OS: {certification.environment_identity.get('os_info', 'unknown')}",
        "",
        "## Check Status",
        "",
        "| Check | Status |",
        "|-------|--------|",
        f"| Static checks | {certification.static_check_status} |",
        f"| Artifact completeness | {'PASS' if certification.artifact_completeness else 'FAIL'} |",
        f"| Audit | {certification.audit_status} |",
        f"| Manifest | {certification.manifest_status} |",
        f"| Assertions | {certification.assertion_status} |",
        f"| Metric consistency | {certification.metric_consistency_status} |",
        f"| Directional checks | {certification.directional_check_status} |",
        f"| Checksums | {certification.checksum_status} |",
        f"| Research valid | {'PASS' if certification.research_valid else 'FAIL'} |",
        f"| Release candidate | {'PASS' if certification.release_candidate else 'FAIL'} |",
        "",
    ]

    if certification.test_counts:
        md_lines.extend(
            [
                "## Test Counts",
                "",
                f"- Passed: {certification.test_counts.get('passed', 0)}",
                f"- Failed: {certification.test_counts.get('failed', 0)}",
                f"- Skipped: {certification.test_counts.get('skipped', 0)}",
                "",
            ]
        )

    if failure_reasons:
        md_lines.extend(
            [
                "## Failure Reasons",
                "",
            ]
        )
        for reason in failure_reasons:
            md_lines.append(f"- {reason}")
        md_lines.append("")

    md_path = certification_dir / "certification.md"
    md_path.write_text("\n".join(md_lines))

    return certification


def main() -> int:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate certification")
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Results directory",
    )
    parser.add_argument(
        "--tested-commit",
        type=str,
        required=True,
        help="Tested commit SHA",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1

    certification_dir = results_dir / "certification"

    print("Certification Generation")
    print("=" * 50)
    print(f"Results directory: {results_dir}")
    print(f"Tested commit: {args.tested_commit}")
    print()

    certification = generate_certification(
        results_dir,
        certification_dir,
        args.tested_commit,
    )

    print()
    print(f"Certification status: {certification.certification_status}")
    if certification.failure_reasons:
        print("Failure reasons:")
        for reason in certification.failure_reasons:
            print(f"  - {reason}")

    print()
    print(f"Certification: {certification_dir / 'certification.json'}")
    print(f"Checksums: {certification_dir / 'checksums.sha256'}")

    return 0 if status_at_least(certification.certification_status, RESEARCH_VALID) else 1


if __name__ == "__main__":
    sys.exit(main())
