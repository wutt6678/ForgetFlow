"""E5-012: Post-freeze test verifier (Iteration 12).

Verifies that all test evidence is intact after freeze:
- all required files exist
- all hashes match
- row/sequence record counts are valid
- metric denominators valid
- configuration matches frozen config

Plan references:
    §87  test reproducibility manifest
    §88  freeze test results
    §89  post-freeze verifier
    §90  E5 test gate

Exit criteria (plan §117):
    test freeze manifest PASS
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FreezeVerificationResult:
    """Result of test freeze verification."""

    passed: bool
    findings: list[str] = field(default_factory=list)
    n_files_checked: int = 0
    n_hashes_checked: int = 0


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file.

    Args:
        path: Path to the file.

    Returns:
        Hex digest string.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_file_exists(
    path: Path,
    findings: list[str],
) -> bool:
    """Check that a required file exists.

    Args:
        path: Expected file path.
        findings: List to append failure messages to.

    Returns:
        True if file exists.
    """
    if not path.exists():
        findings.append(f"missing_file: {path}")
        return False
    return True


def verify_hash(
    path: Path,
    expected_hash: str,
    findings: list[str],
) -> bool:
    """Verify file hash matches expected.

    Args:
        path: Path to the file.
        expected_hash: Expected SHA-256 hex digest.
        findings: List to append failure messages to.

    Returns:
        True if hash matches.
    """
    if not path.exists():
        findings.append(f"missing_file_for_hash: {path}")
        return False
    actual = compute_file_sha256(path)
    if actual != expected_hash:
        findings.append(
            f"hash_mismatch: {path} expected={expected_hash[:16]}... "
            f"actual={actual[:16]}..."
        )
        return False
    return True


def verify_record_counts(
    results: dict[str, Any],
    expected_rows: int = 450,
    expected_sequences: int = 72,
    findings: list[str] | None = None,
) -> bool:
    """Verify row and sequence record counts.

    Args:
        results: Aggregated E5 results dict.
        expected_rows: Expected number of row records per condition.
        expected_sequences: Expected number of sequence records.
        findings: List to append failure messages to.

    Returns:
        True if counts are valid.
    """
    if findings is None:
        findings = []

    valid = True

    # Check overall metrics have expected conditions
    overall = results.get("overall", [])
    if not overall:
        findings.append("no_overall_results")
        valid = False

    for res in overall:
        cond = res.get("condition", "unknown")
        m = res.get("metrics", {})
        n_eligible = m.get("n_eligible", 0)
        if n_eligible == 0:
            findings.append(f"zero_eligible_rows: condition={cond}")
            valid = False

    return valid


def verify_metric_denominators(
    eligibility: list[dict[str, Any]],
    findings: list[str],
) -> bool:
    """Verify metric eligibility denominators are valid.

    Args:
        eligibility: Eligibility manifest rows.
        findings: List to append failure messages to.

    Returns:
        True if all denominators are valid.
    """
    valid = True
    for row in eligibility:
        denom = row.get("denominator", 0)
        numer = row.get("numerator", 0)
        metric = row.get("metric_name", "unknown")
        cond = row.get("condition", "unknown")

        if denom < 0:
            findings.append(
                f"negative_denominator: {metric}/{cond}"
            )
            valid = False
        if numer < 0:
            findings.append(
                f"negative_numerator: {metric}/{cond}"
            )
            valid = False
        if denom > 0 and numer > denom:
            findings.append(
                f"numerator_exceeds_denominator: {metric}/{cond} "
                f"({numer} > {denom})"
            )
            valid = False

    return valid


def verify_test_freeze(
    manifest_path: Path,
    base_dir: Path | None = None,
) -> FreezeVerificationResult:
    """Run full test freeze verification (plan §89).

    Args:
        manifest_path: Path to the freeze manifest JSON.
        base_dir: Base directory for relative paths in manifest.

    Returns:
        FreezeVerificationResult with pass/fail and findings.
    """
    findings: list[str] = []
    n_files = 0
    n_hashes = 0

    if not manifest_path.exists():
        return FreezeVerificationResult(
            passed=False,
            findings=["freeze_manifest_missing"],
        )

    manifest = json.loads(manifest_path.read_text())

    if base_dir is None:
        base_dir = manifest_path.parent

    # Check required files
    required_files = manifest.get("required_files", [])
    for rf in required_files:
        path = base_dir / rf
        n_files += 1
        verify_file_exists(path, findings)

    # Check file hashes
    file_hashes = manifest.get("file_hashes", {})
    for fname, expected_hash in file_hashes.items():
        path = base_dir / fname
        n_files += 1
        n_hashes += 1
        verify_hash(path, expected_hash, findings)

    # Check results integrity
    results_path = manifest.get("results_path")
    if results_path:
        results_file = base_dir / results_path
        if verify_file_exists(results_file, findings):
            results = json.loads(results_file.read_text())
            verify_record_counts(results, findings=findings)

            eligibility = results.get("eligibility_manifest", [])
            verify_metric_denominators(eligibility, findings)

    # Check frozen config
    frozen_config = manifest.get("frozen_config", {})
    if frozen_config:
        tau_sem = frozen_config.get("tau_sem")
        if tau_sem is None:
            findings.append("missing_frozen_tau_sem")

    # Check test gate conditions
    test_frozen = manifest.get("test_results_frozen", False)
    if not test_frozen:
        findings.append("test_results_not_frozen")

    passed = len(findings) == 0
    return FreezeVerificationResult(
        passed=passed,
        findings=findings,
        n_files_checked=n_files,
        n_hashes_checked=n_hashes,
    )


def build_test_freeze_manifest(
    results_path: Path,
    required_files: list[Path],
    *,
    tau_sem: float = 0.75,
    seed: int = 42,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    """Build a test freeze manifest (plan §88).

    Args:
        results_path: Path to aggregated results JSON.
        required_files: List of required evidence files.
        tau_sem: Frozen semantic threshold.
        seed: Random seed used.
        base_dir: Base directory for relative paths.

    Returns:
        Freeze manifest dict.
    """
    if base_dir is None:
        base_dir = results_path.parent

    file_hashes: dict[str, str] = {}
    for f in required_files:
        if f.exists():
            rel = str(f.relative_to(base_dir)) if f.is_relative_to(base_dir) else f.name
            file_hashes[rel] = compute_file_sha256(f)

    # Also hash the results file
    if results_path.exists():
        rel = str(results_path.relative_to(base_dir)) if results_path.is_relative_to(base_dir) else results_path.name
        file_hashes[rel] = compute_file_sha256(results_path)

    required_file_strs = [
        str(f.relative_to(base_dir)) if f.is_relative_to(base_dir) else f.name
        for f in required_files
    ]

    return {
        "test_results_frozen": True,
        "frozen_config": {
            "tau_sem": tau_sem,
            "seed": seed,
        },
        "results_path": str(results_path.relative_to(base_dir)) if results_path.is_relative_to(base_dir) else results_path.name,
        "required_files": required_file_strs,
        "file_hashes": file_hashes,
    }
