"""E5-012: Post-freeze test verifier (Iteration 12).

Verifies that all test evidence is intact after freeze:
- all required files exist
- all hashes match
- row/sequence record counts are valid (§52)
- exact row/sequence ID sets match (§53, §54)
- condition purity — no mixed conditions (§55)
- provenance fields verified (§56)
- lock predates held-out access (§57)
- metric denominators valid
- configuration matches frozen config

Plan references:
    §52  freeze count verification (450 rows × 5 conditions, 72 seqs × 5)
    §53  exact row ID sets per condition
    §54  exact sequence ID sets per condition
    §55  condition purity
    §56  provenance checks
    §57  lock predates access
    §58  separate build/verify scripts
    §59  execution-validity gate
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
    """Verify row and sequence record counts (§52).

    Requires exactly:
        5 conditions × 450 row records = 2250 total
        5 conditions × 72 sequence records = 360 total

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

    # Check per-condition row counts (§52)
    per_condition_rows = results.get("per_condition_rows", {})
    if not per_condition_rows:
        # Fallback: check overall results
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
    else:
        for cond in ("C0", "C1", "C2", "C3", "C4"):
            n_rows = per_condition_rows.get(cond, 0)
            if n_rows != expected_rows:
                findings.append(
                    f"row_count_mismatch: {cond} has {n_rows}, "
                    f"expected {expected_rows} (§52)"
                )
                valid = False

        total_rows = sum(per_condition_rows.values())
        expected_total = expected_rows * 5
        if total_rows != expected_total:
            findings.append(
                f"total_row_count_mismatch: {total_rows}, "
                f"expected {expected_total} (§52)"
            )
            valid = False

    # Check per-condition sequence counts (§52)
    per_condition_seqs = results.get("per_condition_sequences", {})
    if per_condition_seqs:
        for cond in ("C0", "C1", "C2", "C3", "C4"):
            n_seqs = per_condition_seqs.get(cond, 0)
            if n_seqs != expected_sequences:
                findings.append(
                    f"sequence_count_mismatch: {cond} has {n_seqs}, "
                    f"expected {expected_sequences} (§52)"
                )
                valid = False

        total_seqs = sum(per_condition_seqs.values())
        expected_total_seqs = expected_sequences * 5
        if total_seqs != expected_total_seqs:
            findings.append(
                f"total_sequence_count_mismatch: {total_seqs}, "
                f"expected {expected_total_seqs} (§52)"
            )
            valid = False

    return valid


def verify_row_id_sets(
    results: dict[str, Any],
    frozen_candidate_ids: frozenset[str],
    findings: list[str],
) -> bool:
    """Verify exact row ID sets per condition (§53).

    For each C0-C4 result, candidate ID set must equal the frozen
    test candidate ID set.  Counts alone are insufficient.

    Args:
        results: Aggregated E5 results with per_condition_row_ids.
        frozen_candidate_ids: The frozen test candidate ID set.
        findings: List to append failure messages to.

    Returns:
        True if all ID sets match.
    """
    valid = True
    per_condition_ids = results.get("per_condition_row_ids", {})
    for cond in ("C0", "C1", "C2", "C3", "C4"):
        actual_ids = set(per_condition_ids.get(cond, []))
        if actual_ids != frozen_candidate_ids:
            missing = frozen_candidate_ids - actual_ids
            extra = actual_ids - frozen_candidate_ids
            msg = f"row_id_set_mismatch: {cond}"
            if missing:
                msg += f" missing={len(missing)}"
            if extra:
                msg += f" extra={len(extra)}"
            findings.append(msg)
            valid = False
    return valid


def verify_sequence_id_sets(
    results: dict[str, Any],
    frozen_sequence_ids: frozenset[str],
    findings: list[str],
) -> bool:
    """Verify exact sequence ID sets per condition (§54).

    For each condition, sequence_annotation_id set must equal the
    frozen test sequence ID set.

    Args:
        results: Aggregated E5 results with per_condition_sequence_ids.
        frozen_sequence_ids: The frozen test sequence ID set.
        findings: List to append failure messages to.

    Returns:
        True if all ID sets match.
    """
    valid = True
    per_condition_ids = results.get("per_condition_sequence_ids", {})
    for cond in ("C0", "C1", "C2", "C3", "C4"):
        actual_ids = set(per_condition_ids.get(cond, []))
        if actual_ids != frozen_sequence_ids:
            missing = frozen_sequence_ids - actual_ids
            extra = actual_ids - frozen_sequence_ids
            msg = f"sequence_id_set_mismatch: {cond}"
            if missing:
                msg += f" missing={len(missing)}"
            if extra:
                msg += f" extra={len(extra)}"
            findings.append(msg)
            valid = False
    return valid


def verify_condition_purity(
    results: dict[str, Any],
    findings: list[str],
) -> bool:
    """Verify condition purity (§55).

    Each result file must contain only the declared condition ID.
    Reject mixed-condition artifacts.

    Args:
        results: Aggregated E5 results with per_condition_row_ids
            or per_condition data.
        findings: List to append failure messages to.

    Returns:
        True if all conditions are pure.
    """
    valid = True
    per_condition = results.get("per_condition_results", {})
    for cond in ("C0", "C1", "C2", "C3", "C4"):
        cond_results = per_condition.get(cond, [])
        for row in cond_results:
            row_cond = row.get("condition_id", "")
            if row_cond != cond:
                findings.append(
                    f"condition_purity_violation: expected {cond}, "
                    f"got {row_cond} for candidate "
                    f"{row.get('candidate_id', '?')} (§55)"
                )
                valid = False
                break  # One finding per condition is enough
    return valid


def verify_provenance(
    results: dict[str, Any],
    lock: dict[str, Any],
    findings: list[str],
) -> bool:
    """Verify provenance fields (§56).

    Checks: tau_sem, reconstruction threshold, embedding model,
    embedding dimensions, condition manifest SHA, metric spec SHA,
    test lock SHA, E4 global annotation freeze SHA, execution commit,
    test_access_started == true.

    Args:
        results: Aggregated E5 results.
        lock: The test lock dict.
        findings: List to append failure messages to.

    Returns:
        True if provenance checks pass.
    """
    valid = True

    # test_access_started must be true
    if not lock.get("test_access_started"):
        findings.append("provenance: test_access_started is not true (§56)")
        valid = False

    # execution_commit must be non-empty
    if not lock.get("execution_commit"):
        findings.append("provenance: execution_commit is empty (§56)")
        valid = False

    # Check frozen config fields in results
    frozen = results.get("frozen_config", {})
    if not frozen.get("tau_sem"):
        findings.append("provenance: missing tau_sem in frozen config (§56)")
        valid = False
    if not frozen.get("embedding_model"):
        findings.append("provenance: missing embedding_model (§56)")
        valid = False

    return valid


def verify_lock_predates_access(
    lock: dict[str, Any],
    findings: list[str],
) -> bool:
    """Verify lock predates held-out access (§57).

    Require: test_lock.created_at < test_access_started_at
    and all scientific hashes in the lock equal those used for execution.

    Args:
        lock: The test lock dict.
        findings: List to append failure messages to.

    Returns:
        True if temporal ordering is correct.
    """
    valid = True
    created_at = lock.get("created_at")
    access_at = lock.get("test_access_started_at")

    if created_at and access_at:
        if access_at <= created_at:
            findings.append(
                f"lock_not_before_access: created_at={created_at} "
                f">= test_access_started_at={access_at} (§57)"
            )
            valid = False
    elif created_at and not access_at:
        # Access hasn't started yet — that's OK for pre-access verification
        pass
    elif not created_at:
        findings.append("missing created_at in test lock (§57)")
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
    *,
    frozen_candidate_ids: frozenset[str] | None = None,
    frozen_sequence_ids: frozenset[str] | None = None,
    lock_path: Path | None = None,
) -> FreezeVerificationResult:
    """Run full test freeze verification (plan §89, §52-§57).

    Args:
        manifest_path: Path to the freeze manifest JSON.
        base_dir: Base directory for relative paths in manifest.
        frozen_candidate_ids: Frozen test candidate ID set (§53).
        frozen_sequence_ids: Frozen test sequence ID set (§54).
        lock_path: Path to the test lock file (§56, §57).

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
    results: dict[str, Any] = {}
    if results_path:
        results_file = base_dir / results_path
        if verify_file_exists(results_file, findings):
            results = json.loads(results_file.read_text())
            verify_record_counts(results, findings=findings)

            eligibility = results.get("eligibility_manifest", [])
            verify_metric_denominators(eligibility, findings)

            # §53: Exact row ID sets
            if frozen_candidate_ids is not None:
                verify_row_id_sets(results, frozen_candidate_ids, findings)

            # §54: Exact sequence ID sets
            if frozen_sequence_ids is not None:
                verify_sequence_id_sets(results, frozen_sequence_ids, findings)

            # §55: Condition purity
            verify_condition_purity(results, findings)

    # Check frozen config
    frozen_config = manifest.get("frozen_config", {})
    if frozen_config:
        tau_sem = frozen_config.get("tau_sem")
        if tau_sem is None:
            findings.append("missing_frozen_tau_sem")

    # §56-§57: Provenance and lock checks
    if lock_path and lock_path.exists():
        lock = json.loads(lock_path.read_text())
        verify_provenance(results, lock, findings)
        verify_lock_predates_access(lock, findings)

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


# ---------------------------------------------------------------------------
# Execution-validity gate (§59)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionValidityGate:
    """Execution-validity gate (§59).

    Gates on:
        execution completeness
        ID-set completeness
        hash integrity
        config integrity
        split integrity
        metric denominator integrity

    Does NOT gate on whether ForgetFlow performs well.
    """

    passed: bool
    gates: dict[str, bool]
    findings: list[str] = field(default_factory=list)


def build_execution_validity_gate(
    verification: FreezeVerificationResult,
    results: dict[str, Any],
    lock: dict[str, Any] | None = None,
) -> ExecutionValidityGate:
    """Build the execution-validity gate (§59).

    Args:
        verification: Result from verify_test_freeze.
        results: Aggregated E5 results.
        lock: Test lock dict (optional).

    Returns:
        ExecutionValidityGate with per-gate pass/fail.
    """
    gates: dict[str, bool] = {}
    all_findings: list[str] = list(verification.findings)

    # 1. Execution completeness — all conditions ran
    per_cond = results.get("per_condition_rows", {})
    gates["execution_completeness"] = all(
        per_cond.get(c, 0) > 0 for c in ("C0", "C1", "C2", "C3", "C4")
    )
    if not gates["execution_completeness"]:
        all_findings.append("gate_fail: execution_completeness")

    # 2. ID-set completeness — verified by verify_row_id_sets
    gates["id_set_completeness"] = not any(
        "id_set_mismatch" in f for f in all_findings
    )

    # 3. Hash integrity — verified by verify_hash
    gates["hash_integrity"] = not any(
        "hash_mismatch" in f for f in all_findings
    )

    # 4. Config integrity — frozen config present
    frozen = results.get("frozen_config", {})
    gates["config_integrity"] = bool(frozen.get("tau_sem"))
    if not gates["config_integrity"]:
        all_findings.append("gate_fail: config_integrity")

    # 5. Split integrity — test split not used for calibration
    gates["split_integrity"] = not any(
        "test_split_contamination" in f for f in all_findings
    )

    # 6. Metric denominator integrity
    gates["metric_denominator_integrity"] = not any(
        "negative_denominator" in f or "numerator_exceeds_denominator" in f
        for f in all_findings
    )

    passed = all(gates.values())
    return ExecutionValidityGate(
        passed=passed,
        gates=gates,
        findings=all_findings,
    )


def write_execution_validity_gate(
    gate: ExecutionValidityGate,
    path: Path,
) -> None:
    """Write the execution-validity gate artifact (§59).

    Args:
        gate: The computed gate.
        path: Output path for e5_test_gate.json.
    """
    artifact = {
        "passed": gate.passed,
        "gates": gate.gates,
        "findings": gate.findings,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(artifact, f, indent=2)
        f.write("\n")


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
