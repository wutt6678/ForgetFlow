"""E5-007: Held-out test evaluation (Iteration 8).

Executes the frozen experimental configuration on the held-out test split
across all conditions C0–C4.  Produces per-row and per-sequence results
with full provenance, and writes a test-run manifest binding all SHAs.

Plan references:
    §47  official held-out test run
    §48  test output immutability
    §49  test run manifest
    §50  expected test scientific units (450 rows, 72 sequences)
    §51  test result completeness gate

Exit criteria (plan §113):
    450 row records per condition
    72 sequence records per condition
    no tuning
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .e5_conditions import (
    CONDITION_ORDER,
    CONDITIONS,
    ConditionSpec,
    RowResult,
    row_result_to_dict,
)
from .e5_loaders import (
    CorpusCandidate,
    RowLabel,
    SplitData,
    load_split,
)
from .e5_metrics import is_detected
from .e5_sequence_evaluation import (
    SequenceResult,
    evaluate_sequences,
    sequence_result_to_dict,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_E5_DIR = Path(__file__).resolve().parents[2] / "results" / "empirical_v2" / "e5"
_TEST_DIR = _E5_DIR / "test"
_CONFIG_DIR = _E5_DIR / "config"

# ---------------------------------------------------------------------------
# Expected counts (plan §50, §51)
# ---------------------------------------------------------------------------

EXPECTED_ROW_COUNT = 450
EXPECTED_SEQUENCE_COUNT = 72
EXPECTED_CONDITIONS = 5


# ---------------------------------------------------------------------------
# Condition-specific detection logic
# ---------------------------------------------------------------------------


def apply_condition_detection(
    features: dict[str, Any],
    condition: ConditionSpec,
    tau_sem: float,
) -> tuple[bool, bool, bool, bool]:
    """Apply condition-specific detection rule to pre-computed features.

    Each condition enables a different subset of detectors:
        C0: no detection (pass-through)
        C1: exact only
        C2: exact + alias
        C3: exact + alias + semantic
        C4: exact + alias + semantic (full system)

    Args:
        features: Pre-computed detector features for one candidate.
        condition: The condition specification.
        tau_sem: Frozen semantic threshold.

    Returns:
        Tuple of (exact_used, alias_used, semantic_used, detected).
    """
    exact = features.get("exact_match", False)
    alias = features.get("alias_match", False)
    sim = features.get("semantic_similarity", 0.0)

    # Apply condition-specific masking
    exact_used = exact and condition.exact_enabled
    alias_used = alias and condition.alias_enabled
    semantic_used = sim if condition.semantic_enabled else 0.0

    # Detection rule
    if not condition.firewall_enabled:
        detected = False
    else:
        detected = is_detected(exact_used, alias_used, semantic_used, tau_sem)

    return exact_used, alias_used, semantic_used, detected


def determine_policy_action(
    detected: bool,
    condition: ConditionSpec,
) -> tuple[str, bool, bool]:
    """Determine the policy action from detection result.

    For C0 (no firewall) and conditions without rich policy actions,
    the action is binary: block or allow.

    For C4 (rich policy actions), the action can be more nuanced,
    but for this runner we use the same binary mapping since the
    rich actions require additional context not available here.

    Args:
        detected: Whether the candidate was detected.
        condition: The condition specification.

    Returns:
        Tuple of (policy_action, blocked, allowed).
    """
    if not condition.firewall_enabled:
        return "allow", False, True

    if detected:
        return "block", True, False
    return "allow", False, True


# ---------------------------------------------------------------------------
# Row-level evaluation
# ---------------------------------------------------------------------------


def evaluate_row(
    row_label: RowLabel,
    corpus: CorpusCandidate,
    features: dict[str, Any],
    condition: ConditionSpec,
    tau_sem: float,
    *,
    split: str,
    condition_manifest_sha: str = "unknown",
    detector_config_sha: str = "unknown",
) -> RowResult:
    """Evaluate one candidate row under one condition.

    Args:
        row_label: Frozen annotation label.
        corpus: Frozen corpus candidate.
        features: Pre-computed detector features.
        condition: Condition specification.
        tau_sem: Frozen semantic threshold.
        split: Split name (e.g. "test").
        condition_manifest_sha: SHA of condition manifest.
        detector_config_sha: SHA of detector config.

    Returns:
        Frozen RowResult.
    """
    _, _, _, detected = apply_condition_detection(features, condition, tau_sem)
    action, blocked, allowed = determine_policy_action(detected, condition)

    return RowResult(
        candidate_id=row_label.candidate_id,
        split=split,
        condition_id=condition.condition_id,
        scenario_id=corpus.scenario_id,
        trust_level=corpus.trust_level,
        exact_match=features.get("exact_match", False),
        alias_match=features.get("alias_match", False),
        semantic_similarity=features.get("semantic_similarity", 0.0),
        policy_action=action,
        blocked=blocked,
        allowed=allowed,
        input_content_sha=corpus.content_sha256,
        output_content_sha=_compute_output_sha(corpus.content_sha256, action),
        detector_config_sha=detector_config_sha,
        condition_manifest_sha=condition_manifest_sha,
        embedding_model=features.get("embedding_model", "unknown"),
    )


def _compute_output_sha(input_sha: str, action: str) -> str:
    """Compute output content SHA based on policy action.

    For allow: output = input (unchanged).
    For block: output = sha("BLOCKED:<input_sha>").
    For redact/abstract: output = sha("MODIFIED:<input_sha>").
    """
    if action == "allow":
        return input_sha
    return hashlib.sha256(f"{action.upper()}:{input_sha}".encode()).hexdigest()


# ---------------------------------------------------------------------------
# Full condition evaluation
# ---------------------------------------------------------------------------


def run_condition(
    split_data: SplitData,
    condition: ConditionSpec,
    features_by_id: dict[str, dict[str, Any]],
    tau_sem: float,
    *,
    condition_manifest_sha: str = "unknown",
    detector_config_sha: str = "unknown",
) -> tuple[list[RowResult], list[SequenceResult]]:
    """Run one condition on the full split.

    Args:
        split_data: Frozen split data (labels + corpus).
        condition: Condition specification.
        features_by_id: Pre-computed features keyed by candidate_id.
        tau_sem: Frozen semantic threshold.
        condition_manifest_sha: SHA of condition manifest.
        detector_config_sha: SHA of detector config.

    Returns:
        Tuple of (row_results, sequence_results).
    """
    row_results: list[RowResult] = []

    # Build corpus lookup
    corpus_by_id = {c.candidate_id: c for c in split_data.corpus}

    for row_label in split_data.row_labels:
        corpus = corpus_by_id.get(row_label.candidate_id)
        if corpus is None:
            continue

        features = features_by_id.get(row_label.candidate_id, {})
        result = evaluate_row(
            row_label=row_label,
            corpus=corpus,
            features=features,
            condition=condition,
            tau_sem=tau_sem,
            split=split_data.split,
            condition_manifest_sha=condition_manifest_sha,
            detector_config_sha=detector_config_sha,
        )
        row_results.append(result)

    # Sequence evaluation
    sequence_results = evaluate_sequences(
        sequence_labels=list(split_data.sequence_labels),
        features_by_id=features_by_id,
        tau_sem=tau_sem,
        condition_id=condition.condition_id,
    )

    return row_results, sequence_results


# ---------------------------------------------------------------------------
# Full test evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeldOutEvaluationResult:
    """Complete held-out evaluation result across all conditions."""

    split: str
    condition_ids: tuple[str, ...]
    row_results: dict[str, tuple[RowResult, ...]]  # condition_id → results
    sequence_results: dict[str, tuple[SequenceResult, ...]]  # condition_id → results
    tau_sem: float
    n_rows_per_condition: int
    n_sequences_per_condition: int


def run_test_evaluation(
    split: str,
    features_by_id: dict[str, dict[str, Any]],
    tau_sem: float,
    *,
    condition_manifest_sha: str = "unknown",
    detector_config_sha: str = "unknown",
) -> HeldOutEvaluationResult:
    """Run the full held-out test evaluation (plan §47).

    Executes all conditions C0–C4 on the specified split.

    Args:
        split: Split name (should be "test" for held-out).
        features_by_id: Pre-computed detector features.
        tau_sem: Frozen semantic threshold.
        condition_manifest_sha: SHA of condition manifest.
        detector_config_sha: SHA of detector config.

    Returns:
        TestEvaluationResult with all row and sequence results.
    """
    split_data = load_split(split)

    all_row_results: dict[str, tuple[RowResult, ...]] = {}
    all_sequence_results: dict[str, tuple[SequenceResult, ...]] = {}

    for cid in CONDITION_ORDER:
        condition = CONDITIONS[cid]
        row_res, seq_res = run_condition(
            split_data=split_data,
            condition=condition,
            features_by_id=features_by_id,
            tau_sem=tau_sem,
            condition_manifest_sha=condition_manifest_sha,
            detector_config_sha=detector_config_sha,
        )
        all_row_results[cid] = tuple(row_res)
        all_sequence_results[cid] = tuple(seq_res)

    n_rows = len(all_row_results.get("C0", ()))
    n_seqs = len(all_sequence_results.get("C0", ()))

    return HeldOutEvaluationResult(
        split=split,
        condition_ids=CONDITION_ORDER,
        row_results=all_row_results,
        sequence_results=all_sequence_results,
        tau_sem=tau_sem,
        n_rows_per_condition=n_rows,
        n_sequences_per_condition=n_seqs,
    )


# ---------------------------------------------------------------------------
# Test run manifest (plan §49)
# ---------------------------------------------------------------------------


def build_test_run_manifest(
    *,
    result: HeldOutEvaluationResult,
    code_commit: str = "unknown",
    test_lock_sha: str = "unknown",
    condition_manifest_sha: str = "unknown",
    embedding_manifest_sha: str = "unknown",
    selected_config_sha: str = "unknown",
    annotation_freeze_sha: str = "unknown",
    start_timestamp: str | None = None,
    end_timestamp: str | None = None,
) -> dict[str, Any]:
    """Build the test run manifest (plan §49).

    Binds all provenance SHAs and result counts.

    Returns the manifest dict.
    """
    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": code_commit,
        "test_lock_sha": test_lock_sha,
        "condition_manifest_sha": condition_manifest_sha,
        "embedding_manifest_sha": embedding_manifest_sha,
        "selected_config_sha": selected_config_sha,
        "annotation_freeze_sha": annotation_freeze_sha,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp or datetime.now(timezone.utc).isoformat(),
        "split": result.split,
        "condition_count": len(result.condition_ids),
        "condition_ids": list(result.condition_ids),
        "row_count_per_condition": result.n_rows_per_condition,
        "sequence_count_per_condition": result.n_sequences_per_condition,
        "tau_sem": result.tau_sem,
        "completeness": {
            "expected_rows": EXPECTED_ROW_COUNT,
            "expected_sequences": EXPECTED_SEQUENCE_COUNT,
            "expected_conditions": EXPECTED_CONDITIONS,
            "rows_complete": result.n_rows_per_condition == EXPECTED_ROW_COUNT,
            "sequences_complete": (
                result.n_sequences_per_condition == EXPECTED_SEQUENCE_COUNT
            ),
        },
    }

    return manifest


# ---------------------------------------------------------------------------
# Completeness validation (plan §51)
# ---------------------------------------------------------------------------


def validate_completeness(result: HeldOutEvaluationResult) -> list[str]:
    """Validate test result completeness (plan §51).

    Returns a list of error messages (empty = all OK).
    """
    errors: list[str] = []

    # Check all conditions present
    for cid in CONDITION_ORDER:
        if cid not in result.row_results:
            errors.append(f"Missing row results for condition {cid}")
        if cid not in result.sequence_results:
            errors.append(f"Missing sequence results for condition {cid}")

    # Check row counts
    for cid in CONDITION_ORDER:
        n_rows = len(result.row_results.get(cid, ()))
        if n_rows != EXPECTED_ROW_COUNT:
            errors.append(
                f"{cid}: expected {EXPECTED_ROW_COUNT} rows, got {n_rows}"
            )

    # Check sequence counts
    for cid in CONDITION_ORDER:
        n_seqs = len(result.sequence_results.get(cid, ()))
        if n_seqs != EXPECTED_SEQUENCE_COUNT:
            errors.append(
                f"{cid}: expected {EXPECTED_SEQUENCE_COUNT} sequences, "
                f"got {n_seqs}"
            )

    return errors


# ---------------------------------------------------------------------------
# Disk I/O
# ---------------------------------------------------------------------------


def write_test_results(
    result: HeldOutEvaluationResult,
    run_dir: Path | None = None,
) -> Path:
    """Write test results to disk (plan §48).

    Creates a run directory with per-condition JSONL files.

    Args:
        result: Complete test evaluation result.
        run_dir: Optional explicit run directory.  If None, uses
            ``results/empirical_v2/e5/test/run_<timestamp>/``.

    Returns:
        Path to the run directory.
    """
    if run_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = _TEST_DIR / f"run_{ts}"

    run_dir.mkdir(parents=True, exist_ok=True)

    for cid in CONDITION_ORDER:
        # Row results
        row_path = run_dir / f"{cid}_row_results.jsonl"
        with open(row_path, "w") as f:
            for rr in result.row_results.get(cid, ()):
                f.write(json.dumps(row_result_to_dict(rr)) + "\n")

        # Sequence results
        seq_path = run_dir / f"{cid}_sequence_results.jsonl"
        with open(seq_path, "w") as f:
            for sr in result.sequence_results.get(cid, ()):
                f.write(json.dumps(sequence_result_to_dict(sr)) + "\n")

    return run_dir
