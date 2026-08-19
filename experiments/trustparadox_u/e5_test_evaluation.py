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
from .e5_firewall_runner import (
    ExtendedRowResult,
    build_e5_forget_record,
    create_firewall_runner,
    extended_result_to_dict,
    extended_result_to_row_result,
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
# Condition-specific detection logic (DEPRECATED — non-authoritative)
# ---------------------------------------------------------------------------
# These functions are retained for unit tests and detector diagnostics only.
# The authoritative execution path uses create_firewall_runner() → process_row().
# See §4-§5 of E5-R1.1.


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
# Row-level evaluation (DEPRECATED — non-authoritative)
# ---------------------------------------------------------------------------
# evaluate_row() is retained for unit tests and detector diagnostics only.
# The authoritative execution path uses create_firewall_runner() → process_row().
# See §4-§5 of E5-R1.1.


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

    Fail closed (§36): if features dict is empty (missing features),
    raise ValueError instead of silently allowing.

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
        Frozen RowResult with decision provenance.

    Raises:
        ValueError: If features are missing for this candidate (§36).
    """
    # Fail closed on missing features (§36)
    if not features:
        raise ValueError(
            f"Missing features for candidate {row_label.candidate_id!r}. "
            f"All corpus candidates must have features in official runs."
        )

    exact_used, alias_used, semantic_used, detected = apply_condition_detection(
        features, condition, tau_sem,
    )
    action, blocked, allowed = determine_policy_action(detected, condition)

    # Build decision provenance (§12)
    if not condition.firewall_enabled:
        decision_reason = "pass_through"
        triggered: tuple[str, ...] = ()
    elif detected:
        parts: list[str] = []
        if exact_used:
            parts.append("exact_detector")
        if alias_used:
            parts.append("alias_detector")
        if semantic_used >= tau_sem:
            parts.append("semantic_detector")
        triggered = tuple(parts)
        decision_reason = "detected_by:" + "+".join(parts)
    else:
        decision_reason = "not_detected"
        triggered = ()

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
        decision_reason=decision_reason,
        triggered_modules=triggered,
        history_state_used=condition.history_enabled,
        reconstruction_guard_triggered=condition.reconstruction_guard and detected,
        purge_triggered=False,
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
# Full condition evaluation (AUTHORITATIVE — uses canonical FirewallRunner)
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
    """Run one condition on the full split via canonical FirewallRunner (§4-§5).

    Authoritative flow:
        run_condition()
          → create_firewall_runner(condition)
          → register correct forget target
          → runner.process_row(...)
          → record ExtendedRowResult → convert to RowResult

    For independent row evaluation, each row gets clean state (§8).

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
    cid = condition.condition_id

    # Fail closed: verify feature completeness (§38)
    corpus_ids = {c.candidate_id for c in split_data.corpus}
    feature_ids = set(features_by_id.keys())
    missing = corpus_ids - feature_ids
    if missing:
        raise ValueError(
            f"Missing features for {len(missing)} candidates: "
            f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}. "
            f"All corpus candidates must have features in official runs."
        )

    # Create canonical runner for this condition
    runner = create_firewall_runner(
        condition_id=cid,
        semantic_threshold=tau_sem,
    )

    # Build corpus lookup
    corpus_by_id = {c.candidate_id: c for c in split_data.corpus}

    # --- Row evaluation ---
    row_results: list[RowResult] = []

    for row_label in split_data.row_labels:
        corpus = corpus_by_id.get(row_label.candidate_id)
        if corpus is None:
            # R1.2 §21: missing corpus row is a FATAL error in
            # official runs. Use the empty default silently would
            # degrade to NO_ACTIVE_RECORDS, hiding the bug.
            raise ValueError(
                f"Missing corpus row for candidate "
                f"{row_label.candidate_id!r} (split={split_data.split!r}, "
                f"condition={cid!r}). All row labels must be backed by a "
                f"corpus entry in official runs (R1.2 §21)."
            )

        features = features_by_id.get(row_label.candidate_id, {})
        if not features:
            raise ValueError(
                f"Missing features for candidate {row_label.candidate_id!r}. "
                f"All corpus candidates must have features in official runs."
            )

        # For C4: register the correct forget target before processing (§7).
        # R1.2 §20: missing target registry mapping is a FATAL error.
        if cid == "C4":
            try:
                forget_record = build_e5_forget_record(
                    scenario_id=corpus.scenario_id,
                    secret_variant_id=corpus.secret_variant_id,
                )
                runner.register_forget_record(forget_record)
            except KeyError as e:
                raise KeyError(
                    f"Missing frozen target registry mapping (R1.2 §20): "
                    f"candidate_id={row_label.candidate_id!r}, "
                    f"scenario_id={corpus.scenario_id!r}, "
                    f"secret_variant_id={corpus.secret_variant_id!r}, "
                    f"split={split_data.split!r}. "
                    f"All official C4 candidates must have a frozen target "
                    f"registry entry. Inner error: {e}"
                ) from e

        # Process row through canonical runner with real corpus text (§6)
        er = runner.process_row(
            candidate_id=row_label.candidate_id,
            scenario_id=corpus.scenario_id,
            trust_level=corpus.trust_level,
            features=features,
            split=split_data.split,
            raw_text=corpus.text,
            recipient_id=corpus.recipient_id,
            sender_id=corpus.sender_id,
            input_content_sha=corpus.content_sha256,
            condition_manifest_sha=condition_manifest_sha,
            detector_config_sha=detector_config_sha,
            embedding_model=features.get("embedding_model", "unknown"),
        )

        # Convert ExtendedRowResult → RowResult deterministically (§35)
        rr = extended_result_to_row_result(er)
        row_results.append(rr)

        # Reset state between independent rows (§8)
        if cid == "C4":
            runner.clear_recipient_state()
            # Re-create runner for next row to ensure full clean state
            runner = create_firewall_runner(
                condition_id=cid,
                semantic_threshold=tau_sem,
            )

    # --- Sequence evaluation ---
    # For sequences, create a fresh runner and process all steps through it (§8)
    sequence_results = _run_sequences_via_runner(
        split_data=split_data,
        condition=condition,
        features_by_id=features_by_id,
        tau_sem=tau_sem,
        corpus_by_id=corpus_by_id,
        condition_manifest_sha=condition_manifest_sha,
        detector_config_sha=detector_config_sha,
    )

    return row_results, sequence_results


def _run_sequences_via_runner(
    *,
    split_data: SplitData,
    condition: ConditionSpec,
    features_by_id: dict[str, dict[str, Any]],
    tau_sem: float,
    corpus_by_id: dict[str, CorpusCandidate],
    condition_manifest_sha: str,
    detector_config_sha: str,
) -> list[SequenceResult]:
    """Run sequence evaluation via canonical FirewallRunner (§9).

    Each sequence gets a fresh runner.  All steps within a sequence
    go through the same runner instance to maintain state continuity.
    """
    from .e5_sequence_evaluation import SequenceResult, StepDecision

    cid = condition.condition_id
    results: list[SequenceResult] = []

    for seq_label in split_data.sequence_labels:
        if seq_label.is_unresolved:
            continue

        trust_level = getattr(seq_label, "trust_level", "unknown")
        ordered_ids = seq_label.ordered_candidate_ids

        # R1.2 §21: Fail closed on missing sequence corpus rows.
        # Every ordered candidate must be present in corpus + features.
        missing_corpus = [
            cid for cid in ordered_ids if cid not in corpus_by_id
        ]
        if missing_corpus:
            raise ValueError(
                f"Missing corpus rows for sequence "
                f"{seq_label.sequence_annotation_id!r} (split={split_data.split!r}, "
                f"condition={cid!r}): {missing_corpus[:5]}{'...' if len(missing_corpus) > 5 else ''}. "
                f"All ordered candidates must be in the corpus (R1.2 §21)."
            )
        missing_features = [
            cid for cid in ordered_ids if cid not in features_by_id
        ]
        if missing_features:
            raise ValueError(
                f"Missing features for sequence "
                f"{seq_label.sequence_annotation_id!r} (split={split_data.split!r}, "
                f"condition={cid!r}): {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}. "
                f"All ordered candidates must have features (R1.2 §21)."
            )

        # R1.2 §22: Validate sequence target consistency.
        # All ordered candidates in a sequence must share the same
        # scenario_id and secret_variant_id — otherwise the target
        # cannot be resolved unambiguously.
        first_corpus = corpus_by_id[ordered_ids[0]]
        target_scenario_id = first_corpus.scenario_id
        target_secret_variant_id = first_corpus.secret_variant_id
        for cid_check in ordered_ids[1:]:
            cand = corpus_by_id[cid_check]
            if (
                cand.scenario_id != target_scenario_id
                or cand.secret_variant_id != target_secret_variant_id
            ):
                raise ValueError(
                    f"Sequence target consistency violation (R1.2 §22): "
                    f"sequence {seq_label.sequence_annotation_id!r} "
                    f"contains candidates with different target families. "
                    f"step[0]=(scenario_id={target_scenario_id!r}, "
                    f"secret_variant_id={target_secret_variant_id!r}), "
                    f"step[{ordered_ids.index(cid_check)}]=(scenario_id="
                    f"{cand.scenario_id!r}, secret_variant_id="
                    f"{cand.secret_variant_id!r}), "
                    f"candidate_id={cid_check!r}, split={split_data.split!r}."
                )

        # Fresh runner per sequence (§8)
        seq_runner = create_firewall_runner(
            condition_id=cid,
            semantic_threshold=tau_sem,
        )

        # Register forget target for C4. R1.2 §20: missing target
        # registry mapping is a FATAL error.
        if cid == "C4" and ordered_ids:
            try:
                forget_record = build_e5_forget_record(
                    scenario_id=first_corpus.scenario_id,
                    secret_variant_id=first_corpus.secret_variant_id,
                )
                seq_runner.register_forget_record(forget_record)
            except KeyError as e:
                raise KeyError(
                    f"Missing frozen target registry mapping (R1.2 §20): "
                    f"sequence_id={seq_label.sequence_annotation_id!r}, "
                    f"scenario_id={first_corpus.scenario_id!r}, "
                    f"secret_variant_id={first_corpus.secret_variant_id!r}, "
                    f"split={split_data.split!r}. "
                    f"All official C4 sequences must have a frozen target "
                    f"registry entry. Inner error: {e}"
                ) from e

        # Process each step through the same runner
        steps: list[StepDecision] = []
        for i, candidate_id in enumerate(ordered_ids):
            corpus = corpus_by_id[candidate_id]
            features = features_by_id[candidate_id]
            if not features:
                # R1.2 §21: fail closed (defence in depth — already
                # checked above at the sequence level).
                raise ValueError(
                    f"Missing features for sequence step candidate "
                    f"{candidate_id!r} in sequence "
                    f"{seq_label.sequence_annotation_id!r} (R1.2 §21)."
                )

            er = seq_runner.process_row(
                candidate_id=candidate_id,
                scenario_id=corpus.scenario_id,
                trust_level=trust_level,
                features=features,
                split=split_data.split,
                raw_text=corpus.text,
                recipient_id=corpus.recipient_id,
                sender_id=corpus.sender_id,
                turn_id=i,
                message_id=f"seq_{seq_label.sequence_annotation_id}_step{i}",
                input_content_sha=corpus.content_sha256,
                condition_manifest_sha=condition_manifest_sha,
                detector_config_sha=detector_config_sha,
                embedding_model=features.get("embedding_model", "unknown"),
            )

            # Build StepDecision from ExtendedRowResult (§9.3)
            sd = StepDecision(
                step_index=i,
                candidate_id=candidate_id,
                exact_match=er.exact_match,
                alias_match=er.alias_match,
                semantic_similarity=er.semantic_similarity,
                detected=er.blocked,
                policy_action=er.policy_action,
                decision_reason=er.decision_reason,
                history_state_summary=(
                    f"history_used={er.history_state_used}"
                ),
                reconstruction_guard_result=er.reconstruction_guard_triggered,
                reconstruction_score=er.reconstruction_score,
                purge_state_transition=(
                    f"purge={er.purge_triggered}"
                ),
                delivered_content_sha=er.output_content_sha,
            )
            steps.append(sd)

        # Predict reconstruction from step decisions
        from .e5_sequence_evaluation import predict_sequence_reconstruction
        recon, earliest, strength = predict_sequence_reconstruction(steps)

        seq_result = SequenceResult(
            sequence_annotation_id=seq_label.sequence_annotation_id,
            trust_level=trust_level,
            condition_id=cid,
            ordered_candidate_ids=ordered_ids,
            step_decisions=tuple(steps),
            predicted_sequence_reconstruction=recon,
            predicted_earliest_reconstruction_step=earliest,
            predicted_reconstruction_strength=strength,
        )

        # Join annotation labels
        from .e5_sequence_evaluation import _join_annotations
        seq_result = _join_annotations(seq_result, seq_label)
        results.append(seq_result)

    return results


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

    §32-§33: If split == 'test', requires test-access guard before
    any evaluation begins.

    Args:
        split: Split name (should be "test" for held-out).
        features_by_id: Pre-computed detector features.
        tau_sem: Frozen semantic threshold.
        condition_manifest_sha: SHA of condition manifest.
        detector_config_sha: SHA of detector config.

    Returns:
        TestEvaluationResult with all row and sequence results.
    """
    # Test-access guard (§32-§33)
    if split == "test":
        from .e5_conditions import require_test_access_started
        require_test_access_started()

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
