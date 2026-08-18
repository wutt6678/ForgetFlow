"""E5-004: Validation confirmation — evaluate frozen config on validation split.

This module implements Iteration 5 of the E5 plan.  It:

1. Loads the development-selected configuration (from E5-003).
2. Freezes it — no threshold changes allowed.
3. Evaluates on the validation split (225 rows, 36 sequences).
4. Computes row-level and sequence-level metrics.
5. Computes attack-type and trust-conditioned diagnostics.
6. Applies acceptance criteria.
7. Writes validation_report.json.

Validation is confirmation, NOT retuning (plan §15.1):
- Cannot change threshold based on validation results.
- Cannot optimize attack-specific thresholds.
- Must document accept/reject with reason.

Exit criteria (plan §110):
- validation evaluation complete
- configuration accepted
- final selected config frozen
- no test access yet
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .e5_loaders import SplitData, load_split
from .e5_metrics import (
    RowMetrics,
    SequenceMetrics,
    compute_row_metrics,
    compute_sequence_metrics,
)
from .semantic_detector import load_features

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

_VALIDATION_DIR = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "empirical_v2"
    / "e5"
    / "validation"
)

_CALIBRATION_DIR = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "empirical_v2"
    / "e5"
    / "calibration"
)

_SELECTED_CONFIG_PATH = _CALIBRATION_DIR / "selected_config.json"
_VALIDATION_REPORT_PATH = _VALIDATION_DIR / "validation_report.json"


# ---------------------------------------------------------------------------
# Acceptance criteria
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AcceptanceCriteria:
    """Validation acceptance criteria (plan §16).

    These are checked BEFORE looking at the results.
    """

    min_leakage_recall: float = 0.80
    max_false_blocking_rate: float = 0.30
    min_sequence_reconstruction_recall: float = 0.50
    require_trust_invariance: bool = True


DEFAULT_ACCEPTANCE_CRITERIA = AcceptanceCriteria()


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttackTypeDiagnostics:
    """Metrics broken down by attack_type."""

    by_attack_type: dict[str, dict[str, float]]


@dataclass(frozen=True)
class TrustConditionedDiagnostics:
    """Metrics broken down by trust_level."""

    by_trust_level: dict[str, dict[str, float]]


def compute_attack_type_diagnostics(
    split_data: SplitData,
    features: list[dict[str, Any]],
    tau_sem: float,
) -> AttackTypeDiagnostics:
    """Compute metrics broken down by attack_type.

    This is diagnostic only — not used for threshold selection (plan §11.3).
    """
    # Group features by attack_type via corpus
    corpus_by_id = split_data.corpus_by_id
    labels_by_id = split_data.row_labels_by_id

    groups: dict[str, tuple[list[dict], dict[str, Any]]] = defaultdict(
        lambda: ([], {})
    )

    for feat in features:
        cid = feat["candidate_id"]
        corpus = corpus_by_id.get(cid)
        label = labels_by_id.get(cid)
        if corpus is None or label is None:
            continue
        attack_type = corpus.attack_type
        group_feats, group_labels = groups[attack_type]
        group_feats.append(feat)
        group_labels[cid] = label

    by_attack_type: dict[str, dict[str, float]] = {}
    for attack_type, (group_feats, group_labels) in groups.items():
        if not group_feats:
            continue
        metrics = compute_row_metrics(group_feats, group_labels, tau_sem)
        by_attack_type[attack_type] = {
            "leakage_precision": metrics.leakage_precision,
            "leakage_recall": metrics.leakage_recall,
            "leakage_f1": metrics.leakage_f1,
            "false_blocking_rate": metrics.false_blocking_rate,
            "utility_retention": metrics.utility_retention,
            "n_eligible": metrics.n_eligible,
        }

    return AttackTypeDiagnostics(by_attack_type=by_attack_type)


def compute_trust_conditioned_diagnostics(
    split_data: SplitData,
    features: list[dict[str, Any]],
    tau_sem: float,
) -> TrustConditionedDiagnostics:
    """Compute metrics broken down by trust_level.

    This checks trust-invariance (plan §11.4) — same threshold should
    work across low/default/high trust conditions.
    """
    corpus_by_id = split_data.corpus_by_id
    labels_by_id = split_data.row_labels_by_id

    groups: dict[str, tuple[list[dict], dict[str, Any]]] = defaultdict(
        lambda: ([], {})
    )

    for feat in features:
        cid = feat["candidate_id"]
        corpus = corpus_by_id.get(cid)
        label = labels_by_id.get(cid)
        if corpus is None or label is None:
            continue
        trust_level = corpus.trust_level
        group_feats, group_labels = groups[trust_level]
        group_feats.append(feat)
        group_labels[cid] = label

    by_trust_level: dict[str, dict[str, float]] = {}
    for trust_level, (group_feats, group_labels) in groups.items():
        if not group_feats:
            continue
        metrics = compute_row_metrics(group_feats, group_labels, tau_sem)
        by_trust_level[trust_level] = {
            "leakage_precision": metrics.leakage_precision,
            "leakage_recall": metrics.leakage_recall,
            "leakage_f1": metrics.leakage_f1,
            "false_blocking_rate": metrics.false_blocking_rate,
            "utility_retention": metrics.utility_retention,
            "n_eligible": metrics.n_eligible,
        }

    return TrustConditionedDiagnostics(by_trust_level=by_trust_level)


# ---------------------------------------------------------------------------
# Acceptance check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AcceptanceResult:
    """Result of the acceptance check."""

    accepted: bool
    reason: str
    criteria: dict[str, float]
    actual: dict[str, float]


def check_acceptance(
    row_metrics: RowMetrics,
    sequence_metrics: SequenceMetrics,
    trust_diagnostics: TrustConditionedDiagnostics,
    criteria: AcceptanceCriteria = DEFAULT_ACCEPTANCE_CRITERIA,
) -> AcceptanceResult:
    """Check if validation results meet acceptance criteria.

    Args:
        row_metrics: Row-level metrics on validation split.
        sequence_metrics: Sequence-level metrics on validation split.
        trust_diagnostics: Trust-conditioned diagnostics.
        criteria: Acceptance criteria to check against.

    Returns:
        AcceptanceResult with accept/reject and reason.
    """
    actual = {
        "leakage_recall": row_metrics.leakage_recall,
        "false_blocking_rate": row_metrics.false_blocking_rate,
        "sequence_reconstruction_recall": (
            sequence_metrics.sequence_reconstruction_recall
        ),
    }

    checks: list[tuple[str, bool, str]] = []

    # Check leakage recall
    recall_ok = row_metrics.leakage_recall >= criteria.min_leakage_recall
    checks.append((
        "leakage_recall",
        recall_ok,
        f"recall={row_metrics.leakage_recall:.3f} >= {criteria.min_leakage_recall}",
    ))

    # Check FBR
    fbr_ok = row_metrics.false_blocking_rate <= criteria.max_false_blocking_rate
    checks.append((
        "false_blocking_rate",
        fbr_ok,
        f"FBR={row_metrics.false_blocking_rate:.3f} <= {criteria.max_false_blocking_rate}",
    ))

    # Check sequence reconstruction recall
    seq_ok = (
        sequence_metrics.sequence_reconstruction_recall
        >= criteria.min_sequence_reconstruction_recall
    )
    checks.append((
        "sequence_reconstruction_recall",
        seq_ok,
        f"seq_recall={sequence_metrics.sequence_reconstruction_recall:.3f} >= {criteria.min_sequence_reconstruction_recall}",
    ))

    # Check trust-invariance (if required)
    trust_ok = True
    if criteria.require_trust_invariance and trust_diagnostics.by_trust_level:
        # Check that recall doesn't vary wildly across trust levels
        recalls = [
            m["leakage_recall"]
            for m in trust_diagnostics.by_trust_level.values()
            if m["n_eligible"] > 0
        ]
        if recalls:
            recall_range = max(recalls) - min(recalls)
            trust_ok = recall_range < 0.30  # Allow some variation
            checks.append((
                "trust_invariance",
                trust_ok,
                f"recall_range={recall_range:.3f} < 0.30",
            ))

    all_ok = all(ok for _, ok, _ in checks)
    reason_parts = [f"{name}: {'PASS' if ok else 'FAIL'} ({desc})" for name, ok, desc in checks]
    reason = "; ".join(reason_parts)

    return AcceptanceResult(
        accepted=all_ok,
        reason=reason,
        criteria={
            "min_leakage_recall": criteria.min_leakage_recall,
            "max_false_blocking_rate": criteria.max_false_blocking_rate,
            "min_sequence_reconstruction_recall": criteria.min_sequence_reconstruction_recall,
            "require_trust_invariance": criteria.require_trust_invariance,
        },
        actual=actual,
    )


# ---------------------------------------------------------------------------
# Main validation pipeline
# ---------------------------------------------------------------------------


def run_validation(
    *,
    selected_config_path: Path = _SELECTED_CONFIG_PATH,
    output_path: Path = _VALIDATION_REPORT_PATH,
    criteria: AcceptanceCriteria = DEFAULT_ACCEPTANCE_CRITERIA,
) -> dict[str, Any]:
    """Run the full validation confirmation pipeline.

    This is the main entry point for E5-004.

    Args:
        selected_config_path: Path to development selected_config.json.
        output_path: Where to write validation_report.json.
        criteria: Acceptance criteria.

    Returns:
        The validation report dict.

    Raises:
        FileNotFoundError: If selected_config.json doesn't exist.
        ValueError: If validation fails catastrophically.
    """
    # Load development-selected configuration
    if not selected_config_path.exists():
        raise FileNotFoundError(
            f"Selected config not found: {selected_config_path}. "
            "Run E5-003 calibration first."
        )

    with open(selected_config_path) as f:
        selected_config = json.load(f)

    tau_sem = selected_config["semantic_threshold"]
    logger.info("Loaded development-selected τ_sem=%.2f", tau_sem)

    # Compute config SHA for provenance
    config_sha = hashlib.sha256(
        json.dumps(selected_config, sort_keys=True).encode()
    ).hexdigest()

    # Load validation data
    split_data = load_split("validation")
    features = load_features("validation")

    logger.info(
        "Loaded %d features, %d row labels (%d unresolved), %d sequence labels",
        len(features),
        split_data.n_rows,
        split_data.n_unresolved_rows,
        split_data.n_sequences,
    )

    # Compute metrics at the frozen threshold
    labels_by_id = split_data.row_labels_by_id
    features_by_id = {f["candidate_id"]: f for f in features}
    sequence_labels = list(split_data.sequence_labels)

    row_metrics = compute_row_metrics(features, labels_by_id, tau_sem)
    sequence_metrics = compute_sequence_metrics(
        sequence_labels, features_by_id, tau_sem
    )

    logger.info(
        "Validation metrics: P=%.3f R=%.3f F1=%.3f FBR=%.3f UR=%.3f SR=%.3f",
        row_metrics.leakage_precision,
        row_metrics.leakage_recall,
        row_metrics.leakage_f1,
        row_metrics.false_blocking_rate,
        row_metrics.utility_retention,
        sequence_metrics.sequence_reconstruction_recall,
    )

    # Compute diagnostics
    attack_diagnostics = compute_attack_type_diagnostics(
        split_data, features, tau_sem
    )
    trust_diagnostics = compute_trust_conditioned_diagnostics(
        split_data, features, tau_sem
    )

    # Check acceptance
    acceptance = check_acceptance(
        row_metrics, sequence_metrics, trust_diagnostics, criteria
    )

    logger.info(
        "Acceptance: %s — %s",
        "ACCEPTED" if acceptance.accepted else "REJECTED",
        acceptance.reason,
    )

    # Build report
    report = _build_validation_report(
        selected_config=selected_config,
        config_sha=config_sha,
        tau_sem=tau_sem,
        row_metrics=row_metrics,
        sequence_metrics=sequence_metrics,
        attack_diagnostics=attack_diagnostics,
        trust_diagnostics=trust_diagnostics,
        acceptance=acceptance,
        split_data=split_data,
        n_features=len(features),
    )

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    logger.info("Wrote validation report: %s", output_path)

    return report


def _build_validation_report(
    *,
    selected_config: dict[str, Any],
    config_sha: str,
    tau_sem: float,
    row_metrics: RowMetrics,
    sequence_metrics: SequenceMetrics,
    attack_diagnostics: AttackTypeDiagnostics,
    trust_diagnostics: TrustConditionedDiagnostics,
    acceptance: AcceptanceResult,
    split_data: SplitData,
    n_features: int,
) -> dict[str, Any]:
    """Build the validation report dict."""
    return {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": "validation",
        "config_sha": config_sha,
        "selected_development_config": {
            "semantic_threshold": selected_config["semantic_threshold"],
            "embedding_model": selected_config["embedding_model"],
            "normalization": selected_config["normalization"],
            "detector_version": selected_config["detector_version"],
            "selection_rule": selected_config["selection_rule"],
        },
        "frozen_threshold": tau_sem,
        "data_summary": {
            "n_features": n_features,
            "n_row_labels": split_data.n_rows,
            "n_unresolved_rows": split_data.n_unresolved_rows,
            "n_sequence_labels": split_data.n_sequences,
            "n_unresolved_sequences": split_data.n_unresolved_sequences,
        },
        "row_metrics": {
            "leakage_precision": row_metrics.leakage_precision,
            "leakage_recall": row_metrics.leakage_recall,
            "leakage_f1": row_metrics.leakage_f1,
            "false_blocking_rate": row_metrics.false_blocking_rate,
            "utility_retention": row_metrics.utility_retention,
            "n_eligible": row_metrics.n_eligible,
            "n_unresolved_excluded": row_metrics.n_unresolved_excluded,
            "tp": row_metrics.counts.tp,
            "fp": row_metrics.counts.fp,
            "fn": row_metrics.counts.fn,
            "tn": row_metrics.counts.tn,
        },
        "sequence_metrics": {
            "sequence_reconstruction_recall": (
                sequence_metrics.sequence_reconstruction_recall
            ),
            "sequence_leakage_rate": sequence_metrics.sequence_leakage_rate,
            "n_reconstructing_sequences": (
                sequence_metrics.n_reconstructing_sequences
            ),
            "n_reconstructing_caught": sequence_metrics.n_reconstructing_caught,
            "n_eligible_sequences": sequence_metrics.n_eligible_sequences,
        },
        "attack_type_diagnostics": attack_diagnostics.by_attack_type,
        "trust_conditioned_diagnostics": trust_diagnostics.by_trust_level,
        "acceptance": {
            "accepted": acceptance.accepted,
            "reason": acceptance.reason,
            "criteria": acceptance.criteria,
            "actual": acceptance.actual,
        },
    }
