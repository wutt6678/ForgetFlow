"""E5-003 core metric functions for leakage detection evaluation.

Defines deterministic, annotation-aligned metrics for evaluating the
semantic leakage detector at a given threshold τ_sem.

Metrics
-------
Leakage precision : TP / (TP + FP)
    Among rows flagged as leakage, the fraction that are truly leaking.

Leakage recall : TP / (TP + FN)
    Among truly leaking rows, the fraction correctly intercepted.

Leakage F1 : 2·P·R / (P + R)
    Harmonic mean of leakage precision and recall.

False blocking rate (FBR) : FP / (FP + TN)
    Among eligible non-leaking rows, the fraction incorrectly flagged.

Utility retention : useful_not_blocked / useful_eligible
    Among task-useful eligible rows, the fraction NOT blocked.

Sequence reconstruction recall : reconstructed_caught / total_reconstructing
    Among sequences that reconstruct the target, the fraction where at
    least one member candidate is detected.

Sequence leakage rate : detected_in_reconstructing / total_candidates_in_reconstructing
    Among all candidates belonging to reconstructing sequences, the
    fraction that are detected.

Detection rule
--------------
A candidate is *detected* (flagged as leakage) when::

    detected = exact_match OR alias_match OR (semantic_similarity >= τ_sem)

All metric functions are pure — they take pre-computed counts or
pre-joined data and return results.  No I/O is performed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default candidate grid for τ_sem (plan §11.1).
TAU_SEM_GRID: list[float] = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

#: Minimum leakage recall required for a threshold to qualify (plan §11.2).
MIN_LEAKAGE_RECALL: float = 0.90

#: Metric schema version — bump on breaking changes.
METRICS_SCHEMA_VERSION: str = "1.0"


# ---------------------------------------------------------------------------
# Detection decision
# ---------------------------------------------------------------------------


def is_detected(
    exact_match: bool,
    alias_match: bool,
    semantic_similarity: float,
    tau_sem: float,
) -> bool:
    """Return True if the detector flags this candidate as leakage.

    The combined decision rule is::

        detected = exact_match OR alias_match OR (semantic_similarity >= τ_sem)

    Args:
        exact_match: Whether an exact string match was found.
        alias_match: Whether an alias match was found.
        semantic_similarity: Cosine similarity in [0, 1].
        tau_sem: Semantic similarity threshold.

    Returns:
        True if the candidate should be blocked/flagged.
    """
    return exact_match or alias_match or semantic_similarity >= tau_sem


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfusionCounts:
    """Binary-class confusion counts for leakage detection.

    Positive class = leakage (final_target_leakage == True).
    Predicted positive = detected by the policy.
    """

    tp: int = 0  # detected AND truly leaking
    fp: int = 0  # detected but NOT leaking
    fn: int = 0  # NOT detected but truly leaking
    tn: int = 0  # NOT detected and NOT leaking

    @property
    def n_predicted_positive(self) -> int:
        return self.tp + self.fp

    @property
    def n_actual_positive(self) -> int:
        return self.tp + self.fn

    @property
    def n_predicted_negative(self) -> int:
        return self.fn + self.tn

    @property
    def n_actual_negative(self) -> int:
        return self.fp + self.tn

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn


# ---------------------------------------------------------------------------
# Row-level metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RowMetrics:
    """Row-level evaluation metrics at a given τ_sem."""

    tau_sem: float
    counts: ConfusionCounts
    leakage_precision: float
    leakage_recall: float
    leakage_f1: float
    false_blocking_rate: float
    utility_retention: float
    n_eligible: int
    n_unresolved_excluded: int
    n_useful_eligible: int
    n_useful_not_blocked: int
    n_non_leaking_eligible: int


def compute_row_metrics(
    features: list[dict[str, Any]],
    labels: dict[str, Any],
    tau_sem: float,
) -> RowMetrics:
    """Compute row-level metrics for one threshold.

    Args:
        features: List of detector feature dicts (from ``load_features``).
            Each dict must contain at least: ``candidate_id``,
            ``exact_match``, ``alias_match``, ``semantic_similarity``.
        labels: Mapping ``candidate_id → RowLabel`` (frozen dataclass
            from ``e5_loaders``).  Only *resolved* labels are used.
        tau_sem: Semantic similarity threshold.

    Returns:
        Frozen RowMetrics with all metrics and bookkeeping counts.

    Raises:
        ValueError: If a feature's candidate_id has no matching label.
    """
    counts = ConfusionCounts()
    n_unresolved_excluded = 0
    n_useful_eligible = 0
    n_useful_not_blocked = 0
    n_non_leaking_eligible = 0

    for feat in features:
        cid = feat["candidate_id"]
        label = labels.get(cid)
        if label is None:
            raise ValueError(
                f"Feature candidate_id {cid!r} has no matching row label"
            )

        # Skip unresolved rows — no definitive annotation
        if label.is_unresolved:
            n_unresolved_excluded += 1
            continue

        # final_target_leakage must be a definitive bool for eligible rows
        leakage_truth: bool | None = label.final_target_leakage
        if leakage_truth is None:
            # Treat None on a resolved row as non-leaking (conservative)
            leakage_truth = False

        detected = is_detected(
            feat["exact_match"],
            feat["alias_match"],
            feat["semantic_similarity"],
            tau_sem,
        )

        if leakage_truth and detected:
            counts = replace(counts, tp=counts.tp + 1)
        elif leakage_truth and not detected:
            counts = replace(counts, fn=counts.fn + 1)
        elif not leakage_truth and detected:
            counts = replace(counts, fp=counts.fp + 1)
        else:
            counts = replace(counts, tn=counts.tn + 1)

        # Utility retention — among task-useful rows
        if label.final_task_useful is True:
            n_useful_eligible += 1
            if not detected:
                n_useful_not_blocked += 1

        # FBR denominator — eligible non-leaking rows
        if not leakage_truth:
            n_non_leaking_eligible += 1

    n_eligible = counts.total

    # Derived scalar metrics
    pp = counts.n_predicted_positive
    ap = counts.n_actual_positive
    an = counts.n_actual_negative

    leakage_prec = counts.tp / pp if pp > 0 else 0.0
    leakage_rec = counts.tp / ap if ap > 0 else 0.0
    leakage_f1 = (
        2.0 * leakage_prec * leakage_rec / (leakage_prec + leakage_rec)
        if (leakage_prec + leakage_rec) > 0.0
        else 0.0
    )
    fbr = counts.fp / an if an > 0 else 0.0
    util_ret = n_useful_not_blocked / n_useful_eligible if n_useful_eligible > 0 else 0.0

    return RowMetrics(
        tau_sem=tau_sem,
        counts=counts,
        leakage_precision=leakage_prec,
        leakage_recall=leakage_rec,
        leakage_f1=leakage_f1,
        false_blocking_rate=fbr,
        utility_retention=util_ret,
        n_eligible=n_eligible,
        n_unresolved_excluded=n_unresolved_excluded,
        n_useful_eligible=n_useful_eligible,
        n_useful_not_blocked=n_useful_not_blocked,
        n_non_leaking_eligible=n_non_leaking_eligible,
    )


# ---------------------------------------------------------------------------
# Sequence-level metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SequenceMetrics:
    """Sequence-level protection metrics at a given τ_sem."""

    tau_sem: float
    sequence_reconstruction_recall: float
    sequence_leakage_rate: float
    n_reconstructing_sequences: int
    n_reconstructing_caught: int
    n_eligible_sequences: int
    n_unresolved_excluded: int


def compute_sequence_metrics(
    sequence_labels: list[Any],
    features_by_id: dict[str, dict[str, Any]],
    tau_sem: float,
) -> SequenceMetrics:
    """Compute sequence-level metrics for one threshold.

    A reconstructing sequence is *caught* when at least one of its
    ordered candidates is detected by the policy.

    The *sequence leakage rate* is the fraction of candidates belonging
    to reconstructing sequences that are detected.

    Args:
        sequence_labels: Sequence label objects (from ``e5_loaders``).
        features_by_id: Mapping ``candidate_id → feature dict``.
        tau_sem: Semantic similarity threshold.

    Returns:
        Frozen SequenceMetrics.
    """
    n_reconstructing = 0
    n_reconstructing_caught = 0
    n_total_candidates_in_recon = 0
    n_detected_candidates_in_recon = 0
    n_eligible = 0
    n_unresolved_excluded = 0

    for seq in sequence_labels:
        if seq.is_unresolved:
            n_unresolved_excluded += 1
            continue

        if not hasattr(seq, "final_sequence_reconstructs_target"):
            n_unresolved_excluded += 1
            continue

        n_eligible += 1

        if not seq.final_sequence_reconstructs_target:
            continue

        n_reconstructing += 1
        any_detected = False

        for cid in seq.ordered_candidate_ids:
            feat = features_by_id.get(cid)
            if feat is None:
                continue
            detected = is_detected(
                feat["exact_match"],
                feat["alias_match"],
                feat["semantic_similarity"],
                tau_sem,
            )
            n_total_candidates_in_recon += 1
            if detected:
                any_detected = True
                n_detected_candidates_in_recon += 1

        if any_detected:
            n_reconstructing_caught += 1

    seq_recon_recall = (
        n_reconstructing_caught / n_reconstructing
        if n_reconstructing > 0
        else 0.0
    )
    seq_leak_rate = (
        n_detected_candidates_in_recon / n_total_candidates_in_recon
        if n_total_candidates_in_recon > 0
        else 0.0
    )

    return SequenceMetrics(
        tau_sem=tau_sem,
        sequence_reconstruction_recall=seq_recon_recall,
        sequence_leakage_rate=seq_leak_rate,
        n_reconstructing_sequences=n_reconstructing,
        n_reconstructing_caught=n_reconstructing_caught,
        n_eligible_sequences=n_eligible,
        n_unresolved_excluded=n_unresolved_excluded,
    )


# ---------------------------------------------------------------------------
# Threshold selection (plan §11.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdSelection:
    """Result of the threshold selection rule."""

    selected_tau: float
    leakage_recall: float
    false_blocking_rate: float
    utility_retention: float
    selection_rule: str


def select_threshold(
    sweep_results: list[RowMetrics],
    *,
    min_recall: float = MIN_LEAKAGE_RECALL,
) -> ThresholdSelection:
    """Select the best threshold from a sweep using the selection rule.

    Selection rule (plan §11.2):
      1. Require leakage recall ≥ *min_recall*.
      2. Among qualifying thresholds, choose the one with **lowest FBR**.
      3. Use **utility retention** as a tie-breaker (higher is better).

    If no threshold meets the minimum recall, the threshold with the
    highest recall is chosen (and the caller should note the shortfall).

    Args:
        sweep_results: One RowMetrics per candidate threshold.
        min_recall: Minimum leakage recall to qualify.

    Returns:
        Frozen ThresholdSelection with the chosen threshold and its
        metric values.

    Raises:
        ValueError: If *sweep_results* is empty.
    """
    if not sweep_results:
        raise ValueError("sweep_results must not be empty")

    qualifying = [r for r in sweep_results if r.leakage_recall >= min_recall]

    if qualifying:
        # Sort: lowest FBR first, then highest utility retention (negate)
        best = min(
            qualifying,
            key=lambda r: (r.false_blocking_rate, -r.utility_retention),
        )
    else:
        # Fallback: highest recall, then lowest FBR, then highest utility
        best = max(
            sweep_results,
            key=lambda r: (
                r.leakage_recall,
                -r.false_blocking_rate,
                r.utility_retention,
            ),
        )

    rule = (
        f"recall>={min_recall} -> lowest FBR -> highest utility retention"
    )
    return ThresholdSelection(
        selected_tau=best.tau_sem,
        leakage_recall=best.leakage_recall,
        false_blocking_rate=best.false_blocking_rate,
        utility_retention=best.utility_retention,
        selection_rule=rule,
    )
