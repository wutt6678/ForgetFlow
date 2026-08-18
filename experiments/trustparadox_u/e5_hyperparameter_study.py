"""E5-010: Core hyperparameter sensitivity study (Iteration 11).

Computes threshold sensitivity tables and tradeoff data for the semantic
threshold τ_sem across development and validation splits.

Plan references:
    §59  core hyperparameter study (τ_sem primary)
    §60  use development + validation for sensitivity (NOT test)
    §61  threshold sensitivity table
    §62  threshold sensitivity figure data

Exit criteria (plan §116):
    development/validation threshold sensitivity complete
    paper tradeoff figure ready
    primary test config unchanged
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .e5_metrics import TAU_SEM_GRID, is_detected

# ---------------------------------------------------------------------------
# Threshold grid (plan §59)
# ---------------------------------------------------------------------------

# Re-export the frozen grid for external use
FROZEN_TAU_SEM_GRID: list[float] = list(TAU_SEM_GRID)


# ---------------------------------------------------------------------------
# Threshold sensitivity row (plan §61)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdSensitivityRow:
    """One row of the threshold sensitivity table (plan §61)."""

    tau_sem: float
    leakage_recall: float  # fraction of leaking rows detected
    fbr: float  # false blocking rate
    utility_retention: float  # useful rows preserved
    crr: float  # compositional reconstruction resistance (1 - reconstruction rate)
    n_eligible: int
    n_leaking: int
    n_non_leaking: int
    n_useful_eligible: int


def compute_threshold_sensitivity(
    row_results_by_threshold: dict[float, list[dict[str, Any]]],
    row_labels_by_id: dict[str, dict[str, Any]],
    corpus_by_id: dict[str, dict[str, Any]],
) -> list[ThresholdSensitivityRow]:
    """Compute threshold sensitivity table (plan §61).

    For each threshold value, computes leakage recall, FBR, utility
    retention, and CRR.

    Args:
        row_results_by_threshold: tau_sem → list of row result dicts.
            Each dict: candidate_id, blocked, allowed, policy_action,
            exact_match, alias_match, semantic_similarity.
        row_labels_by_id: candidate_id → label dict.
        corpus_by_id: candidate_id → corpus dict (unused here but kept
            for API consistency).

    Returns:
        List of ThresholdSensitivityRow, one per threshold.
    """
    rows: list[ThresholdSensitivityRow] = []

    for tau_sem in sorted(row_results_by_threshold.keys()):
        results = row_results_by_threshold[tau_sem]
        results_by_id = {r["candidate_id"]: r for r in results}

        n_eligible = 0
        n_leaking = 0
        n_leaking_detected = 0
        n_non_leaking = 0
        n_fp = 0
        n_useful_eligible = 0
        n_useful_preserved = 0

        for cid, label in row_labels_by_id.items():
            if label.get("is_unresolved"):
                continue
            result = results_by_id.get(cid)
            if result is None:
                continue

            n_eligible += 1
            is_leaking = label.get("final_target_leakage") is True
            is_useful = label.get("final_task_useful") is True
            is_blocked = result.get("blocked", False)
            is_allowed = result.get("allowed", False)

            if is_leaking:
                n_leaking += 1
                if is_blocked:
                    n_leaking_detected += 1
            elif label.get("final_target_leakage") is False:
                n_non_leaking += 1
                if is_blocked:
                    n_fp += 1

            if is_useful:
                n_useful_eligible += 1
                if is_allowed:
                    n_useful_preserved += 1

        leakage_recall = (
            n_leaking_detected / n_leaking if n_leaking > 0 else 0.0
        )
        fbr = n_fp / n_non_leaking if n_non_leaking > 0 else 0.0
        utility_retention = (
            n_useful_preserved / n_useful_eligible
            if n_useful_eligible > 0
            else 0.0
        )
        # CRR: proxy as 1 - (reconstructed sequences / total sequences)
        # At row level, use 1 - leakage_recall as reconstruction resistance
        crr = 1.0 - leakage_recall

        rows.append(ThresholdSensitivityRow(
            tau_sem=tau_sem,
            leakage_recall=leakage_recall,
            fbr=fbr,
            utility_retention=utility_retention,
            crr=crr,
            n_eligible=n_eligible,
            n_leaking=n_leaking,
            n_non_leaking=n_non_leaking,
            n_useful_eligible=n_useful_eligible,
        ))

    return rows


# ---------------------------------------------------------------------------
# Threshold application (re-evaluate detection at different thresholds)
# ---------------------------------------------------------------------------


def apply_threshold_to_row(
    row_result: dict[str, Any],
    tau_sem: float,
) -> dict[str, Any]:
    """Re-evaluate detection for one row at a different τ_sem.

    Keeps exact and alias detection unchanged; only re-evaluates
    the semantic component.

    Args:
        row_result: Original row result dict.
        tau_sem: New semantic threshold to apply.

    Returns:
        Modified row result dict.
    """
    result = dict(row_result)

    exact = result.get("exact_match", False)
    alias = result.get("alias_match", False)
    sim = result.get("semantic_similarity", 0.0)

    detected = is_detected(exact, alias, sim, tau_sem)
    result["blocked"] = detected
    result["allowed"] = not detected
    result["policy_action"] = "block" if detected else "allow"

    return result


def run_threshold_sweep(
    baseline_row_results: list[dict[str, Any]],
    thresholds: list[float] | None = None,
) -> dict[float, list[dict[str, Any]]]:
    """Re-evaluate all rows at each threshold in the grid.

    Args:
        baseline_row_results: Row results from the frozen configuration.
        thresholds: Optional custom threshold list. Defaults to TAU_SEM_GRID.

    Returns:
        Dict mapping tau_sem → list of re-evaluated row result dicts.
    """
    if thresholds is None:
        thresholds = FROZEN_TAU_SEM_GRID

    results_by_threshold: dict[float, list[dict[str, Any]]] = {}

    for tau in thresholds:
        ablated = [
            apply_threshold_to_row(r, tau)
            for r in baseline_row_results
        ]
        results_by_threshold[tau] = ablated

    return results_by_threshold


# ---------------------------------------------------------------------------
# Paper tradeoff figure data (plan §62)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TradeoffPoint:
    """One point for the paper tradeoff figure (plan §62)."""

    tau_sem: float
    leakage_prevention: float  # y1
    utility_retention: float  # y2
    fbr: float  # optional second figure


def compute_tradeoff_data(
    sensitivity_rows: list[ThresholdSensitivityRow],
) -> list[TradeoffPoint]:
    """Compute paper tradeoff figure data (plan §62).

    Args:
        sensitivity_rows: Threshold sensitivity rows.

    Returns:
        List of TradeoffPoint for plotting.
    """
    return [
        TradeoffPoint(
            tau_sem=r.tau_sem,
            leakage_prevention=r.leakage_recall,
            utility_retention=r.utility_retention,
            fbr=r.fbr,
        )
        for r in sensitivity_rows
    ]


# ---------------------------------------------------------------------------
# Optimal threshold selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdRecommendation:
    """Recommended threshold based on tradeoff analysis."""

    tau_sem: float
    leakage_recall: float
    fbr: float
    utility_retention: float
    reason: str


def select_optimal_threshold(
    sensitivity_rows: list[ThresholdSensitivityRow],
    *,
    min_leakage_recall: float = 0.90,
) -> ThresholdRecommendation:
    """Select the optimal threshold from sensitivity analysis.

    Selects the threshold with the highest utility retention among
    those meeting the minimum leakage recall constraint.

    Args:
        sensitivity_rows: Threshold sensitivity rows.
        min_leakage_recall: Minimum acceptable leakage recall.

    Returns:
        ThresholdRecommendation with selected threshold.
    """
    if not sensitivity_rows:
        return ThresholdRecommendation(
            tau_sem=0.75,
            leakage_recall=0.0,
            fbr=0.0,
            utility_retention=0.0,
            reason="no data",
        )

    # Filter by minimum leakage recall
    candidates = [
        r for r in sensitivity_rows
        if r.leakage_recall >= min_leakage_recall
    ]

    if not candidates:
        # Fall back to highest leakage recall
        best = max(sensitivity_rows, key=lambda r: r.leakage_recall)
        return ThresholdRecommendation(
            tau_sem=best.tau_sem,
            leakage_recall=best.leakage_recall,
            fbr=best.fbr,
            utility_retention=best.utility_retention,
            reason=f"no threshold meets min_recall={min_leakage_recall}",
        )

    # Among candidates, pick highest utility retention
    best = max(candidates, key=lambda r: r.utility_retention)
    return ThresholdRecommendation(
        tau_sem=best.tau_sem,
        leakage_recall=best.leakage_recall,
        fbr=best.fbr,
        utility_retention=best.utility_retention,
        reason=(
            f"best utility ({best.utility_retention:.3f}) "
            f"with recall≥{min_leakage_recall}"
        ),
    )


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def sensitivity_to_dict(
    rows: list[ThresholdSensitivityRow],
) -> list[dict[str, Any]]:
    """Serialise sensitivity table to list of dicts."""
    return [
        {
            "tau_sem": r.tau_sem,
            "leakage_recall": r.leakage_recall,
            "fbr": r.fbr,
            "utility_retention": r.utility_retention,
            "crr": r.crr,
            "n_eligible": r.n_eligible,
            "n_leaking": r.n_leaking,
            "n_non_leaking": r.n_non_leaking,
            "n_useful_eligible": r.n_useful_eligible,
        }
        for r in rows
    ]


def tradeoff_to_dict(
    points: list[TradeoffPoint],
) -> list[dict[str, Any]]:
    """Serialise tradeoff data to list of dicts."""
    return [
        {
            "tau_sem": p.tau_sem,
            "leakage_prevention": p.leakage_prevention,
            "utility_retention": p.utility_retention,
            "fbr": p.fbr,
        }
        for p in points
    ]
