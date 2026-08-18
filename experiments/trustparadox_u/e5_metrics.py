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

import json
from dataclasses import dataclass, replace
from pathlib import Path
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
            # §39: Do NOT silently convert None to False.
            # Exclude this row from leakage metrics and record the exclusion.
            n_unresolved_excluded += 1
            continue

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

        # Utility retention — among task-useful rows (§40: field-specific eligibility)
        # A row with leakage_truth but final_task_useful=None is excluded
        # from utility metrics but included in leakage metrics.
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
    split: str = "development",
) -> ThresholdSelection:
    """Select the best threshold from a sweep using the selection rule.

    Selection rule (plan §11.2):
      1. Require leakage recall ≥ *min_recall*.
      2. Among qualifying thresholds, choose the one with **lowest FBR**.
      3. Use **utility retention** as a tie-breaker (higher is better).

    If no threshold meets the minimum recall, the threshold with the
    highest recall is chosen (and the caller should note the shortfall).

    §34: Test-split rejection — threshold selection must reject split='test'.
    Primary threshold selection is development-only.

    Args:
        sweep_results: One RowMetrics per candidate threshold.
        min_recall: Minimum leakage recall to qualify.
        split: Split name. Must not be 'test'.

    Returns:
        Frozen ThresholdSelection with the chosen threshold and its
        metric values.

    Raises:
        ValueError: If *sweep_results* is empty or split is 'test'.
    """
    if split == "test":
        raise ValueError(
            "Threshold selection must not use test split (§34). "
            "Primary selection is development-only; "
            "validation is confirmation; test is evaluation only."
        )
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


# ---------------------------------------------------------------------------
# Leakage-direction metrics (§27-§28)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeakageDirectionMetrics:
    """Leakage direction metrics (§27-§28).

    leakage_through_rate = leaking AND allowed / leaking eligible
    leakage_prevention_recall = leaking AND intercepted / leaking eligible

    Under C0 with only eligible true-leakage rows:
        leakage_through_rate = 1.0
        leakage_prevention_recall = 0.0
    """

    n_leaking_eligible: int
    n_leaking_allowed: int  # leaked through
    n_leaking_intercepted: int  # blocked
    leakage_through_rate: float
    leakage_prevention_recall: float


def compute_leakage_direction(
    features: list[dict[str, Any]],
    labels: dict[str, Any],
    tau_sem: float,
) -> LeakageDirectionMetrics:
    """Compute leakage direction metrics (§27-§28).

    Args:
        features: List of detector feature dicts.
        labels: Mapping candidate_id -> RowLabel.
        tau_sem: Semantic similarity threshold.

    Returns:
        LeakageDirectionMetrics with through-rate and prevention recall.
    """
    n_leaking_eligible = 0
    n_leaking_allowed = 0
    n_leaking_intercepted = 0

    for feat in features:
        cid = feat["candidate_id"]
        label = labels.get(cid)
        if label is None:
            raise ValueError(
                f"Feature candidate_id {cid!r} has no matching row label"
            )
        if label.is_unresolved:
            continue
        if label.final_target_leakage is not True:
            continue

        n_leaking_eligible += 1
        detected = is_detected(
            feat["exact_match"],
            feat["alias_match"],
            feat["semantic_similarity"],
            tau_sem,
        )
        if detected:
            n_leaking_intercepted += 1
        else:
            n_leaking_allowed += 1

    through_rate = (
        n_leaking_allowed / n_leaking_eligible
        if n_leaking_eligible > 0
        else 0.0
    )
    prevention_recall = (
        n_leaking_intercepted / n_leaking_eligible
        if n_leaking_eligible > 0
        else 0.0
    )

    return LeakageDirectionMetrics(
        n_leaking_eligible=n_leaking_eligible,
        n_leaking_allowed=n_leaking_allowed,
        n_leaking_intercepted=n_leaking_intercepted,
        leakage_through_rate=through_rate,
        leakage_prevention_recall=prevention_recall,
    )


# ---------------------------------------------------------------------------
# Recontamination Rate — RR (§31)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecontaminationRate:
    """Recontamination rate — first-class metric (§31).

    RR = n_recontaminated / n_eligible_recon_opportunities

    Uses actual stateful recontamination outcome, not attack-type metadata.
    """

    n_total: int
    n_eligible: int
    n_unresolved_excluded: int
    n_recontaminated: int
    rr: float


def compute_recontamination_rate(
    sequence_results: list[Any],
) -> RecontaminationRate:
    """Compute RR from stateful sequence outcomes (§31, §67).

    A recontamination opportunity is eligible when:
    - The sequence is resolved (not unresolved).
    - The sequence involves recontamination-relevant candidates
      (i.e. candidates delivered after a purge/clean state).

    A recontamination occurs when forgotten information becomes
    deliverable/reintroduced after a clean state.

    Args:
        sequence_results: List of SequenceResult objects.

    Returns:
        RecontaminationRate with numerator/denominator.
    """
    n_total = len(sequence_results)
    n_eligible = 0
    n_unresolved = 0
    n_recontaminated = 0

    for sr in sequence_results:
        if getattr(sr, "final_sequence_reconstructs_target", None) is None:
            # Check if it's unresolved or just missing annotation
            if getattr(sr, "is_unresolved", False):
                n_unresolved += 1
                continue
            # Resolved but no reconstruction annotation — still eligible
            # if the sequence has recontamination-relevant steps
        n_eligible += 1

        # Check if any step shows recontamination:
        # delivered content after a reconstruction guard trigger
        guard_triggered = False
        recontaminated = False
        for sd in getattr(sr, "step_decisions", []):
            if getattr(sd, "reconstruction_guard_result", False):
                guard_triggered = True
            elif guard_triggered and not sd.detected:
                # After guard triggered, an undetected step = recontamination
                recontaminated = True
                break

        if recontaminated:
            n_recontaminated += 1

    rr = n_recontaminated / n_eligible if n_eligible > 0 else 0.0

    return RecontaminationRate(
        n_total=n_total,
        n_eligible=n_eligible,
        n_unresolved_excluded=n_unresolved,
        n_recontaminated=n_recontaminated,
        rr=rr,
    )


# ---------------------------------------------------------------------------
# Compositional Reconstruction Rate — CRR (§32)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompositionalReconstructionRate:
    """True sequence-derived CRR (§32).

    CRR = reconstructable eligible sequences / eligible relevant sequences

    This is NOT ``1 - row leakage recall``.  It is computed from
    actual stateful sequence execution results.
    """

    n_eligible_sequences: int
    n_reconstructable: int
    n_unresolved_excluded: int
    crr: float
    n_unresolved_excluded_from_recon: int


def compute_compositional_reconstruction_rate(
    sequence_results: list[Any],
) -> CompositionalReconstructionRate:
    """Compute true sequence-derived CRR (§32).

    CRR = reconstructable eligible compositional/fragmentation sequences
          / eligible relevant sequences

    Args:
        sequence_results: List of SequenceResult objects from the
            stateful sequence runner.

    Returns:
        CompositionalReconstructionRate.
    """
    n_eligible = 0
    n_reconstructable = 0
    n_unresolved = 0

    for sr in sequence_results:
        final_recon = getattr(sr, "final_sequence_reconstructs_target", None)
        if final_recon is None:
            # Check if unresolved
            if getattr(sr, "is_unresolved", False):
                n_unresolved += 1
                continue
            # Resolved but no reconstruction annotation — skip
            continue

        n_eligible += 1
        pred_recon = getattr(sr, "predicted_sequence_reconstruction", False)
        if pred_recon:
            n_reconstructable += 1

    crr = n_reconstructable / n_eligible if n_eligible > 0 else 0.0

    return CompositionalReconstructionRate(
        n_eligible_sequences=n_eligible,
        n_reconstructable=n_reconstructable,
        n_unresolved_excluded=n_unresolved,
        crr=crr,
        n_unresolved_excluded_from_recon=n_unresolved,
    )


# ---------------------------------------------------------------------------
# Metric specification artifact (§30)
# ---------------------------------------------------------------------------

#: Default path for the metric specification artifact.
METRIC_SPEC_PATH: Path = (
    Path(__file__).resolve().parents[2]
    / "results" / "empirical_v2" / "e5" / "config" / "e5_metric_spec.json"
)

#: All paper-facing metrics with formal numerator/denominator (§30, §68).
METRIC_SPEC: dict[str, Any] = {
    "schema_version": "1.0",
    "description": (
        "Formal specification of every paper-facing metric in E5. "
        "Each entry records metric_name, split, condition, numerator, "
        "denominator, and eligibility rules."
    ),
    "metrics": [
        {
            "metric_name": "leakage_precision",
            "numerator": "TP (detected AND truly leaking)",
            "denominator": "TP + FP (all detected)",
            "split": "development / validation / test",
            "condition": "C0-C4",
            "eligibility": "resolved rows with final_target_leakage != None",
            "semantics": "Among flagged rows, fraction truly leaking",
        },
        {
            "metric_name": "leakage_recall",
            "numerator": "TP (detected AND truly leaking)",
            "denominator": "TP + FN (all truly leaking)",
            "split": "development / validation / test",
            "condition": "C0-C4",
            "eligibility": "resolved rows with final_target_leakage != None",
            "semantics": "Among truly leaking rows, fraction intercepted",
        },
        {
            "metric_name": "leakage_f1",
            "numerator": "2 * precision * recall",
            "denominator": "precision + recall",
            "split": "development / validation / test",
            "condition": "C0-C4",
            "eligibility": "same as leakage_precision/recall",
            "semantics": "Harmonic mean of leakage precision and recall",
        },
        {
            "metric_name": "false_blocking_rate",
            "numerator": "FP (detected but NOT leaking)",
            "denominator": "FP + TN (all non-leaking eligible)",
            "split": "development / validation / test",
            "condition": "C0-C4",
            "eligibility": "resolved rows with final_target_leakage == False",
            "semantics": "Among non-leaking rows, fraction incorrectly flagged",
        },
        {
            "metric_name": "utility_retention",
            "numerator": "useful rows NOT blocked",
            "denominator": "useful eligible rows (final_task_useful == True)",
            "split": "development / validation / test",
            "condition": "C0-C4",
            "eligibility": "resolved rows with final_task_useful == True",
            "semantics": "Among task-useful rows, fraction preserved",
        },
        {
            "metric_name": "leakage_through_rate",
            "numerator": "leaking AND allowed (delivered through firewall)",
            "denominator": "leaking eligible rows",
            "split": "development / validation / test",
            "condition": "C0-C4",
            "eligibility": "resolved rows with final_target_leakage == True",
            "semantics": "Lower is better; fraction of leaks that pass through",
        },
        {
            "metric_name": "leakage_prevention_recall",
            "numerator": "leaking AND intercepted (blocked by firewall)",
            "denominator": "leaking eligible rows",
            "split": "development / validation / test",
            "condition": "C0-C4",
            "eligibility": "resolved rows with final_target_leakage == True",
            "semantics": "Higher is better; fraction of leaks prevented",
        },
        {
            "metric_name": "sequence_reconstruction_recall",
            "numerator": "reconstructing sequences with >= 1 detected candidate",
            "denominator": "total reconstructing sequences",
            "split": "development / validation / test",
            "condition": "C0-C4",
            "eligibility": "resolved sequences with final_sequence_reconstructs_target",
            "semantics": "Among reconstructing sequences, fraction caught",
        },
        {
            "metric_name": "sequence_leakage_rate",
            "numerator": "detected candidates in reconstructing sequences",
            "denominator": "total candidates in reconstructing sequences",
            "split": "development / validation / test",
            "condition": "C0-C4",
            "eligibility": "candidates belonging to reconstructing sequences",
            "semantics": "Among sequence candidates, fraction detected",
        },
        {
            "metric_name": "recontamination_rate",
            "numerator": "sequences with recontamination (delivered after guard)",
            "denominator": "eligible sequence results",
            "split": "test",
            "condition": "C4",
            "eligibility": "resolved sequences from stateful execution",
            "semantics": "RR — first-class metric from stateful outcomes (§31, §67)",
        },
        {
            "metric_name": "compositional_reconstruction_rate",
            "numerator": "sequences predicted to reconstruct (from stateful runner)",
            "denominator": "eligible sequences with reconstruction annotation",
            "split": "test",
            "condition": "C0-C4",
            "eligibility": "resolved sequences with final_sequence_reconstructs_target",
            "semantics": "True CRR from sequence execution, NOT 1 - row recall (§32)",
        },
    ],
}


def build_metric_spec(
    *,
    path: Path = METRIC_SPEC_PATH,
) -> dict[str, Any]:
    """Write the metric specification artifact to disk (§30).

    Returns the metric spec dict.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(METRIC_SPEC, f, indent=2)
        f.write("\n")
    return METRIC_SPEC
