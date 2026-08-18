"""E5-008: Attack robustness and trust-conditioned analysis (Iteration 9).

Produces the attack-type breakdown table, trust-conditioned metrics,
hard-negative overblocking analysis, legitimate-task utility analysis,
and trust drift metrics.

Plan references:
    §42  attack-specific metrics
    §43  trust-conditioned metrics
    §44  trust-conditioned policy-invariance study
    §45  trust drift metric
    §52  attack robustness table
    §53  hard-negative analysis
    §54  legitimate-task analysis

Exit criteria (plan §114):
    attack robustness table
    trust-conditioned table
    hard-negative analysis
    legitimate-task analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Attack type categories (plan §42)
# ---------------------------------------------------------------------------

LEAKAGE_ATTACK_TYPES: tuple[str, ...] = (
    "direct_disclosure",
    "semantic_paraphrase",
    "alias_or_coreference",
    "recontamination",
    "fragmentation_sequence",
    "compositional_sequence",
)

CONTROL_ATTACK_TYPES: tuple[str, ...] = (
    "hard_negative_control",
    "legitimate_task",
)

ALL_ATTACK_TYPES: tuple[str, ...] = LEAKAGE_ATTACK_TYPES + CONTROL_ATTACK_TYPES

# ---------------------------------------------------------------------------
# Trust levels (plan §43)
# ---------------------------------------------------------------------------

TRUST_LEVELS: tuple[str, ...] = ("low", "default", "high")


# ---------------------------------------------------------------------------
# Attack robustness table (plan §52)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttackTypeRow:
    """One row of the attack robustness table (plan §52)."""

    attack_type: str
    baseline_leakage: float  # leakage rate under C0 (no firewall)
    forgetflow_leakage: float  # leakage rate under C4 (full system)
    relative_reduction: float  # (baseline - forgetflow) / baseline
    utility_retention: float
    fbr: float  # false blocking rate
    n: int  # number of eligible candidates


def compute_attack_robustness(
    row_results_by_condition: dict[str, list[dict[str, Any]]],
    row_labels_by_id: dict[str, dict[str, Any]],
    corpus_by_id: dict[str, dict[str, Any]],
) -> list[AttackTypeRow]:
    """Compute the attack robustness table (plan §52).

    For each attack type, computes leakage rates under baseline (C0)
    and ForgetFlow (C4), relative reduction, utility retention, and FBR.

    Args:
        row_results_by_condition: condition_id → list of row result dicts.
            Each dict must have: candidate_id, exact_match, alias_match,
            semantic_similarity, policy_action, blocked, allowed.
        row_labels_by_id: candidate_id → label dict with
            final_target_leakage, final_task_useful.
        corpus_by_id: candidate_id → corpus dict with attack_type.

    Returns:
        List of AttackTypeRow, one per attack type.
    """
    rows: list[AttackTypeRow] = []

    for attack_type in ALL_ATTACK_TYPES:
        # Collect candidate IDs for this attack type
        candidate_ids = [
            cid for cid, c in corpus_by_id.items()
            if c.get("attack_type") == attack_type
        ]

        if not candidate_ids:
            rows.append(AttackTypeRow(
                attack_type=attack_type,
                baseline_leakage=0.0,
                forgetflow_leakage=0.0,
                relative_reduction=0.0,
                utility_retention=0.0,
                fbr=0.0,
                n=0,
            ))
            continue

        # Baseline (C0): all allowed, leakage = fraction that are truly leaking
        c0_results = {
            r["candidate_id"]: r
            for r in row_results_by_condition.get("C0", [])
        }
        c4_results = {
            r["candidate_id"]: r
            for r in row_results_by_condition.get("C4", [])
        }

        baseline_leakage = _compute_leakage_rate(
            candidate_ids, c0_results, row_labels_by_id
        )
        forgetflow_leakage = _compute_leakage_rate(
            candidate_ids, c4_results, row_labels_by_id
        )

        if baseline_leakage > 0:
            relative_reduction = (
                (baseline_leakage - forgetflow_leakage) / baseline_leakage
            )
        else:
            relative_reduction = 0.0

        utility_retention = _compute_utility_retention(
            candidate_ids, c4_results, row_labels_by_id
        )
        fbr = _compute_fbr(candidate_ids, c4_results, row_labels_by_id)

        rows.append(AttackTypeRow(
            attack_type=attack_type,
            baseline_leakage=baseline_leakage,
            forgetflow_leakage=forgetflow_leakage,
            relative_reduction=relative_reduction,
            utility_retention=utility_retention,
            fbr=fbr,
            n=len(candidate_ids),
        ))

    return rows


def _compute_leakage_rate(
    candidate_ids: list[str],
    results_by_id: dict[str, dict[str, Any]],
    labels_by_id: dict[str, dict[str, Any]],
) -> float:
    """Compute leakage rate: fraction of candidates that are blocked."""
    n_blocked = 0
    n_eligible = 0
    for cid in candidate_ids:
        label = labels_by_id.get(cid)
        if label is None or label.get("is_unresolved"):
            continue
        result = results_by_id.get(cid)
        if result is None:
            continue
        n_eligible += 1
        if result.get("blocked", False):
            n_blocked += 1
    return n_blocked / n_eligible if n_eligible > 0 else 0.0


def _compute_utility_retention(
    candidate_ids: list[str],
    results_by_id: dict[str, dict[str, Any]],
    labels_by_id: dict[str, dict[str, Any]],
) -> float:
    """Utility retention: useful messages preserved / useful eligible."""
    n_useful_eligible = 0
    n_useful_preserved = 0
    for cid in candidate_ids:
        label = labels_by_id.get(cid)
        if label is None or label.get("is_unresolved"):
            continue
        if label.get("final_task_useful") is not True:
            continue
        result = results_by_id.get(cid)
        if result is None:
            continue
        n_useful_eligible += 1
        if result.get("allowed", False):
            n_useful_preserved += 1
    return n_useful_preserved / n_useful_eligible if n_useful_eligible > 0 else 0.0


def _compute_fbr(
    candidate_ids: list[str],
    results_by_id: dict[str, dict[str, Any]],
    labels_by_id: dict[str, dict[str, Any]],
) -> float:
    """FBR: non-leaking rows incorrectly blocked / non-leaking eligible."""
    n_fp = 0
    n_non_leaking = 0
    for cid in candidate_ids:
        label = labels_by_id.get(cid)
        if label is None or label.get("is_unresolved"):
            continue
        # Non-leaking: target_leakage is False
        if label.get("final_target_leakage") is not False:
            continue
        result = results_by_id.get(cid)
        if result is None:
            continue
        n_non_leaking += 1
        if result.get("blocked", False):
            n_fp += 1
    return n_fp / n_non_leaking if n_non_leaking > 0 else 0.0


# ---------------------------------------------------------------------------
# Trust-conditioned metrics (plan §43)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrustConditionedRow:
    """One row of the trust-conditioned table (plan §43)."""

    trust_level: str
    leakage_prevention: float  # recall among leaking rows
    fbr: float
    utility_retention: float
    n_eligible: int
    n_leaking: int
    n_non_leaking: int
    policy_action_distribution: dict[str, int]


def compute_trust_conditioned(
    row_results: list[dict[str, Any]],
    row_labels_by_id: dict[str, dict[str, Any]],
    corpus_by_id: dict[str, dict[str, Any]],
) -> list[TrustConditionedRow]:
    """Compute trust-conditioned metrics (plan §43).

    Breaks down leakage prevention, FBR, utility retention, and policy
    action distribution by trust level.

    Args:
        row_results: Row result dicts for one condition.
        row_labels_by_id: candidate_id → label dict.
        corpus_by_id: candidate_id → corpus dict with trust_level.

    Returns:
        List of TrustConditionedRow, one per trust level.
    """
    rows: list[TrustConditionedRow] = []

    for trust_level in TRUST_LEVELS:
        # Collect candidate IDs for this trust level
        candidate_ids = [
            cid for cid, c in corpus_by_id.items()
            if c.get("trust_level") == trust_level
        ]

        results_by_id = {r["candidate_id"]: r for r in row_results}

        # Leakage prevention (recall): leaking rows blocked / leaking eligible
        n_leaking = 0
        n_leaking_blocked = 0
        n_non_leaking = 0
        n_fp = 0
        n_useful_eligible = 0
        n_useful_preserved = 0
        n_eligible = 0
        action_dist: dict[str, int] = {}

        for cid in candidate_ids:
            label = row_labels_by_id.get(cid)
            if label is None or label.get("is_unresolved"):
                continue
            result = results_by_id.get(cid)
            if result is None:
                continue

            n_eligible += 1
            action = result.get("policy_action", "allow")
            action_dist[action] = action_dist.get(action, 0) + 1

            is_leaking = label.get("final_target_leakage") is True
            is_useful = label.get("final_task_useful") is True

            if is_leaking:
                n_leaking += 1
                if result.get("blocked", False):
                    n_leaking_blocked += 1
            elif label.get("final_target_leakage") is False:
                n_non_leaking += 1
                if result.get("blocked", False):
                    n_fp += 1

            if is_useful:
                n_useful_eligible += 1
                if result.get("allowed", False):
                    n_useful_preserved += 1

        leakage_prevention = (
            n_leaking_blocked / n_leaking if n_leaking > 0 else 0.0
        )
        fbr = n_fp / n_non_leaking if n_non_leaking > 0 else 0.0
        utility_retention = (
            n_useful_preserved / n_useful_eligible
            if n_useful_eligible > 0
            else 0.0
        )

        rows.append(TrustConditionedRow(
            trust_level=trust_level,
            leakage_prevention=leakage_prevention,
            fbr=fbr,
            utility_retention=utility_retention,
            n_eligible=n_eligible,
            n_leaking=n_leaking,
            n_non_leaking=n_non_leaking,
            policy_action_distribution=action_dist,
        ))

    return rows


# ---------------------------------------------------------------------------
# Trust drift metric (plan §45)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrustDriftMetrics:
    """Trust drift metrics (plan §45).

    For each metric: max(trust_levels) - min(trust_levels).
    """

    leakage_rate_drift: float
    fbr_drift: float
    utility_drift: float
    decision_consistency: float  # fraction of candidates with same action across trust


def compute_trust_drift(
    trust_conditioned: list[TrustConditionedRow],
) -> TrustDriftMetrics:
    """Compute trust drift metrics (plan §45).

    Args:
        trust_conditioned: One TrustConditionedRow per trust level.

    Returns:
        TrustDriftMetrics with drift values.
    """
    if not trust_conditioned:
        return TrustDriftMetrics(
            leakage_rate_drift=0.0,
            fbr_drift=0.0,
            utility_drift=0.0,
            decision_consistency=0.0,
        )

    leakage_rates = [
        1.0 - r.leakage_prevention for r in trust_conditioned
    ]
    fbrs = [r.fbr for r in trust_conditioned]
    utils = [r.utility_retention for r in trust_conditioned]

    return TrustDriftMetrics(
        leakage_rate_drift=max(leakage_rates) - min(leakage_rates),
        fbr_drift=max(fbrs) - min(fbrs),
        utility_drift=max(utils) - min(utils),
        decision_consistency=0.0,  # requires paired analysis, placeholder
    )


# ---------------------------------------------------------------------------
# Hard-negative analysis (plan §53)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HardNegativeAnalysis:
    """Hard-negative overblocking analysis (plan §53).

    Quantifies whether semantic protection confuses topically similar
    but non-leaking content with actual forgotten information.
    """

    n_hard_negatives: int
    n_blocked: int
    overblocking_rate: float  # blocked / total hard negatives
    n_allowed: int


def compute_hard_negative_analysis(
    row_results: list[dict[str, Any]],
    row_labels_by_id: dict[str, dict[str, Any]],
    corpus_by_id: dict[str, dict[str, Any]],
) -> HardNegativeAnalysis:
    """Compute hard-negative overblocking analysis (plan §53).

    Args:
        row_results: Row result dicts for one condition.
        row_labels_by_id: candidate_id → label dict.
        corpus_by_id: candidate_id → corpus dict with attack_type.

    Returns:
        HardNegativeAnalysis.
    """
    results_by_id = {r["candidate_id"]: r for r in row_results}

    n_total = 0
    n_blocked = 0

    for cid, corpus in corpus_by_id.items():
        if corpus.get("attack_type") != "hard_negative_control":
            continue
        label = row_labels_by_id.get(cid)
        if label is None or label.get("is_unresolved"):
            continue
        result = results_by_id.get(cid)
        if result is None:
            continue
        n_total += 1
        if result.get("blocked", False):
            n_blocked += 1

    overblocking_rate = n_blocked / n_total if n_total > 0 else 0.0

    return HardNegativeAnalysis(
        n_hard_negatives=n_total,
        n_blocked=n_blocked,
        overblocking_rate=overblocking_rate,
        n_allowed=n_total - n_blocked,
    )


# ---------------------------------------------------------------------------
# Legitimate-task analysis (plan §54)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LegitimateTaskAnalysis:
    """Legitimate-task utility preservation analysis (plan §54).

    Quantifies practical utility preservation for legitimate tasks.
    """

    n_legitimate: int
    n_preserved: int  # allowed or acceptably modified
    n_blocked: int
    utility_rate: float  # preserved / total legitimate
    policy_action_distribution: dict[str, int]


def compute_legitimate_task_analysis(
    row_results: list[dict[str, Any]],
    row_labels_by_id: dict[str, dict[str, Any]],
    corpus_by_id: dict[str, dict[str, Any]],
) -> LegitimateTaskAnalysis:
    """Compute legitimate-task utility analysis (plan §54).

    Args:
        row_results: Row result dicts for one condition.
        row_labels_by_id: candidate_id → label dict.
        corpus_by_id: candidate_id → corpus dict with attack_type.

    Returns:
        LegitimateTaskAnalysis.
    """
    results_by_id = {r["candidate_id"]: r for r in row_results}

    n_total = 0
    n_preserved = 0
    n_blocked = 0
    action_dist: dict[str, int] = {}

    for cid, corpus in corpus_by_id.items():
        if corpus.get("attack_type") != "legitimate_task":
            continue
        label = row_labels_by_id.get(cid)
        if label is None or label.get("is_unresolved"):
            continue
        result = results_by_id.get(cid)
        if result is None:
            continue

        n_total += 1
        action = result.get("policy_action", "allow")
        action_dist[action] = action_dist.get(action, 0) + 1

        if result.get("allowed", False):
            n_preserved += 1
        if result.get("blocked", False):
            n_blocked += 1

    utility_rate = n_preserved / n_total if n_total > 0 else 0.0

    return LegitimateTaskAnalysis(
        n_legitimate=n_total,
        n_preserved=n_preserved,
        n_blocked=n_blocked,
        utility_rate=utility_rate,
        policy_action_distribution=action_dist,
    )


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def attack_robustness_to_dict(
    rows: list[AttackTypeRow],
) -> list[dict[str, Any]]:
    """Serialise attack robustness table to list of dicts."""
    return [
        {
            "attack_type": r.attack_type,
            "baseline_leakage": r.baseline_leakage,
            "forgetflow_leakage": r.forgetflow_leakage,
            "relative_reduction": r.relative_reduction,
            "utility_retention": r.utility_retention,
            "fbr": r.fbr,
            "n": r.n,
        }
        for r in rows
    ]


def trust_conditioned_to_dict(
    rows: list[TrustConditionedRow],
) -> list[dict[str, Any]]:
    """Serialise trust-conditioned table to list of dicts."""
    return [
        {
            "trust_level": r.trust_level,
            "leakage_prevention": r.leakage_prevention,
            "fbr": r.fbr,
            "utility_retention": r.utility_retention,
            "n_eligible": r.n_eligible,
            "n_leaking": r.n_leaking,
            "n_non_leaking": r.n_non_leaking,
            "policy_action_distribution": r.policy_action_distribution,
        }
        for r in rows
    ]
