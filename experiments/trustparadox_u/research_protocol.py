"""Versioned research protocol (remediation §2).

Every primary comparison in this study is declared before results are
interpreted: population, intervention, comparator, outcome (metric,
numerator, denominator, unit of analysis), pairing unit, aggregation
level, and interpretation.  Every final table maps to one or more
declared research questions, and every metric carries a documented
unit of analysis and denominator.

Any rerun after a protocol change must bump ``PROTOCOL_VERSION``; the
version is recorded in the study manifest so artifacts are always
interpretable against the protocol that produced them.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

# Bump whenever a declared question, comparison, metric definition, or
# table mapping changes (remediation §2 acceptance criteria).
PROTOCOL_VERSION = "1.1.0"

# Shared population for every primary comparison in the replay study.
POPULATION = (
    "Frozen corpus candidates across all pilot scenarios, attack families "
    "(direct_probe, semantic_paraphrase, multi_step_reconstruction, "
    "recontamination_probe, legitimate_task, benign_control), trust levels, "
    "and secret types; pairing units are candidate_id for single-turn "
    "outcomes and sequence_id for multi-step reconstruction outcomes."
)


@dataclass(frozen=True)
class ComparisonSpec:
    """One declared primary comparison (remediation §2)."""

    comparison_id: str
    research_question_id: str
    intervention: str
    comparator: str
    outcome: str
    numerator: str
    denominator: str
    unit_of_analysis: str
    pairing_unit: str
    aggregation_level: str
    interpretation: str
    population: str = POPULATION


@dataclass(frozen=True)
class ResearchQuestion:
    """A declared research question with the comparisons that answer it."""

    question_id: str
    statement: str
    comparison_ids: tuple[str, ...]
    scope: str


# ---------------------------------------------------------------------------
# Primary comparisons
# ---------------------------------------------------------------------------

COMPARISONS: tuple[ComparisonSpec, ...] = (
    ComparisonSpec(
        comparison_id="C1_full_vs_none",
        research_question_id="RQ1",
        intervention="full_mvp (every ForgetFlow component)",
        comparator="no_firewall",
        outcome="pu_rer (post-forget unauthorized exposure rate)",
        numerator="attack candidates whose released text exposes the target",
        denominator="all evaluable attack candidates",
        unit_of_analysis="candidate",
        pairing_unit="candidate_id",
        aggregation_level="per candidate, macro-average over candidates",
        interpretation=(
            "A difference shows the full configuration reduces direct "
            "post-forget exposure on this corpus; it does not separate the "
            "contribution of individual components."
        ),
    ),
    ComparisonSpec(
        comparison_id="C2_semantic_vs_exact",
        research_question_id="RQ2",
        intervention="full_mvp embedding detector",
        comparator="exact_only (no entity/embedding/claim detection)",
        outcome="pu_rer restricted to semantic_paraphrase candidates",
        numerator="paraphrase candidates whose released text exposes the target",
        denominator="all evaluable semantic_paraphrase candidates",
        unit_of_analysis="candidate",
        pairing_unit="candidate_id",
        aggregation_level="per candidate, macro-average over candidates",
        interpretation=(
            "A difference isolates semantic detection only if the paired "
            "candidates are identical; exact_only additionally disables "
            "history/policy/monitoring, so C3 is the clean embedding test."
        ),
    ),
    ComparisonSpec(
        comparison_id="C3_embedding_ablation",
        research_question_id="RQ2",
        intervention="full_mvp",
        comparator="no_embedding (embedding detector disabled only)",
        outcome="pu_rer restricted to semantic_paraphrase candidates",
        numerator="paraphrase candidates whose released text exposes the target",
        denominator="all evaluable semantic_paraphrase candidates",
        unit_of_analysis="candidate",
        pairing_unit="candidate_id",
        aggregation_level="per candidate, macro-average over candidates",
        interpretation=(
            "A difference measures the embedding detector alone: the two "
            "conditions differ at detector.embedding_enabled only."
        ),
    ),
    ComparisonSpec(
        comparison_id="C4_history_vs_stateless",
        research_question_id="RQ3",
        intervention="full_mvp recipient history",
        comparator="stateless (history disabled only)",
        outcome="crr (compositional reconstruction rate)",
        numerator="reconstruction sequences whose final release exposes the target",
        denominator="all evaluable reconstruction sequences",
        unit_of_analysis="sequence",
        pairing_unit="sequence_id",
        aggregation_level="per sequence, macro-average over sequences",
        interpretation=(
            "A difference shows recipient history reduces multi-message "
            "reconstruction; the conditions differ at history.enabled only."
        ),
    ),
    ComparisonSpec(
        comparison_id="C5_monitoring_ladder_continuous_vs_bounded",
        research_question_id="RQ4",
        intervention="full_mvp continuous monitoring",
        comparator="one_time_monitoring (bounded, one recontamination round)",
        outcome="rr (recontamination rate) and probe recontamination recovery",
        numerator="cleaned agents recontaminated after post-forget collaboration",
        denominator="cleaned agents exposed to post-forget collaboration",
        unit_of_analysis="candidate",
        pairing_unit="candidate_id",
        aggregation_level="per candidate, macro-average over candidates",
        interpretation=(
            "Monitoring ladder step; conditions differ only in monitoring "
            "fields, so a difference is attributable to monitoring duration."
        ),
    ),
    ComparisonSpec(
        comparison_id="C6_monitoring_ladder_bounded_vs_none",
        research_question_id="RQ4",
        intervention="one_time_monitoring (bounded monitoring)",
        comparator="no_monitoring (monitoring disabled)",
        outcome="rr (recontamination rate) and probe recontamination recovery",
        numerator="cleaned agents recontaminated after post-forget collaboration",
        denominator="cleaned agents exposed to post-forget collaboration",
        unit_of_analysis="candidate",
        pairing_unit="candidate_id",
        aggregation_level="per candidate, macro-average over candidates",
        interpretation=(
            "Monitoring ladder step; a difference shows even bounded "
            "monitoring catches recontamination that no monitoring misses."
        ),
    ),
    ComparisonSpec(
        comparison_id="C7_rich_vs_binary_policy",
        research_question_id="RQ5",
        intervention="full_mvp rich trust-independent policy",
        comparator="binary_policy (allow/block only)",
        outcome="paired utility retention and utility false-block rate, "
        "conditional on comparable exposure (pu_rer)",
        numerator="legitimate candidates blocked (false block) / paired task successes",
        denominator="legitimate candidates paired against the no_firewall baseline",
        unit_of_analysis="candidate",
        pairing_unit="candidate_id",
        aggregation_level="per candidate, macro-average over candidates",
        interpretation=(
            "Utility may only be compared at comparable privacy protection: "
            "report exposure for both conditions alongside utility."
        ),
    ),
    ComparisonSpec(
        comparison_id="C8_trust_invariance",
        research_question_id="RQ6",
        intervention="candidate trust level (low/high)",
        comparator="same candidate message content across trust levels",
        outcome="per-trust exposure rate within one firewall condition",
        numerator="candidates whose released text exposes the target, by trust level",
        denominator="evaluable candidates within each trust level",
        unit_of_analysis="candidate",
        pairing_unit="candidate_id (identical content across trust levels)",
        aggregation_level="per trust level within condition",
        interpretation=(
            "Enforcement is trust-invariant when the per-trust rates match "
            "for identical candidate content; any gap is a policy failure, "
            "not a generator effect."
        ),
    ),
    ComparisonSpec(
        comparison_id="C9_generator_trust_effect",
        research_question_id="RQ7",
        intervention="trust level presented to the generating agent",
        comparator="no_firewall (pre-enforcement candidate text)",
        outcome="candidate-level exposure rate before firewall enforcement",
        numerator="candidates whose generated candidate text contains the target",
        denominator="all candidates generated under no_firewall",
        unit_of_analysis="candidate",
        pairing_unit="candidate_id",
        aggregation_level="per trust level under no_firewall",
        interpretation=(
            "Measures the generating agent's behavior only; firewall "
            "enforcement is absent, so differences are upstream of policy."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Declared research questions (remediation §2 minimum primary claims)
# ---------------------------------------------------------------------------

QUESTIONS: tuple[ResearchQuestion, ...] = (
    ResearchQuestion(
        question_id="RQ1",
        statement=(
            "Does the full ForgetFlow configuration reduce direct post-forget "
            "exposure relative to no firewall?"
        ),
        comparison_ids=("C1_full_vs_none",),
        scope="system-level enforcement effect",
    ),
    ResearchQuestion(
        question_id="RQ2",
        statement=(
            "Does semantic detection reduce semantic-paraphrase exposure "
            "relative to exact-only or no-embedding conditions?"
        ),
        comparison_ids=("C2_semantic_vs_exact", "C3_embedding_ablation"),
        scope="detector component effect",
    ),
    ResearchQuestion(
        question_id="RQ3",
        statement=(
            "Does recipient history reduce multi-message reconstruction "
            "relative to a stateless condition?"
        ),
        comparison_ids=("C4_history_vs_stateless",),
        scope="history component effect",
    ),
    ResearchQuestion(
        question_id="RQ4",
        statement=(
            "Does continuous monitoring reduce recontamination relative to "
            "bounded or absent monitoring?"
        ),
        comparison_ids=(
            "C5_monitoring_ladder_continuous_vs_bounded",
            "C6_monitoring_ladder_bounded_vs_none",
        ),
        scope="monitoring component effect",
    ),
    ResearchQuestion(
        question_id="RQ5",
        statement=(
            "Does a rich policy preserve task utility better than a binary "
            "allow/block policy at comparable privacy protection?"
        ),
        comparison_ids=("C7_rich_vs_binary_policy",),
        scope="policy component effect",
    ),
    ResearchQuestion(
        question_id="RQ6",
        statement=(
            "Is enforcement invariant to trust level after controlling for "
            "the candidate message presented to the firewall?"
        ),
        comparison_ids=("C8_trust_invariance",),
        scope="policy invariance property",
    ),
    ResearchQuestion(
        question_id="RQ7",
        statement=(
            "Does trust level alter the behavior of the message-generating "
            "agent before firewall enforcement?"
        ),
        comparison_ids=("C9_generator_trust_effect",),
        scope="pre-enforcement generator behavior",
    ),
)

# Every final table maps to at least one declared research question.
TABLE_QUESTION_MAP: dict[str, tuple[str, ...]] = {
    "table1_main_results": ("RQ1", "RQ2", "RQ3", "RQ4", "RQ5"),
    "table2_leakage_breakdown": ("RQ1",),
    "table3_parameter_sensitivity": ("RQ1",),
    "table4_statistical_comparisons": ("RQ1", "RQ2", "RQ3", "RQ4", "RQ5"),
    # Remediation §36: per-target-type and per-scenario results are a
    # declared final table; pooled summaries are secondary.
    "table5_target_type_results": ("RQ1",),
}


def protocol_as_dict() -> dict[str, Any]:
    """Serialize the protocol for embedding in study artifacts."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "population": POPULATION,
        "questions": [
            {
                "question_id": q.question_id,
                "statement": q.statement,
                "comparison_ids": list(q.comparison_ids),
                "scope": q.scope,
            }
            for q in QUESTIONS
        ],
        "comparisons": [{f.name: getattr(c, f.name) for f in fields(c)} for c in COMPARISONS],
        "table_question_map": {k: list(v) for k, v in TABLE_QUESTION_MAP.items()},
    }


def validate_protocol() -> list[str]:
    """Acceptance checks for the protocol itself (remediation §2).

    Returns a list of findings; an empty list means the protocol is
    internally consistent.
    """
    findings: list[str] = []
    question_ids = {q.question_id for q in QUESTIONS}
    comparison_ids = {c.comparison_id for c in COMPARISONS}

    if len(question_ids) != len(QUESTIONS):
        findings.append("duplicate question ids")
    if len(comparison_ids) != len(COMPARISONS):
        findings.append("duplicate comparison ids")

    for comparison in COMPARISONS:
        if comparison.research_question_id not in question_ids:
            findings.append(
                f"{comparison.comparison_id}: unknown question "
                f"{comparison.research_question_id!r}"
            )
        for f in fields(comparison):
            if not str(getattr(comparison, f.name)).strip():
                findings.append(f"{comparison.comparison_id}: empty field {f.name}")

    for question in QUESTIONS:
        if not question.comparison_ids:
            findings.append(f"{question.question_id}: no comparisons declared")
        for cid in question.comparison_ids:
            if cid not in comparison_ids:
                findings.append(f"{question.question_id}: unknown comparison {cid!r}")

    for table, qids in TABLE_QUESTION_MAP.items():
        if not qids:
            findings.append(f"{table}: maps to no research question")
        for qid in qids:
            if qid not in question_ids:
                findings.append(f"{table}: maps to unknown question {qid!r}")

    return findings
