"""Iteration 1: Assertion suite contracts and exposure classification.

This module implements:
- AssertionSuiteSummary: Top-level suite result contract
- AssertionCaseResult: Per-case actual-result schema
- classify_candidate_exposure: Canonical exposure classification function
- ExposureClassification: Structured exposure evidence
"""
from __future__ import annotations

from dataclasses import dataclass


# Iteration 1: Top-level assertion suite summary
@dataclass(frozen=True)
class AssertionSuiteSummary:
    """Top-level assertion suite result contract."""
    suite_type: str
    seed: int

    total_cases: int

    execution_completed: int
    execution_skipped: int
    execution_failed: int

    assertion_cases_passed: int
    assertion_cases_failed: int

    individual_assertions_passed: int
    individual_assertions_failed: int

    # Iteration 1: Add consistency failure tracking
    assertion_consistency_failures: int

    audit_failures: int
    suite_passed: bool

    @classmethod
    def from_results(cls, suite_type: str, seed: int, results: list[dict]) -> "AssertionSuiteSummary":
        """Build summary from case results."""
        total = len(results)
        completed = sum(1 for r in results if r.get("execution_status") == "completed")
        skipped = sum(1 for r in results if r.get("execution_status") == "skipped")
        failed = sum(1 for r in results if r.get("execution_status") == "failed")

        assertion_passed = sum(1 for r in results if r.get("assertion_passed", False))
        assertion_failed = total - assertion_passed

        # Count individual assertions
        ind_passed = 0
        ind_failed = 0
        for r in results:
            for assertion in r.get("assertions", []):
                if assertion.get("passed", False):
                    ind_passed += 1
                else:
                    ind_failed += 1

        audit_failures = sum(1 for r in results if r.get("audit_failed", False))

        # Iteration 1: Count consistency violations
        consistency_failures = sum(
            len(r.get("consistency_violations", []))
            for r in results
        )

        suite_passed = (
            assertion_failed == 0
            and ind_failed == 0
            and consistency_failures == 0
            and audit_failures == 0
            and completed == total
        )

        return cls(
            suite_type=suite_type,
            seed=seed,
            total_cases=total,
            execution_completed=completed,
            execution_skipped=skipped,
            execution_failed=failed,
            assertion_cases_passed=assertion_passed,
            assertion_cases_failed=assertion_failed,
            individual_assertions_passed=ind_passed,
            individual_assertions_failed=ind_failed,
            assertion_consistency_failures=consistency_failures,
            audit_failures=audit_failures,
            suite_passed=suite_passed,
        )


# Iteration 1: Per-case actual-result schema
@dataclass(frozen=True)
class AssertionCaseResult:
    """Per-case actual-result schema with complete evidence."""
    case_id: str
    execution_status: str
    assertion_passed: bool

    expected_exposure_class: str
    expected_action: str | None

    candidate_text: str
    released_text: str | None

    actual_candidate_exposure_class: str
    actual_released_exposure_class: str

    action: str | None
    reason_codes: tuple[str, ...]

    matched_forget_ids: tuple[str, ...]

    exact_score: float
    entity_score: float
    embedding_score: float
    proposition_relevant: bool
    proposition_entailed: bool
    proposition_score: float
    reconstruction_score: float

    message_delivered: bool
    history_updated: bool

    state_before: str
    state_after: str
    state_transition_reason: str | None

    task_success: bool
    assertion_failures: tuple[str, ...]

    # Additional evidence for debugging
    audit_failed: bool = False
    audit_errors: tuple[str, ...] = ()


# Iteration 1: Exposure classification
@dataclass(frozen=True)
class ExposureClassification:
    """Structured exposure evidence."""
    exposure_class: str
    target_ids: tuple[str, ...]
    evidence_source: str
    reason_codes: tuple[str, ...]


def classify_candidate_exposure(
    *,
    exact_score: float,
    entity_score: float,
    embedding_score: float,
    proposition_relevant: bool,
    proposition_entailed: bool,
    reconstruction_score: float,
    recontamination_attempt: bool,
    permitted_residual: bool,
    attack_request: bool,
    matched_forget_ids: tuple[str, ...],
) -> ExposureClassification:
    """Canonical exposure classification function.

    Precedence order (per spec):
    1. direct_exact
    2. direct_alias
    3. direct_claim
    4. direct_embedding
    5. fragment_reconstruction
    6. fact_chain_reconstruction
    7. recontamination
    8. permitted_residual
    9. attack_request
    10. none
    """
    # 1. Direct exact match
    if exact_score >= 1.0 and matched_forget_ids:
        return ExposureClassification(
            exposure_class="direct_exact",
            target_ids=matched_forget_ids,
            evidence_source="exact_detector",
            reason_codes=("EXACT_CANONICAL_MATCH",),
        )

    # 2. Direct alias match
    if entity_score > 0.0 and matched_forget_ids:
        return ExposureClassification(
            exposure_class="direct_alias",
            target_ids=matched_forget_ids,
            evidence_source="entity_detector",
            reason_codes=("ALIAS_MATCH",),
        )

    # 3. Direct claim match
    if proposition_entailed and matched_forget_ids:
        return ExposureClassification(
            exposure_class="direct_claim",
            target_ids=matched_forget_ids,
            evidence_source="claim_detector",
            reason_codes=("PROPOSITION_ENTAILED",),
        )

    # 4. Direct embedding match
    if embedding_score > 0.0 and matched_forget_ids:
        return ExposureClassification(
            exposure_class="direct_embedding",
            target_ids=matched_forget_ids,
            evidence_source="embedding_detector",
            reason_codes=("SEMANTIC_SIMILARITY",),
        )

    # 5. Fragment reconstruction
    if reconstruction_score > 0.0 and matched_forget_ids:
        return ExposureClassification(
            exposure_class="fragment_reconstruction",
            target_ids=matched_forget_ids,
            evidence_source="reconstruction_checker",
            reason_codes=("FRAGMENT_RECONSTRUCTION",),
        )

    # 6. Fact chain reconstruction (not implemented yet)
    # Skip for now

    # 7. Recontamination
    if recontamination_attempt:
        return ExposureClassification(
            exposure_class="recontamination",
            target_ids=matched_forget_ids if matched_forget_ids else (),
            evidence_source="recontamination_detector",
            reason_codes=("RECONTAMINATION_ATTEMPT",),
        )

    # 8. Permitted residual
    if permitted_residual:
        return ExposureClassification(
            exposure_class="permitted_residual",
            target_ids=(),
            evidence_source="policy_check",
            reason_codes=("PERMITTED_RESIDUAL",),
        )

    # 9. Attack request
    if attack_request:
        return ExposureClassification(
            exposure_class="attack_request",
            target_ids=(),
            evidence_source="speech_act_check",
            reason_codes=("ATTACK_REQUEST",),
        )

    # 10. None
    return ExposureClassification(
        exposure_class="none",
        target_ids=(),
        evidence_source="no_match",
        reason_codes=(),
    )


def classify_released_exposure(
    *,
    candidate_classification: ExposureClassification,
    message_delivered: bool,
    released_text: str | None,
) -> ExposureClassification:
    """Classify released exposure independently from candidate exposure.

    A blocked unsafe candidate can be direct_exact at candidate level
    and none at released level.
    """
    if not message_delivered or released_text is None:
        return ExposureClassification(
            exposure_class="none",
            target_ids=(),
            evidence_source="blocked_or_not_delivered",
            reason_codes=("NOT_DELIVERED",),
        )

    # If delivered, released exposure matches candidate exposure
    return candidate_classification
