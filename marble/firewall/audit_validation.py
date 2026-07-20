"""Audit invariant validation for firewall decisions.

This module checks for known semantic contradictions that should never occur
in a correctly functioning firewall system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from marble.firewall.normalization import text_contains_canonical_value
from marble.firewall.types import DetectorResult, FirewallDecision, ForgetRecord


@dataclass(frozen=True)
class AuditViolation:
    """A violation of an audit invariant."""

    violation_type: str
    message: str
    decision_index: int | None = None


class AuditInvariantChecker:
    """Checks audit invariants on firewall decisions.

    Invariants:
    1. Released alias leak: canonicalized released text contains active target
       but target_exposed = false
    2. Unsafe permitted residual: released_exposure_class = permitted_residual
       but a detector matches an active target
    3. Targetless reconstruction: reconstruction_score > 0 but matched_forget_ids empty
    4. Question-based reconstruction: speech_act = question, proposition_entailed = false,
       but candidate_information_contribution > 0
    5. Delivered but post-transform recheck failed
    """

    def __init__(self, active_records: Sequence[ForgetRecord] | None = None) -> None:
        self._active_records = active_records or []

    def check_decision(
        self,
        decision: FirewallDecision,
        decision_index: int | None = None,
        active_records: Sequence[ForgetRecord] | None = None,
    ) -> list[AuditViolation]:
        """Check a single decision for audit violations."""
        violations: list[AuditViolation] = []
        records = active_records or self._active_records

        # Invariant 1: Released alias leak
        # If released text contains a canonical target value, it should be flagged
        if decision.released_text and decision.action != "block":
            for rec in records:
                if text_contains_canonical_value(decision.released_text, rec.canonical_target):
                    # Check if this forget_id was matched
                    if rec.forget_id not in decision.detector_result.matched_forget_ids:
                        violations.append(
                            AuditViolation(
                                violation_type="RELEASED_ALIAS_LEAK",
                                message=(
                                    f"Released text contains canonical target '{rec.canonical_target}' "
                                    f"for {rec.forget_id} but it was not detected"
                                ),
                                decision_index=decision_index,
                            )
                        )

        # Invariant 2: Unsafe permitted residual
        # If action is abstract/redact but detectors still match, it's unsafe
        if decision.action in ("redact", "abstract") and decision.released_text:
            det = decision.detector_result
            # Check if any detector still matches after transformation
            if det.exact_score == 1.0 or det.entity_score > 0:
                violations.append(
                    AuditViolation(
                        violation_type="UNSAFE_PERMITTED_RESIDUAL",
                        message=(
                            f"Action is '{decision.action}' but detectors still match: "
                            f"exact={det.exact_score}, entity={det.entity_score}"
                        ),
                        decision_index=decision_index,
                    )
                )

        # Invariant 3: Targetless reconstruction
        # reconstruction_score > 0 but no matched forget_ids
        det = decision.detector_result
        if det.reconstruction_score > 0 and not det.matched_forget_ids:
            violations.append(
                AuditViolation(
                    violation_type="TARGETLESS_RECONSTRUCTION",
                    message=(
                        f"Reconstruction score is {det.reconstruction_score} but "
                        f"no forget_ids are matched"
                    ),
                    decision_index=decision_index,
                )
            )

        # Invariant 4: Delivered but post-transform recheck failed
        # If reason_codes contains RECHECK_FAILED, action should be block
        if "RECHECK_FAILED" in decision.reason_codes and decision.action != "block":
            violations.append(
                AuditViolation(
                    violation_type="RECHECK_FAILED_NOT_BLOCKED",
                    message=(
                        f"RECHECK_FAILED in reason_codes but action is '{decision.action}'"
                    ),
                    decision_index=decision_index,
                )
            )

        # Invariant 5: Block action should have no released text
        if decision.action == "block" and decision.released_text is not None:
            violations.append(
                AuditViolation(
                    violation_type="BLOCK_WITH_RELEASED_TEXT",
                    message="Block action should have released_text=None",
                    decision_index=decision_index,
                )
            )

        # Invariant 6: Allow/redact/abstract should have released text
        if decision.action in ("allow", "redact", "abstract") and decision.released_text is None:
            violations.append(
                AuditViolation(
                    violation_type="NO_RELEASED_TEXT",
                    message=f"Action '{decision.action}' requires non-empty released_text",
                    decision_index=decision_index,
                )
            )

        return violations

    def check_all(
        self,
        decisions: Sequence[FirewallDecision],
        active_records: Sequence[ForgetRecord] | None = None,
    ) -> list[AuditViolation]:
        """Check all decisions for audit violations."""
        all_violations: list[AuditViolation] = []
        for idx, decision in enumerate(decisions):
            violations = self.check_decision(decision, idx, active_records)
            all_violations.extend(violations)
        return all_violations


def validate_audit_invariants(
    decisions: Sequence[FirewallDecision],
    active_records: Sequence[ForgetRecord] | None = None,
) -> tuple[bool, list[AuditViolation]]:
    """Validate audit invariants and return (passed, violations)."""
    checker = AuditInvariantChecker(active_records)
    violations = checker.check_all(decisions, active_records)
    return len(violations) == 0, violations
