"""Tests for audit invariant validation (Iteration 3 of repair spec)."""

from __future__ import annotations

import pytest

from marble.firewall.audit_validation import (
    AuditInvariantChecker,
    AuditViolation,
    validate_audit_invariants,
)
from marble.firewall.types import (
    DetectorResult,
    FirewallDecision,
    ForgetRecord,
    RecordDetectionEvidence,
)


def _make_record(
    forget_id: str = "F001",
    canonical_target: str = "0107",
) -> ForgetRecord:
    return ForgetRecord(
        forget_id=forget_id,
        canonical_target=canonical_target,
        target_type="credential",
        aliases=(),
        semantic_variants=(),
        permitted_residuals=(),
        active_from_turn=0,
    )


def _make_decision(
    action: str = "allow",
    released_text: str | None = "Hello",
    exact_score: float = 0.0,
    entity_score: float = 0.0,
    semantic_score: float = 0.0,
    reconstruction_score: float = 0.0,
    matched_forget_ids: tuple[str, ...] = (),
    reason_codes: tuple[str, ...] = (),
) -> FirewallDecision:
    det = DetectorResult(
        exact_score=exact_score,
        entity_score=entity_score,
        semantic_score=semantic_score,
        reconstruction_score=reconstruction_score,
        matched_forget_ids=matched_forget_ids,
        evidence=(),
    )
    return FirewallDecision(
        action=action,
        released_text=released_text,
        detector_result=det,
        reason_codes=reason_codes,
        policy_version="test",
        latency_ms=1.0,
    )


class TestReleasedAliasLeak:
    """Test invariant: released text should not contain undetected targets."""

    def test_released_alias_leak_detected(self) -> None:
        """Released text containing target should be flagged if not detected."""
        record = _make_record()
        checker = AuditInvariantChecker([record])

        # Released text contains target but it wasn't detected
        decision = _make_decision(
            action="allow",
            released_text="The credential is 0107.",
            matched_forget_ids=(),  # Not detected!
        )
        violations = checker.check_decision(decision, active_records=[record])

        assert len(violations) == 1
        assert violations[0].violation_type == "RELEASED_ALIAS_LEAK"

    def test_no_violation_when_target_detected(self) -> None:
        """No violation when target is properly detected."""
        record = _make_record()
        checker = AuditInvariantChecker([record])

        decision = _make_decision(
            action="block",
            released_text=None,
            exact_score=1.0,
            matched_forget_ids=("F001",),
        )
        violations = checker.check_decision(decision, active_records=[record])

        # Block with no released text is fine
        assert len(violations) == 0

    def test_spoken_digit_leak_detected(self) -> None:
        """Spoken digit form should also be caught."""
        record = _make_record()
        checker = AuditInvariantChecker([record])

        decision = _make_decision(
            action="allow",
            released_text="The code is zero one zero seven.",
            matched_forget_ids=(),
        )
        violations = checker.check_decision(decision, active_records=[record])

        assert len(violations) == 1
        assert violations[0].violation_type == "RELEASED_ALIAS_LEAK"


class TestUnsafePermittedResidual:
    """Test invariant: permitted residual should not have active detector matches."""

    def test_unsafe_permitted_residual(self) -> None:
        """Abstract with exact match should be flagged."""
        decision = _make_decision(
            action="abstract",
            released_text="The credential must be replaced.",
            exact_score=1.0,  # Still matches!
            matched_forget_ids=("F001",),
        )
        violations = AuditInvariantChecker().check_decision(decision)

        assert len(violations) == 1
        assert violations[0].violation_type == "UNSAFE_PERMITTED_RESIDUAL"

    def test_safe_permitted_residual(self) -> None:
        """Abstract with no matches should be fine."""
        decision = _make_decision(
            action="abstract",
            released_text="The credential must be replaced.",
            exact_score=0.0,
            entity_score=0.0,
            matched_forget_ids=(),
        )
        violations = AuditInvariantChecker().check_decision(decision)

        assert len(violations) == 0


class TestTargetlessReconstruction:
    """Test invariant: reconstruction score > 0 requires matched forget_ids."""

    def test_targetless_reconstruction_flagged(self) -> None:
        """Reconstruction without targets should be flagged."""
        decision = _make_decision(
            reconstruction_score=1.0,
            matched_forget_ids=(),  # No targets!
        )
        violations = AuditInvariantChecker().check_decision(decision)

        assert len(violations) == 1
        assert violations[0].violation_type == "TARGETLESS_RECONSTRUCTION"

    def test_targeted_reconstruction_ok(self) -> None:
        """Reconstruction with targets should be fine."""
        decision = _make_decision(
            reconstruction_score=1.0,
            matched_forget_ids=("F001",),
        )
        violations = AuditInvariantChecker().check_decision(decision)

        assert len(violations) == 0


class TestBlockConsistency:
    """Test block action consistency.

    Note: FirewallDecision.__post_init__ already prevents creating invalid
    decisions. These tests verify the audit validator catches edge cases
    that might slip through via other code paths.
    """

    def test_block_with_released_text_caught_by_post_init(self) -> None:
        """FirewallDecision prevents block with released_text."""
        with pytest.raises(ValueError, match="block action requires"):
            _make_decision(
                action="block",
                released_text="Some text",
            )

    def test_allow_without_released_text_caught_by_post_init(self) -> None:
        """FirewallDecision prevents allow without released_text."""
        with pytest.raises(ValueError, match="requires non-empty"):
            _make_decision(
                action="allow",
                released_text=None,
            )


class TestValidateAuditInvariants:
    """Test the validate_audit_invariants function."""

    def test_all_pass(self) -> None:
        """All decisions pass validation."""
        decisions = [
            _make_decision(action="allow", released_text="Hello"),
            _make_decision(action="block", released_text=None, exact_score=1.0, matched_forget_ids=("F001",)),
        ]
        passed, violations = validate_audit_invariants(decisions)
        assert passed is True
        assert len(violations) == 0

    def test_some_fail(self) -> None:
        """Some decisions fail validation."""
        decisions = [
            _make_decision(action="allow", released_text="Hello"),
            _make_decision(reconstruction_score=1.0, matched_forget_ids=()),  # Targetless
        ]
        passed, violations = validate_audit_invariants(decisions)
        assert passed is False
        assert len(violations) == 1
