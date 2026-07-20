"""Tests for release validation gates (Iteration 10 of repair spec)."""

from __future__ import annotations

import pytest

from marble.firewall.audit_validation import (
    ReleaseStatus,
    validate_release,
)
from marble.firewall.types import DetectorResult, FirewallDecision, ForgetRecord


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
    released_text: str | None = "safe message",
    exact_score: float = 0.0,
    entity_score: float = 0.0,
    semantic_score: float = 0.0,
    reconstruction_score: float = 0.0,
    matched_forget_ids: tuple[str, ...] = (),
    reason_codes: tuple[str, ...] = (),
) -> FirewallDecision:
    return FirewallDecision(
        action=action,
        released_text=released_text,
        detector_result=DetectorResult(
            exact_score=exact_score,
            entity_score=entity_score,
            semantic_score=semantic_score,
            reconstruction_score=reconstruction_score,
            matched_forget_ids=matched_forget_ids,
            evidence=(),
            record_evidence=(),
        ),
        reason_codes=reason_codes,
        policy_version="test",
        latency_ms=0.0,
    )


class TestReleaseValidationGates:
    """Test release validation with semantic no-go gates."""

    def test_release_candidate_no_violations(self) -> None:
        """Clean decision should be RELEASE_CANDIDATE."""
        decision = _make_decision(
            action="allow",
            released_text="safe operational instruction",
            exact_score=0.0,
            entity_score=0.0,
            semantic_score=0.0,
            reconstruction_score=0.0,
        )

        result = validate_release(decision)

        assert result.status == ReleaseStatus.RELEASE_CANDIDATE
        assert result.passed is True
        assert result.no_go_gates == ()
        assert result.warnings == ()

    def test_diagnostic_only_exact_match(self) -> None:
        """Exact match should be DIAGNOSTIC_ONLY."""
        decision = _make_decision(
            action="block",
            released_text=None,
            exact_score=1.0,
            matched_forget_ids=("F001",),
        )

        result = validate_release(decision, [_make_record()])

        assert result.status == ReleaseStatus.DIAGNOSTIC_ONLY
        assert result.passed is False
        assert "EXACT_MATCH_DETECTED" in result.no_go_gates

    def test_diagnostic_only_alias_match(self) -> None:
        """Alias match should be DIAGNOSTIC_ONLY."""
        decision = _make_decision(
            action="block",
            released_text=None,
            entity_score=0.8,
            matched_forget_ids=("F001",),
        )

        result = validate_release(decision, [_make_record()])

        assert result.status == ReleaseStatus.DIAGNOSTIC_ONLY
        assert result.passed is False
        assert "ALIAS_MATCH_DETECTED" in result.no_go_gates

    def test_diagnostic_only_high_reconstruction(self) -> None:
        """High reconstruction score should be DIAGNOSTIC_ONLY."""
        decision = _make_decision(
            action="allow",
            released_text="01 07",
            reconstruction_score=0.8,
            matched_forget_ids=("F001",),
        )

        result = validate_release(decision, [_make_record()])

        assert result.status == ReleaseStatus.DIAGNOSTIC_ONLY
        assert result.passed is False
        assert "HIGH_RECONSTRUCTION_SCORE" in result.no_go_gates

    def test_diagnostic_only_recheck_failed(self) -> None:
        """RECHECK_FAILED should be DIAGNOSTIC_ONLY."""
        decision = _make_decision(
            action="block",
            released_text=None,
            reason_codes=("RECHECK_FAILED",),
        )

        result = validate_release(decision)

        assert result.status == ReleaseStatus.DIAGNOSTIC_ONLY
        assert result.passed is False
        assert "RECHECK_FAILED" in result.no_go_gates

    def test_research_valid_high_embedding(self) -> None:
        """High embedding similarity should be RESEARCH_VALID."""
        decision = _make_decision(
            action="allow",
            released_text="semantically similar text",
            semantic_score=0.8,
        )

        result = validate_release(decision)

        assert result.status == ReleaseStatus.RESEARCH_VALID
        assert result.passed is True
        assert "HIGH_EMBEDDING_SIMILARITY" in result.warnings

    def test_release_candidate_with_low_embedding(self) -> None:
        """Low embedding similarity should still be RELEASE_CANDIDATE."""
        decision = _make_decision(
            action="allow",
            released_text="unrelated text",
            semantic_score=0.3,
        )

        result = validate_release(decision)

        assert result.status == ReleaseStatus.RELEASE_CANDIDATE
        assert result.passed is True
        assert result.warnings == ()

    def test_canonical_value_leak(self) -> None:
        """Canonical value leak should be DIAGNOSTIC_ONLY."""
        record = _make_record(canonical_target="0107")
        decision = _make_decision(
            action="allow",
            released_text="the code is zero one zero seven",
        )

        result = validate_release(decision, [record])

        assert result.status == ReleaseStatus.DIAGNOSTIC_ONLY
        assert result.passed is False
        assert any("CANONICAL_VALUE_LEAK" in gate for gate in result.no_go_gates)

    def test_multiple_no_go_gates(self) -> None:
        """Multiple violations should all be reported."""
        decision = _make_decision(
            action="block",
            released_text=None,
            exact_score=1.0,
            entity_score=0.8,
            reconstruction_score=0.8,
            matched_forget_ids=("F001",),
            reason_codes=("RECHECK_FAILED",),
        )

        result = validate_release(decision, [_make_record()])

        assert result.status == ReleaseStatus.DIAGNOSTIC_ONLY
        assert result.passed is False
        assert "EXACT_MATCH_DETECTED" in result.no_go_gates
        assert "ALIAS_MATCH_DETECTED" in result.no_go_gates
        assert "HIGH_RECONSTRUCTION_SCORE" in result.no_go_gates
        assert "RECHECK_FAILED" in result.no_go_gates
