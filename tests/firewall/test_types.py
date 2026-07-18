"""Tests for marble.firewall.types."""

import pytest

from marble.firewall.types import (
    ContaminationStatus,
    DetectorResult,
    FirewallDecision,
    ForgetRecord,
    MessageEnvelope,
    RecipientHistoryItem,
)

# ── ForgetRecord ──────────────────────────────────────────────


class TestForgetRecord:
    def test_valid_creation(self) -> None:
        rec = ForgetRecord(
            forget_id="F001",
            canonical_target="0107",
            target_type="credential",
            aliases=("warehouse access code",),
            semantic_variants=("the code for January seventh",),
            permitted_residuals=("Request a new credential.",),
            active_from_turn=2,
        )
        assert rec.forget_id == "F001"
        assert rec.active_from_turn == 2
        assert rec.scoped_agent_ids == ()

    def test_to_dict(self) -> None:
        rec = ForgetRecord(
            forget_id="F001",
            canonical_target="0107",
            target_type="credential",
            aliases=(),
            semantic_variants=(),
            permitted_residuals=(),
            active_from_turn=0,
        )
        d = rec.to_dict()
        assert d["forget_id"] == "F001"
        assert d["canonical_target"] == "0107"

    def test_empty_forget_id_raises(self) -> None:
        with pytest.raises(ValueError, match="forget_id"):
            ForgetRecord(
                forget_id="",
                canonical_target="0107",
                target_type="credential",
                aliases=(),
                semantic_variants=(),
                permitted_residuals=(),
                active_from_turn=0,
            )

    def test_empty_canonical_target_raises(self) -> None:
        with pytest.raises(ValueError, match="canonical_target"):
            ForgetRecord(
                forget_id="F001",
                canonical_target="",
                target_type="credential",
                aliases=(),
                semantic_variants=(),
                permitted_residuals=(),
                active_from_turn=0,
            )

    def test_negative_turn_raises(self) -> None:
        with pytest.raises(ValueError, match="active_from_turn"):
            ForgetRecord(
                forget_id="F001",
                canonical_target="0107",
                target_type="credential",
                aliases=(),
                semantic_variants=(),
                permitted_residuals=(),
                active_from_turn=-1,
            )

    def test_empty_alias_raises(self) -> None:
        with pytest.raises(ValueError, match="aliases"):
            ForgetRecord(
                forget_id="F001",
                canonical_target="0107",
                target_type="credential",
                aliases=("",),
                semantic_variants=(),
                permitted_residuals=(),
                active_from_turn=0,
            )

    def test_empty_variant_raises(self) -> None:
        with pytest.raises(ValueError, match="semantic_variants"):
            ForgetRecord(
                forget_id="F001",
                canonical_target="0107",
                target_type="credential",
                aliases=(),
                semantic_variants=("",),
                permitted_residuals=(),
                active_from_turn=0,
            )

    def test_frozen(self) -> None:
        rec = ForgetRecord(
            forget_id="F001",
            canonical_target="0107",
            target_type="credential",
            aliases=(),
            semantic_variants=(),
            permitted_residuals=(),
            active_from_turn=0,
        )
        with pytest.raises(AttributeError):
            rec.forget_id = "F002"  # type: ignore[misc]


# ── MessageEnvelope ───────────────────────────────────────────


class TestMessageEnvelope:
    def test_valid_creation(self) -> None:
        env = MessageEnvelope(
            message_id="m1",
            episode_id="ep1",
            session_id="s1",
            turn_id=0,
            sender_id="CK",
            recipient_id="SK",
            raw_text="Hello",
            trust_level="high",
        )
        assert env.trust_level == "high"

    def test_all_trust_levels(self) -> None:
        for level in ("low", "default", "high"):
            env = MessageEnvelope(
                message_id="m1",
                episode_id="ep1",
                session_id="s1",
                turn_id=0,
                sender_id="A",
                recipient_id="B",
                raw_text="hi",
                trust_level=level,
            )
            assert env.trust_level == level

    def test_empty_message_id_raises(self) -> None:
        with pytest.raises(ValueError, match="message_id"):
            MessageEnvelope(
                message_id="",
                episode_id="ep1",
                session_id="s1",
                turn_id=0,
                sender_id="A",
                recipient_id="B",
                raw_text="hi",
                trust_level="low",
            )

    def test_negative_turn_raises(self) -> None:
        with pytest.raises(ValueError, match="turn_id"):
            MessageEnvelope(
                message_id="m1",
                episode_id="ep1",
                session_id="s1",
                turn_id=-1,
                sender_id="A",
                recipient_id="B",
                raw_text="hi",
                trust_level="low",
            )

    def test_invalid_trust_raises(self) -> None:
        with pytest.raises(ValueError, match="trust_level"):
            MessageEnvelope(
                message_id="m1",
                episode_id="ep1",
                session_id="s1",
                turn_id=0,
                sender_id="A",
                recipient_id="B",
                raw_text="hi",
                trust_level="ultra",
            )

    def test_empty_sender_raises(self) -> None:
        with pytest.raises(ValueError, match="sender_id"):
            MessageEnvelope(
                message_id="m1",
                episode_id="ep1",
                session_id="s1",
                turn_id=0,
                sender_id="",
                recipient_id="B",
                raw_text="hi",
                trust_level="low",
            )


# ── DetectorResult ────────────────────────────────────────────


class TestDetectorResult:
    def test_valid(self) -> None:
        dr = DetectorResult(
            exact_score=1.0,
            entity_score=0.0,
            semantic_score=0.5,
            reconstruction_score=0.0,
            matched_forget_ids=("F001",),
            evidence=("0107",),
        )
        assert dr.exact_score == 1.0

    def test_score_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="exact_score"):
            DetectorResult(
                exact_score=1.5,
                entity_score=0.0,
                semantic_score=0.0,
                reconstruction_score=0.0,
                matched_forget_ids=(),
                evidence=(),
            )

    def test_negative_score_raises(self) -> None:
        with pytest.raises(ValueError, match="semantic_score"):
            DetectorResult(
                exact_score=0.0,
                entity_score=0.0,
                semantic_score=-0.1,
                reconstruction_score=0.0,
                matched_forget_ids=(),
                evidence=(),
            )


# ── FirewallDecision ──────────────────────────────────────────


class TestFirewallDecision:
    def _det(self) -> DetectorResult:
        return DetectorResult(
            exact_score=0.0,
            entity_score=0.0,
            semantic_score=0.0,
            reconstruction_score=0.0,
            matched_forget_ids=(),
            evidence=(),
        )

    def test_allow(self) -> None:
        fd = FirewallDecision(
            action="allow",
            released_text="Hello",
            detector_result=self._det(),
            reason_codes=(),
            policy_version="v1",
            latency_ms=1.0,
        )
        assert fd.action == "allow"

    def test_block_requires_none(self) -> None:
        with pytest.raises(ValueError, match="block"):
            FirewallDecision(
                action="block",
                released_text="still here",
                detector_result=self._det(),
                reason_codes=(),
                policy_version="v1",
                latency_ms=1.0,
            )

    def test_block_with_none_ok(self) -> None:
        fd = FirewallDecision(
            action="block",
            released_text=None,
            detector_result=self._det(),
            reason_codes=("EXACT",),
            policy_version="v1",
            latency_ms=1.0,
        )
        assert fd.released_text is None

    def test_allow_requires_text(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            FirewallDecision(
                action="allow",
                released_text=None,
                detector_result=self._det(),
                reason_codes=(),
                policy_version="v1",
                latency_ms=1.0,
            )

    def test_negative_latency_raises(self) -> None:
        with pytest.raises(ValueError, match="latency"):
            FirewallDecision(
                action="allow",
                released_text="ok",
                detector_result=self._det(),
                reason_codes=(),
                policy_version="v1",
                latency_ms=-1.0,
            )


# ── RecipientHistoryItem ──────────────────────────────────────


class TestRecipientHistoryItem:
    def test_creation(self) -> None:
        item = RecipientHistoryItem(
            message_id="m1",
            turn_id=3,
            sender_id="CK",
            released_text=" sanitized text ",
        )
        assert item.turn_id == 3

    def test_to_dict(self) -> None:
        item = RecipientHistoryItem(
            message_id="m1",
            turn_id=0,
            sender_id="A",
            released_text="hello",
        )
        d = item.to_dict()
        assert d["sender_id"] == "A"


# ── ContaminationStatus ───────────────────────────────────────


class TestContaminationStatus:
    def test_values(self) -> None:
        assert ContaminationStatus.UNKNOWN == "unknown"
        assert ContaminationStatus.CONTAMINATED == "contaminated"
        assert ContaminationStatus.CLEAN == "clean"
        assert ContaminationStatus.VERIFIED == "verified"
        assert ContaminationStatus.AT_RISK == "at_risk"
        assert ContaminationStatus.RECONTAMINATED == "recontaminated"

    def test_is_str(self) -> None:
        assert isinstance(ContaminationStatus.CLEAN, str)


# ── r7: Record Evidence Completeness Invariant ──────────────


class TestRecordEvidenceCompleteness:
    """r10: Regression tests for validate_record_evidence_completeness."""

    def test_valid_evidence_passes(self) -> None:
        """Matched IDs with corresponding evidence entries pass."""
        from marble.firewall.types import (
            DetectorResult,
            RecordDetectionEvidence,
            validate_record_evidence_completeness,
        )

        dr = DetectorResult(
            exact_score=1.0,
            entity_score=0.0,
            semantic_score=0.0,
            reconstruction_score=0.0,
            matched_forget_ids=("F001",),
            evidence=("EXACT",),
            record_evidence=(
                RecordDetectionEvidence(
                    forget_id="F001",
                    exact_score=1.0,
                    entity_score=0.0,
                    semantic_score=0.0,
                    reconstruction_score=0.0,
                    matched=True,
                ),
            ),
        )
        # Should not raise
        validate_record_evidence_completeness(dr)

    def test_missing_evidence_for_matched_id_fails(self) -> None:
        """Matched ID without evidence entry fails."""
        from marble.firewall.types import (
            DetectorResult,
            validate_record_evidence_completeness,
        )

        dr = DetectorResult(
            exact_score=1.0,
            entity_score=0.0,
            semantic_score=0.0,
            reconstruction_score=0.0,
            matched_forget_ids=("F001",),
            evidence=("EXACT",),
            record_evidence=(),  # No evidence
        )
        import pytest

        with pytest.raises(ValueError, match="Missing record evidence"):
            validate_record_evidence_completeness(dr)

    def test_matched_false_for_matched_id_fails(self) -> None:
        """Matched ID with matched=False in evidence fails."""
        from marble.firewall.types import (
            DetectorResult,
            RecordDetectionEvidence,
            validate_record_evidence_completeness,
        )

        dr = DetectorResult(
            exact_score=1.0,
            entity_score=0.0,
            semantic_score=0.0,
            reconstruction_score=0.0,
            matched_forget_ids=("F001",),
            evidence=("EXACT",),
            record_evidence=(
                RecordDetectionEvidence(
                    forget_id="F001",
                    exact_score=1.0,
                    entity_score=0.0,
                    semantic_score=0.0,
                    reconstruction_score=0.0,
                    matched=False,  # Wrong!
                ),
            ),
        )
        import pytest

        with pytest.raises(ValueError, match="matched=False"):
            validate_record_evidence_completeness(dr)

    def test_unmatched_id_with_matched_true_fails(self) -> None:
        """Unmatched ID with matched=True in evidence fails."""
        from marble.firewall.types import (
            DetectorResult,
            RecordDetectionEvidence,
            validate_record_evidence_completeness,
        )

        dr = DetectorResult(
            exact_score=0.0,
            entity_score=0.0,
            semantic_score=0.0,
            reconstruction_score=0.0,
            matched_forget_ids=(),  # No matched IDs
            evidence=(),
            record_evidence=(
                RecordDetectionEvidence(
                    forget_id="F001",
                    exact_score=0.0,
                    entity_score=0.0,
                    semantic_score=0.0,
                    reconstruction_score=0.0,
                    matched=True,  # Wrong!
                ),
            ),
        )
        import pytest

        with pytest.raises(ValueError, match="matched=True"):
            validate_record_evidence_completeness(dr)

    def test_duplicate_evidence_fails(self) -> None:
        """Duplicate evidence entries fail."""
        from marble.firewall.types import (
            DetectorResult,
            RecordDetectionEvidence,
            validate_record_evidence_completeness,
        )

        ev = RecordDetectionEvidence(
            forget_id="F001",
            exact_score=1.0,
            entity_score=0.0,
            semantic_score=0.0,
            reconstruction_score=0.0,
            matched=True,
        )
        dr = DetectorResult(
            exact_score=1.0,
            entity_score=0.0,
            semantic_score=0.0,
            reconstruction_score=0.0,
            matched_forget_ids=("F001",),
            evidence=("EXACT",),
            record_evidence=(ev, ev),  # Duplicate!
        )
        import pytest

        with pytest.raises(ValueError, match="Duplicate"):
            validate_record_evidence_completeness(dr)

    def test_unmatched_id_with_matched_false_passes(self) -> None:
        """Unmatched ID with matched=False in evidence passes."""
        from marble.firewall.types import (
            DetectorResult,
            RecordDetectionEvidence,
            validate_record_evidence_completeness,
        )

        dr = DetectorResult(
            exact_score=0.0,
            entity_score=0.0,
            semantic_score=0.0,
            reconstruction_score=0.0,
            matched_forget_ids=(),
            evidence=(),
            record_evidence=(
                RecordDetectionEvidence(
                    forget_id="F001",
                    exact_score=0.0,
                    entity_score=0.0,
                    semantic_score=0.0,
                    reconstruction_score=0.0,
                    matched=False,  # Correct - not matched
                ),
            ),
        )
        # Should not raise
        validate_record_evidence_completeness(dr)
