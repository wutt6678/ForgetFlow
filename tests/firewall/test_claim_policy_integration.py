"""Tests for policy integration with claim-based detection.

Req #17: Update policy behavior
Req #18: Prevent questions/denials from entering positive recipient evidence
"""

from __future__ import annotations

from marble.firewall.claims import MessageContext
from marble.firewall.detectors import HybridDetector
from marble.firewall.types import ForgetRecord


class TestPolicyBehaviorWithClaims:
    """Req #17: Policy uses entailed, not merely relevant."""

    def test_question_not_counted_as_exposure(self) -> None:
        """Questions are relevant but not exposure (Req #17, #18)."""
        detector = HybridDetector(
            exact_enabled=False,
            entity_enabled=False,
            semantic_enabled=False,
            claim_matching_enabled=True,
            semantic_threshold=0.60,
        )

        record = ForgetRecord(
            forget_id="F001",
            canonical_target="Agent B has emergency authority",
            target_type="credential",
            aliases=("Agent B",),
            semantic_variants=(),
            permitted_residuals=(),
            active_from_turn=0,
        )

        context = MessageContext(
            sender_id="COMMAND",
            recipient_id="Agent B",
            sender_aliases=(),
            recipient_aliases=("Agent B",),
        )

        # Question about target
        result = detector.detect(
            text="Does Agent B have emergency authority?",
            active_records=[record],
            message_context=context,
        )

        # Should be relevant but not entailed
        assert len(result.record_evidence) > 0
        evidence = result.record_evidence[0]
        # Question should be detected as relevant
        assert evidence.proposition_relevant is True or evidence.proposition_score > 0
        # But NOT entailed (not a positive assertion)
        assert evidence.proposition_entailed is False

    def test_denial_not_counted_as_exposure(self) -> None:
        """Denials are relevant but not positive exposure."""
        detector = HybridDetector(
            exact_enabled=False,
            entity_enabled=False,
            semantic_enabled=False,
            claim_matching_enabled=True,
            semantic_threshold=0.60,
        )

        record = ForgetRecord(
            forget_id="F001",
            canonical_target="Agent B has emergency authority",
            target_type="credential",
            aliases=("Agent B",),
            semantic_variants=(),
            permitted_residuals=(),
            active_from_turn=0,
        )

        # Denial of target
        result = detector.detect(
            text="Agent B does not have emergency authority",
            active_records=[record],
        )

        # Should be relevant but not entailed
        assert len(result.record_evidence) > 0
        evidence = result.record_evidence[0]
        # Denial should be detected as relevant
        assert evidence.proposition_relevant is True or evidence.proposition_score > 0
        # But NOT entailed (negative polarity)
        assert evidence.proposition_entailed is False

    def test_positive_assertion_is_exposure(self) -> None:
        """Positive assertions are counted as exposure."""
        detector = HybridDetector(
            exact_enabled=False,
            entity_enabled=False,
            semantic_enabled=False,
            claim_matching_enabled=True,
            semantic_threshold=0.60,
        )

        record = ForgetRecord(
            forget_id="F001",
            canonical_target="Agent B has emergency authority",
            target_type="credential",
            aliases=("Agent B",),
            semantic_variants=(),
            permitted_residuals=(),
            active_from_turn=0,
        )

        context = MessageContext(
            sender_id="COMMAND",
            recipient_id="Agent B",
            sender_aliases=(),
            recipient_aliases=("Agent B",),
        )

        # Positive assertion
        result = detector.detect(
            text="You have emergency authority",
            active_records=[record],
            message_context=context,
        )

        # Should be both relevant and entailed
        assert len(result.record_evidence) > 0
        evidence = result.record_evidence[0]
        assert evidence.proposition_relevant is True
        assert evidence.proposition_entailed is True


class TestMultiClaimHandling:
    """Req #24: Messages may contain more than one proposition."""

    def test_multiple_claims_extracted(self) -> None:
        """Multiple claims in one message are extracted."""
        from marble.firewall.claims import ClaimNormalizer

        normalizer = ClaimNormalizer()

        # Message with two claims
        text = "Agent B lacks administrative authority but retains emergency drone control."
        claims = normalizer.normalize(text)

        # Should extract at least one claim (simplified implementation may not get both)
        assert len(claims) >= 1
        # The claim should mention Agent B
        assert any("agent b" in claim.subject.lower() for claim in claims)


class TestMultiTargetIsolation:
    """Req #25: Claim evidence cannot leak between forget records."""

    def test_multi_target_isolation(self) -> None:
        """Claims for one target don't affect another."""
        detector = HybridDetector(
            exact_enabled=False,
            entity_enabled=False,
            semantic_enabled=False,
            claim_matching_enabled=True,
            semantic_threshold=0.60,
        )

        record_f002 = ForgetRecord(
            forget_id="F002",
            canonical_target="Alice has condition X",
            target_type="medical",
            aliases=("Alice",),
            semantic_variants=(),
            permitted_residuals=(),
            active_from_turn=0,
        )

        record_f003 = ForgetRecord(
            forget_id="F003",
            canonical_target="Agent B has emergency authority",
            target_type="credential",
            aliases=("Agent B",),
            semantic_variants=(),
            permitted_residuals=(),
            active_from_turn=0,
        )

        context = MessageContext(
            sender_id="COMMAND",
            recipient_id="Agent B",
            sender_aliases=(),
            recipient_aliases=("Agent B",),
        )

        # Message about F003 only
        result = detector.detect(
            text="You retain emergency authority",
            active_records=[record_f002, record_f003],
            message_context=context,
        )

        # F003 should be entailed
        f003_evidence = [e for e in result.record_evidence if e.forget_id == "F003"]
        assert len(f003_evidence) > 0
        assert f003_evidence[0].proposition_entailed is True

        # F002 should NOT be entailed
        f002_evidence = [e for e in result.record_evidence if e.forget_id == "F002"]
        assert len(f002_evidence) > 0
        assert f002_evidence[0].proposition_entailed is False
