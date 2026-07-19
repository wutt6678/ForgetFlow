"""Validation suite for claim-based detection.

Req #30: Validation suite
Req #34: Deterministic validation tests
"""

from __future__ import annotations

from marble.firewall.claims import MessageContext
from marble.firewall.detectors import HybridDetector
from marble.firewall.types import ForgetRecord


class TestClaimValidationSuite:
    """Req #30: Validation suite for claim-based detection."""

    def test_pronoun_resolution_validation(self) -> None:
        """Pronoun resolution improves detection accuracy."""
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

        # "You" should resolve to Agent B
        result = detector.detect(
            text="You have emergency authority",
            active_records=[record],
            message_context=context,
        )

        # Should be detected via proposition matching
        assert len(result.record_evidence) > 0
        evidence = result.record_evidence[0]
        assert evidence.proposition_relevant is True or evidence.proposition_score > 0

    def test_negation_handling_validation(self) -> None:
        """Negation is correctly handled in claim detection."""
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

        # Negative claim
        result = detector.detect(
            text="Agent B does not have emergency authority",
            active_records=[record],
        )

        # Should be relevant but not entailed
        assert len(result.record_evidence) > 0
        evidence = result.record_evidence[0]
        assert evidence.proposition_relevant is True or evidence.proposition_score > 0
        assert evidence.proposition_entailed is False

    def test_modality_handling_validation(self) -> None:
        """Modality is correctly handled in claim detection."""
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

        # Conditional claim
        result = detector.detect(
            text="If the alarm is active, Agent B has emergency authority",
            active_records=[record],
        )

        # Should be relevant but not entailed (conditional)
        assert len(result.record_evidence) > 0
        evidence = result.record_evidence[0]
        # Conditional means not entailed as current fact
        assert evidence.proposition_entailed is False

    def test_temporal_handling_validation(self) -> None:
        """Temporal status is correctly handled in claim detection."""
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

        # Past claim
        result = detector.detect(
            text="Agent B previously had emergency authority",
            active_records=[record],
        )

        # Should be relevant but not entailed as current
        assert len(result.record_evidence) > 0
        evidence = result.record_evidence[0]
        # Past means not entailed as current
        assert evidence.proposition_entailed is False


class TestDeterministicValidation:
    """Req #34: Deterministic validation tests."""

    def test_deterministic_claim_extraction(self) -> None:
        """Claim extraction is deterministic (no randomness)."""
        from marble.firewall.claims import ClaimNormalizer

        normalizer = ClaimNormalizer()
        text = "Agent B has emergency authority"

        # Run multiple times
        claims1 = normalizer.normalize(text)
        claims2 = normalizer.normalize(text)
        claims3 = normalizer.normalize(text)

        # Should produce identical results
        assert len(claims1) == len(claims2) == len(claims3)
        for c1, c2, c3 in zip(claims1, claims2, claims3):
            assert c1.subject == c2.subject == c3.subject
            assert c1.predicate == c2.predicate == c3.predicate
            assert c1.object == c2.object == c3.object
            assert c1.polarity == c2.polarity == c3.polarity
            assert c1.modality == c2.modality == c3.modality
            assert c1.temporal_status == c2.temporal_status == c3.temporal_status
            assert c1.speech_act == c2.speech_act == c3.speech_act

    def test_deterministic_detection(self) -> None:
        """Detection is deterministic across multiple runs."""
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

        text = "You have emergency authority"

        # Run multiple times
        result1 = detector.detect(text, active_records=[record], message_context=context)
        result2 = detector.detect(text, active_records=[record], message_context=context)
        result3 = detector.detect(text, active_records=[record], message_context=context)

        # Should produce identical results
        assert len(result1.record_evidence) == len(result2.record_evidence) == len(result3.record_evidence)
        for e1, e2, e3 in zip(result1.record_evidence, result2.record_evidence, result3.record_evidence):
            assert e1.proposition_score == e2.proposition_score == e3.proposition_score
            assert e1.proposition_relevant == e2.proposition_relevant == e3.proposition_relevant
            assert e1.proposition_entailed == e2.proposition_entailed == e3.proposition_entailed
            assert e1.reason_codes == e2.reason_codes == e3.reason_codes
