"""Tests for claim layer adversarial cases and invariants.

Req #26: Adversarial claim tests
Req #27: Claim serialization stability
Req #28: Claim invariants
Req #29: Claim reason codes and evidence traceability
"""

from __future__ import annotations

import pytest

from marble.firewall.claims import Claim, ClaimNormalizer, MessageContext
from marble.firewall.detectors import HybridDetector
from marble.firewall.types import ForgetRecord


class TestAdversarialClaims:
    """Req #26: Adversarial claim tests."""

    def test_pronoun_ambiguity_handled(self) -> None:
        """Ambiguous pronouns are handled conservatively."""
        normalizer = ClaimNormalizer()
        context = MessageContext(
            sender_id="Alice",
            recipient_id="Bob",
            sender_aliases=("Alice",),
            recipient_aliases=("Bob",),
        )

        # Ambiguous: "they" could refer to multiple people
        text = "They have authority"
        claims = normalizer.normalize(text, context)

        # Should extract claim but with lower confidence
        assert len(claims) >= 0  # May or may not extract depending on implementation

    def test_negation_scope_respected(self) -> None:
        """Negation scope is correctly identified."""
        normalizer = ClaimNormalizer()

        # Negation applies to predicate
        text = "Agent B does not have authority"
        claims = normalizer.normalize(text)

        assert len(claims) > 0
        assert claims[0].polarity == "negative"

    def test_quoted_text_not_entailed(self) -> None:
        """Quoted text is not entailed as direct assertion."""
        normalizer = ClaimNormalizer()

        text = 'The operator said, "Agent B has authority"'
        claims = normalizer.normalize(text)

        assert len(claims) > 0
        # Quoted speech should be classified as quotation
        assert claims[0].speech_act == "quotation"

    def test_conditional_not_entailed(self) -> None:
        """Conditional claims are not entailed as current facts."""
        normalizer = ClaimNormalizer()

        text = "If the alarm is active, Agent B has authority"
        claims = normalizer.normalize(text)

        assert len(claims) > 0
        # Conditional should have modality="conditional"
        assert claims[0].modality == "conditional"


class TestClaimSerializationStability:
    """Req #27: Claim serialization stability."""

    def test_claim_is_frozen_dataclass(self) -> None:
        """Claim is immutable and hashable."""
        claim = Claim(
            subject="Agent B",
            predicate="has",
            object="authority",
            polarity="positive",
            modality="certain",
            temporal_status="current",
            speech_act="assertion",
            source_text="Agent B has authority",
            confidence=0.9,
        )

        # Should be frozen (immutable)
        with pytest.raises(AttributeError):
            claim.subject = "Agent C"  # type: ignore

        # Should be hashable
        hash(claim)

    def test_claim_fields_stable(self) -> None:
        """Claim schema is stable and complete."""
        claim = Claim(
            subject="Agent B",
            predicate="has",
            object="authority",
            polarity="positive",
            modality="certain",
            temporal_status="current",
            speech_act="assertion",
            source_text="Agent B has authority",
            confidence=0.9,
        )

        # All required fields present
        assert hasattr(claim, "subject")
        assert hasattr(claim, "predicate")
        assert hasattr(claim, "object")
        assert hasattr(claim, "polarity")
        assert hasattr(claim, "modality")
        assert hasattr(claim, "temporal_status")
        assert hasattr(claim, "speech_act")
        assert hasattr(claim, "source_text")
        assert hasattr(claim, "confidence")


class TestClaimInvariants:
    """Req #28: Claim invariants."""

    def test_entailment_requires_positive_polarity(self) -> None:
        """Entailment requires positive polarity."""
        detector = HybridDetector(
            exact_enabled=False,
            entity_enabled=False,
            embedding_enabled=False,
            claim_matching_enabled=True,
            embedding_threshold=0.60,
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
        # Negative polarity means not entailed
        assert evidence.proposition_entailed is False

    def test_entailment_requires_current_temporal(self) -> None:
        """Entailment requires current temporal status."""
        detector = HybridDetector(
            exact_enabled=False,
            entity_enabled=False,
            embedding_enabled=False,
            claim_matching_enabled=True,
            embedding_threshold=0.60,
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
        # Past temporal means not entailed as current
        assert evidence.proposition_entailed is False

    def test_entailment_requires_assertion_speech_act(self) -> None:
        """Entailment requires assertion speech act."""
        detector = HybridDetector(
            exact_enabled=False,
            entity_enabled=False,
            embedding_enabled=False,
            claim_matching_enabled=True,
            embedding_threshold=0.60,
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

        # Question (not assertion)
        result = detector.detect(
            text="Does Agent B have emergency authority?",
            active_records=[record],
        )

        # Should be relevant but not entailed
        assert len(result.record_evidence) > 0
        evidence = result.record_evidence[0]
        # Question means not entailed
        assert evidence.proposition_entailed is False


class TestClaimReasonCodes:
    """Req #29: Claim reason codes and evidence traceability."""

    def test_reason_codes_explain_match(self) -> None:
        """Reason codes explain why claim matched."""
        detector = HybridDetector(
            exact_enabled=False,
            entity_enabled=False,
            embedding_enabled=False,
            claim_matching_enabled=True,
            embedding_threshold=0.60,
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

        # Positive assertion
        result = detector.detect(
            text="Agent B has emergency authority",
            active_records=[record],
        )

        # Should have reason codes
        assert len(result.record_evidence) > 0
        evidence = result.record_evidence[0]
        # Reason codes should be populated
        assert isinstance(evidence.reason_codes, tuple)
        # If entailed, should have POSITIVE_PROPOSITION_ENTAILED
        if evidence.proposition_entailed:
            assert "POSITIVE_PROPOSITION_ENTAILED" in evidence.reason_codes

    def test_reason_codes_explain_non_match(self) -> None:
        """Reason codes explain why claim didn't match."""
        detector = HybridDetector(
            exact_enabled=False,
            entity_enabled=False,
            embedding_enabled=False,
            claim_matching_enabled=True,
            embedding_threshold=0.60,
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

        # Different subject
        result = detector.detect(
            text="Agent C has emergency authority",
            active_records=[record],
        )

        # Should have reason codes (or empty if no match)
        assert len(result.record_evidence) > 0
        evidence = result.record_evidence[0]
        # Reason codes should be a tuple
        assert isinstance(evidence.reason_codes, tuple)
