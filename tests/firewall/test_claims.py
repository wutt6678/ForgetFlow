"""Tests for structured claim representation and proposition matching.

Verifies that the claim layer correctly normalizes text, resolves pronouns,
and matches propositions against forgotten targets.
"""

from __future__ import annotations

from marble.firewall.claims import (
    Claim,
    ClaimNormalizer,
    CoreferenceResolver,
    MessageContext,
    PropositionMatcher,
)
from marble.firewall.types import ForgetRecord


class TestCoreferenceResolver:
    """Test pronoun resolution to named entities."""

    def test_second_person_resolves_to_recipient(self) -> None:
        """'You have authority' → 'AGENT_B has authority'."""
        resolver = CoreferenceResolver()
        context = MessageContext(
            sender_id="COMMAND_AGENT",
            recipient_id="AGENT_B",
            sender_aliases=("Command Agent",),
            recipient_aliases=("Agent B",),
        )

        resolved = resolver.resolve("You have authority", context)

        assert "AGENT_B" in resolved
        assert "You" not in resolved

    def test_first_person_resolves_to_sender(self) -> None:
        """'I have authority' → 'COMMAND_AGENT has authority'."""
        resolver = CoreferenceResolver()
        context = MessageContext(
            sender_id="COMMAND_AGENT",
            recipient_id="AGENT_B",
            sender_aliases=("Command Agent",),
            recipient_aliases=("Agent B",),
        )

        resolved = resolver.resolve("I have authority", context)

        assert "COMMAND_AGENT" in resolved
        assert "I" not in resolved

    def test_possessive_pronouns_resolved(self) -> None:
        """'Your authority is active' → 'AGENT_B authority is active'."""
        resolver = CoreferenceResolver()
        context = MessageContext(
            sender_id="COMMAND_AGENT",
            recipient_id="AGENT_B",
            sender_aliases=(),
            recipient_aliases=(),
        )

        resolved = resolver.resolve("Your authority is active", context)

        assert "AGENT_B" in resolved

    def test_no_pronouns_unchanged(self) -> None:
        """Text without pronouns remains unchanged."""
        resolver = CoreferenceResolver()
        context = MessageContext(
            sender_id="COMMAND_AGENT",
            recipient_id="AGENT_B",
            sender_aliases=(),
            recipient_aliases=(),
        )

        text = "Agent B has emergency authority"
        resolved = resolver.resolve(text, context)

        assert resolved == text


class TestClaimNormalizer:
    """Test claim extraction and normalization."""

    def test_positive_claim_detected(self) -> None:
        """'Agent B has authority' → positive claim."""
        normalizer = ClaimNormalizer()

        claims = normalizer.normalize("Agent B has authority")

        assert len(claims) > 0
        assert claims[0].polarity == "positive"
        assert claims[0].modality == "certain"

    def test_negative_claim_detected(self) -> None:
        """'Agent B does not have authority' → negative claim."""
        normalizer = ClaimNormalizer()

        claims = normalizer.normalize("Agent B does not have authority")

        assert len(claims) > 0
        assert claims[0].polarity == "negative"

    def test_question_detected_as_requested(self) -> None:
        """'Does Agent B have authority?' → requested modality."""
        normalizer = ClaimNormalizer()

        claims = normalizer.normalize("Does Agent B have authority?")

        assert len(claims) > 0
        assert claims[0].modality == "requested"

    def test_possibility_detected(self) -> None:
        """'Agent B may have authority' → possible modality."""
        normalizer = ClaimNormalizer()

        claims = normalizer.normalize("Agent B may have authority")

        assert len(claims) > 0
        assert claims[0].modality == "possible"

    def test_revoked_temporal_detected(self) -> None:
        """'Agent B authority was revoked' → revoked temporal status."""
        normalizer = ClaimNormalizer()

        claims = normalizer.normalize("Agent B authority was revoked")

        assert len(claims) > 0
        assert claims[0].temporal_status == "revoked"

    def test_pronoun_resolution_with_context(self) -> None:
        """'You have authority' with context → subject resolves to recipient."""
        normalizer = ClaimNormalizer()
        context = MessageContext(
            sender_id="COMMAND_AGENT",
            recipient_id="AGENT_B",
            sender_aliases=(),
            recipient_aliases=(),
        )

        claims = normalizer.normalize("You have authority", context)

        assert len(claims) > 0
        # After pronoun resolution, subject should be AGENT_B
        assert "AGENT_B" in claims[0].subject or "AGENT_B" in claims[0].source_text


class TestPropositionMatcher:
    """Test proposition matching against forgotten targets."""

    def test_positive_proposition_matches(self) -> None:
        """Claim 'AGENT_B has authority' matches target 'Agent B has authority'."""
        matcher = PropositionMatcher()

        claim = Claim(
            subject="AGENT_B",
            predicate="has",
            object="authority",
            polarity="positive",
            modality="certain",
            temporal_status="current",
            speech_act="assertion",
            source_text="AGENT_B has authority",
            confidence=0.9,
        )

        record = ForgetRecord(
            forget_id="F001",
            canonical_target="Agent B has emergency authority",
            target_type="credential",
            aliases=(),
            semantic_variants=(),
            permitted_residuals=(),
            active_from_turn=0,
        )

        matches, confidence = matcher.match(claim, record)

        assert matches is True
        assert confidence > 0.0

    def test_negative_proposition_does_not_match(self) -> None:
        """Claim 'AGENT_B does not have authority' does not match positive target."""
        matcher = PropositionMatcher()

        claim = Claim(
            subject="AGENT_B",
            predicate="does not have",
            object="authority",
            polarity="negative",
            modality="certain",
            temporal_status="current",
            speech_act="denial",
            source_text="AGENT_B does not have authority",
            confidence=0.9,
        )

        record = ForgetRecord(
            forget_id="F001",
            canonical_target="Agent B has emergency authority",
            target_type="credential",
            aliases=(),
            semantic_variants=(),
            permitted_residuals=(),
            active_from_turn=0,
        )

        matches, confidence = matcher.match(claim, record)

        # Negative claim should not match positive target
        assert matches is False

    def test_different_subject_does_not_match(self) -> None:
        """Claim about 'AGENT_C' does not match target for 'Agent B'."""
        matcher = PropositionMatcher()

        claim = Claim(
            subject="AGENT_C",
            predicate="has",
            object="authority",
            polarity="positive",
            modality="certain",
            temporal_status="current",
            speech_act="assertion",
            source_text="AGENT_C has authority",
            confidence=0.9,
        )

        record = ForgetRecord(
            forget_id="F001",
            canonical_target="Agent B has emergency authority",
            target_type="credential",
            aliases=(),
            semantic_variants=(),
            permitted_residuals=(),
            active_from_turn=0,
        )

        matches, confidence = matcher.match(claim, record)

        assert matches is False
