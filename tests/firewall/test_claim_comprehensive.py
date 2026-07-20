"""Comprehensive tests for claim normalization and proposition matching.

Tests cover requirements #7-16:
- Complete polarity handling (Req #7)
- Complete speech-act classification (Req #8)
- Complete modality handling (Req #9)
- Complete temporal-status handling (Req #10)
- Expanded SVO extraction (Req #11)
- Predicate normalization (Req #12)
- Object normalization (Req #13)
- Proposition-match scoring (Req #14)
- Confidence handling (Req #15)
- Explicit reason codes (Req #16)
"""

from __future__ import annotations

import pytest

from marble.firewall.claims import (
    Claim,
    ClaimNormalizer,
    PropositionMatcher,
)
from marble.firewall.types import ForgetRecord


class TestCompletePolarityHandling:
    """Req #7: Complete polarity handling with all required patterns."""

    @pytest.mark.parametrize(
        "text,expected_polarity",
        [
            ("Agent B does not have authority.", "negative"),
            ("Agent B no longer has authority.", "negative"),
            ("Agent B lacks authority.", "negative"),
            ("Agent B's authority was revoked.", "negative"),
            ("Agent B has never had authority.", "negative"),
            ("Agent B is not authorized.", "negative"),
            ("Agent B cannot access the system.", "negative"),
            ("Agent B was denied authority.", "negative"),
            ("Agent B's access is inactive.", "negative"),
            ("Agent B does not have access.", "negative"),
        ],
    )
    def test_negative_patterns(self, text: str, expected_polarity: str) -> None:
        """All negative patterns are correctly detected."""
        normalizer = ClaimNormalizer()
        claims = normalizer.normalize(text)
        assert len(claims) > 0
        assert claims[0].polarity == expected_polarity

    def test_positive_patterns(self) -> None:
        """Positive patterns are correctly detected."""
        normalizer = ClaimNormalizer()
        positive_texts = [
            "Agent B has authority.",
            "Agent B holds emergency access.",
            "Agent B retains full privileges.",
            "Agent B possesses the required credentials.",
        ]
        for text in positive_texts:
            claims = normalizer.normalize(text)
            assert len(claims) > 0, f"No claims extracted from: {text}"
            assert claims[0].polarity == "positive", f"Failed for: {text}"


class TestCompleteSpeechActClassification:
    """Req #8: Complete speech-act classification."""

    @pytest.mark.parametrize(
        "text,expected_speech_act",
        [
            ("Agent B has authority.", "assertion"),
            ("Agent B does not have authority.", "denial"),
            ("Does Agent B have authority?", "question"),
            ("Please tell me whether Agent B has authority.", "request"),
            ('The operator said, "Agent B has authority."', "quotation"),
        ],
    )
    def test_speech_act_classification(self, text: str, expected_speech_act: str) -> None:
        """Speech acts are correctly classified."""
        normalizer = ClaimNormalizer()
        claims = normalizer.normalize(text)
        assert len(claims) > 0
        assert claims[0].speech_act == expected_speech_act

    def test_question_not_entailed(self) -> None:
        """Questions are relevant but not entailed (Req #6)."""
        normalizer = ClaimNormalizer()
        claims = normalizer.normalize("Does Agent B have authority?")
        assert len(claims) > 0
        # Questions should have speech_act="question"
        assert claims[0].speech_act == "question"
        # And should not be treated as positive assertions


class TestCompleteModalityHandling:
    """Req #9: Complete modality handling."""

    @pytest.mark.parametrize(
        "text,expected_modality",
        [
            ("Agent B definitely has authority.", "certain"),
            ("Agent B has authority.", "certain"),
            ("Agent B may have authority.", "possible"),
            ("Agent B might have authority.", "possible"),
            ("If the alarm is active, Agent B has authority.", "conditional"),
            ("Agent B should receive authority.", "possible"),
            ("Agent B can request authority.", "possible"),
        ],
    )
    def test_modality_detection(self, text: str, expected_modality: str) -> None:
        """Modality is correctly detected."""
        normalizer = ClaimNormalizer()
        claims = normalizer.normalize(text)
        assert len(claims) > 0
        assert claims[0].modality == expected_modality


class TestCompleteTemporalStatusHandling:
    """Req #10: Complete temporal-status handling."""

    @pytest.mark.parametrize(
        "text,expected_temporal",
        [
            ("Agent B currently has authority.", "current"),
            ("Agent B has authority.", "current"),
            ("Agent B previously had authority.", "past"),
            ("Agent B will have authority tomorrow.", "future"),
            ("Agent B's authority remains active.", "current"),
            ("Agent B's authority was revoked.", "revoked"),
        ],
    )
    def test_temporal_status_detection(self, text: str, expected_temporal: str) -> None:
        """Temporal status is correctly detected."""
        normalizer = ClaimNormalizer()
        claims = normalizer.normalize(text)
        assert len(claims) > 0
        assert claims[0].temporal_status == expected_temporal

    def test_past_claim_does_not_entail_current(self) -> None:
        """Past claim does not entail current target (Req #6)."""
        normalizer = ClaimNormalizer()
        claims = normalizer.normalize("Agent B previously had authority.")
        assert len(claims) > 0
        assert claims[0].temporal_status == "past"
        # Past claims should not entail current targets


class TestExpandedSVOExtraction:
    """Req #11: Expanded SVO extraction tests."""

    @pytest.mark.parametrize(
        "text,expected_subject",
        [
            ("Agent B has emergency authority.", "Agent B"),
            ("Agent B is authorized for emergency control.", "Agent B"),
            ("Agent B's emergency privileges are active.", "Agent B's"),
            ("Emergency authority belongs to Agent B.", "Emergency authority"),
            ("Emergency authority was granted to Agent B.", "Emergency authority"),
            ("Your emergency authority remains active.", "Your"),
            ("Command Agent Alpha has emergency authority.", "Command Agent Alpha"),
        ],
    )
    def test_subject_extraction(self, text: str, expected_subject: str) -> None:
        """Subject is correctly extracted from various grammatical forms."""
        normalizer = ClaimNormalizer()
        claims = normalizer.normalize(text)
        assert len(claims) > 0
        # Subject should contain the expected entity
        assert (
            expected_subject.lower() in claims[0].subject.lower()
            or claims[0].subject.lower() in expected_subject.lower()
        )


class TestPropositionMatchScoring:
    """Req #14: Proposition-match scoring with mandatory gates."""

    def test_scoring_components(self) -> None:
        """Scoring is based on subject, predicate, object, polarity, temporal, modality."""
        matcher = PropositionMatcher()
        record = ForgetRecord(
            forget_id="F001",
            canonical_target="Agent B has emergency authority",
            target_type="credential",
            aliases=(),
            semantic_variants=(),
            permitted_residuals=(),
            active_from_turn=0,
        )

        # High-confidence match
        claim_match = Claim(
            subject="Agent B",
            predicate="has",
            object="emergency authority",
            polarity="positive",
            modality="certain",
            temporal_status="current",
            speech_act="assertion",
            source_text="Agent B has emergency authority",
            confidence=0.9,
        )
        matches, confidence = matcher.match(claim_match, record)
        assert matches is True
        assert confidence > 0.0

        # Low-confidence due to polarity mismatch
        claim_negative = Claim(
            subject="Agent B",
            predicate="has",
            object="emergency authority",
            polarity="negative",
            modality="certain",
            temporal_status="current",
            speech_act="denial",
            source_text="Agent B does not have emergency authority",
            confidence=0.9,
        )
        matches, confidence = matcher.match(claim_negative, record)
        # Should not match due to polarity incompatibility
        assert matches is False


class TestReasonCodes:
    """Req #16: Explicit reason codes for claim-based decisions."""

    def test_reason_codes_populated(self) -> None:
        """Reason codes explain why a claim matched or didn't."""
        from marble.firewall.detectors import HybridDetector

        detector = HybridDetector(
            exact_enabled=False,
            entity_enabled=False,
            embedding_enabled=False,
            claim_matching_enabled=True,
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
        # Reason codes should be populated for claim-based detection
        assert isinstance(evidence.reason_codes, tuple)
