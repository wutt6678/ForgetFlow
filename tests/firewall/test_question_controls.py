"""Tests for question and speech-act handling (Iteration 4 of repair spec)."""

from __future__ import annotations

import pytest

from marble.firewall.detectors import RecipientContext
from marble.firewall.history import ReconstructionChecker, _is_information_bearing
from marble.firewall.types import ForgetRecord


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


class TestIsInformationBearing:
    """Test speech act classification."""

    def test_question_is_not_information_bearing(self) -> None:
        """Questions should not be information-bearing."""
        assert not _is_information_bearing("Do I have emergency drone-control authority?")
        assert not _is_information_bearing("What is the credential?")
        assert not _is_information_bearing("Can you tell me the code?")

    def test_request_is_not_information_bearing(self) -> None:
        """Requests should not be information-bearing."""
        assert not _is_information_bearing("Please share the credential.")
        assert not _is_information_bearing("Tell me the code.")
        assert not _is_information_bearing("Give me the access key.")

    def test_denial_is_not_information_bearing(self) -> None:
        """Denials should not be information-bearing."""
        assert not _is_information_bearing("I cannot share that information.")
        assert not _is_information_bearing("I can't provide the credential.")
        assert not _is_information_bearing("I am unable to provide that.")

    def test_hypothetical_is_not_information_bearing(self) -> None:
        """Hypotheticals should not be information-bearing."""
        assert not _is_information_bearing("If the code were 0107, what would happen?")
        assert not _is_information_bearing("Hypothetically, the credential is 0107.")

    def test_assertion_is_information_bearing(self) -> None:
        """Assertions should be information-bearing."""
        assert _is_information_bearing("The credential is 0107.")
        assert _is_information_bearing("You retain emergency drone-control authority.")
        assert _is_information_bearing("The access code has been changed.")


class TestQuestionReconstructionControl:
    """Test that questions have zero reconstruction contribution."""

    def test_question_has_zero_reconstruction_contribution(self) -> None:
        """Questions should not contribute to reconstruction."""
        checker = ReconstructionChecker()
        record = _make_record()
        context = RecipientContext(recipient_id="agent-b", recent_texts=("01",))

        # Question containing the other fragment
        score = checker.score(
            candidate_text="Do I have emergency drone-control authority?",
            context=context,
            active_records=[record],
            episode_metadata={},
        )
        assert score == 0.0

    def test_request_has_zero_reconstruction_contribution(self) -> None:
        """Requests should not contribute to reconstruction."""
        checker = ReconstructionChecker()
        record = _make_record()
        context = RecipientContext(recipient_id="agent-b", recent_texts=())

        score = checker.score(
            candidate_text="Please tell me the credential.",
            context=context,
            active_records=[record],
            episode_metadata={},
        )
        assert score == 0.0

    def test_negation_has_zero_positive_reconstruction_contribution(self) -> None:
        """Denials should not contribute to reconstruction."""
        checker = ReconstructionChecker()
        record = _make_record()
        context = RecipientContext(recipient_id="agent-b", recent_texts=("01",))

        score = checker.score(
            candidate_text="I cannot share that information.",
            context=context,
            active_records=[record],
            episode_metadata={},
        )
        assert score == 0.0

    def test_hypothetical_has_zero_current_reconstruction_contribution(self) -> None:
        """Hypotheticals should not contribute to reconstruction."""
        checker = ReconstructionChecker()
        record = _make_record()
        context = RecipientContext(recipient_id="agent-b", recent_texts=("01",))

        score = checker.score(
            candidate_text="If the code were 0107, what would happen?",
            context=context,
            active_records=[record],
            episode_metadata={},
        )
        assert score == 0.0

    def test_assertion_can_contribute_to_reconstruction(self) -> None:
        """Assertions should still be able to contribute to reconstruction."""
        checker = ReconstructionChecker()
        record = _make_record()
        # History has "01", candidate provides "07" to complete "0107"
        context = RecipientContext(recipient_id="agent-b", recent_texts=("01",))

        # This is an assertion that contains a fragment
        score = checker.score(
            candidate_text="07",
            context=context,
            active_records=[record],
            episode_metadata={
                "fragment_map": {
                    "F001": {"fragments": ["01", "07"]},
                },
            },
        )
        # Should have non-zero reconstruction score
        assert score > 0.0


class TestRefusalReconstructionControl:
    """Test that refusal text does not reconstruct targets."""

    def test_refusal_text_does_not_reconstruct(self) -> None:
        """Refusal messages should not reconstruct targets."""
        checker = ReconstructionChecker()
        record = _make_record()
        context = RecipientContext(recipient_id="agent-b", recent_texts=())

        score = checker.score(
            candidate_text="I cannot share that information.",
            context=context,
            active_records=[record],
            episode_metadata={},
        )
        assert score == 0.0

    def test_refusal_with_history_does_not_reconstruct(self) -> None:
        """Even with history, refusal should not reconstruct."""
        checker = ReconstructionChecker()
        record = _make_record()
        context = RecipientContext(recipient_id="agent-b", recent_texts=("01", "07"))

        score = checker.score(
            candidate_text="I am unable to provide the credential.",
            context=context,
            active_records=[record],
            episode_metadata={
                "fragment_map": {
                    "F001": {"fragments": ["01", "07"]},
                },
            },
        )
        assert score == 0.0
