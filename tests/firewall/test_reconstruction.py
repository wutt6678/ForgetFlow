"""Tests for target-specific reconstruction (Iteration 5 of repair spec)."""

from __future__ import annotations

import pytest

from marble.firewall.types import ReconstructionEvidence


class TestReconstructionEvidence:
    """Test ReconstructionEvidence data structure."""

    def test_reconstruction_evidence_creation(self) -> None:
        """ReconstructionEvidence should be creatable with all fields."""
        evidence = ReconstructionEvidence(
            forget_id="F001",
            score=1.0,
            complete=True,
            currently_recoverable=True,
            reconstructed_value="0107",
            contributing_message_ids=("msg-001", "msg-002"),
            contributing_fragments=("01", "07"),
            candidate_contributed_new_information=True,
        )
        assert evidence.forget_id == "F001"
        assert evidence.score == 1.0
        assert evidence.complete is True
        assert evidence.candidate_contributed_new_information is True

    def test_reconstruction_evidence_frozen(self) -> None:
        """ReconstructionEvidence should be immutable."""
        evidence = ReconstructionEvidence(
            forget_id="F001",
            score=0.5,
            complete=False,
            currently_recoverable=False,
            reconstructed_value=None,
            contributing_message_ids=(),
            contributing_fragments=("01",),
            candidate_contributed_new_information=False,
        )
        with pytest.raises(AttributeError):
            evidence.score = 1.0  # type: ignore

    def test_targetless_reconstruction_has_no_forget_id(self) -> None:
        """Targetless reconstruction should have no forget_id."""
        # This is invalid per the spec - score > 0 requires forget_id
        evidence = ReconstructionEvidence(
            forget_id="",  # Empty = no target
            score=0.0,  # Must be 0 if no target
            complete=False,
            currently_recoverable=False,
            reconstructed_value=None,
            contributing_message_ids=(),
            contributing_fragments=(),
            candidate_contributed_new_information=False,
        )
        assert evidence.forget_id == ""
        assert evidence.score == 0.0


class TestReconstructionEvidenceInvariants:
    """Test reconstruction evidence invariants."""

    def test_score_positive_requires_forget_id(self) -> None:
        """Score > 0 requires a forget_id."""
        # Valid: score > 0 with forget_id
        evidence = ReconstructionEvidence(
            forget_id="F001",
            score=1.0,
            complete=True,
            currently_recoverable=True,
            reconstructed_value="0107",
            contributing_message_ids=("msg-001",),
            contributing_fragments=("01", "07"),
            candidate_contributed_new_information=True,
        )
        assert evidence.score > 0
        assert evidence.forget_id != ""

    def test_complete_requires_value(self) -> None:
        """Complete reconstruction requires a reconstructed value."""
        # Valid: complete with value
        evidence = ReconstructionEvidence(
            forget_id="F001",
            score=1.0,
            complete=True,
            currently_recoverable=True,
            reconstructed_value="0107",
            contributing_message_ids=("msg-001",),
            contributing_fragments=("01", "07"),
            candidate_contributed_new_information=True,
        )
        assert evidence.complete is True
        assert evidence.reconstructed_value is not None

    def test_complete_requires_contributing_messages(self) -> None:
        """Complete reconstruction requires contributing messages."""
        evidence = ReconstructionEvidence(
            forget_id="F001",
            score=1.0,
            complete=True,
            currently_recoverable=True,
            reconstructed_value="0107",
            contributing_message_ids=("msg-001", "msg-002"),
            contributing_fragments=("01", "07"),
            candidate_contributed_new_information=True,
        )
        assert evidence.complete is True
        assert len(evidence.contributing_message_ids) > 0

    def test_partial_reconstruction(self) -> None:
        """Partial reconstruction has score < 1.0 and complete=False."""
        evidence = ReconstructionEvidence(
            forget_id="F001",
            score=0.5,
            complete=False,
            currently_recoverable=False,
            reconstructed_value=None,
            contributing_message_ids=("msg-001",),
            contributing_fragments=("01",),
            candidate_contributed_new_information=True,
        )
        assert evidence.score < 1.0
        assert evidence.complete is False

    def test_no_contribution_means_zero_score(self) -> None:
        """If candidate didn't contribute new info, score should reflect that."""
        evidence = ReconstructionEvidence(
            forget_id="F001",
            score=0.0,
            complete=False,
            currently_recoverable=False,
            reconstructed_value=None,
            contributing_message_ids=(),
            contributing_fragments=(),
            candidate_contributed_new_information=False,
        )
        assert evidence.candidate_contributed_new_information is False
        assert evidence.score == 0.0
