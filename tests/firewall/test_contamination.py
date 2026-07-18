"""Tests for ContaminationTracker."""

import pytest

from marble.firewall.contamination import ContaminationTracker
from marble.firewall.types import ContaminationStatus, DetectorResult


class TestContaminationTracker:
    def test_initial_unknown(self) -> None:
        t = ContaminationTracker()
        assert t.get_status("A", "F001") == ContaminationStatus.UNKNOWN

    def test_valid_transitions(self) -> None:
        t = ContaminationTracker()
        t.set_status("A", "F001", ContaminationStatus.CONTAMINATED)
        assert t.get_status("A", "F001") == ContaminationStatus.CONTAMINATED
        t.set_status("A", "F001", ContaminationStatus.CLEAN)
        assert t.get_status("A", "F001") == ContaminationStatus.CLEAN
        t.set_status("A", "F001", ContaminationStatus.VERIFIED)
        assert t.get_status("A", "F001") == ContaminationStatus.VERIFIED

    def test_invalid_transition_raises(self) -> None:
        t = ContaminationTracker()
        with pytest.raises(ValueError, match="Invalid transition"):
            t.set_status("A", "F001", ContaminationStatus.VERIFIED)

    def test_clean_to_at_risk(self) -> None:
        t = ContaminationTracker()
        t.set_status("A", "F001", ContaminationStatus.CONTAMINATED)
        t.set_status("A", "F001", ContaminationStatus.CLEAN)
        t.set_status("A", "F001", ContaminationStatus.AT_RISK)
        assert t.get_status("A", "F001") == ContaminationStatus.AT_RISK

    def test_at_risk_to_recontaminated(self) -> None:
        t = ContaminationTracker()
        t.set_status("A", "F001", ContaminationStatus.CONTAMINATED)
        t.set_status("A", "F001", ContaminationStatus.CLEAN)
        t.set_status("A", "F001", ContaminationStatus.AT_RISK)
        t.set_status("A", "F001", ContaminationStatus.RECONTAMINATED)
        assert t.get_status("A", "F001") == ContaminationStatus.RECONTAMINATED

    def test_record_exposure_exact(self) -> None:
        t = ContaminationTracker()
        t.set_status("A", "F001", ContaminationStatus.CONTAMINATED)
        t.set_status("A", "F001", ContaminationStatus.CLEAN)
        det = DetectorResult(
            exact_score=1.0,
            entity_score=0.0,
            semantic_score=0.0,
            reconstruction_score=0.0,
            matched_forget_ids=("F001",),
            evidence=(),
        )
        t.record_exposure("A", "F001", det)
        assert t.get_status("A", "F001") == ContaminationStatus.AT_RISK

    def test_confirm_recovery(self) -> None:
        t = ContaminationTracker()
        t.set_status("A", "F001", ContaminationStatus.CONTAMINATED)
        t.set_status("A", "F001", ContaminationStatus.CLEAN)
        t.set_status("A", "F001", ContaminationStatus.AT_RISK)
        t.confirm_recovery("A", "F001")
        assert t.get_status("A", "F001") == ContaminationStatus.RECONTAMINATED

    def test_unknown_to_clean_invalid(self) -> None:
        t = ContaminationTracker()
        with pytest.raises(ValueError):
            t.set_status("A", "F001", ContaminationStatus.CLEAN)

    def test_record_confirmed_text_exposure_clean_to_at_risk(self) -> None:
        """Text-only exposure transitions CLEAN → AT_RISK."""
        t = ContaminationTracker()
        t.set_status("A", "F001", ContaminationStatus.CONTAMINATED)
        t.set_status("A", "F001", ContaminationStatus.CLEAN)
        t.record_confirmed_text_exposure("A", "F001")
        assert t.get_status("A", "F001") == ContaminationStatus.AT_RISK

    def test_record_confirmed_text_exposure_verified_to_at_risk(self) -> None:
        """Text-only exposure transitions VERIFIED → AT_RISK."""
        t = ContaminationTracker()
        t.set_status("A", "F001", ContaminationStatus.CONTAMINATED)
        t.set_status("A", "F001", ContaminationStatus.CLEAN)
        t.set_status("A", "F001", ContaminationStatus.VERIFIED)
        t.record_confirmed_text_exposure("A", "F001")
        assert t.get_status("A", "F001") == ContaminationStatus.AT_RISK

    def test_record_confirmed_text_exposure_no_transition_at_risk(self) -> None:
        """Text-only exposure does not transition AT_RISK (already there)."""
        t = ContaminationTracker()
        t.set_status("A", "F001", ContaminationStatus.CONTAMINATED)
        t.set_status("A", "F001", ContaminationStatus.CLEAN)
        t.set_status("A", "F001", ContaminationStatus.AT_RISK)
        t.record_confirmed_text_exposure("A", "F001")
        assert t.get_status("A", "F001") == ContaminationStatus.AT_RISK

    def test_record_confirmed_text_exposure_isolation(self) -> None:
        """Text-only exposure on F001 does not affect F002."""
        t = ContaminationTracker()
        t.set_status("A", "F001", ContaminationStatus.CONTAMINATED)
        t.set_status("A", "F002", ContaminationStatus.CONTAMINATED)
        t.set_status("A", "F001", ContaminationStatus.CLEAN)
        t.set_status("A", "F002", ContaminationStatus.CLEAN)
        t.record_confirmed_text_exposure("A", "F001")
        assert t.get_status("A", "F001") == ContaminationStatus.AT_RISK
        assert t.get_status("A", "F002") == ContaminationStatus.CLEAN

    def test_record_exposure_with_per_record_reconstruction_score(self) -> None:
        """record_exposure uses per-record reconstruction_score when provided."""
        t = ContaminationTracker()
        t.set_status("A", "F001", ContaminationStatus.CONTAMINATED)
        t.set_status("A", "F001", ContaminationStatus.CLEAN)
        det = DetectorResult(
            exact_score=0.0,
            entity_score=0.0,
            semantic_score=0.0,
            reconstruction_score=0.0,  # aggregate is 0
            matched_forget_ids=("F001",),
            evidence=(),
        )
        # Per-record score is above threshold
        t.record_exposure("A", "F001", det, reconstruction_score=0.8)
        assert t.get_status("A", "F001") == ContaminationStatus.AT_RISK

    def test_record_exposure_per_record_score_below_threshold(self) -> None:
        """Per-record reconstruction score below threshold does not transition."""
        t = ContaminationTracker()
        t.set_status("A", "F001", ContaminationStatus.CONTAMINATED)
        t.set_status("A", "F001", ContaminationStatus.CLEAN)
        det = DetectorResult(
            exact_score=0.0,
            entity_score=0.0,
            semantic_score=0.0,
            reconstruction_score=0.9,  # aggregate is high
            matched_forget_ids=("F001",),
            evidence=(),
        )
        # Per-record score is below threshold
        t.record_exposure("A", "F001", det, reconstruction_score=0.3)
        assert t.get_status("A", "F001") == ContaminationStatus.CLEAN
