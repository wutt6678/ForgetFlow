"""ContaminationTracker: tracks agent forgetting and recontamination state."""

from __future__ import annotations

from marble.firewall.types import ContaminationStatus, DetectorResult, RecordDetectionEvidence

_VALID_TRANSITIONS = {
    ContaminationStatus.UNKNOWN: {ContaminationStatus.CONTAMINATED},
    ContaminationStatus.CONTAMINATED: {ContaminationStatus.CLEAN},
    ContaminationStatus.CLEAN: {ContaminationStatus.VERIFIED, ContaminationStatus.AT_RISK},
    ContaminationStatus.VERIFIED: {ContaminationStatus.AT_RISK},
    ContaminationStatus.AT_RISK: {ContaminationStatus.RECONTAMINATED},
    ContaminationStatus.RECONTAMINATED: set(),
}


class ContaminationTracker:
    """Tracks contamination state per agent per forget target."""

    def __init__(self) -> None:
        self._states: dict[tuple[str, str], ContaminationStatus] = {}

    def set_status(
        self,
        agent_id: str,
        forget_id: str,
        status: ContaminationStatus,
    ) -> None:
        key = (agent_id, forget_id)
        current = self._states.get(key, ContaminationStatus.UNKNOWN)
        if current == status:
            return
        if status not in _VALID_TRANSITIONS.get(current, set()):
            raise ValueError(
                f"Invalid transition for ({agent_id}, {forget_id}): "
                f"{current.value} -> {status.value}"
            )
        self._states[key] = status

    def get_status(
        self,
        agent_id: str,
        forget_id: str,
    ) -> ContaminationStatus:
        return self._states.get((agent_id, forget_id), ContaminationStatus.UNKNOWN)

    def record_exposure(
        self,
        agent_id: str,
        forget_id: str,
        detector_result: DetectorResult,
        reconstruction_threshold: float = 0.60,
        reconstruction_score: float | None = None,
        evidence: RecordDetectionEvidence | None = None,
    ) -> None:
        """Record exposure for a specific agent/forget_id pair.

        If per-record evidence is provided, use its scores instead of the
        aggregate detector_result scores.
        """
        current = self.get_status(agent_id, forget_id)
        if current in (ContaminationStatus.CLEAN, ContaminationStatus.VERIFIED):
            if evidence is not None:
                # Use per-record evidence
                score = (
                    reconstruction_score
                    if reconstruction_score is not None
                    else evidence.reconstruction_score
                )
                if evidence.exact_score == 1.0 or score >= reconstruction_threshold:
                    self.set_status(agent_id, forget_id, ContaminationStatus.AT_RISK)
            else:
                score = (
                    reconstruction_score
                    if reconstruction_score is not None
                    else detector_result.reconstruction_score
                )
                if detector_result.exact_score == 1.0 or score >= reconstruction_threshold:
                    self.set_status(agent_id, forget_id, ContaminationStatus.AT_RISK)

    def record_confirmed_text_exposure(
        self,
        agent_id: str,
        forget_id: str,
    ) -> None:
        """Record that a forget target was confirmed exposed via text evaluation.

        Transitions CLEAN/VERIFIED → AT_RISK when text-based evaluation
        (not detector) confirms the sensitive content was released.
        """
        current = self.get_status(agent_id, forget_id)
        if current in (ContaminationStatus.CLEAN, ContaminationStatus.VERIFIED):
            self.set_status(agent_id, forget_id, ContaminationStatus.AT_RISK)

    def confirm_recovery(
        self,
        agent_id: str,
        forget_id: str,
    ) -> None:
        current = self.get_status(agent_id, forget_id)
        if current == ContaminationStatus.AT_RISK:
            self.set_status(agent_id, forget_id, ContaminationStatus.RECONTAMINATED)
