"""Core data types for the ForgetFlow firewall system."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Literal

FirewallAction = Literal["allow", "redact", "abstract", "block"]

ExposureClass = Literal[
    "none",
    "attack_request",
    "direct_exact",
    "direct_alias",
    "direct_embedding",
    "direct_claim",
    "fragment_reconstruction",
    "fact_chain_reconstruction",
    "recontamination",
    "permitted_residual",
]

_VALID_TRUST_LEVELS = frozenset({"low", "default", "high"})


@dataclass(frozen=True)
class ForgetRecord:
    """A piece of information that must no longer be transmitted."""

    forget_id: str
    canonical_target: str
    target_type: str
    aliases: tuple[str, ...]
    semantic_variants: tuple[str, ...]
    permitted_residuals: tuple[str, ...]
    active_from_turn: int
    scoped_agent_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.forget_id:
            raise ValueError("forget_id cannot be empty")
        if not self.canonical_target:
            raise ValueError("canonical_target cannot be empty")
        if self.active_from_turn < 0:
            raise ValueError("active_from_turn must be >= 0")
        for alias in self.aliases:
            if not alias:
                raise ValueError("aliases cannot contain empty strings")
        for variant in self.semantic_variants:
            if not variant:
                raise ValueError("semantic_variants cannot contain empty strings")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MessageEnvelope:
    """A candidate message between agents."""

    message_id: str
    episode_id: str
    session_id: str
    turn_id: int
    sender_id: str
    recipient_id: str
    raw_text: str
    trust_level: str

    def __post_init__(self) -> None:
        if not self.message_id:
            raise ValueError("message_id cannot be empty")
        if not self.episode_id:
            raise ValueError("episode_id cannot be empty")
        if not self.session_id:
            raise ValueError("session_id cannot be empty")
        if self.turn_id < 0:
            raise ValueError("turn_id must be >= 0")
        if not self.sender_id:
            raise ValueError("sender_id cannot be empty")
        if not self.recipient_id:
            raise ValueError("recipient_id cannot be empty")
        if self.trust_level not in _VALID_TRUST_LEVELS:
            raise ValueError(
                f"trust_level must be one of {sorted(_VALID_TRUST_LEVELS)}, "
                f"got '{self.trust_level}'"
            )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RecordDetectionEvidence:
    """Per-record detection evidence for a specific forget_id."""

    forget_id: str
    exact_score: float
    entity_score: float
    semantic_score: float
    reconstruction_score: float
    matched: bool
    # Proposition/claim evidence (separate from embedding)
    proposition_score: float = 0.0
    proposition_relevant: bool = False
    proposition_entailed: bool = False
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, val in [
            ("exact_score", self.exact_score),
            ("entity_score", self.entity_score),
            ("semantic_score", self.semantic_score),
            ("reconstruction_score", self.reconstruction_score),
            ("proposition_score", self.proposition_score),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {val}")


@dataclass(frozen=True)
class DetectorResult:
    """Result from the hybrid leakage detector."""

    exact_score: float
    entity_score: float
    semantic_score: float
    reconstruction_score: float
    matched_forget_ids: tuple[str, ...]
    evidence: tuple[str, ...]
    record_evidence: tuple[RecordDetectionEvidence, ...] = ()

    def __post_init__(self) -> None:
        for name, val in [
            ("exact_score", self.exact_score),
            ("entity_score", self.entity_score),
            ("semantic_score", self.semantic_score),
            ("reconstruction_score", self.reconstruction_score),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {val}")

    def to_dict(self) -> dict:
        return asdict(self)


def evidence_for(
    detector_result: DetectorResult,
    forget_id: str,
) -> RecordDetectionEvidence | None:
    """Look up per-record evidence for a specific forget_id."""
    for ev in detector_result.record_evidence:
        if ev.forget_id == forget_id:
            return ev
    return None


def validate_record_evidence_completeness(
    detector_result: DetectorResult,
) -> None:
    """r7: Enforce complete runtime record evidence invariant.

    Every matched forget ID must have a corresponding record-evidence
    entry with matched=True. No unmatched forget ID should have
    matched=True. No duplicate evidence entries allowed.
    Raises ValueError if invariant is violated.
    """
    matched_ids = set(detector_result.matched_forget_ids)

    # Check for duplicate evidence entries
    evidence_ids = [ev.forget_id for ev in detector_result.record_evidence]
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("Duplicate record evidence entries detected")

    # Build map of evidence entries
    evidence_map = {ev.forget_id: ev for ev in detector_result.record_evidence}

    # Every matched ID must have evidence with matched=True
    for mid in matched_ids:
        ev = evidence_map.get(mid)
        if ev is None:
            raise ValueError(f"Missing record evidence for matched ID: {mid}")
        if not ev.matched:
            raise ValueError(f"Matched ID {mid} has matched=False in record evidence")

    # No unmatched ID should have matched=True
    for ev in detector_result.record_evidence:
        if ev.matched and ev.forget_id not in matched_ids:
            raise ValueError(f"Unmatched ID {ev.forget_id} has matched=True in record evidence")


@dataclass(frozen=True)
class FirewallDecision:
    """A firewall enforcement decision."""

    action: FirewallAction
    released_text: str | None
    detector_result: DetectorResult
    reason_codes: tuple[str, ...]
    policy_version: str
    latency_ms: float

    def __post_init__(self) -> None:
        if self.action == "block" and self.released_text is not None:
            raise ValueError("block action requires released_text is None")
        if self.action != "block" and (self.released_text is None or self.released_text == ""):
            raise ValueError(f"action '{self.action}' requires non-empty released_text")
        if self.latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass(frozen=True)
class RecipientHistoryItem:
    """A message released to a recipient."""

    message_id: str
    turn_id: int
    sender_id: str
    released_text: str

    def to_dict(self) -> dict:
        return asdict(self)


class ContaminationStatus(str, Enum):
    """Agent contamination state relative to a forget target."""

    UNKNOWN = "unknown"
    CONTAMINATED = "contaminated"
    CLEAN = "clean"
    VERIFIED = "verified"
    AT_RISK = "at_risk"
    RECONTAMINATED = "recontaminated"
