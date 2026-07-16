"""Core data types for the ForgetFlow firewall system."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Literal

FirewallAction = Literal["allow", "redact", "abstract", "block"]

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
class DetectorResult:
    """Result from the hybrid leakage detector."""

    exact_score: float
    entity_score: float
    semantic_score: float
    reconstruction_score: float
    matched_forget_ids: tuple[str, ...]
    evidence: tuple[str, ...]

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
