"""ForgetPolicy: deterministic enforcement policy."""

from __future__ import annotations

import re
from typing import Sequence

from marble.firewall.types import DetectorResult, FirewallAction, ForgetRecord


class ForgetPolicy:
    """Deterministic policy choosing allow/redact/abstract/block."""

    def __init__(
        self,
        rich_actions_enabled: bool = True,
        embedding_threshold: float = 0.80,
        reconstruction_threshold: float = 0.60,
        trust_independent: bool = True,
    ) -> None:
        self.rich_actions_enabled = rich_actions_enabled
        self.embedding_threshold = embedding_threshold
        self.reconstruction_threshold = reconstruction_threshold
        self.trust_independent = trust_independent

    def decide(
        self,
        detector_result: DetectorResult,
        active_records: Sequence[ForgetRecord],
        policy_version: str,
    ) -> tuple[FirewallAction, str | None, tuple[str, ...]]:
        """Return (action, released_text_or_none, reason_codes)."""
        reasons: list[str] = []

        # Exact match → block or abstract (if rich actions enabled and residual available)
        if detector_result.exact_score == 1.0:
            reasons.append("EXACT_TARGET_MATCH")
            if self.rich_actions_enabled:
                residual = self._find_residual(active_records, detector_result)
                if residual:
                    return ("abstract", residual, tuple(reasons))
            return ("block", None, tuple(reasons))

        # Reconstruction above threshold → block
        if detector_result.reconstruction_score >= self.reconstruction_threshold:
            reasons.append("RECONSTRUCTION_RISK")
            return ("block", None, tuple(reasons))

        # Semantic above threshold
        if detector_result.semantic_score >= self.embedding_threshold:
            reasons.append("SEMANTIC_SIMILARITY")
            if self.rich_actions_enabled:
                residual = self._find_residual(active_records, detector_result)
                if residual:
                    return ("abstract", residual, tuple(reasons))
            return ("block", None, tuple(reasons))

        # Entity/alias match
        if detector_result.entity_score > 0:
            reasons.append("ENTITY_MATCH")
            if self.rich_actions_enabled:
                redacted = self._redact(detector_result, active_records)
                return ("redact", redacted, tuple(reasons))
            return ("block", None, tuple(reasons))

        reasons.append("NO_LEAKAGE_DETECTED")
        return ("allow", None, tuple(reasons))

    def _find_residual(
        self,
        active_records: Sequence[ForgetRecord],
        detector_result: DetectorResult,
    ) -> str | None:
        for rec in active_records:
            if rec.forget_id in detector_result.matched_forget_ids:
                if rec.permitted_residuals:
                    return rec.permitted_residuals[0]
        return None

    def _redact(
        self,
        detector_result: DetectorResult,
        active_records: Sequence[ForgetRecord],
    ) -> str | None:
        return None  # Redaction requires original text, handled in FlowGate

    def redact_text(
        self,
        text: str,
        active_records: Sequence[ForgetRecord],
        detector_result: DetectorResult,
    ) -> str:
        result = text
        for rec in active_records:
            if rec.forget_id not in detector_result.matched_forget_ids:
                continue
            pattern = re.compile(re.escape(rec.canonical_target), re.IGNORECASE)
            result = pattern.sub("[REDACTED]", result)
            for alias in rec.aliases:
                pattern = re.compile(re.escape(alias), re.IGNORECASE)
                result = pattern.sub("[REDACTED]", result)
        return result
