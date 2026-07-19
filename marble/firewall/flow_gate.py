"""FlowGate: the single message-decision point combining all firewall components."""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from experiments.trustparadox_u.config import ExperimentConfig
from marble.firewall.audit import AuditLogger
from marble.firewall.detectors import HybridDetector
from marble.firewall.history import RecipientHistory, ReconstructionChecker
from marble.firewall.policy import ForgetPolicy
from marble.firewall.registry import ForgetLedger
from marble.firewall.types import (
    DetectorResult,
    FirewallDecision,
    ForgetRecord,
    MessageEnvelope,
    RecipientHistoryItem,
    RecordDetectionEvidence,
)


class FlowGate:
    """Combines ledger, detector, history, reconstruction, and policy."""

    def __init__(
        self,
        ledger: ForgetLedger,
        detector: HybridDetector,
        history: RecipientHistory,
        reconstruction_checker: ReconstructionChecker,
        policy: ForgetPolicy,
        audit_logger: AuditLogger,
        config: ExperimentConfig,
        episode_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.ledger = ledger
        self.detector = detector
        self.history = history
        self.reconstruction_checker = reconstruction_checker
        self.policy = policy
        self.audit_logger = audit_logger
        self.config = config
        self.episode_metadata = episode_metadata or {}

    def inspect(self, envelope: MessageEnvelope) -> FirewallDecision:
        start = time.monotonic()

        # 1. Load active records
        active = self.ledger.active_records(
            envelope.turn_id, envelope.sender_id, envelope.recipient_id
        )

        if not active:
            elapsed = (time.monotonic() - start) * 1000
            det = DetectorResult(
                exact_score=0.0,
                entity_score=0.0,
                semantic_score=0.0,
                reconstruction_score=0.0,
                matched_forget_ids=(),
                evidence=(),
            )
            decision = FirewallDecision(
                action="allow",
                released_text=envelope.raw_text,
                detector_result=det,
                reason_codes=("NO_ACTIVE_RECORDS",),
                policy_version=self.ledger.policy_version(),
                latency_ms=elapsed,
            )
            self._record(envelope, decision)
            return decision

        # 2. Load recipient context
        ctx = self.history.get_context(envelope.recipient_id, self.config.history.window_size)

        # 3. Run detector
        det_result = self.detector.detect(envelope.raw_text, active, ctx)

        # 4. Run reconstruction checker (per-record scoring for P0 #4)
        # Compute per-record reconstruction scores to ensure target-specificity.
        # A candidate like "7391" must not be treated as reconstruction of "0107".
        per_record_recon_scores: dict[str, float] = {}
        for rec in active:
            per_record_recon_scores[rec.forget_id] = self.reconstruction_checker.score(
                envelope.raw_text,
                ctx,
                active,
                self.episode_metadata,
                history_enabled=self.config.history.enabled,
                reconstruction_threshold=self.config.history.reconstruction_threshold,
                forget_id=rec.forget_id,
            )
        # Aggregate: max score across all records (for policy decision)
        recon_score = max(per_record_recon_scores.values()) if per_record_recon_scores else 0.0

        # 5. Merge reconstruction score and update per-record evidence
        updated_record_evidence = []
        for ev in det_result.record_evidence:
            ev_recon = per_record_recon_scores.get(ev.forget_id, 0.0)
            updated_record_evidence.append(
                RecordDetectionEvidence(
                    forget_id=ev.forget_id,
                    exact_score=ev.exact_score,
                    entity_score=ev.entity_score,
                    semantic_score=ev.semantic_score,
                    reconstruction_score=ev_recon,
                    matched=ev.matched,
                )
            )
        det_result = DetectorResult(
            exact_score=det_result.exact_score,
            entity_score=det_result.entity_score,
            semantic_score=det_result.semantic_score,
            reconstruction_score=recon_score,
            matched_forget_ids=det_result.matched_forget_ids,
            evidence=det_result.evidence,
            record_evidence=tuple(updated_record_evidence),
        )

        # 6. Run policy
        action, released_text, reasons = self.policy.decide(
            det_result, active, self.ledger.policy_version()
        )

        # Handle redaction
        if action == "redact" and released_text is None:
            released_text = self.policy.redact_text(envelope.raw_text, active, det_result)
            if not released_text or released_text == envelope.raw_text:
                action = "block"
                released_text = None
                reasons = reasons + ("REDACT_FAILED",)

        # Handle allow
        if action == "allow":
            released_text = envelope.raw_text

        # 7. Recheck transformed output
        if action in ("redact", "abstract") and released_text is not None:
            recheck_det = self.detector.detect(released_text, active, ctx)
            # Per-record reconstruction recheck (P0 #4)
            recheck_recon_scores = []
            for rec in active:
                recheck_recon_scores.append(
                    self.reconstruction_checker.score(
                        released_text,
                        ctx,
                        active,
                        self.episode_metadata,
                        history_enabled=self.config.history.enabled,
                        reconstruction_threshold=self.config.history.reconstruction_threshold,
                        forget_id=rec.forget_id,
                    )
                )
            recheck_recon = max(recheck_recon_scores) if recheck_recon_scores else 0.0

            # Determine if released text is a permitted residual
            permitted_residual = _is_permitted_residual(released_text, active)

            # Exact or alias match always blocks (even for permitted residuals)
            exact_or_alias = recheck_det.exact_score == 1.0 or recheck_det.entity_score > 0.0
            # Reconstruction risk always blocks
            reconstructive = recheck_recon >= self.config.history.reconstruction_threshold
            # Semantic unsafe: above threshold AND not a permitted residual
            embedding_threshold = self.config.detector.embedding_threshold
            semantic_unsafe = (
                self.config.detector.embedding_enabled
                and recheck_det.semantic_score >= embedding_threshold
                and not permitted_residual
            )

            unsafe = exact_or_alias or reconstructive or semantic_unsafe
            if unsafe:
                action = "block"
                released_text = None
                reasons = reasons + ("RECHECK_FAILED",)

        elapsed = (time.monotonic() - start) * 1000
        decision = FirewallDecision(
            action=action,
            released_text=released_text,
            detector_result=det_result,
            reason_codes=reasons,
            policy_version=self.ledger.policy_version(),
            latency_ms=elapsed,
        )

        # 8. Log and record
        self._record(envelope, decision)
        return decision

    def _record(self, envelope: MessageEnvelope, decision: FirewallDecision) -> None:
        self.audit_logger.log(envelope, decision)
        if decision.released_text is not None:
            self.history.append(
                envelope.recipient_id,
                RecipientHistoryItem(
                    message_id=envelope.message_id,
                    turn_id=envelope.turn_id,
                    sender_id=envelope.sender_id,
                    released_text=decision.released_text,
                ),
            )


def _is_permitted_residual(
    text: str,
    active_records: Sequence[ForgetRecord],
) -> bool:
    """Check if text matches a permitted residual (normalized exact match)."""
    text_lower = text.lower().strip()
    for rec in active_records:
        for residual in rec.permitted_residuals:
            if residual.lower().strip() == text_lower:
                return True
    return False
