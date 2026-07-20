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
from marble.firewall.normalization import text_contains_canonical_value
from marble.firewall.types import (
    DetectorResult,
    FirewallDecision,
    ForgetRecord,
    MessageEnvelope,
    RecipientHistoryItem,
    RecordDetectionEvidence,
    TransformationAttempt,
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

        # 7. Recheck transformed output with escalation
        transformation_attempts: list[TransformationAttempt] = []
        if action in ("redact", "abstract") and released_text is not None:
            action, released_text, reasons, transformation_attempts = self._recheck_and_escalate(
                action, released_text, envelope.raw_text, active, ctx, reasons
            )

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

    def _recheck_and_escalate(
        self,
        action: str,
        released_text: str,
        original_text: str,
        active: Sequence[ForgetRecord],
        ctx: Any,
        reasons: tuple[str, ...],
    ) -> tuple[str, str | None, tuple[str, ...], list[TransformationAttempt]]:
        """Recheck transformed output and escalate if unsafe.

        Escalation path: redact -> abstract -> block
        Maximum 2 transformation attempts before blocking.
        """
        attempts: list[TransformationAttempt] = []
        max_attempts = 2
        current_action = action
        current_text = released_text

        for attempt_idx in range(max_attempts):
            # Run ALL detectors on transformed output
            recheck_det = self.detector.detect(current_text, active, ctx)

            # Per-record reconstruction recheck
            recheck_recon_scores = []
            for rec in active:
                recheck_recon_scores.append(
                    self.reconstruction_checker.score(
                        current_text,
                        ctx,
                        active,
                        self.episode_metadata,
                        history_enabled=self.config.history.enabled,
                        reconstruction_threshold=self.config.history.reconstruction_threshold,
                        forget_id=rec.forget_id,
                    )
                )
            recheck_recon = max(recheck_recon_scores) if recheck_recon_scores else 0.0

            # Check each safety dimension
            exact_safe = recheck_det.exact_score < 1.0
            alias_safe = recheck_det.entity_score == 0.0
            embedding_threshold = self.config.detector.embedding_threshold
            embedding_safe = (
                not self.config.detector.embedding_enabled
                or recheck_det.semantic_score < embedding_threshold
            )
            # Claim safe: no proposition entailed
            claim_safe = not any(ev.proposition_entailed for ev in recheck_det.record_evidence)
            reconstruction_safe = recheck_recon < self.config.history.reconstruction_threshold

            # Also check canonical value presence (spoken digits, etc.)
            for rec in active:
                if text_contains_canonical_value(current_text, rec.canonical_target):
                    exact_safe = False
                    alias_safe = False

            passed = exact_safe and alias_safe and embedding_safe and claim_safe and reconstruction_safe

            # Record this attempt
            attempt = TransformationAttempt(
                attempt_index=attempt_idx,
                transformation_type=current_action,
                input_text=original_text if attempt_idx == 0 else current_text,
                output_text=current_text,
                exact_safe=exact_safe,
                alias_safe=alias_safe,
                embedding_safe=embedding_safe,
                claim_safe=claim_safe,
                reconstruction_safe=reconstruction_safe,
                matched_forget_ids=recheck_det.matched_forget_ids,
                passed=passed,
            )
            attempts.append(attempt)

            if passed:
                # Transformation succeeded
                return current_action, current_text, reasons, attempts

            # Transformation failed - escalate
            if current_action == "redact":
                # Try abstract (use permitted residual)
                current_action = "abstract"
                residual = self.policy._find_residual(active, recheck_det)
                if residual:
                    current_text = residual
                else:
                    # No permitted residual available, block
                    return "block", None, reasons + ("ESCALATION_NO_RESIDUAL",), attempts
            elif current_action == "abstract":
                # Abstract also failed, block
                return "block", None, reasons + ("ESCALATION_FAILED",), attempts

        # Max attempts exhausted
        return "block", None, reasons + ("MAX_TRANSFORMATION_ATTEMPTS_EXCEEDED",), attempts


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
