"""FlowGate: the single message-decision point combining all firewall components."""

from __future__ import annotations

import time
from typing import Any, Mapping

from experiments.trustparadox_u.config import ExperimentConfig
from marble.firewall.audit import AuditLogger
from marble.firewall.detectors import HybridDetector
from marble.firewall.history import RecipientHistory, ReconstructionChecker
from marble.firewall.policy import ForgetPolicy
from marble.firewall.registry import ForgetLedger
from marble.firewall.types import (
    DetectorResult,
    FirewallDecision,
    MessageEnvelope,
    RecipientHistoryItem,
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

        # 4. Run reconstruction checker
        recon_score = self.reconstruction_checker.score(
            envelope.raw_text,
            ctx,
            active,
            self.episode_metadata,
            history_enabled=self.config.history.enabled,
            reconstruction_threshold=self.config.history.reconstruction_threshold,
        )

        # 5. Merge reconstruction score
        det_result = DetectorResult(
            exact_score=det_result.exact_score,
            entity_score=det_result.entity_score,
            semantic_score=det_result.semantic_score,
            reconstruction_score=recon_score,
            matched_forget_ids=det_result.matched_forget_ids,
            evidence=det_result.evidence,
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
            recheck_recon = self.reconstruction_checker.score(
                released_text,
                ctx,
                active,
                self.episode_metadata,
                history_enabled=self.config.history.enabled,
                reconstruction_threshold=self.config.history.reconstruction_threshold,
            )
            # Check all risk dimensions: exact, entity, semantic, reconstruction
            semantic_threshold = self.config.detector.semantic_threshold
            unsafe = (
                recheck_det.exact_score == 1.0
                or recheck_det.entity_score > 0.0
                or (
                    self.config.detector.semantic_enabled
                    and recheck_det.semantic_score >= semantic_threshold
                )
                or recheck_recon >= self.config.history.reconstruction_threshold
            )
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
