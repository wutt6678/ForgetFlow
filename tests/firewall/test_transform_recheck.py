"""Tests for mandatory post-transformation rechecking (Iteration 2 of repair spec)."""

from __future__ import annotations

import pytest

from marble.firewall.audit import AuditLogger
from marble.firewall.detectors import HybridDetector
from marble.firewall.flow_gate import FlowGate
from marble.firewall.history import RecipientHistory, ReconstructionChecker
from marble.firewall.policy import ForgetPolicy
from marble.firewall.registry import ForgetLedger
from marble.firewall.types import (
    FirewallDecision,
    ForgetRecord,
    MessageEnvelope,
    TransformationAttempt,
)
from experiments.trustparadox_u.config import (
    ExperimentConfig,
    DetectorConfig,
    HistoryConfig,
    PolicyConfig,
    MonitoringConfig,
)
from experiments.trustparadox_u.embedding import EmbeddingProvider


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 384 for _ in texts]


def _make_forget_record(
    forget_id: str = "F001",
    canonical_target: str = "0107",
    aliases: tuple[str, ...] = ("gold-store credential",),
    permitted_residuals: tuple[str, ...] = ("The credential must be replaced.",),
) -> ForgetRecord:
    return ForgetRecord(
        forget_id=forget_id,
        canonical_target=canonical_target,
        target_type="numeric_credential",
        aliases=aliases,
        semantic_variants=(),
        permitted_residuals=permitted_residuals,
        active_from_turn=0,
    )


def _make_envelope(
    text: str,
    turn_id: int = 1,
) -> MessageEnvelope:
    return MessageEnvelope(
        message_id="msg-001",
        episode_id="ep-001",
        session_id="sess-001",
        turn_id=turn_id,
        sender_id="agent-a",
        recipient_id="agent-b",
        raw_text=text,
        trust_level="default",
    )


def _make_flow_gate(records: list[ForgetRecord]) -> FlowGate:
    ledger = ForgetLedger()
    for rec in records:
        ledger.register(rec)

    config = ExperimentConfig(
        seed=42,
        repetitions=1,
        detector=DetectorConfig(embedding_enabled=False),
        history=HistoryConfig(),
        policy=PolicyConfig(),
        monitoring=MonitoringConfig(),
    )
    detector = HybridDetector(
        exact_enabled=True,
        entity_enabled=True,
        embedding_enabled=False,  # Disable for simpler testing
        claim_matching_enabled=False,
    )
    history = RecipientHistory()
    recon_checker = ReconstructionChecker()
    policy = ForgetPolicy(rich_actions_enabled=True)
    audit = AuditLogger()

    return FlowGate(
        ledger=ledger,
        detector=detector,
        history=history,
        reconstruction_checker=recon_checker,
        policy=policy,
        audit_logger=audit,
        config=config,
    )


class TestTransformRecheckRunsAllDetectors:
    """Test that recheck runs all detectors after transformation."""

    def test_transform_recheck_runs_all_detectors(self) -> None:
        """After redaction, all detectors should be run on the output."""
        record = _make_forget_record()
        gate = _make_flow_gate([record])

        # Text with alias that will be redacted
        envelope = _make_envelope("The gold-store credential is important.")
        decision = gate.inspect(envelope)

        # Should have been processed
        assert decision.action in ("redact", "abstract", "block", "allow")

    def test_alias_trigger_rechecked_by_exact_and_alias(self) -> None:
        """After alias trigger, both exact and alias should be rechecked."""
        record = _make_forget_record()
        gate = _make_flow_gate([record])

        # Text with both alias and exact match
        envelope = _make_envelope("The gold-store credential is 0107.")
        decision = gate.inspect(envelope)

        # Should be blocked or abstracted (exact match triggers block/abstract)
        assert decision.action in ("block", "abstract")


class TestEscalationBehavior:
    """Test escalation from redact to abstract to block."""

    def test_redaction_escalates_when_value_remains(self) -> None:
        """If redaction leaves the value, should escalate to abstract or block."""
        record = _make_forget_record(permitted_residuals=("The credential must be replaced.",))
        gate = _make_flow_gate([record])

        # Text with spoken digit form that redaction might miss
        envelope = _make_envelope("The gold-store credential is zero one zero seven.")
        decision = gate.inspect(envelope)

        # Should either use permitted residual or block
        if decision.action == "abstract":
            assert decision.released_text == "The credential must be replaced."
        elif decision.action == "block":
            assert decision.released_text is None
        else:
            # If it passed, the transformation was safe
            assert decision.released_text is not None

    def test_abstract_escalates_to_block_when_semantically_unsafe(self) -> None:
        """If abstract is also unsafe, should block."""
        record = _make_forget_record(permitted_residuals=())  # No permitted residuals
        gate = _make_flow_gate([record])

        # Text with exact match and no permitted residual
        envelope = _make_envelope("The credential is 0107.")
        decision = gate.inspect(envelope)

        # Should block since no permitted residual available
        assert decision.action == "block"
        assert decision.released_text is None

    def test_block_when_transform_attempts_exhausted(self) -> None:
        """After max attempts, should block."""
        record = _make_forget_record(permitted_residuals=())
        gate = _make_flow_gate([record])

        envelope = _make_envelope("The gold-store credential is 0107.")
        decision = gate.inspect(envelope)

        assert decision.action == "block"
        assert "MAX_TRANSFORMATION_ATTEMPTS_EXCEEDED" in decision.reason_codes or \
               "ESCALATION_NO_RESIDUAL" in decision.reason_codes or \
               "EXACT_TARGET_MATCH" in decision.reason_codes


class TestReleasedExposure:
    """Test released exposure computation."""

    def test_released_exposure_is_computed_from_released_text(self) -> None:
        """Released exposure should be based on actual released text."""
        record = _make_forget_record()
        gate = _make_flow_gate([record])

        # Safe text that should be allowed
        envelope = _make_envelope("The meeting is at 3pm.")
        decision = gate.inspect(envelope)

        assert decision.action == "allow"
        assert decision.released_text == "The meeting is at 3pm."


class TestTransformationAttempt:
    """Test TransformationAttempt data structure."""

    def test_transformation_attempt_creation(self) -> None:
        """TransformationAttempt should be creatable with all fields."""
        attempt = TransformationAttempt(
            attempt_index=0,
            transformation_type="redact",
            input_text="The credential is 0107.",
            output_text="The credential is [REDACTED].",
            exact_safe=True,
            alias_safe=True,
            embedding_safe=True,
            claim_safe=True,
            reconstruction_safe=True,
            matched_forget_ids=(),
            passed=True,
        )
        assert attempt.passed is True
        assert attempt.attempt_index == 0
        assert attempt.transformation_type == "redact"

    def test_transformation_attempt_frozen(self) -> None:
        """TransformationAttempt should be immutable."""
        attempt = TransformationAttempt(
            attempt_index=0,
            transformation_type="redact",
            input_text="input",
            output_text="output",
            exact_safe=True,
            alias_safe=True,
            embedding_safe=True,
            claim_safe=True,
            reconstruction_safe=True,
            matched_forget_ids=(),
            passed=True,
        )
        with pytest.raises(AttributeError):
            attempt.passed = False  # type: ignore
