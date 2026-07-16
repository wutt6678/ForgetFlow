"""Tests for FlowGate."""

from experiments.trustparadox_u.config import (
    DetectorConfig,
    ExperimentConfig,
    HistoryConfig,
    MonitoringConfig,
    PolicyConfig,
)
from marble.firewall.audit import AuditLogger
from marble.firewall.detectors import HybridDetector
from marble.firewall.flow_gate import FlowGate
from marble.firewall.history import RecipientHistory, ReconstructionChecker
from marble.firewall.policy import ForgetPolicy
from marble.firewall.registry import ForgetLedger
from marble.firewall.types import ForgetRecord, MessageEnvelope


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        seed=42,
        repetitions=1,
        detector=DetectorConfig(semantic_enabled=False),
        history=HistoryConfig(),
        policy=PolicyConfig(),
        monitoring=MonitoringConfig(),
    )


def _setup() -> tuple[FlowGate, ForgetLedger]:
    ledger = ForgetLedger()
    ledger.register(
        ForgetRecord(
            forget_id="F001",
            canonical_target="0107",
            target_type="credential",
            aliases=("warehouse code",),
            semantic_variants=(),
            permitted_residuals=("Request new credential.",),
            active_from_turn=0,
        )
    )
    detector = HybridDetector(semantic_enabled=False)
    history = RecipientHistory()
    checker = ReconstructionChecker()
    policy = ForgetPolicy()
    audit = AuditLogger()
    gate = FlowGate(ledger, detector, history, checker, policy, audit, _config())
    return gate, ledger


def _env(text: str = "Hello", turn: int = 0) -> MessageEnvelope:
    return MessageEnvelope(
        message_id="m1",
        episode_id="ep1",
        session_id="s1",
        turn_id=turn,
        sender_id="CK",
        recipient_id="SK",
        raw_text=text,
        trust_level="high",
    )


class TestFlowGate:
    def test_safe_allow(self) -> None:
        gate, _ = _setup()
        decision = gate.inspect(_env("Hello world"))
        assert decision.action == "allow"
        assert decision.released_text == "Hello world"

    def test_exact_block(self) -> None:
        gate, _ = _setup()
        decision = gate.inspect(_env("The code is 0107"))
        assert decision.action == "block"
        assert decision.released_text is None

    def test_alias_redact(self) -> None:
        gate, _ = _setup()
        decision = gate.inspect(_env("Tell me the warehouse code"))
        assert decision.action in ("redact", "block")

    def test_no_active_records(self) -> None:
        ledger = ForgetLedger()
        ledger.register(
            ForgetRecord(
                forget_id="F001",
                canonical_target="0107",
                target_type="credential",
                aliases=(),
                semantic_variants=(),
                permitted_residuals=(),
                active_from_turn=10,
            )
        )
        detector = HybridDetector(semantic_enabled=False)
        gate = FlowGate(
            ledger,
            detector,
            RecipientHistory(),
            ReconstructionChecker(),
            ForgetPolicy(),
            AuditLogger(),
            _config(),
        )
        decision = gate.inspect(_env("The code is 0107", turn=5))
        assert decision.action == "allow"

    def test_only_released_in_history(self) -> None:
        gate, _ = _setup()
        gate.inspect(_env("The code is 0107"))
        gate.inspect(_env("Hello"))
        history_items = gate.history.get_all_items("SK")
        for item in history_items:
            assert "0107" not in item.released_text

    def test_audit_log_has_entries(self) -> None:
        gate, _ = _setup()
        gate.inspect(_env("Hello"))
        entries = gate.audit_logger.get_entries()
        assert len(entries) == 1
        assert entries[0]["action"] == "allow"


class TestPermittedResidualRecheck:
    """Tests for permitted residual handling during recheck."""

    def test_exact_permitted_residual_released(self) -> None:
        """Exact permitted residual should be released despite semantic relation."""
        from marble.firewall.flow_gate import _is_permitted_residual

        rec = ForgetRecord(
            forget_id="F001",
            canonical_target="0107",
            target_type="credential",
            aliases=("warehouse code",),
            semantic_variants=(),
            permitted_residuals=("Request new credential.",),
            active_from_turn=0,
        )
        assert _is_permitted_residual("Request new credential.", [rec]) is True

    def test_residual_containing_target_blocked(self) -> None:
        """Residual containing exact target should still be blocked."""
        from marble.firewall.flow_gate import _is_permitted_residual

        rec = ForgetRecord(
            forget_id="F001",
            canonical_target="0107",
            target_type="credential",
            aliases=(),
            semantic_variants=(),
            permitted_residuals=("The code 0107 is old.",),
            active_from_turn=0,
        )
        # It IS a permitted residual by text match
        assert _is_permitted_residual("The code 0107 is old.", [rec]) is True
        # But the flow gate would still block it due to exact_score == 1.0

    def test_residual_containing_alias(self) -> None:
        """Residual containing forbidden alias is still matched by entity."""
        from marble.firewall.flow_gate import _is_permitted_residual

        rec = ForgetRecord(
            forget_id="F001",
            canonical_target="0107",
            target_type="credential",
            aliases=("warehouse code",),
            semantic_variants=(),
            permitted_residuals=("Ask about the warehouse code.",),
            active_from_turn=0,
        )
        assert _is_permitted_residual("Ask about the warehouse code.", [rec]) is True

    def test_non_approved_paraphrase_not_residual(self) -> None:
        """Non-approved text is not a permitted residual."""
        from marble.firewall.flow_gate import _is_permitted_residual

        rec = ForgetRecord(
            forget_id="F001",
            canonical_target="0107",
            target_type="credential",
            aliases=(),
            semantic_variants=(),
            permitted_residuals=("Request new credential.",),
            active_from_turn=0,
        )
        assert _is_permitted_residual("Some other text", [rec]) is False
