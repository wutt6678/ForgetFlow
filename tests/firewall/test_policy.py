"""Tests for ForgetPolicy."""

from marble.firewall.policy import ForgetPolicy
from marble.firewall.types import DetectorResult, ForgetRecord


def _rec() -> ForgetRecord:
    return ForgetRecord(
        forget_id="F001",
        canonical_target="0107",
        target_type="credential",
        aliases=("warehouse code",),
        semantic_variants=(),
        permitted_residuals=("Request a new credential.",),
        active_from_turn=0,
    )


def _det(**kwargs: object) -> DetectorResult:
    defaults = dict(
        exact_score=0.0,
        entity_score=0.0,
        semantic_score=0.0,
        reconstruction_score=0.0,
        matched_forget_ids=(),
        evidence=(),
    )
    defaults.update(kwargs)
    return DetectorResult(**defaults)  # type: ignore[arg-type]


class TestForgetPolicy:
    def test_exact_blocks(self) -> None:
        policy = ForgetPolicy()
        action, text, reasons = policy.decide(
            _det(exact_score=1.0, matched_forget_ids=("F001",)),
            [_rec()],
            "v1",
        )
        assert action == "block"
        assert text is None

    def test_reconstruction_blocks(self) -> None:
        policy = ForgetPolicy(reconstruction_threshold=0.6)
        action, text, reasons = policy.decide(
            _det(reconstruction_score=0.8, matched_forget_ids=("F001",)),
            [_rec()],
            "v1",
        )
        assert action == "block"

    def test_semantic_abstracts(self) -> None:
        policy = ForgetPolicy(semantic_threshold=0.8)
        action, text, reasons = policy.decide(
            _det(semantic_score=0.9, matched_forget_ids=("F001",)),
            [_rec()],
            "v1",
        )
        assert action == "abstract"
        assert text == "Request a new credential."

    def test_alias_redacts(self) -> None:
        policy = ForgetPolicy()
        action, text, reasons = policy.decide(
            _det(entity_score=1.0, matched_forget_ids=("F001",)),
            [_rec()],
            "v1",
        )
        assert action == "redact"

    def test_safe_allows(self) -> None:
        policy = ForgetPolicy()
        action, text, reasons = policy.decide(_det(), [_rec()], "v1")
        assert action == "allow"

    def test_binary_policy_blocks_alias(self) -> None:
        policy = ForgetPolicy(rich_actions_enabled=False)
        action, text, reasons = policy.decide(
            _det(entity_score=1.0, matched_forget_ids=("F001",)),
            [_rec()],
            "v1",
        )
        assert action == "block"

    def test_binary_policy_blocks_semantic(self) -> None:
        policy = ForgetPolicy(rich_actions_enabled=False, semantic_threshold=0.8)
        action, text, reasons = policy.decide(
            _det(semantic_score=0.9, matched_forget_ids=("F001",)),
            [_rec()],
            "v1",
        )
        assert action == "block"

    def test_reason_codes_present(self) -> None:
        policy = ForgetPolicy()
        _, _, reasons = policy.decide(
            _det(exact_score=1.0, matched_forget_ids=("F001",)),
            [_rec()],
            "v1",
        )
        assert "EXACT_TARGET_MATCH" in reasons

    def test_deterministic(self) -> None:
        policy = ForgetPolicy()
        r1 = policy.decide(_det(exact_score=1.0), [_rec()], "v1")
        r2 = policy.decide(_det(exact_score=1.0), [_rec()], "v1")
        assert r1 == r2

    def test_redact_text(self) -> None:
        policy = ForgetPolicy()
        det = _det(entity_score=1.0, matched_forget_ids=("F001",))
        result = policy.redact_text("The warehouse code is 0107", [_rec()], det)
        assert "0107" not in result
        assert "[REDACTED]" in result
