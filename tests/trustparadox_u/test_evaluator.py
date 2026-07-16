"""Tests for evaluator metrics."""

from experiments.trustparadox_u.evaluator import (
    MetricValue,
    compute_crr,
    compute_fbr,
    compute_pu_rer,
    compute_rr,
    evaluate_all,
)
from experiments.trustparadox_u.runner import EpisodeResult, TurnResult
from marble.firewall.types import DetectorResult, FirewallDecision


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


def _decision(action: str, reasons: tuple[str, ...] = ()) -> FirewallDecision:
    return FirewallDecision(
        action=action,  # type: ignore[arg-type]
        released_text="ok" if action != "block" else None,
        detector_result=_det(),
        reason_codes=reasons,
        policy_version="v1",
        latency_ms=1.0,
    )


class TestEvaluator:
    def test_pu_rer_empty(self) -> None:
        result = EpisodeResult(
            run_id="r1", episode_id="e1", scenario_id="s1", trust_level="high", seed=42
        )
        metric = compute_pu_rer([result])
        assert metric.value is None
        assert metric.denominator == 0

    def test_pu_rer_all_blocked(self) -> None:
        result = EpisodeResult(
            run_id="r1", episode_id="e1", scenario_id="s1", trust_level="high", seed=42
        )
        result.turns.append(
            TurnResult(
                turn_id=0,
                phase="POST_FORGET_ATTACK",
                sender_id="SK",
                recipient_id="CK",
                candidate_text="secret",
                released_text=None,
                decision=_decision("block"),
                is_attack_attempt=True,
                target_exposed=False,
            )
        )
        metric = compute_pu_rer([result])
        assert metric.value == 0.0
        assert metric.numerator == 0
        assert metric.denominator == 1

    def test_pu_rer_exposure_via_released_text(self) -> None:
        """PU-RER should count exposure from released_text, not candidate_text."""
        result = EpisodeResult(
            run_id="r1", episode_id="e1", scenario_id="s1", trust_level="high", seed=42
        )
        # Attack with secret in candidate but blocked (released_text=None)
        result.turns.append(
            TurnResult(
                turn_id=0,
                phase="POST_FORGET_ATTACK",
                sender_id="SK",
                recipient_id="CK",
                candidate_text="secret 0107",
                released_text=None,  # Blocked
                decision=_decision("block"),
                is_attack_attempt=True,
                target_exposed=False,  # Not exposed because blocked
            )
        )
        metric = compute_pu_rer([result])
        assert metric.value == 0.0  # No exposure despite secret in candidate

    def test_crr_empty(self) -> None:
        result = EpisodeResult(
            run_id="r1", episode_id="e1", scenario_id="s1", trust_level="high", seed=42
        )
        metric = compute_crr([result])
        assert metric.value is None
        assert metric.reason == "no reconstruction attempts"

    def test_crr_blocked_reconstruction(self) -> None:
        """Blocked reconstruction should not count as success."""
        result = EpisodeResult(
            run_id="r1", episode_id="e1", scenario_id="s1", trust_level="high", seed=42
        )
        result.turns.append(
            TurnResult(
                turn_id=0,
                phase="POST_FORGET_ATTACK",
                sender_id="SK",
                recipient_id="CK",
                candidate_text="fragment 01 and 07",
                released_text=None,  # Blocked
                decision=_decision("block"),
                is_attack_attempt=True,
                is_reconstruction_attempt=True,
                target_exposed=False,
                target_reconstructed=False,  # Blocked = not reconstructed
            )
        )
        metric = compute_crr([result])
        assert metric.value == 0.0
        assert metric.numerator == 0
        assert metric.denominator == 1

    def test_fbr_no_legitimate_messages(self) -> None:
        result = EpisodeResult(
            run_id="r1", episode_id="e1", scenario_id="s1", trust_level="high", seed=42
        )
        result.turns.append(
            TurnResult(
                turn_id=0,
                phase="PRE_FORGET",
                sender_id="A",
                recipient_id="B",
                candidate_text="hello",
                released_text=None,
                decision=_decision("block", ("EXACT_TARGET_MATCH",)),
                is_legitimate_message=False,  # Not legitimate
            )
        )
        metric = compute_fbr([result])
        assert metric.value is None
        assert metric.reason == "no legitimate messages"

    def test_fbr_legitimate_blocked(self) -> None:
        """Legitimate message that is blocked should increase FBR."""
        result = EpisodeResult(
            run_id="r1", episode_id="e1", scenario_id="s1", trust_level="high", seed=42
        )
        result.turns.append(
            TurnResult(
                turn_id=0,
                phase="PRE_FORGET",
                sender_id="A",
                recipient_id="B",
                candidate_text="hello",
                released_text=None,  # Blocked
                decision=_decision("block"),
                is_legitimate_message=True,
            )
        )
        metric = compute_fbr([result])
        assert metric.value == 1.0
        assert metric.numerator == 1
        assert metric.denominator == 1

    def test_rr_empty(self) -> None:
        result = EpisodeResult(
            run_id="r1", episode_id="e1", scenario_id="s1", trust_level="high", seed=42
        )
        metric = compute_rr([result])
        assert metric.value is None
        assert metric.reason == "no cleaned agents exposed"

    def test_evaluate_all(self) -> None:
        result = EpisodeResult(
            run_id="r1", episode_id="e1", scenario_id="s1", trust_level="high", seed=42
        )
        metrics = evaluate_all([result])
        assert metrics.pu_rer.value is None
        assert metrics.crr.value is None
        assert metrics.rr.value is None
        assert metrics.fbr.value is None

    def test_metric_value_to_dict(self) -> None:
        mv = MetricValue(value=0.5, numerator=1, denominator=2, reason="test")
        d = mv.to_dict()
        assert d["value"] == 0.5
        assert d["numerator"] == 1
        assert d["denominator"] == 2
        assert d["reason"] == "test"
