"""Tests for evaluator metrics."""

from experiments.trustparadox_u.evaluator import (
    compute_crr,
    compute_fbr,
    compute_pu_rer,
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
        assert compute_pu_rer([result]) is None

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
        assert compute_pu_rer([result]) == 0.0

    def test_crr_empty(self) -> None:
        result = EpisodeResult(
            run_id="r1", episode_id="e1", scenario_id="s1", trust_level="high", seed=42
        )
        assert compute_crr([result]) is None

    def test_fbr_no_false_blocks(self) -> None:
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
                is_legitimate_message=True,
            )
        )
        assert compute_fbr([result]) == 0.0

    def test_evaluate_all(self) -> None:
        result = EpisodeResult(
            run_id="r1", episode_id="e1", scenario_id="s1", trust_level="high", seed=42
        )
        metrics = evaluate_all([result])
        assert metrics.pu_rer is None
