"""Tests for experiment result auditor."""

from experiments.trustparadox_u.audit_results import (
    audit_episode_result,
    audit_results,
    validate_for_aggregation,
)
from experiments.trustparadox_u.runner import EpisodeResult, TurnResult
from marble.firewall.types import DetectorResult, FirewallDecision


def _det() -> DetectorResult:
    return DetectorResult(
        exact_score=0.0,
        entity_score=0.0,
        semantic_score=0.0,
        reconstruction_score=0.0,
        matched_forget_ids=(),
        evidence=(),
    )


def _decision(action: str, released_text: str | None = None) -> FirewallDecision:
    return FirewallDecision(
        action=action,  # type: ignore[arg-type]
        released_text=released_text if action != "block" else None,
        detector_result=_det(),
        reason_codes=(),
        policy_version="v1",
        latency_ms=1.0,
    )


class TestAuditor:
    def test_valid_episode_passes(self) -> None:
        """Valid episode with consistent turns should pass audit."""
        result = EpisodeResult(
            run_id="r1",
            episode_id="e1",
            scenario_id="s1",
            trust_level="high",
            seed=42,
        )
        result.metadata = {
            "forbidden_strings": ["secret"],
            "seed": 42,
            "config_hash": "test_hash",
        }
        result.turns.append(
            TurnResult(
                turn_id=0,
                phase="POST_FORGET_ATTACK",
                sender_id="A",
                recipient_id="B",
                candidate_text="secret",
                released_text=None,
                decision=_decision("block"),
                is_attack_attempt=True,
                target_exposed=False,
            )
        )
        findings = audit_episode_result(result)
        errors = [f for f in findings if f.level == "error"]
        assert len(errors) == 0

    def test_block_with_released_text_is_error(self) -> None:
        """Blocked message with released_text is inconsistent."""
        result = EpisodeResult(
            run_id="r1",
            episode_id="e1",
            scenario_id="s1",
            trust_level="high",
            seed=42,
        )
        result.metadata = {"forbidden_strings": ["secret"], "seed": 42, "config_hash": "h"}
        result.turns.append(
            TurnResult(
                turn_id=0,
                phase="POST_FORGET_ATTACK",
                sender_id="A",
                recipient_id="B",
                candidate_text="secret",
                released_text="some text",  # Error: block should have None
                decision=_decision("block"),
                is_attack_attempt=True,
                target_exposed=False,
            )
        )
        findings = audit_episode_result(result)
        errors = [f for f in findings if f.level == "error"]
        assert any(f.code == "BLOCK_WITH_RELEASED_TEXT" for f in errors)

    def test_exposed_without_released_text_is_error(self) -> None:
        """target_exposed=True without released_text is inconsistent."""
        result = EpisodeResult(
            run_id="r1",
            episode_id="e1",
            scenario_id="s1",
            trust_level="high",
            seed=42,
        )
        result.metadata = {"forbidden_strings": ["secret"], "seed": 42, "config_hash": "h"}
        result.turns.append(
            TurnResult(
                turn_id=0,
                phase="POST_FORGET_ATTACK",
                sender_id="A",
                recipient_id="B",
                candidate_text="secret",
                released_text=None,
                decision=_decision("block"),
                is_attack_attempt=True,
                target_exposed=True,  # Error: can't be exposed with no released_text
            )
        )
        findings = audit_episode_result(result)
        errors = [f for f in findings if f.level == "error"]
        assert any(f.code == "EXPOSED_WITHOUT_RELEASED_TEXT" for f in errors)

    def test_reconstructed_without_attempt_is_error(self) -> None:
        """target_reconstructed without is_reconstruction_attempt is inconsistent."""
        result = EpisodeResult(
            run_id="r1",
            episode_id="e1",
            scenario_id="s1",
            trust_level="high",
            seed=42,
        )
        result.metadata = {"forbidden_strings": ["secret"], "seed": 42, "config_hash": "h"}
        result.turns.append(
            TurnResult(
                turn_id=0,
                phase="POST_FORGET_ATTACK",
                sender_id="A",
                recipient_id="B",
                candidate_text="text",
                released_text="text",
                is_reconstruction_attempt=False,
                target_reconstructed=True,  # Error: no attempt
            )
        )
        findings = audit_episode_result(result)
        errors = [f for f in findings if f.level == "error"]
        assert any(f.code == "RECONSTRUCTED_WITHOUT_ATTEMPT" for f in errors)

    def test_missing_forbidden_strings_is_error(self) -> None:
        """Missing forbidden_strings in metadata is an error."""
        result = EpisodeResult(
            run_id="r1",
            episode_id="e1",
            scenario_id="s1",
            trust_level="high",
            seed=42,
        )
        result.metadata = {"seed": 42}  # Missing forbidden_strings
        findings = audit_episode_result(result)
        errors = [f for f in findings if f.level == "error"]
        assert any(f.code == "MISSING_FORBIDDEN_STRINGS" for f in errors)

    def test_validate_for_aggregation(self) -> None:
        """validate_for_aggregation returns False for results with errors."""
        result = EpisodeResult(
            run_id="r1",
            episode_id="e1",
            scenario_id="s1",
            trust_level="high",
            seed=42,
        )
        result.metadata = {}  # Missing forbidden_strings
        is_valid, report = validate_for_aggregation([result])
        assert not is_valid
        assert report.has_errors

    def test_validate_for_aggregation_with_override(self) -> None:
        """validate_for_aggregation with allow_errors=True passes even with errors."""
        result = EpisodeResult(
            run_id="r1",
            episode_id="e1",
            scenario_id="s1",
            trust_level="high",
            seed=42,
        )
        result.metadata = {}
        is_valid, report = validate_for_aggregation([result], allow_errors=True)
        assert is_valid

    def test_audit_results_counts(self) -> None:
        """audit_results correctly counts episodes and errors."""
        r1 = EpisodeResult(
            run_id="r1", episode_id="e1", scenario_id="s1", trust_level="high", seed=42
        )
        r1.metadata = {"forbidden_strings": ["s"], "seed": 42, "config_hash": "h"}
        r2 = EpisodeResult(
            run_id="r2", episode_id="e2", scenario_id="s1", trust_level="high", seed=42
        )
        r2.metadata = {}  # Error
        report = audit_results([r1, r2])
        assert report.episodes_audited == 2
        assert report.episodes_with_errors == 1
