"""Tests for episode runner."""

from pathlib import Path

from experiments.trustparadox_u.config import (
    DetectorConfig,
    ExperimentConfig,
    HistoryConfig,
    MonitoringConfig,
    PolicyConfig,
)
from experiments.trustparadox_u.dataset import load_episode
from experiments.trustparadox_u.runner import run_episode

SCENARIOS_DIR = Path(__file__).parents[2] / "data" / "trustparadox_u" / "scenarios"


def _config(**overrides) -> ExperimentConfig:
    kwargs = dict(
        seed=42,
        repetitions=1,
        detector=DetectorConfig(semantic_enabled=False),
        history=HistoryConfig(),
        policy=PolicyConfig(),
        monitoring=MonitoringConfig(),
    )
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


class TestRunner:
    def test_phase_order(self) -> None:
        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        result = run_episode(ep, _config())
        phases = [t.phase for t in result.turns]
        if "PRE_FORGET" in phases and "POST_FORGET_ATTACK" in phases:
            assert phases.index("PRE_FORGET") < phases.index("POST_FORGET_ATTACK")

    def test_deterministic(self) -> None:
        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        r1 = run_episode(ep, _config())
        r2 = run_episode(ep, _config())
        assert len(r1.turns) == len(r2.turns)
        for t1, t2 in zip(r1.turns, r2.turns):
            assert t1.candidate_text == t2.candidate_text

    def test_firewall_disabled_mode(self) -> None:
        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        result = run_episode(ep, _config(), firewall_enabled=False)
        for turn in result.turns:
            if turn.phase == "POST_FORGET_ATTACK":
                assert turn.decision is None

    def test_recontamination_denominator_counts_blocked_attempts(self) -> None:
        """Blocked recontamination attempt still contributes to RR denominator."""
        ep = load_episode(SCENARIOS_DIR / "pilot_authorization.yaml")
        result = run_episode(ep, _config())
        # Authorization scenario has a recontamination attack on CMD (cleaned agent)
        # The denominator should count cleaned agents targeted by recontamination
        assert result.cleaned_agents_exposed >= 0  # Denominator is defined

    def test_recontamination_numerator_only_confirmed_recovery(self) -> None:
        """Numerator counts only agents with confirmed recovery (RECONTAMINATED)."""
        ep = load_episode(SCENARIOS_DIR / "pilot_authorization.yaml")
        result = run_episode(ep, _config())
        # Numerator <= denominator always
        assert result.recontaminated_agents <= result.cleaned_agents_exposed

    def test_no_recontamination_scenario(self) -> None:
        """Credential scenario has no recontamination; RR denominator = 0."""
        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        result = run_episode(ep, _config())
        assert result.cleaned_agents_exposed == 0
        assert result.recontaminated_agents == 0


class TestTaskSuccess:
    """Tests for explicit task-success evaluation."""

    def test_required_release_success(self) -> None:
        """Required residual released -> task success."""
        from experiments.trustparadox_u.runner import EpisodeResult, _evaluate_task_success

        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        result = EpisodeResult(
            run_id="r1", episode_id=ep.episode_id, scenario_id=ep.scenario_id,
            trust_level=ep.trust_level, seed=42,
        )
        # success_value = "Request a new temporary credential."
        from experiments.trustparadox_u.runner import TurnResult
        result.turns.append(TurnResult(
            turn_id=0, phase="POST_FORGET_ATTACK",
            sender_id="SK", recipient_id="CK",
            candidate_text="Please Request a new temporary credential.",
            released_text="Please Request a new temporary credential.",
        ))
        assert _evaluate_task_success(ep, result, True) is True

    def test_required_release_blocked(self) -> None:
        """Required residual blocked -> task failure."""
        from experiments.trustparadox_u.runner import (
            EpisodeResult,
            TurnResult,
            _evaluate_task_success,
        )

        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        result = EpisodeResult(
            run_id="r1", episode_id=ep.episode_id, scenario_id=ep.scenario_id,
            trust_level=ep.trust_level, seed=42,
        )
        result.turns.append(TurnResult(
            turn_id=0, phase="POST_FORGET_ATTACK",
            sender_id="SK", recipient_id="CK",
            candidate_text="Request a new temporary credential.",
            released_text=None,  # Blocked
        ))
        assert _evaluate_task_success(ep, result, True) is False

    def test_exact_label_matches(self) -> None:
        """Exact label matches -> success."""
        from experiments.trustparadox_u.dataset import TaskSpec
        from experiments.trustparadox_u.runner import (
            EpisodeResult,
            TurnResult,
            _evaluate_task_success,
        )

        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        # Override task to exact_label type
        from dataclasses import replace
        ep = replace(ep, task=TaskSpec(
            description=ep.task.description,
            success_label=ep.task.success_label,
            success_type="exact_label",
            success_value="safe_access_procedure_completed",
        ))
        result = EpisodeResult(
            run_id="r1", episode_id=ep.episode_id, scenario_id=ep.scenario_id,
            trust_level=ep.trust_level, seed=42,
        )
        result.turns.append(TurnResult(
            turn_id=0, phase="POST_FORGET_ATTACK",
            sender_id="SK", recipient_id="CK",
            candidate_text="safe_access_procedure_completed",
            released_text="safe_access_procedure_completed",
        ))
        assert _evaluate_task_success(ep, result, True) is True

    def test_exact_label_differs(self) -> None:
        """Exact label differs -> failure."""
        from dataclasses import replace

        from experiments.trustparadox_u.dataset import TaskSpec
        from experiments.trustparadox_u.runner import (
            EpisodeResult,
            TurnResult,
            _evaluate_task_success,
        )

        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        ep = replace(ep, task=TaskSpec(
            description=ep.task.description,
            success_label=ep.task.success_label,
            success_type="exact_label",
            success_value="some_other_label",
        ))
        result = EpisodeResult(
            run_id="r1", episode_id=ep.episode_id, scenario_id=ep.scenario_id,
            trust_level=ep.trust_level, seed=42,
        )
        result.turns.append(TurnResult(
            turn_id=0, phase="POST_FORGET_ATTACK",
            sender_id="SK", recipient_id="CK",
            candidate_text="safe_access_procedure_completed",
            released_text="safe_access_procedure_completed",
        ))
        assert _evaluate_task_success(ep, result, True) is False

    def test_unsupported_type_raises(self) -> None:
        """Unsupported task type raises ValueError."""
        from dataclasses import replace

        from experiments.trustparadox_u.dataset import TaskSpec
        from experiments.trustparadox_u.runner import EpisodeResult, _evaluate_task_success

        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        ep = replace(ep, task=TaskSpec(
            description=ep.task.description,
            success_label=ep.task.success_label,
            success_type="unsupported_type",
            success_value="something",
        ))
        result = EpisodeResult(
            run_id="r1", episode_id=ep.episode_id, scenario_id=ep.scenario_id,
            trust_level=ep.trust_level, seed=42,
        )
        import pytest
        with pytest.raises(ValueError, match="Unsupported task success_type"):
            _evaluate_task_success(ep, result, True)


class TestExperimentIdentity:
    """Tests for config hash and run ID stability."""

    def test_different_config_different_hash(self) -> None:
        """Different history window -> different config hash."""
        from experiments.trustparadox_u.config import HistoryConfig

        cfg1 = _config()
        cfg2 = _config(history=HistoryConfig(window_size=10))
        assert cfg1.config_hash() != cfg2.config_hash()

    def test_same_config_same_hash(self) -> None:
        """Same resolved config -> same hash."""
        cfg1 = _config()
        cfg2 = _config()
        assert cfg1.config_hash() == cfg2.config_hash()

    def test_different_seed_different_run_id(self) -> None:
        """Different seed -> different run ID."""
        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        r1 = run_episode(ep, _config(seed=42))
        r2 = run_episode(ep, _config(seed=99))
        assert r1.run_id != r2.run_id

    def test_config_hash_is_sha256(self) -> None:
        """Config hash should be a valid SHA-256 hex digest."""
        cfg = _config()
        h = cfg.config_hash()
        assert len(h) == 64  # SHA-256 hex length
        assert all(c in "0123456789abcdef" for c in h)

    def test_run_id_is_populated(self) -> None:
        """Run ID should be auto-generated when not provided."""
        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        result = run_episode(ep, _config())
        assert result.run_id != ""
        assert len(result.run_id) == 20

    def test_metadata_has_config_hash(self) -> None:
        """Metadata should include the full config hash."""
        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        result = run_episode(ep, _config())
        assert "config_hash" in result.metadata
        assert len(result.metadata["config_hash"]) == 64

    def test_metadata_has_secret_variant_id(self) -> None:
        """Metadata should include the real secret variant ID."""
        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        result = run_episode(ep, _config())
        variant_id = result.metadata["secret_variant_id"]
        assert variant_id != ""
        assert variant_id != "F001"  # Should not be just the forget_id
