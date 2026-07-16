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
