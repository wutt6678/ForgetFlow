"""Tests for episode runner."""

import pytest
from pathlib import Path

from experiments.trustparadox_u.config import ExperimentConfig, DetectorConfig, HistoryConfig, PolicyConfig, MonitoringConfig
from experiments.trustparadox_u.dataset import load_episode
from experiments.trustparadox_u.runner import run_episode
from experiments.trustparadox_u.agent import ScriptedResponder


SCENARIOS_DIR = Path(__file__).parents[2] / "data" / "trustparadox_u" / "scenarios"


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        seed=42, repetitions=1,
        detector=DetectorConfig(semantic_enabled=False),
        history=HistoryConfig(),
        policy=PolicyConfig(),
        monitoring=MonitoringConfig(),
    )


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
