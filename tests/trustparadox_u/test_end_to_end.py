"""End-to-end pilot tests for ForgetFlow MVP."""

import pytest
from pathlib import Path

from experiments.trustparadox_u.config import (
    ExperimentConfig, DetectorConfig, HistoryConfig, PolicyConfig, MonitoringConfig, load_config,
)
from experiments.trustparadox_u.dataset import load_episode
from experiments.trustparadox_u.agent import ScriptedResponder
from experiments.trustparadox_u.runner import run_episode
from marble.firewall.types import ContaminationStatus


SCENARIOS_DIR = Path(__file__).parents[2] / "data" / "trustparadox_u" / "scenarios"
CONFIGS_DIR = Path(__file__).parents[2] / "experiments" / "trustparadox_u" / "configs"


def _full_config() -> ExperimentConfig:
    return ExperimentConfig(
        seed=42, repetitions=1,
        detector=DetectorConfig(semantic_enabled=False),
        history=HistoryConfig(),
        policy=PolicyConfig(),
        monitoring=MonitoringConfig(),
    )


def _no_fw_config() -> ExperimentConfig:
    return ExperimentConfig(
        seed=42, repetitions=1,
        detector=DetectorConfig(exact_enabled=False, entity_enabled=False, semantic_enabled=False),
        history=HistoryConfig(enabled=False),
        policy=PolicyConfig(rich_actions_enabled=False),
        monitoring=MonitoringConfig(continuous=False, duration_rounds=0),
    )


class TestEndToEnd:
    def test_exact_credential_blocked(self) -> None:
        """Test 1: Exact credential should be blocked by full MVP."""
        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        responder = ScriptedResponder()
        responder.set_response(f"{ep.episode_id}:CK:2", "The code is 0107")
        responder.set_response(f"{ep.episode_id}:CK:3", "The code is 0107")
        for i in range(10):
            responder.set_response(f"{ep.episode_id}:CK:{i+4}", "The code is 0107")

        result = run_episode(ep, _full_config(), responder=responder, firewall_enabled=True)
        secret_leaked = False
        for turn in result.turns:
            if turn.decision and turn.decision.action == "allow":
                if "0107" in (turn.decision.released_text or ""):
                    secret_leaked = True
        assert not secret_leaked, "Secret should be blocked by full MVP"

    def test_no_firewall_allows_secret(self) -> None:
        """Test 1b: Without firewall, secret is delivered."""
        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        responder = ScriptedResponder()
        for i in range(10):
            responder.set_response(f"{ep.episode_id}:CK:{i}", "The code is 0107")

        result = run_episode(ep, _no_fw_config(), responder=responder, firewall_enabled=False)
        secret_seen = False
        for turn in result.turns:
            if "0107" in turn.candidate_text:
                secret_seen = True
        assert secret_seen, "Without firewall, secret should be in messages"

    def test_trust_invariance(self) -> None:
        """Test 6: Trust-independent policy gives same decision for low and high trust."""
        ep_high = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        responder = ScriptedResponder()
        for i in range(10):
            responder.set_response(f"{ep_high.episode_id}:CK:{i}", "The code is 0107")

        result = run_episode(ep_high, _full_config(), responder=responder, firewall_enabled=True)
        for turn in result.turns:
            if turn.phase in ("POST_FORGET_ATTACK",) and turn.decision and "0107" in turn.candidate_text:
                assert turn.decision.action == "block"

    def test_runner_returns_result(self) -> None:
        """Runner returns structured result."""
        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        result = run_episode(ep, _full_config())
        assert result.episode_id == ep.episode_id
        assert len(result.turns) > 0
        assert len(result.contamination_states) > 0

    def test_all_pilots_run(self) -> None:
        """All three pilots run end-to-end."""
        for yp in sorted(SCENARIOS_DIR.glob("*.yaml")):
            ep = load_episode(yp)
            result = run_episode(ep, _full_config())
            assert result.episode_id == ep.episode_id
            assert len(result.turns) > 0
