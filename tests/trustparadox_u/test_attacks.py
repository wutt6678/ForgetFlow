"""Tests for attack library."""

import pytest
from pathlib import Path

from experiments.trustparadox_u.dataset import load_episode
from experiments.trustparadox_u.attacks import build_attack


SCENARIOS_DIR = Path(__file__).parents[2] / "data" / "trustparadox_u" / "scenarios"


class TestAttacks:
    def test_direct_attack(self) -> None:
        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        plan = build_attack(ep, "direct", seed=42)
        assert plan.attack_type == "direct"
        assert len(plan.steps) > 0

    def test_alias_attack(self) -> None:
        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        plan = build_attack(ep, "alias", seed=42)
        assert plan.attack_type == "alias"

    def test_deterministic(self) -> None:
        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        p1 = build_attack(ep, "direct", seed=42)
        p2 = build_attack(ep, "direct", seed=42)
        assert p1 == p2

    def test_unknown_attack_raises(self) -> None:
        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        with pytest.raises(ValueError, match="Unknown attack"):
            build_attack(ep, "nonexistent")

    def test_fragmentation_has_fragments(self) -> None:
        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        plan = build_attack(ep, "temporal_fragmentation", seed=42)
        assert len(plan.fragments) > 0

    def test_valid_agents(self) -> None:
        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        plan = build_attack(ep, "direct", seed=42)
        agent_ids = ep.agent_ids()
        for step in plan.steps:
            assert step.sender in agent_ids
            assert step.recipient in agent_ids
