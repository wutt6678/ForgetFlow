"""Tests for TrustParadox-U schema and dataset loading."""

from pathlib import Path

import pytest

from experiments.trustparadox_u.dataset import load_episode, load_episodes_from_dir

SCENARIOS_DIR = Path(__file__).parents[2] / "data" / "trustparadox_u" / "scenarios"


class TestSchema:
    def test_credential_loads(self) -> None:
        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        assert ep.episode_id == "credential_001_high_direct"
        assert len(ep.agents) == 2
        assert len(ep.sensitive_items) == 1

    def test_attribute_loads(self) -> None:
        ep = load_episode(SCENARIOS_DIR / "pilot_private_attribute.yaml")
        assert ep.episode_id == "attribute_001_high_direct"

    def test_authorization_loads(self) -> None:
        ep = load_episode(SCENARIOS_DIR / "pilot_authorization.yaml")
        assert ep.episode_id == "auth_001_high_direct"

    def test_has_forget_event(self) -> None:
        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        assert ep.phases.forget.turn >= 0
        assert len(ep.phases.forget.clean_agents) > 0

    def test_has_prohibited_output(self) -> None:
        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        assert len(ep.expected.forbidden_strings) > 0

    def test_has_permitted_output(self) -> None:
        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        assert len(ep.expected.permitted_strings) > 0

    def test_has_post_forget_attack(self) -> None:
        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        assert len(ep.phases.post_forget) >= 1

    def test_agent_ids(self) -> None:
        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        ids = ep.agent_ids()
        assert "CK" in ids
        assert "SK" in ids


class TestDataset:
    def test_load_all_pilots(self) -> None:
        episodes = load_episodes_from_dir(SCENARIOS_DIR)
        assert len(episodes) == 4

    def test_get_agent(self) -> None:
        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        ck = ep.get_agent("CK")
        assert ck.role == "custodian"

    def test_unknown_agent_raises(self) -> None:
        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        with pytest.raises(ValueError, match="Unknown agent"):
            ep.get_agent("UNKNOWN")
