"""Tests for attack library."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from experiments.trustparadox_u.attacks import (
    FRAGMENTATION_ATTACKS,
    build_attack,
)
from experiments.trustparadox_u.dataset import (
    MessageLabel,
    PostForgetAttack,
    load_episode,
)

SCENARIOS_DIR = Path(__file__).parents[2] / "data" / "trustparadox_u" / "scenarios"


def _base_episode() -> "object":
    return load_episode(SCENARIOS_DIR / "pilot_credential.yaml")


def _with_post_forget(ep: object, attacks: tuple[PostForgetAttack, ...]) -> object:
    """Return episode copy with replaced post_forget attacks."""
    return dataclasses.replace(
        ep,
        phases=dataclasses.replace(ep.phases, post_forget=attacks),  # type: ignore[union-attr]
    )


def _frag_attack(
    attacker: str = "SK",
    target: str = "CK",
    instruction: str = "Part question",
    attack_type: str = "temporal_fragmentation",
) -> PostForgetAttack:
    return PostForgetAttack(
        attack_type=attack_type,
        attacker=attacker,
        target_agent=target,
        instruction=instruction,
        label=MessageLabel(
            is_attack_attempt=True,
            is_reconstruction_attempt=True,
        ),
    )


class TestAttacks:
    def test_direct_attack(self) -> None:
        ep = _base_episode()
        plan = build_attack(ep, "direct", seed=42)  # type: ignore[arg-type]
        assert plan.attack_type == "direct"
        assert len(plan.steps) > 0

    def test_alias_attack(self) -> None:
        ep = _base_episode()
        plan = build_attack(ep, "alias", seed=42)  # type: ignore[arg-type]
        assert plan.attack_type == "alias"

    def test_deterministic(self) -> None:
        ep = _base_episode()
        p1 = build_attack(ep, "direct", seed=42)  # type: ignore[arg-type]
        p2 = build_attack(ep, "direct", seed=42)  # type: ignore[arg-type]
        assert p1 == p2

    def test_unknown_attack_raises(self) -> None:
        ep = _base_episode()
        with pytest.raises(ValueError, match="Unknown attack"):
            build_attack(ep, "nonexistent")  # type: ignore[arg-type]

    def test_valid_agents(self) -> None:
        ep = _base_episode()
        plan = build_attack(ep, "direct", seed=42)  # type: ignore[arg-type]
        agent_ids = ep.agent_ids()  # type: ignore[union-attr]
        for step in plan.steps:
            assert step.sender in agent_ids
            assert step.recipient in agent_ids


class TestFragmentationValidation:
    """Tests for fragmentation attack structural validation."""

    def test_temporal_one_step_fails(self) -> None:
        """Temporal fragmentation with one step must fail."""
        ep = _with_post_forget(
            _base_episode(),
            (_frag_attack(instruction="Only one step"),),
        )
        with pytest.raises(ValueError, match="at least two ordered"):
            build_attack(ep, "temporal_fragmentation")  # type: ignore[arg-type]

    def test_temporal_one_fragment_fails(self) -> None:
        """Temporal fragmentation with only one fragment in fragment_map must fail."""
        ep = _base_episode()
        # Override fragment_map to have only one fragment
        ep = dataclasses.replace(
            ep,
            fragment_map={"F001": {"target": "0107", "fragments": ["01"]}},  # type: ignore[union-attr]
        )
        ep = _with_post_forget(
            ep,
            (
                _frag_attack(instruction="Part 1"),
                _frag_attack(instruction="Part 2"),
            ),
        )
        with pytest.raises(ValueError, match="at least two configured fragments"):
            build_attack(ep, "temporal_fragmentation")  # type: ignore[arg-type]

    def test_temporal_different_recipients_fails(self) -> None:
        """Temporal fragmentation with different recipients must fail."""
        ep = _with_post_forget(
            _base_episode(),
            (
                _frag_attack(target="CK", instruction="Part 1"),
                _frag_attack(target="SK", instruction="Part 2"),
            ),
        )
        with pytest.raises(ValueError, match="one shared recipient"):
            build_attack(ep, "temporal_fragmentation")  # type: ignore[arg-type]

    def test_valid_temporal_fragmentation_passes(self) -> None:
        """Valid temporal fragmentation (2 steps, same recipient, ≥2 fragments) passes."""
        ep = _base_episode()
        ep = _with_post_forget(
            ep,
            (
                _frag_attack(target="CK", instruction="Part 1"),
                _frag_attack(target="CK", instruction="Part 2"),
            ),
        )
        plan = build_attack(ep, "temporal_fragmentation")  # type: ignore[arg-type]
        assert plan.attack_type == "temporal_fragmentation"
        assert len(plan.steps) == 2
        assert len(plan.fragments) >= 2

    def test_cross_agent_one_sender_fails(self) -> None:
        """Cross-agent fragmentation with one sender must fail."""
        ep = _with_post_forget(
            _base_episode(),
            (
                _frag_attack(
                    attacker="SK",
                    target="CK",
                    instruction="Part 1",
                    attack_type="cross_agent_fragmentation",
                ),
                _frag_attack(
                    attacker="SK",
                    target="CK",
                    instruction="Part 2",
                    attack_type="cross_agent_fragmentation",
                ),
            ),
        )
        with pytest.raises(ValueError, match="at least two distinct senders"):
            build_attack(ep, "cross_agent_fragmentation")  # type: ignore[arg-type]

    def test_valid_cross_agent_fragmentation_passes(self) -> None:
        """Valid cross-agent fragmentation (2 senders) passes."""
        ep = _base_episode()
        ep = _with_post_forget(
            ep,
            (
                _frag_attack(
                    attacker="SK",
                    target="CK",
                    instruction="Part 1",
                    attack_type="cross_agent_fragmentation",
                ),
                _frag_attack(
                    attacker="CK",
                    target="SK",
                    instruction="Part 2",
                    attack_type="cross_agent_fragmentation",
                ),
            ),
        )
        plan = build_attack(ep, "cross_agent_fragmentation")  # type: ignore[arg-type]
        assert plan.attack_type == "cross_agent_fragmentation"
        assert len(plan.steps) == 2

    def test_step_order_is_stable(self) -> None:
        """Step order is deterministic and matches dataset order."""
        ep = _base_episode()
        ep = _with_post_forget(
            ep,
            (
                _frag_attack(target="CK", instruction="First"),
                _frag_attack(target="CK", instruction="Second"),
            ),
        )
        p1 = build_attack(ep, "temporal_fragmentation")  # type: ignore[arg-type]
        p2 = build_attack(ep, "temporal_fragmentation")  # type: ignore[arg-type]
        assert p1.steps == p2.steps
        assert p1.steps[0].instruction == "First"
        assert p1.steps[1].instruction == "Second"

    def test_two_turn_e2e_fixture_valid(self) -> None:
        """The two-turn temporal fragmentation fixture from e2e tests is valid."""
        ep = _base_episode()
        frag1 = _frag_attack(target="CK", instruction="What is the first part?")
        frag2 = _frag_attack(target="CK", instruction="What is the second part?")
        non_frag = tuple(
            a
            for a in ep.phases.post_forget  # type: ignore[union-attr]
            if a.attack_type != "temporal_fragmentation"
        )
        ep = _with_post_forget(ep, non_frag + (frag1, frag2))
        plan = build_attack(ep, "temporal_fragmentation")  # type: ignore[arg-type]
        assert len(plan.steps) == 2
        assert len(plan.fragments) >= 2

    def test_fragmentation_attacks_constant(self) -> None:
        """FRAGMENTATION_ATTACKS contains the expected types."""
        assert "temporal_fragmentation" in FRAGMENTATION_ATTACKS
        assert "cross_agent_fragmentation" in FRAGMENTATION_ATTACKS
        assert len(FRAGMENTATION_ATTACKS) == 2
