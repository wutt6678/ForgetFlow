"""Tests for experiments.trustparadox_u.identity module."""

import pytest

from experiments.trustparadox_u.identity import (
    ResearchRunIdentity,
    normalize_attack_type,
    normalize_identity_component,
    normalize_pairing_key,
    pairing_key_from_result,
    research_run_identity_from_result,
    run_identity_from_result,
)
from experiments.trustparadox_u.runner import EpisodeResult


def _valid_result(**overrides) -> EpisodeResult:
    """Create a minimal valid EpisodeResult for testing."""
    result = EpisodeResult(
        run_id="run_0001",
        episode_id="ep1",
        scenario_id="s1",
        trust_level="default",
        seed=42,
    )
    result.metadata = {
        "secret_variant_id": "sv1",
        "attack_type": "direct",
        "config_hash": "a" * 64,
    }
    for k, v in overrides.items():
        setattr(result, k, v)
    return result


class TestNormalizeIdentityComponent:
    """Tests for normalize_identity_component."""

    def test_scalar_string(self) -> None:
        assert normalize_identity_component("test") == "test"

    def test_scalar_int(self) -> None:
        assert normalize_identity_component(42) == "42"

    def test_list_sorted(self) -> None:
        assert normalize_identity_component(["b", "a"]) == '["a","b"]'

    def test_list_deterministic(self) -> None:
        # Same list in different order should produce same output
        assert normalize_identity_component(["a", "b"]) == normalize_identity_component(["b", "a"])

    def test_tuple_sorted(self) -> None:
        """Tuples are normalized like lists."""
        assert normalize_identity_component(("b", "a")) == '["a","b"]'

    def test_set_sorted(self) -> None:
        """Sets are normalized and sorted."""
        assert normalize_identity_component({"b", "a"}) == '["a","b"]'

    def test_mapping_sorted_keys(self) -> None:
        """Mappings are serialized with sorted keys."""
        result = normalize_identity_component({"b": 2, "a": 1})
        assert result == '{"a":1,"b":2}'

    def test_nested_list_normalized(self) -> None:
        """Nested structures are recursively normalized."""
        result = normalize_identity_component([["b", "a"], ["d", "c"]])
        # Each inner list is normalized, then sorted as strings
        assert result == '["[\\"a\\",\\"b\\"]","[\\"c\\",\\"d\\"]"]'


class TestNormalizeAttackType:
    """Tests for normalize_attack_type."""

    def test_scalar_attack_type(self) -> None:
        assert normalize_attack_type("direct") == "direct"

    def test_list_attack_type(self) -> None:
        assert normalize_attack_type(["direct", "indirect"]) == '["direct","indirect"]'

    def test_list_order_normalized(self) -> None:
        # Order should not matter
        assert normalize_attack_type(["a", "b"]) == normalize_attack_type(["b", "a"])


class TestPairingKeyFromResult:
    """Tests for pairing_key_from_result."""

    def test_basic_pairing_key(self) -> None:
        result = _valid_result()
        key = pairing_key_from_result(result)
        assert key == ("s1", "sv1", "default", "direct", 42)

    def test_list_attack_type_normalized(self) -> None:
        result = _valid_result()
        result.metadata["attack_type"] = ["direct", "indirect"]
        key = pairing_key_from_result(result)
        assert key[3] == '["direct","indirect"]'

    def test_list_secret_variant_normalized(self) -> None:
        result = _valid_result()
        result.metadata["secret_variant_id"] = ["sv1", "sv2"]
        key = pairing_key_from_result(result)
        assert key[1] == '["sv1","sv2"]'


class TestRunIdentityFromResult:
    """Tests for run_identity_from_result."""

    def test_basic_run_identity(self) -> None:
        result = _valid_result()
        identity = run_identity_from_result(result)
        assert identity == (("s1", "sv1", "default", "direct", 42), "a" * 64)

    def test_missing_config_hash_raises(self) -> None:
        result = _valid_result()
        del result.metadata["config_hash"]
        with pytest.raises(ValueError, match="missing config_hash"):
            run_identity_from_result(result)

    def test_empty_config_hash_raises(self) -> None:
        result = _valid_result()
        result.metadata["config_hash"] = ""
        with pytest.raises(ValueError, match="missing config_hash"):
            run_identity_from_result(result)


class TestPairingKeyVsRunIdentity:
    """Tests demonstrating the difference between PairingKey and RunIdentity."""

    def test_same_pairing_key_different_config_hash(self) -> None:
        """Same pairing key but different config_hash => distinct run identities."""
        r1 = _valid_result()
        r2 = _valid_result(episode_id="ep2")
        r2.metadata["config_hash"] = "b" * 64

        # Same pairing key
        assert pairing_key_from_result(r1) == pairing_key_from_result(r2)

        # Different run identities
        assert run_identity_from_result(r1) != run_identity_from_result(r2)

    def test_different_seed_different_pairing_key(self) -> None:
        """Different seed => different pairing key and run identity."""
        r1 = _valid_result()
        r2 = _valid_result(episode_id="ep2", seed=99)
        r2.metadata["seed"] = 99

        assert pairing_key_from_result(r1) != pairing_key_from_result(r2)
        assert run_identity_from_result(r1) != run_identity_from_result(r2)


class TestEvaluatorAuditorConsistency:
    """Tests ensuring evaluator and auditor use the same pairing logic."""

    def test_evaluator_auditor_pairing_key_consistency(self) -> None:
        """Evaluator and auditor should produce identical pairing keys."""
        result = _valid_result()
        result.metadata["pairing_key"] = {
            "scenario_id": "s1",
            "secret_variant_id": "sv1",
            "trust_level": "default",
            "attack_type": "direct",
            "seed": 42,
        }

        # pairing_key_from_result should match normalize_pairing_key
        from_result = pairing_key_from_result(result)
        from_dict = normalize_pairing_key(result.metadata["pairing_key"])
        assert from_result == from_dict

    def test_list_valued_attack_type_consistent(self) -> None:
        """List-valued attack types normalize consistently."""
        r1 = _valid_result()
        r1.metadata["attack_type"] = ["direct", "indirect"]

        r2 = _valid_result(episode_id="ep2")
        r2.metadata["attack_type"] = ["indirect", "direct"]  # different order

        # Should produce same pairing key due to normalization
        assert pairing_key_from_result(r1) == pairing_key_from_result(r2)

    def test_multi_target_secret_variant_consistent(self) -> None:
        """Multi-target secret variants normalize consistently."""
        r1 = _valid_result()
        r1.metadata["secret_variant_id"] = ["sv1", "sv2"]

        r2 = _valid_result(episode_id="ep2")
        r2.metadata["secret_variant_id"] = ["sv2", "sv1"]  # different order

        # Should produce same pairing key due to normalization
        assert pairing_key_from_result(r1) == pairing_key_from_result(r2)


class TestResearchRunIdentity:
    """Tests for ResearchRunIdentity dataclass."""

    def test_basic_construction(self) -> None:
        identity = ResearchRunIdentity(
            scenario_id="s1",
            secret_variant_id="sv1",
            trust_level="default",
            attack_type="direct",
            seed=42,
            condition_id="full_mvp",
        )
        assert identity.scenario_id == "s1"
        assert identity.condition_id == "full_mvp"
        assert identity.seed == 42

    def test_frozen(self) -> None:
        identity = ResearchRunIdentity("s1", "sv1", "default", "direct", 42, "full_mvp")
        with pytest.raises(AttributeError):
            identity.scenario_id = "s2"  # type: ignore[misc]

    def test_orderable(self) -> None:
        a = ResearchRunIdentity("s1", "sv1", "default", "direct", 42, "binary_policy")
        b = ResearchRunIdentity("s1", "sv1", "default", "direct", 42, "full_mvp")
        assert a < b  # binary_policy < full_mvp alphabetically

    def test_different_conditions_are_distinct(self) -> None:
        """Same scenario/seed under different conditions are distinct."""
        a = ResearchRunIdentity("s1", "sv1", "default", "direct", 42, "binary_policy")
        b = ResearchRunIdentity("s1", "sv1", "default", "direct", 42, "full_mvp")
        assert a != b

    def test_same_condition_repeated_is_duplicate(self) -> None:
        """Identical identities are equal (duplicate)."""
        a = ResearchRunIdentity("s1", "sv1", "default", "direct", 42, "full_mvp")
        b = ResearchRunIdentity("s1", "sv1", "default", "direct", 42, "full_mvp")
        assert a == b

    def test_list_attack_type_normalizes_deterministically(self) -> None:
        """List-valued attack types normalize to the same string."""
        from experiments.trustparadox_u.identity import normalize_attack_type

        a = normalize_attack_type(["direct", "indirect"])
        b = normalize_attack_type(["indirect", "direct"])
        assert a == b

    def test_condition_order_does_not_affect_identity(self) -> None:
        """condition_id is a string; order is irrelevant."""
        a = ResearchRunIdentity("s1", "sv1", "default", "direct", 42, "full_mvp")
        b = ResearchRunIdentity("s1", "sv1", "default", "direct", 42, "full_mvp")
        assert a == b


class TestResearchRunIdentityFromResult:
    """Tests for research_run_identity_from_result."""

    def test_basic_identity(self) -> None:
        result = _valid_result()
        result.metadata["smoke_condition"] = "full_mvp"
        identity = research_run_identity_from_result(result)
        assert identity == ResearchRunIdentity(
            scenario_id="s1",
            secret_variant_id="sv1",
            trust_level="default",
            attack_type="direct",
            seed=42,
            condition_id="full_mvp",
        )

    def test_fallback_to_config_hash(self) -> None:
        """Falls back to config_hash when smoke_condition is absent."""
        result = _valid_result()
        identity = research_run_identity_from_result(result)
        assert identity.condition_id == "a" * 64

    def test_missing_both_raises(self) -> None:
        """Raises when neither smoke_condition nor config_hash is present."""
        result = _valid_result()
        del result.metadata["config_hash"]
        with pytest.raises(ValueError, match="missing both"):
            research_run_identity_from_result(result)

    def test_different_conditions_distinct(self) -> None:
        """Same pairing key, different conditions produce distinct identities."""
        r1 = _valid_result()
        r1.metadata["smoke_condition"] = "binary_policy"
        r2 = _valid_result(episode_id="ep2")
        r2.metadata["smoke_condition"] = "full_mvp"

        id1 = research_run_identity_from_result(r1)
        id2 = research_run_identity_from_result(r2)
        assert id1 != id2
        assert id1.condition_id == "binary_policy"
        assert id2.condition_id == "full_mvp"
