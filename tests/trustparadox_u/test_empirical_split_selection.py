"""E3-002: tests for split-aware target-spec selection.

The generation runner must select target specs from the frozen registry
by the ``split`` field — not infer the split from the variant suffix.
Expected counts: development=3, validation=3, test=6.
"""

from __future__ import annotations

import pytest

from experiments.trustparadox_u.empirical_corpus import (
    EMPIRICAL_TARGET_REGISTRY,
    validate_target_registry,
)
from experiments.trustparadox_u.generate_empirical_corpus import (
    development_spec_for_scenario,
    specs_for_split,
)


class TestSpecsForSplit:
    def test_development_returns_three_v1_specs(self) -> None:
        specs = specs_for_split("development")
        assert len(specs) == 3
        for spec in specs:
            assert spec.split == "development"
            assert spec.secret_variant_id.endswith("_v1")

    def test_validation_returns_three_v2_specs(self) -> None:
        specs = specs_for_split("validation")
        assert len(specs) == 3
        for spec in specs:
            assert spec.split == "validation"
            assert spec.secret_variant_id.endswith("_v2")

    def test_test_returns_six_v3_v4_specs(self) -> None:
        specs = specs_for_split("test")
        assert len(specs) == 6
        for spec in specs:
            assert spec.split == "test"
            suffix = spec.secret_variant_id.split("_")[-1]
            assert suffix in ("v3", "v4")

    def test_no_variant_in_two_splits(self) -> None:
        dev_ids = {s.secret_variant_id for s in specs_for_split("development")}
        val_ids = {s.secret_variant_id for s in specs_for_split("validation")}
        test_ids = {s.secret_variant_id for s in specs_for_split("test")}
        assert not (dev_ids & val_ids)
        assert not (dev_ids & test_ids)
        assert not (val_ids & test_ids)

    def test_no_split_is_empty(self) -> None:
        for split in ("development", "validation", "test"):
            assert len(specs_for_split(split)) > 0

    def test_test_never_returns_development_target(self) -> None:
        dev_ids = {s.secret_variant_id for s in specs_for_split("development")}
        test_specs = specs_for_split("test")
        test_ids = {s.secret_variant_id for s in test_specs}
        assert not (test_ids & dev_ids)

    def test_scenario_filter(self) -> None:
        specs = specs_for_split("test", scenario_ids=["credential_001"])
        assert len(specs) == 2  # credential_v3 + credential_v4
        for spec in specs:
            assert spec.scenario_id == "credential_001"

    def test_unknown_split_raises(self) -> None:
        with pytest.raises(ValueError):
            specs_for_split("nonexistent")

    def test_all_12_specs_accounted_for(self) -> None:
        dev = specs_for_split("development")
        val = specs_for_split("validation")
        test = specs_for_split("test")
        all_ids = (
            {s.secret_variant_id for s in dev}
            | {s.secret_variant_id for s in val}
            | {s.secret_variant_id for s in test}
        )
        registry_ids = {s.secret_variant_id for s in EMPIRICAL_TARGET_REGISTRY}
        assert all_ids == registry_ids

    def test_target_registry_valid(self) -> None:
        problems = validate_target_registry(EMPIRICAL_TARGET_REGISTRY)
        assert not problems


class TestDevelopmentSpecBackwardCompat:
    """The legacy development_spec_for_scenario helper still works."""

    def test_credential_v1(self) -> None:
        spec = development_spec_for_scenario("credential_001")
        assert spec.secret_variant_id == "credential_v1"

    def test_private_attribute_v1(self) -> None:
        spec = development_spec_for_scenario("private_attribute_001")
        assert spec.secret_variant_id == "private_attribute_v1"

    def test_authorization_v1(self) -> None:
        spec = development_spec_for_scenario("authorization_001")
        assert spec.secret_variant_id == "authorization_v1"
