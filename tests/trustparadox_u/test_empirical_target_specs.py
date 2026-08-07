"""E1-026/027: empirical target-spec detail tests.

Registry structural validation lives in ``test_empirical_corpus_schema.py``;
this module covers variant naming, split locking, and separation from the
synthetic benchmark values.
"""

from __future__ import annotations

import pytest

from experiments.trustparadox_u.empirical_corpus import (
    EMPIRICAL_TARGET_REGISTRY,
    EmpiricalSplit,
    get_target_spec,
    validate_empirical_target_variant_consistency,
)

_EXPECTED_VARIANT_SPLITS = {
    "credential_v1": "development",
    "credential_v2": "validation",
    "credential_v3": "test",
    "credential_v4": "test",
    "private_attribute_v1": "development",
    "private_attribute_v2": "validation",
    "private_attribute_v3": "test",
    "private_attribute_v4": "test",
    "authorization_v1": "development",
    "authorization_v2": "validation",
    "authorization_v3": "test",
    "authorization_v4": "test",
}

# Synthetic benchmark values must never reappear in empirical targets.
_SYNTHETIC_VALUES = (
    "0107",
    "Alice has condition X",
    "987-65-4321",
    "Agent B has emergency drone-control authority",
)


class TestVariantNamingAndSplits:
    @pytest.mark.parametrize(
        ("variant_id", "expected_split"), sorted(_EXPECTED_VARIANT_SPLITS.items())
    )
    def test_variant_split_assignment(self, variant_id: str, expected_split: str) -> None:
        spec = get_target_spec(variant_id)
        assert spec.split == expected_split

    def test_variant_names_follow_family_convention(self) -> None:
        for spec in EMPIRICAL_TARGET_REGISTRY:
            family, _, suffix = spec.secret_variant_id.rpartition("_")
            assert family in {"credential", "private_attribute", "authorization"}
            assert suffix in {"v1", "v2", "v3", "v4"}

    def test_unknown_variant_lookup_fails(self) -> None:
        with pytest.raises(KeyError):
            get_target_spec("credential_v9")

    def test_test_split_requires_dependency_freezes(self) -> None:
        from experiments.trustparadox_u.empirical_corpus import (
            EmpiricalPhaseLockedError,
            assert_test_split_locked,
        )

        # Unfrozen: test-split access must be refused.
        with pytest.raises(EmpiricalPhaseLockedError):
            assert_test_split_locked(EmpiricalSplit.TEST.value)
        # All six dependencies frozen: access permitted.
        assert_test_split_locked(
            EmpiricalSplit.TEST.value,
            corpus_frozen=True,
            annotations_frozen=True,
            embedding_frozen=True,
            thresholds_frozen=True,
            hypotheses_frozen=True,
            statistics_frozen=True,
        )


class TestVariantConsistencyChecks:
    def test_consistent_context_passes(self) -> None:
        spec = get_target_spec("credential_v1")
        problems = validate_empirical_target_variant_consistency(
            spec, context_texts=(f"the code is {spec.canonical_target}",)
        )
        assert problems == []

    def test_forget_id_mismatch_detected(self) -> None:
        spec = get_target_spec("credential_v1")
        problems = validate_empirical_target_variant_consistency(
            spec, context_texts=("some context",), forget_id="forget_wrong"
        )
        assert any("forget_id_mismatch" in problem for problem in problems)

    def test_foreign_variant_present_detected(self) -> None:
        spec = get_target_spec("credential_v1")
        foreign = get_target_spec("credential_v2")
        problems = validate_empirical_target_variant_consistency(
            spec, context_texts=(f"note {foreign.canonical_target}",)
        )
        assert any("foreign_variant_present" in problem for problem in problems)

    def test_canonical_target_missing_detected(self) -> None:
        spec = get_target_spec("credential_v1")
        problems = validate_empirical_target_variant_consistency(
            spec, context_texts=("nothing relevant here",)
        )
        assert any("canonical_target_missing" in problem for problem in problems)


class TestSyntheticSeparation:
    def test_no_synthetic_value_reuse_in_canonical_targets(self) -> None:
        for spec in EMPIRICAL_TARGET_REGISTRY:
            for synthetic_value in _SYNTHETIC_VALUES:
                assert synthetic_value not in spec.canonical_target

    def test_no_synthetic_value_reuse_in_metadata(self) -> None:
        for spec in EMPIRICAL_TARGET_REGISTRY:
            metadata_texts = (
                *spec.aliases,
                *spec.permitted_residuals,
                *spec.semantic_descriptions,
                *spec.fragments,
                *spec.fact_chain,
            )
            for synthetic_value in _SYNTHETIC_VALUES:
                assert all(synthetic_value not in text for text in metadata_texts)
