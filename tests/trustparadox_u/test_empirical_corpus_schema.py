"""E1-024/025/026/027/028: empirical corpus schema, identity, registry,
contamination, and sequence structure tests."""

from __future__ import annotations

import dataclasses

import pytest

from experiments.trustparadox_u.empirical_corpus import (
    CROSS_VARIANT_CONTAMINATION,
    EMPIRICAL_SCENARIOS,
    EMPIRICAL_TARGET_REGISTRY,
    SEQUENCE_ATTACK_TYPES,
    AttackType,
    EmpiricalGenerationAttempt,
    EmpiricalSplit,
    GenerationStatus,
    TrustLevel,
    accept_generation_attempt,
    compute_target_registry_hash,
    empirical_candidate_family_id,
    empirical_candidate_id,
    empirical_content_hash,
    empirical_sequence_family_id,
    empirical_sequence_id,
    generation_attempt_id,
    get_target_spec,
    validate_sequence_structure,
    validate_target_registry,
)

_DEV_SPEC = get_target_spec("credential_v1")
_VAL_SPEC = get_target_spec("credential_v2")


def make_attempt(
    *,
    trust_level: str = TrustLevel.DEFAULT.value,
    attack_type: str = AttackType.DIRECT_DISCLOSURE.value,
    sample_index: int = 0,
    generation_replicate: int = 0,
    status: str = GenerationStatus.SUCCESS.value,
    text: str | None = "As agreed between us: 5163",
    sequence_step_index: int | None = None,
    sequence_step_count: int | None = None,
    **overrides: object,
) -> EmpiricalGenerationAttempt:
    spec = _DEV_SPEC
    attempt = EmpiricalGenerationAttempt(
        generation_attempt_id=generation_attempt_id(
            scenario_id=spec.scenario_id,
            secret_variant_id=spec.secret_variant_id,
            trust_level=trust_level,
            attack_type=attack_type,
            sample_index=sample_index,
            generation_replicate=generation_replicate,
            sequence_step_index=sequence_step_index,
        ),
        scenario_id=spec.scenario_id,
        secret_variant_id=spec.secret_variant_id,
        split=spec.split,
        trust_level=trust_level,
        attack_type=attack_type,
        sample_index=sample_index,
        generation_replicate=generation_replicate,
        sender_id=spec.custodian_agent_id,
        recipient_id=spec.default_recipient_id,
        candidate_family_id=empirical_candidate_family_id(
            scenario_id=spec.scenario_id,
            secret_variant_id=spec.secret_variant_id,
            attack_type=attack_type,
            sample_index=sample_index,
            generation_replicate=generation_replicate,
            sequence_step_index=sequence_step_index,
        ),
        sequence_family_id=(
            empirical_sequence_family_id(
                scenario_id=spec.scenario_id,
                secret_variant_id=spec.secret_variant_id,
                attack_type=attack_type,
                sample_index=sample_index,
                generation_replicate=generation_replicate,
            )
            if sequence_step_index is not None
            else None
        ),
        sequence_id=None,
        sequence_step_index=sequence_step_index,
        sequence_step_count=sequence_step_count,
        candidate_text=text,
        generation_status=status,
        refusal=status == GenerationStatus.REFUSAL.value,
        malformed=status == GenerationStatus.MALFORMED.value,
        off_topic=status == GenerationStatus.OFF_TOPIC.value,
        generator_provider="mock",
        generator_model="mock-empirical-generator",
        generator_revision=None,
        temperature=0.7,
        seed=None,
        system_prompt_hash="0" * 64,
        user_prompt_hash="1" * 64,
        request_id=None,
        retry_index=0,
        generated_at="1970-01-01T00:00:00+00:00",
    )
    if sequence_step_index is not None:
        assert attempt.sequence_family_id is not None
        attempt = dataclasses.replace(
            attempt,
            sequence_id=empirical_sequence_id(attempt.sequence_family_id, trust_level),
        )
    return dataclasses.replace(attempt, **overrides)  # type: ignore[arg-type]


def make_sequence_step(step_index: int, step_count: int, **overrides: object):
    return make_attempt(
        attack_type=AttackType.FRAGMENTATION_SEQUENCE.value,
        text=_DEV_SPEC.fragments[min(step_index, len(_DEV_SPEC.fragments) - 1)],
        sequence_step_index=step_index,
        sequence_step_count=step_count,
        **overrides,
    )


# ---------------------------------------------------------------------------
# E1-024: generation schema tests
# ---------------------------------------------------------------------------


class TestGenerationSchema:
    def test_valid_non_sequence_attempt(self) -> None:
        attempt = make_attempt()
        assert attempt.validate() == []
        assert attempt.validate_identity() == []
        assert attempt.validate_against_target_spec(_DEV_SPEC) == []

    def test_valid_sequence_attempt(self) -> None:
        attempt = make_sequence_step(0, 2)
        assert attempt.validate() == []
        assert attempt.validate_identity() == []
        assert attempt.is_sequence_attempt

    def test_successful_attempt_requires_text(self) -> None:
        attempt = make_attempt(text=None)
        assert any("candidate text" in problem for problem in attempt.validate())

    def test_refusal_may_omit_candidate_text(self) -> None:
        attempt = make_attempt(status=GenerationStatus.REFUSAL.value, text=None)
        assert attempt.validate() == []

    def test_invalid_trust_rejected(self) -> None:
        attempt = make_attempt()
        bad = dataclasses.replace(attempt, trust_level="extreme")
        assert any("invalid trust_level" in problem for problem in bad.validate())

    def test_invalid_split_rejected(self) -> None:
        attempt = make_attempt()
        bad = dataclasses.replace(attempt, split="holdout")
        assert any("invalid split" in problem for problem in bad.validate())

    def test_sequence_fields_all_present_or_all_null(self) -> None:
        attempt = make_sequence_step(0, 2)
        bad = dataclasses.replace(attempt, sequence_step_index=None)
        assert any("all-present or all-null" in problem for problem in bad.validate())

    def test_candidate_id_unique_across_units(self) -> None:
        ids = {
            empirical_candidate_id(
                empirical_candidate_family_id(
                    scenario_id=spec.scenario_id,
                    secret_variant_id=spec.secret_variant_id,
                    attack_type=AttackType.DIRECT_DISCLOSURE.value,
                    sample_index=sample_index,
                    generation_replicate=0,
                ),
                trust.value,
            )
            for spec in EMPIRICAL_TARGET_REGISTRY
            for sample_index in range(3)
            for trust in TrustLevel
        }
        assert len(ids) == len(EMPIRICAL_TARGET_REGISTRY) * 3 * len(TrustLevel)

    def test_generation_attempt_id_unique_across_units(self) -> None:
        ids = {
            generation_attempt_id(
                scenario_id=spec.scenario_id,
                secret_variant_id=spec.secret_variant_id,
                trust_level=trust.value,
                attack_type=attack.value,
                sample_index=sample_index,
                generation_replicate=replicate,
            )
            for spec in EMPIRICAL_TARGET_REGISTRY
            for trust in TrustLevel
            for attack in (AttackType.DIRECT_DISCLOSURE, AttackType.SEMANTIC_PARAPHRASE)
            for sample_index in range(2)
            for replicate in range(2)
        }
        expected = len(EMPIRICAL_TARGET_REGISTRY) * 3 * 2 * 2 * 2
        assert len(ids) == expected

    def test_split_matches_target_spec(self) -> None:
        attempt = make_attempt()
        mismatched = dataclasses.replace(attempt, split=EmpiricalSplit.VALIDATION.value)
        problems = mismatched.validate_against_target_spec(_DEV_SPEC)
        assert any("split" in problem for problem in problems)


# ---------------------------------------------------------------------------
# E1-025: identity tests
# ---------------------------------------------------------------------------


class TestIdentity:
    def _family_kwargs(self) -> dict[str, object]:
        return {
            "scenario_id": _DEV_SPEC.scenario_id,
            "secret_variant_id": _DEV_SPEC.secret_variant_id,
            "attack_type": AttackType.DIRECT_DISCLOSURE.value,
            "sample_index": 0,
            "generation_replicate": 0,
        }

    def test_trust_levels_share_candidate_family_id(self) -> None:
        # Family IDs have no trust parameter at all: the same family serves
        # every trust level by construction.
        families = {
            empirical_candidate_family_id(**self._family_kwargs())  # type: ignore[arg-type]
            for _ in TrustLevel
        }
        assert len(families) == 1

    def test_trust_specific_candidate_ids_differ(self) -> None:
        family = empirical_candidate_family_id(
            **self._family_kwargs(),  # type: ignore[arg-type]
        )
        ids = {empirical_candidate_id(family, trust.value) for trust in TrustLevel}
        assert len(ids) == len(TrustLevel)

    def test_sequence_family_id_is_trust_independent(self) -> None:
        kwargs = {
            "scenario_id": _DEV_SPEC.scenario_id,
            "secret_variant_id": _DEV_SPEC.secret_variant_id,
            "attack_type": AttackType.FRAGMENTATION_SEQUENCE.value,
            "sample_index": 0,
            "generation_replicate": 0,
        }
        families = {
            empirical_sequence_family_id(**kwargs)  # type: ignore[arg-type]
            for _ in TrustLevel
        }
        assert len(families) == 1
        assert "low" not in next(iter(families))
        assert "high" not in next(iter(families))

    def test_sequence_id_is_trust_specific(self) -> None:
        family = empirical_sequence_family_id(
            scenario_id=_DEV_SPEC.scenario_id,
            secret_variant_id=_DEV_SPEC.secret_variant_id,
            attack_type=AttackType.FRAGMENTATION_SEQUENCE.value,
            sample_index=0,
            generation_replicate=0,
        )
        ids = {empirical_sequence_id(family, trust.value) for trust in TrustLevel}
        assert len(ids) == len(TrustLevel)

    def test_firewall_condition_cannot_affect_candidate_identity(self) -> None:
        """Family/candidate IDs have no firewall-condition parameter at all."""
        signature_fields = set(empirical_candidate_family_id.__code__.co_varnames)
        assert not any("firewall" in field for field in signature_fields)
        assert not any("condition" in field for field in signature_fields)
        # And identical construction under any hypothetical condition yields
        # the identical ID:
        family_a = empirical_candidate_family_id(
            **self._family_kwargs(),  # type: ignore[arg-type]
        )
        family_b = empirical_candidate_family_id(
            **self._family_kwargs(),  # type: ignore[arg-type]
        )
        assert family_a == family_b

    def test_content_hash_changes_when_text_changes(self) -> None:
        assert empirical_content_hash("message one") != empirical_content_hash("message two")

    def test_generation_replicate_changes_attempt_identity(self) -> None:
        kwargs = {
            "scenario_id": _DEV_SPEC.scenario_id,
            "secret_variant_id": _DEV_SPEC.secret_variant_id,
            "trust_level": "default",
            "attack_type": AttackType.DIRECT_DISCLOSURE.value,
            "sample_index": 0,
        }
        assert generation_attempt_id(generation_replicate=0, **kwargs) != generation_attempt_id(  # type: ignore[arg-type]
            generation_replicate=1,
            **kwargs,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# E1-026: target-registry tests
# ---------------------------------------------------------------------------


class TestTargetRegistry:
    def test_registry_has_no_structural_problems(self) -> None:
        assert validate_target_registry(EMPIRICAL_TARGET_REGISTRY) == []

    def test_exactly_twelve_target_specs(self) -> None:
        assert len(EMPIRICAL_TARGET_REGISTRY) == 12

    def test_exactly_four_variants_per_scenario(self) -> None:
        for scenario in EMPIRICAL_SCENARIOS:
            count = sum(1 for spec in EMPIRICAL_TARGET_REGISTRY if spec.scenario_id == scenario)
            assert count == 4, scenario

    def test_split_assignment_matches_protocol(self) -> None:
        split_counts: dict[str, int] = {}
        for spec in EMPIRICAL_TARGET_REGISTRY:
            split_counts[spec.split] = split_counts.get(spec.split, 0) + 1
        assert split_counts == {"development": 3, "validation": 3, "test": 6}

    def test_each_variant_belongs_to_one_split(self) -> None:
        seen: dict[str, str] = {}
        for spec in EMPIRICAL_TARGET_REGISTRY:
            assert spec.secret_variant_id not in seen or seen[spec.secret_variant_id] == spec.split
            seen[spec.secret_variant_id] = spec.split
        assert len(seen) == 12

    def test_forget_ids_unique(self) -> None:
        forget_ids = [spec.forget_id for spec in EMPIRICAL_TARGET_REGISTRY]
        assert len(set(forget_ids)) == len(forget_ids)

    def test_no_canonical_value_reuse(self) -> None:
        targets = [spec.canonical_target for spec in EMPIRICAL_TARGET_REGISTRY]
        assert len(set(targets)) == len(targets)

    def test_permitted_residuals_nonempty(self) -> None:
        for spec in EMPIRICAL_TARGET_REGISTRY:
            assert spec.permitted_residuals, spec.secret_variant_id
            assert all(residual.strip() for residual in spec.permitted_residuals)

    def test_fragment_definitions_internally_consistent(self) -> None:
        for spec in EMPIRICAL_TARGET_REGISTRY:
            assert len(spec.fragments) >= 2, spec.secret_variant_id
            assert len(spec.fact_chain) >= 2, spec.secret_variant_id
            assert all(fragment.strip() for fragment in spec.fragments)
            assert all(fact.strip() for fact in spec.fact_chain)

    def test_target_metadata_hash_deterministic(self) -> None:
        hash_a = compute_target_registry_hash(EMPIRICAL_TARGET_REGISTRY)
        hash_b = compute_target_registry_hash(tuple(reversed(EMPIRICAL_TARGET_REGISTRY)))
        assert hash_a == hash_b
        assert len(hash_a) == 64


# ---------------------------------------------------------------------------
# E1-027: contamination tests
# ---------------------------------------------------------------------------


class TestContamination:
    def test_selected_variant_passes(self) -> None:
        attempt = make_attempt(text=f"Confirmed: {_DEV_SPEC.canonical_target}")
        result = accept_generation_attempt(attempt, _DEV_SPEC)
        assert result.accepted
        assert result.reason == "accepted"

    @pytest.mark.parametrize(
        "foreign_variant",
        ["credential_v2", "private_attribute_v1", "authorization_v3"],
    )
    def test_foreign_target_rejected(self, foreign_variant: str) -> None:
        foreign = get_target_spec(foreign_variant)
        attempt = make_attempt(text=f"By the way: {foreign.canonical_target}")
        result = accept_generation_attempt(attempt, _DEV_SPEC)
        assert not result.accepted
        assert result.reason == CROSS_VARIANT_CONTAMINATION

    def test_multiple_foreign_variants_rejected(self) -> None:
        other_a = get_target_spec("credential_v2")
        other_b = get_target_spec("private_attribute_v2")
        attempt = make_attempt(
            text=f"{other_a.canonical_target} and also {other_b.canonical_target}"
        )
        result = accept_generation_attempt(attempt, _DEV_SPEC)
        assert not result.accepted
        assert result.reason == CROSS_VARIANT_CONTAMINATION


# ---------------------------------------------------------------------------
# E1-028: sequence validator tests
# ---------------------------------------------------------------------------


class TestSequenceValidator:
    def test_complete_two_step_sequence_passes(self) -> None:
        steps = [make_sequence_step(0, 2), make_sequence_step(1, 2)]
        assert validate_sequence_structure(steps) == []

    def test_missing_final_step_fails(self) -> None:
        steps = [make_sequence_step(0, 3), make_sequence_step(1, 3)]
        assert validate_sequence_structure(steps) != []

    def test_duplicate_step_fails(self) -> None:
        steps = [make_sequence_step(0, 2), make_sequence_step(0, 2)]
        assert validate_sequence_structure(steps) != []

    def test_wrong_trust_in_one_step_fails(self) -> None:
        steps = [
            make_sequence_step(0, 2, trust_level=TrustLevel.LOW.value),
            make_sequence_step(1, 2, trust_level=TrustLevel.HIGH.value),
        ]
        assert validate_sequence_structure(steps) != []

    def test_wrong_target_variant_fails(self) -> None:
        foreign = make_sequence_step(1, 2)
        steps = [
            make_sequence_step(0, 2),
            dataclasses.replace(
                foreign,
                secret_variant_id=_VAL_SPEC.secret_variant_id,
                scenario_id=_VAL_SPEC.scenario_id,
                split=_VAL_SPEC.split,
            ),
        ]
        assert validate_sequence_structure(steps) != []

    def test_wrong_sequence_family_id_fails(self) -> None:
        step_b = make_sequence_step(1, 2)
        assert step_b.sequence_family_id is not None
        steps = [
            make_sequence_step(0, 2),
            dataclasses.replace(step_b, sequence_family_id="esf_other_family"),
        ]
        assert validate_sequence_structure(steps) != []

    def test_mixed_sender_recipient_structure_fails(self) -> None:
        step_b = make_sequence_step(1, 2)
        steps = [make_sequence_step(0, 2), dataclasses.replace(step_b, recipient_id="AGENT_X")]
        assert validate_sequence_structure(steps) != []

    def test_noncontiguous_steps_fail(self) -> None:
        steps = [make_sequence_step(0, 3), make_sequence_step(2, 3)]
        assert validate_sequence_structure(steps) != []

    def test_sequence_attack_types_are_structural_only(self) -> None:
        assert AttackType.FRAGMENTATION_SEQUENCE in SEQUENCE_ATTACK_TYPES
        assert AttackType.COMPOSITIONAL_SEQUENCE in SEQUENCE_ATTACK_TYPES
        assert AttackType.DIRECT_DISCLOSURE not in SEQUENCE_ATTACK_TYPES
