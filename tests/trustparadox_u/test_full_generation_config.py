"""E3-003/004: tests for frozen generation config and deterministic plan.

Covers:

- config contains all required frozen fields;
- config is hashable and deterministic;
- plan items have unique IDs;
- plan invariants (no firewall fields, sequence steps correct);
- attack applicability rules;
- expected split/scenario/trust/attack counts;
- plan summary matches the expanded items.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.trustparadox_u.empirical_corpus import (
    EMPIRICAL_TARGET_REGISTRY,
    empirical_candidate_family_id,
)
from experiments.trustparadox_u.empirical_generation_plan import (
    FROZEN_TRUST_LEVELS,
    GENERATOR_MODEL_REQUESTED,
    GENERATOR_TEMPERATURE,
    attack_is_applicable,
    build_full_generation_plan,
    build_generation_config,
    plan_sha256,
    plan_summary,
    planned_units_for_spec,
    validate_generation_plan,
)


class TestGenerationConfig:
    def test_config_has_required_fields(self) -> None:
        config = build_generation_config()
        required_fields = (
            "schema_version",
            "protocol_version",
            "study_version",
            "generator_provider",
            "generator_model_requested",
            "generator_temperature",
            "generator_max_tokens",
            "generator_seed_policy",
            "generation_replicates",
            "retry_policy",
            "request_timeout",
            "trust_levels",
            "attack_types",
            "sample_counts_by_attack",
            "scenario_ids",
            "target_registry_sha256",
            "frozen_prompt_manifest_sha256",
            "environment_lock_hash",
            "phase_manifest_sha256",
            "created_from_commit",
        )
        for field in required_fields:
            assert field in config, f"missing config field: {field}"

    def test_config_model_is_frozen(self) -> None:
        config = build_generation_config()
        assert config["generator_model_requested"] == GENERATOR_MODEL_REQUESTED
        assert config["generator_temperature"] == GENERATOR_TEMPERATURE

    def test_config_is_deterministic(self) -> None:
        c1 = build_generation_config()
        c2 = build_generation_config()
        # Hash-bearing fields are stable within the same tree.
        assert c1["target_registry_sha256"] == c2["target_registry_sha256"]
        assert c1["environment_lock_hash"] == c2["environment_lock_hash"]

    def test_config_trust_levels_match_protocol(self) -> None:
        config = build_generation_config()
        assert config["trust_levels"] == list(FROZEN_TRUST_LEVELS)

    def test_config_attack_types_exclude_discretion(self) -> None:
        config = build_generation_config()
        assert "trust_discretion_task" not in config["attack_types"]

    def test_config_writable(self, tmp_path: Path) -> None:
        from experiments.trustparadox_u.empirical_generation_plan import (
            write_generation_config,
        )

        path = tmp_path / "config.json"
        config = write_generation_config(path)
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["generator_model_requested"] == config["generator_model_requested"]


class TestAttackApplicability:
    def test_alias_not_applicable_for_credential(self) -> None:
        cred_spec = next(
            s
            for s in EMPIRICAL_TARGET_REGISTRY
            if s.scenario_id == "credential_001" and s.split == "development"
        )
        assert not attack_is_applicable("alias_or_coreference", cred_spec)

    def test_alias_applicable_for_private_attribute(self) -> None:
        pa_spec = next(
            s
            for s in EMPIRICAL_TARGET_REGISTRY
            if s.scenario_id == "private_attribute_001" and s.split == "development"
        )
        assert attack_is_applicable("alias_or_coreference", pa_spec)

    def test_alias_applicable_for_authorization(self) -> None:
        auth_spec = next(
            s
            for s in EMPIRICAL_TARGET_REGISTRY
            if s.scenario_id == "authorization_001" and s.split == "development"
        )
        assert attack_is_applicable("alias_or_coreference", auth_spec)

    def test_compositional_requires_fact_chain(self) -> None:
        cred_spec = next(
            s
            for s in EMPIRICAL_TARGET_REGISTRY
            if s.scenario_id == "credential_001" and s.split == "development"
        )
        # Credentials have a fact_chain of length >= 2.
        assert attack_is_applicable("compositional_sequence", cred_spec)

    def test_direct_always_applicable(self) -> None:
        for spec in EMPIRICAL_TARGET_REGISTRY:
            assert attack_is_applicable("direct_disclosure", spec)


class TestPlannedUnits:
    def test_plan_for_development_spec(self) -> None:
        cred_spec = next(
            s
            for s in EMPIRICAL_TARGET_REGISTRY
            if s.scenario_id == "credential_001" and s.split == "development"
        )
        items = planned_units_for_spec(cred_spec)
        assert len(items) > 0
        # All items must reference the correct spec.
        for it in items:
            assert it.scenario_id == "credential_001"
            assert it.secret_variant_id == "credential_v1"
            assert it.split == "development"

    def test_credential_has_no_alias_items(self) -> None:
        cred_spec = next(
            s
            for s in EMPIRICAL_TARGET_REGISTRY
            if s.scenario_id == "credential_001" and s.split == "development"
        )
        items = planned_units_for_spec(cred_spec)
        alias_items = [it for it in items if it.attack_type == "alias_or_coreference"]
        assert len(alias_items) == 0

    def test_sequence_items_have_step_metadata(self) -> None:
        cred_spec = next(
            s
            for s in EMPIRICAL_TARGET_REGISTRY
            if s.scenario_id == "credential_001" and s.split == "development"
        )
        items = planned_units_for_spec(cred_spec)
        frag_items = [it for it in items if it.attack_type == "fragmentation_sequence"]
        assert len(frag_items) > 0
        for it in frag_items:
            assert it.sequence_id is not None
            assert it.sequence_step_index is not None
            assert it.sequence_step_count is not None
            assert it.sequence_step_count == len(cred_spec.fragments)

    def test_trust_symmetry(self) -> None:
        """low/default/high must have the same planned family structure."""
        spec = next(
            s
            for s in EMPIRICAL_TARGET_REGISTRY
            if s.scenario_id == "private_attribute_001" and s.split == "development"
        )
        items = planned_units_for_spec(spec)
        families_by_trust: dict[str, set[str]] = {}
        for trust in FROZEN_TRUST_LEVELS:
            families_by_trust[trust] = {
                empirical_candidate_family_id(
                    scenario_id=it.scenario_id,
                    secret_variant_id=it.secret_variant_id,
                    attack_type=it.attack_type,
                    sample_index=it.sample_index,
                    generation_replicate=it.generation_replicate,
                    sequence_step_index=it.sequence_step_index,
                )
                for it in items
                if it.trust_level == trust
            }
        assert families_by_trust["low"] == families_by_trust["default"]
        assert families_by_trust["default"] == families_by_trust["high"]


class TestFullPlan:
    def test_full_plan_builds_without_error(self) -> None:
        items = build_full_generation_plan()
        assert len(items) > 0

    def test_full_plan_validates_clean(self) -> None:
        items = build_full_generation_plan()
        findings = validate_generation_plan(items)
        assert findings == []

    def test_full_plan_unique_ids(self) -> None:
        items = build_full_generation_plan()
        ids = [it.plan_item_id for it in items]
        assert len(ids) == len(set(ids))

    def test_full_plan_split_counts(self) -> None:
        items = build_full_generation_plan()
        summary = plan_summary(items)
        by_split = summary["by_split"]
        # test should have roughly 2x the items of development
        # (test has 6 specs vs development's 3).
        assert by_split["development"] > 0
        assert by_split["validation"] > 0
        assert by_split["test"] > 0
        # test should be approximately double development.
        ratio = by_split["test"] / by_split["development"]
        assert 1.5 < ratio < 2.5

    def test_full_plan_summary_matches_items(self) -> None:
        items = build_full_generation_plan()
        summary = plan_summary(items)
        assert summary["total_planned_attempts"] == len(items)
        seq_items = [it for it in items if it.sequence_id is not None]
        non_seq = [it for it in items if it.sequence_id is None]
        assert summary["sequence_step_attempts"] == len(seq_items)
        assert summary["non_sequence_attempts"] == len(non_seq)

    def test_plan_sha256_deterministic(self) -> None:
        items = build_full_generation_plan()
        h1 = plan_sha256(items)
        h2 = plan_sha256(items)
        assert h1 == h2
        assert len(h1) == 64

    def test_plan_contains_no_firewall_fields(self) -> None:
        items = build_full_generation_plan()
        for it in items:
            assert "firewall" not in it.plan_item_id
            assert "embedding" not in it.plan_item_id

    def test_plan_writable(self, tmp_path: Path) -> None:
        from experiments.trustparadox_u.empirical_generation_plan import (
            write_generation_plan,
        )

        plan_path = tmp_path / "plan.jsonl"
        summary_path = tmp_path / "summary.json"
        items, summary = write_generation_plan(plan_path, summary_path)
        assert plan_path.exists()
        assert summary_path.exists()
        assert summary["total_planned_attempts"] == len(items)
        assert "plan_sha256" in summary
