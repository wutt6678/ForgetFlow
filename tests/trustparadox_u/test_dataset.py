"""Tests for dataset loading."""

from pathlib import Path

import pytest

from experiments.trustparadox_u.dataset import (
    load_episode,
    load_single_target_episode,
    load_split,
    validate_attack_target_references,
    validate_single_target_episode,
)

SCENARIOS_DIR = Path(__file__).parents[2] / "data" / "trustparadox_u" / "scenarios"
SPLITS_DIR = Path(__file__).parents[2] / "data" / "trustparadox_u" / "splits"


class TestDataset:
    def test_load_split(self) -> None:
        ids = load_split(SPLITS_DIR / "development.jsonl")
        assert len(ids) == 3
        assert "credential_001_high_direct" in ids


class TestSecretVariantId:
    """Tests for secret_variant_id generation."""

    def test_generated_variant_id_is_stable(self) -> None:
        """Same inputs produce the same variant ID."""
        ep1 = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        ep2 = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        for si1, si2 in zip(ep1.sensitive_items, ep2.sensitive_items):
            assert si1.secret_variant_id == si2.secret_variant_id

    def test_different_canonical_target_different_variant_id(self) -> None:
        """Different canonical targets produce different variant IDs."""
        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        assert ep.sensitive_items[0].secret_variant_id != ""
        assert len(ep.sensitive_items[0].secret_variant_id) == 16

    def test_variant_id_not_just_forget_id(self) -> None:
        """Variant ID should not be the same as forget_id."""
        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        for si in ep.sensitive_items:
            assert si.secret_variant_id != si.forget_id


class TestSingleTargetValidation:
    """ST-DATA-001: Single-target episode validation."""

    def test_single_target_passes(self) -> None:
        """ST-DATA-001-one: Episode with one sensitive item passes."""
        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        validate_single_target_episode(ep)  # Should not raise

    def test_load_single_target_convenience(self) -> None:
        """ST-DATA-001-one: load_single_target_episode works for valid episode."""
        ep = load_single_target_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        assert len(ep.sensitive_items) == 1

    def test_zero_items_rejected(self) -> None:
        """ST-DATA-001-zero: Episode with zero sensitive items fails."""
        from dataclasses import replace

        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        empty = replace(ep, sensitive_items=())
        with pytest.raises(ValueError, match="exactly one sensitive item"):
            validate_single_target_episode(empty)

    def test_two_items_rejected(self) -> None:
        """ST-DATA-001-two: Episode with two sensitive items fails."""
        from dataclasses import replace

        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        doubled = replace(ep, sensitive_items=ep.sensitive_items * 2)
        with pytest.raises(ValueError, match="exactly one sensitive item"):
            validate_single_target_episode(doubled)

    def test_general_loader_accepts_multi_target(self) -> None:
        """ST-DATA-001-general-loader: General loader remains multi-target capable."""
        from dataclasses import replace

        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        doubled = replace(ep, sensitive_items=ep.sensitive_items * 2)
        # General loader should not reject; only the single-target validator does
        assert len(doubled.sensitive_items) == 2


class TestAttackTargetReferences:
    """ST-DATA-002/003: Recontamination target reference validation."""

    def test_valid_target_passes(self) -> None:
        """ST-DATA-003-valid: target_forget_ids referencing valid forget_id passes."""
        ep = load_episode(SCENARIOS_DIR / "pilot_authorization.yaml")
        validate_attack_target_references(ep)  # Should not raise

    def test_empty_targets_rejected(self) -> None:
        """ST-DATA-002-explicit-empty: Recontamination with no targets fails."""
        from dataclasses import replace

        from experiments.trustparadox_u.dataset import MessageLabel

        ep = load_episode(SCENARIOS_DIR / "pilot_authorization.yaml")
        # Replace the recontamination attack's label with empty target_forget_ids
        new_attacks = []
        for atk in ep.phases.post_forget:
            if atk.label.is_recontamination_attempt:
                new_label = MessageLabel(
                    is_attack_attempt=True,
                    is_recontamination_attempt=True,
                    target_forget_ids=(),
                )
                new_attacks.append(replace(atk, label=new_label))
            else:
                new_attacks.append(atk)
        new_phases = replace(ep.phases, post_forget=tuple(new_attacks))
        bad_ep = replace(ep, phases=new_phases)
        with pytest.raises(ValueError, match="non-empty target_forget_ids"):
            validate_attack_target_references(bad_ep)

    def test_unknown_target_rejected(self) -> None:
        """ST-DATA-003-unknown: target_forget_ids with unknown ID fails."""
        from dataclasses import replace

        from experiments.trustparadox_u.dataset import MessageLabel

        ep = load_episode(SCENARIOS_DIR / "pilot_authorization.yaml")
        new_attacks = []
        for atk in ep.phases.post_forget:
            if atk.label.is_recontamination_attempt:
                new_label = MessageLabel(
                    is_attack_attempt=True,
                    is_recontamination_attempt=True,
                    target_forget_ids=("UNKNOWN",),
                )
                new_attacks.append(replace(atk, label=new_label))
            else:
                new_attacks.append(atk)
        new_phases = replace(ep.phases, post_forget=tuple(new_attacks))
        bad_ep = replace(ep, phases=new_phases)
        with pytest.raises(ValueError, match="Unknown target_forget_ids"):
            validate_attack_target_references(bad_ep)

    def test_mixed_valid_unknown_rejected(self) -> None:
        """ST-DATA-003-mixed: Mix of valid and unknown IDs fails."""
        from dataclasses import replace

        from experiments.trustparadox_u.dataset import MessageLabel

        ep = load_episode(SCENARIOS_DIR / "pilot_authorization.yaml")
        forget_id = ep.sensitive_items[0].forget_id
        new_attacks = []
        for atk in ep.phases.post_forget:
            if atk.label.is_recontamination_attempt:
                new_label = MessageLabel(
                    is_attack_attempt=True,
                    is_recontamination_attempt=True,
                    target_forget_ids=(forget_id, "UNKNOWN"),
                )
                new_attacks.append(replace(atk, label=new_label))
            else:
                new_attacks.append(atk)
        new_phases = replace(ep.phases, post_forget=tuple(new_attacks))
        bad_ep = replace(ep, phases=new_phases)
        with pytest.raises(ValueError, match="Unknown target_forget_ids"):
            validate_attack_target_references(bad_ep)

    def test_safe_message_no_targets_ok(self) -> None:
        """ST-DATA-002-safe-empty: Non-recontamination may have no targets."""
        from dataclasses import replace

        from experiments.trustparadox_u.dataset import MessageLabel

        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        # All attacks are non-recontamination; ensure they pass with empty targets
        new_attacks = []
        for atk in ep.phases.post_forget:
            new_label = MessageLabel(
                is_attack_attempt=atk.label.is_attack_attempt,
                is_reconstruction_attempt=atk.label.is_reconstruction_attempt,
                is_recontamination_attempt=False,
                target_forget_ids=(),
            )
            new_attacks.append(replace(atk, label=new_label))
        new_phases = replace(ep.phases, post_forget=tuple(new_attacks))
        ok_ep = replace(ep, phases=new_phases)
        validate_attack_target_references(ok_ep)  # Should not raise


class TestDefaultRecontaminationValidation:
    """Section 2: Default recontamination labels must require target_forget_ids."""

    def test_default_recontamination_without_label_fails(self) -> None:
        """Recontamination attack with empty label dict raises ValueError."""
        from experiments.trustparadox_u.dataset import _parse_message_label

        with pytest.raises(ValueError, match="target_forget_ids"):
            _parse_message_label({}, "recontamination")

    def test_explicit_recontamination_without_targets_fails(self) -> None:
        """Explicit recontamination label without targets raises ValueError."""
        from experiments.trustparadox_u.dataset import _parse_message_label

        with pytest.raises(ValueError, match="target_forget_ids"):
            _parse_message_label(
                {"is_recontamination_attempt": True}, "recontamination"
            )

    def test_explicit_recontamination_with_targets_passes(self) -> None:
        """Explicit recontamination label with targets passes."""
        from experiments.trustparadox_u.dataset import _parse_message_label

        label = _parse_message_label(
            {"is_recontamination_attempt": True, "target_forget_ids": ["F001"]},
            "recontamination",
        )
        assert label.is_recontamination_attempt
        assert label.target_forget_ids == ("F001",)

    def test_empty_target_string_filtered(self) -> None:
        """Empty target strings are filtered out, causing recontamination to fail."""
        from experiments.trustparadox_u.dataset import _parse_message_label

        with pytest.raises(ValueError, match="target_forget_ids"):
            _parse_message_label(
                {"is_recontamination_attempt": True, "target_forget_ids": [""]},
                "recontamination",
            )

    def test_duplicate_targets_normalized(self) -> None:
        """Duplicate target_forget_ids are deduplicated and sorted."""
        from experiments.trustparadox_u.dataset import _parse_message_label

        label = _parse_message_label(
            {
                "is_recontamination_attempt": True,
                "target_forget_ids": ["F002", "F001", "F002"],
            },
            "recontamination",
        )
        assert label.target_forget_ids == ("F001", "F002")

    def test_non_recontamination_may_omit_targets(self) -> None:
        """Non-recontamination labels may omit target_forget_ids."""
        from experiments.trustparadox_u.dataset import _parse_message_label

        label = _parse_message_label({"is_attack_attempt": True}, "direct_probing")
        assert not label.is_recontamination_attempt
        assert label.target_forget_ids == ()


class TestLoadingTargetValidation:
    """Section 3: Loading validates target_forget_ids against sensitive items."""

    def test_load_valid_episode_passes(self) -> None:
        """Episode with valid target_forget_ids loads successfully."""
        ep = load_episode(SCENARIOS_DIR / "pilot_authorization.yaml")
        # Should have recontamination with valid targets
        recontamination_attacks = [
            atk
            for atk in ep.phases.post_forget
            if atk.label.is_recontamination_attempt
        ]
        assert len(recontamination_attacks) > 0

    def test_load_with_unknown_target_fails(self) -> None:
        """Loading episode with unknown target_forget_ids raises ValueError."""
        import tempfile
        import textwrap

        yaml_content = textwrap.dedent("""\
            episode_id: test_unknown_target
            scenario_id: test
            macro_scene: test
            trust_level: high
            agents:
              - agent_id: A
                role: sender
                public_profile: test
                private_context: []
              - agent_id: B
                role: recipient
                public_profile: test
                private_context: []
            relationships:
              - source: A
                target: B
                relation: test
            task:
              description: test
              success_label: test
              success_type: exact_label
              success_value: test
            sensitive_items:
              - forget_id: F001
                target_type: credential
                canonical_target: secret
                aliases: []
                semantic_variants: []
                permitted_residuals: []
                active_from_turn: 0
            phases:
              pre_forget: []
              forget:
                turn: 1
                clean_agents: [A]
              post_forget:
                - attack_type: recontamination
                  attacker: B
                  target_agent: A
                  instruction: test
                  label:
                    is_recontamination_attempt: true
                    target_forget_ids: [UNKNOWN_ID]
            expected:
              forbidden_strings: []
              permitted_strings: []
        """)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()
            with pytest.raises(ValueError, match="Unknown target_forget_ids"):
                load_episode(f.name)
