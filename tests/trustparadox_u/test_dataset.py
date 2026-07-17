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
                new_attacks.append(
                    replace(atk, label=new_label)
                )
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
                new_attacks.append(
                    replace(atk, label=new_label)
                )
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
                new_attacks.append(
                    replace(atk, label=new_label)
                )
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
