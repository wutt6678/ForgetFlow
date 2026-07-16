"""Tests for dataset loading."""

from pathlib import Path

from experiments.trustparadox_u.dataset import load_episode, load_split

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
