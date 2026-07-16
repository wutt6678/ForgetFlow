"""Tests for experiment matrix generator."""

import pytest

from experiments.trustparadox_u.generate_matrix import (
    MatrixEntry,
    generate_matrix,
    group_trust_triplets,
    validate_trust_triplet,
)


class TestMatrix:
    def test_deterministic(self) -> None:
        m1 = generate_matrix(["s1"], ["high"], ["direct"], ["full_mvp"], [42])
        m2 = generate_matrix(["s1"], ["high"], ["direct"], ["full_mvp"], [42])
        assert m1 == m2

    def test_no_duplicate_ids(self) -> None:
        entries = generate_matrix(
            ["s1", "s2"],
            ["low", "high"],
            ["direct", "alias"],
            ["full_mvp", "no_firewall"],
            [42, 43],
        )
        ids = [e.run_id for e in entries]
        assert len(ids) == len(set(ids))

    def test_correct_size(self) -> None:
        entries = generate_matrix(
            ["s1", "s2"],
            ["low", "high"],
            ["direct"],
            ["full_mvp"],
            [42],
        )
        assert len(entries) == 2 * 2 * 1 * 1 * 1

    def test_config_path(self) -> None:
        entries = generate_matrix(["s1"], ["high"], ["direct"], ["full_mvp"], [42])
        assert "full_mvp.yaml" in entries[0].config_path


class TestTrustMatrixIntegrity:
    """Ensure trust comparison runs are paired correctly."""

    def _make_triplet(self) -> tuple[MatrixEntry, MatrixEntry, MatrixEntry]:
        """Create a valid trust triplet."""
        low = MatrixEntry(
            run_id="s1_low_direct_full_mvp_42",
            scenario_id="s1",
            trust_level="low",
            attack_type="direct",
            firewall_variant="full_mvp",
            seed=42,
            config_path="configs/full_mvp.yaml",
        )
        default = MatrixEntry(
            run_id="s1_default_direct_full_mvp_42",
            scenario_id="s1",
            trust_level="default",
            attack_type="direct",
            firewall_variant="full_mvp",
            seed=42,
            config_path="configs/full_mvp.yaml",
        )
        high = MatrixEntry(
            run_id="s1_high_direct_full_mvp_42",
            scenario_id="s1",
            trust_level="high",
            attack_type="direct",
            firewall_variant="full_mvp",
            seed=42,
            config_path="configs/full_mvp.yaml",
        )
        return low, default, high

    def test_correct_matched_triplet(self) -> None:
        """Valid triplet passes validation."""
        low, default, high = self._make_triplet()
        assert validate_trust_triplet(low, default, high) is True

    def test_threshold_drift_rejected(self) -> None:
        """Different config_path (threshold drift) is rejected."""
        low, default, high = self._make_triplet()
        # Simulate threshold drift: high has a different config
        high = MatrixEntry(
            run_id="s1_high_direct_full_mvp_42",
            scenario_id="s1",
            trust_level="high",
            attack_type="direct",
            firewall_variant="full_mvp",
            seed=42,
            config_path="configs/full_mvp_v2.yaml",
        )
        with pytest.raises(ValueError, match="config_path"):
            validate_trust_triplet(low, default, high)

    def test_history_window_drift_rejected(self) -> None:
        """Different config_path (history window drift) is rejected."""
        low, default, high = self._make_triplet()
        # Simulate history window drift via different config path
        default = MatrixEntry(
            run_id="s1_default_direct_full_mvp_42",
            scenario_id="s1",
            trust_level="default",
            attack_type="direct",
            firewall_variant="full_mvp",
            seed=42,
            config_path="configs/full_mvp_w10.yaml",
        )
        with pytest.raises(ValueError, match="config_path"):
            validate_trust_triplet(low, default, high)

    def test_different_secret_rejected(self) -> None:
        """Different scenario_id is rejected."""
        low, default, high = self._make_triplet()
        high = MatrixEntry(
            run_id="s2_high_direct_full_mvp_42",
            scenario_id="s2",
            trust_level="high",
            attack_type="direct",
            firewall_variant="full_mvp",
            seed=42,
            config_path="configs/full_mvp.yaml",
        )
        with pytest.raises(ValueError, match="scenario_id"):
            validate_trust_triplet(low, default, high)

    def test_different_seed_rejected(self) -> None:
        """Different seed is rejected."""
        low, default, high = self._make_triplet()
        high = MatrixEntry(
            run_id="s1_high_direct_full_mvp_99",
            scenario_id="s1",
            trust_level="high",
            attack_type="direct",
            firewall_variant="full_mvp",
            seed=99,
            config_path="configs/full_mvp.yaml",
        )
        with pytest.raises(ValueError, match="seed"):
            validate_trust_triplet(low, default, high)

    def test_group_trust_triplets(self) -> None:
        """group_trust_triplets groups entries correctly."""
        low, default, high = self._make_triplet()
        entries = [low, default, high]
        groups = group_trust_triplets(entries)
        assert len(groups) == 1
        key = ("s1", "direct", "full_mvp", 42)
        assert key in groups
        assert len(groups[key]) == 3

    def test_generated_matrix_has_valid_triplets(self) -> None:
        """Generated matrix with 3 trust levels forms valid triplets."""
        entries = generate_matrix(
            ["s1"],
            ["low", "default", "high"],
            ["direct"],
            ["full_mvp"],
            [42],
        )
        groups = group_trust_triplets(entries)
        for key, group in groups.items():
            assert len(group) == 3
            by_trust = {e.trust_level: e for e in group}
            validate_trust_triplet(by_trust["low"], by_trust["default"], by_trust["high"])
