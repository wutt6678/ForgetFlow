"""Tests for experiments.trustparadox_u.aggregate module."""

import pytest

from experiments.trustparadox_u.aggregate import aggregate_summary
from experiments.trustparadox_u.audit_results import InvalidExperimentResults
from experiments.trustparadox_u.runner import EpisodeResult, TurnResult


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
        "forbidden_strings": ["secret"],
        "secret_variant_id": "sv1",
        "attack_type": "direct",
        "config_hash": "a" * 64,
        "seed": 42,
    }
    for k, v in overrides.items():
        setattr(result, k, v)
    return result


class TestAggregateSummary:
    """Tests for aggregate_summary."""

    def test_valid_results_aggregate(self) -> None:
        """Valid results should aggregate without error."""
        results = [_valid_result()]
        variant_results = {"firewall": results}
        summary = aggregate_summary(variant_results)
        assert "firewall" in summary
        assert "pu_rer" in summary["firewall"]

    def test_invalid_audit_blocks_aggregation(self) -> None:
        """Invalid audit results should block aggregation."""
        result = _valid_result()
        # Create a result with invalid metric (numerator > denominator)
        result.turns = [
            TurnResult(
                turn_id=0,
                phase="POST_FORGET_ATTACK",
                sender_id="A",
                recipient_id="B",
                candidate_text="test",
                released_text="test",
                is_attack_attempt=True,
                target_exposed=True,
            ),
            TurnResult(
                turn_id=1,
                phase="POST_FORGET_ATTACK",
                sender_id="A",
                recipient_id="B",
                candidate_text="test2",
                released_text="test2",
                is_attack_attempt=False,  # Not an attack but marked as target exposed
                target_exposed=True,  # This creates invalid metric
            ),
        ]
        # Manually create an invalid state by having inconsistent metrics
        # We need to trigger an audit error
        result.metadata = {}  # Missing required fields

        variant_results = {"firewall": [result]}
        with pytest.raises(InvalidExperimentResults):
            aggregate_summary(variant_results)

    def test_duplicate_run_identity_blocks_aggregation(self) -> None:
        """Duplicate run identities should block aggregation."""
        r1 = _valid_result()
        r2 = _valid_result(episode_id="ep2")
        # Same pairing key and config_hash => duplicate run identity
        variant_results = {"firewall": [r1, r2]}
        with pytest.raises(InvalidExperimentResults):
            aggregate_summary(variant_results)

    def test_allow_errors_permits_aggregation(self) -> None:
        """With allow_errors=True, aggregation proceeds despite audit errors."""
        result = _valid_result()
        result.metadata = {}  # Missing required fields
        variant_results = {"firewall": [result]}
        # Should not raise with allow_errors=True
        summary = aggregate_summary(variant_results, allow_errors=True)
        assert "firewall" in summary

    def test_multiple_variants_aggregate(self) -> None:
        """Multiple variants should aggregate independently."""
        r1 = _valid_result()
        r2 = _valid_result(episode_id="ep2")
        r2.metadata["config_hash"] = "b" * 64  # Different config hash

        variant_results = {
            "firewall": [r1],
            "baseline": [r2],
        }
        summary = aggregate_summary(variant_results)
        assert "firewall" in summary
        assert "baseline" in summary
