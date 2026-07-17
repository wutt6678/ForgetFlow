"""Tests for experiments.trustparadox_u.aggregate module."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from experiments.trustparadox_u.aggregate import aggregate_summary, main
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


def _write_valid_episodes(path: Path) -> None:
    """Write a valid episodes.jsonl file for testing."""
    result = _valid_result()
    data = {
        "run_id": result.run_id,
        "episode_id": result.episode_id,
        "scenario_id": result.scenario_id,
        "trust_level": result.trust_level,
        "seed": result.seed,
        "turns": [],
        "contamination_states": {},
        "audit_entries": [],
        "task_success": result.task_success,
        "task_label": result.task_label,
        "cleaned_agents_exposed": result.cleaned_agents_exposed,
        "recontaminated_agents": result.recontaminated_agents,
        "metadata": result.metadata,
    }
    with open(path, "w") as f:
        f.write(json.dumps(data) + "\n")


def _write_valid_manifest(path: Path, commit: str = "a" * 40) -> None:
    """Write a valid smoke_manifest.json for testing."""
    manifest = {
        "repository_commit": commit,
        "generated_at_utc": "2024-01-01T00:00:00+00:00",
        "run_mode": "test",
        "config_hashes": ["a" * 64],
        "provider": None,
        "model": None,
        "dimension": None,
        "semantic_threshold": 0.8,
        "api_base_sanitized": None,
        "episode_ids": ["ep1"],
        "seeds": [42],
        "result_count": 1,
        "audit_valid": True,
        "audit_error_count": 0,
        "metric_counts": {},
    }
    with open(path, "w") as f:
        json.dump(manifest, f)


class TestAggregationCLI:
    """Tests for the aggregation CLI."""

    def test_valid_aggregation(self, tmp_path: Path) -> None:
        """Valid input should produce all output files."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        _write_valid_episodes(input_dir / "episodes.jsonl")
        _write_valid_manifest(input_dir / "smoke_manifest.json")

        with patch.object(sys, "argv", ["aggregate", "--input", str(input_dir), "--output", str(output_dir)]):
            exit_code = main()

        assert exit_code == 0
        assert (output_dir / "metrics.json").exists()
        assert (output_dir / "metric_counts.json").exists()
        assert (output_dir / "summary.json").exists()
        assert (output_dir / "audit_report.json").exists()
        assert (output_dir / "utility_pairing.json").exists()
        assert (output_dir / "unmatched_pairs.json").exists()
        assert (output_dir / "summary.md").exists()

    def test_missing_episodes_file(self, tmp_path: Path) -> None:
        """Missing episodes.jsonl should fail."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        _write_valid_manifest(input_dir / "smoke_manifest.json")

        with patch.object(sys, "argv", ["aggregate", "--input", str(input_dir), "--output", str(output_dir)]):
            exit_code = main()

        assert exit_code == 1

    def test_missing_manifest(self, tmp_path: Path) -> None:
        """Missing manifest without --allow-missing-manifest should fail."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        _write_valid_episodes(input_dir / "episodes.jsonl")

        with patch.object(sys, "argv", ["aggregate", "--input", str(input_dir), "--output", str(output_dir)]):
            exit_code = main()

        assert exit_code == 1

    def test_malformed_jsonl(self, tmp_path: Path) -> None:
        """Malformed JSONL should fail."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        with open(input_dir / "episodes.jsonl", "w") as f:
            f.write("not valid json\n")
        _write_valid_manifest(input_dir / "smoke_manifest.json")

        with patch.object(sys, "argv", ["aggregate", "--input", str(input_dir), "--output", str(output_dir)]):
            exit_code = main()

        assert exit_code == 1

    def test_manifest_mismatch(self, tmp_path: Path) -> None:
        """Manifest with wrong result count should fail."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        _write_valid_episodes(input_dir / "episodes.jsonl")
        _write_valid_manifest(input_dir / "smoke_manifest.json")

        # Modify manifest to have wrong result count
        manifest_path = input_dir / "smoke_manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["result_count"] = 99
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        with patch.object(sys, "argv", ["aggregate", "--input", str(input_dir), "--output", str(output_dir)]):
            exit_code = main()

        assert exit_code == 1

    def test_allow_missing_manifest_diagnostic(self, tmp_path: Path) -> None:
        """--allow-missing-manifest should produce diagnostic output."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        _write_valid_episodes(input_dir / "episodes.jsonl")

        with patch.object(
            sys, "argv",
            ["aggregate", "--input", str(input_dir), "--output", str(output_dir), "--allow-missing-manifest"],
        ):
            exit_code = main()

        assert exit_code == 0
        assert (output_dir / "metrics.json").exists()
        assert (output_dir / "diagnostic_warning.txt").exists()
        # Publication files should not exist
        assert not (output_dir / "summary.json").exists()

    def test_invalid_audit_blocks_aggregation(self, tmp_path: Path) -> None:
        """Invalid audit results should block aggregation."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        # Write a result with missing required metadata
        data = {
            "run_id": "run_0001",
            "episode_id": "ep1",
            "scenario_id": "s1",
            "trust_level": "default",
            "seed": 42,
            "turns": [],
            "contamination_states": {},
            "audit_entries": [],
            "metadata": {},  # Missing required fields
        }
        with open(input_dir / "episodes.jsonl", "w") as f:
            f.write(json.dumps(data) + "\n")
        _write_valid_manifest(input_dir / "smoke_manifest.json")

        with patch.object(sys, "argv", ["aggregate", "--input", str(input_dir), "--output", str(output_dir)]):
            exit_code = main()

        assert exit_code == 1
