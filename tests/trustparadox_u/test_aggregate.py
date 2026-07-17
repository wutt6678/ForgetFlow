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
    result.metadata["run_mode"] = "test"
    result.metadata["semantic_threshold"] = 0.8
    episode_data = {
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
    data = {
        "schema_version": "1.1",
        "episode": episode_data,
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

        with patch.object(
            sys,
            "argv",
            [
                "aggregate",
                "--input",
                str(input_dir),
                "--output",
                str(output_dir),
                "--skip-commit-check",
            ],
        ):
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

        with patch.object(
            sys, "argv", ["aggregate", "--input", str(input_dir), "--output", str(output_dir)]
        ):
            exit_code = main()

        assert exit_code == 2  # INPUT_MISSING

    def test_missing_manifest(self, tmp_path: Path) -> None:
        """Missing manifest without --allow-missing-manifest should fail."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        _write_valid_episodes(input_dir / "episodes.jsonl")

        with patch.object(
            sys, "argv", ["aggregate", "--input", str(input_dir), "--output", str(output_dir)]
        ):
            exit_code = main()

        assert exit_code == 5  # MANIFEST_VALIDATION

    def test_malformed_jsonl(self, tmp_path: Path) -> None:
        """Malformed JSONL should fail."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        with open(input_dir / "episodes.jsonl", "w") as f:
            f.write("not valid json\n")
        _write_valid_manifest(input_dir / "smoke_manifest.json")

        with patch.object(
            sys,
            "argv",
            [
                "aggregate",
                "--input",
                str(input_dir),
                "--output",
                str(output_dir),
                "--skip-commit-check",
            ],
        ):
            exit_code = main()

        assert exit_code == 3  # RESULT_LOAD

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

        with patch.object(
            sys,
            "argv",
            [
                "aggregate",
                "--input",
                str(input_dir),
                "--output",
                str(output_dir),
                "--skip-commit-check",
            ],
        ):
            exit_code = main()

        assert exit_code == 5  # MANIFEST_VALIDATION

    def test_allow_missing_manifest_diagnostic(self, tmp_path: Path) -> None:
        """--allow-missing-manifest should produce diagnostic output."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        _write_valid_episodes(input_dir / "episodes.jsonl")

        with patch.object(
            sys,
            "argv",
            [
                "aggregate",
                "--input",
                str(input_dir),
                "--output",
                str(output_dir),
                "--allow-missing-manifest",
            ],
        ):
            exit_code = main()

        assert exit_code == 0
        # Full output files are now written with diagnostic provenance
        assert (output_dir / "metrics.json").exists()
        assert (output_dir / "summary.json").exists()
        assert (output_dir / "summary.md").exists()
        assert (output_dir / "aggregation_manifest.json").exists()
        # Verify diagnostic provenance
        metrics = json.loads((output_dir / "metrics.json").read_text())
        prov = metrics["artifact_provenance"]
        assert prov["diagnostic"] is True
        assert prov["release_certifying"] is False
        assert prov["validation_mode"] == "missing_manifest_diagnostic"
        # Verify markdown warning
        md = (output_dir / "summary.md").read_text()
        assert "No authoritative smoke manifest" in md

    def test_invalid_audit_blocks_aggregation(self, tmp_path: Path) -> None:
        """Invalid audit results should block aggregation."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        # Write a result with missing required metadata, wrapped in proper envelope
        episode_data = {
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
        data = {
            "schema_version": "1.1",
            "episode": episode_data,
        }
        with open(input_dir / "episodes.jsonl", "w") as f:
            f.write(json.dumps(data) + "\n")
        _write_valid_manifest(input_dir / "smoke_manifest.json")

        with patch.object(
            sys,
            "argv",
            [
                "aggregate",
                "--input",
                str(input_dir),
                "--output",
                str(output_dir),
                "--skip-commit-check",
            ],
        ):
            exit_code = main()

        assert exit_code == 4  # AUDIT


class TestLocateEpisodeResults:
    """Tests for locate_episode_results."""

    def test_canonical_filename_found(self, tmp_path: Path) -> None:
        """Canonical episodes.jsonl is found."""
        from experiments.trustparadox_u.aggregate import locate_episode_results
        from experiments.trustparadox_u.paths import EPISODE_RESULTS_FILENAME

        canonical = tmp_path / EPISODE_RESULTS_FILENAME
        canonical.touch()

        result = locate_episode_results(tmp_path)
        assert result == canonical

    def test_legacy_filename_found_with_warning(self, tmp_path: Path) -> None:
        """Legacy episode_results.jsonl is found with deprecation warning."""
        import warnings

        from experiments.trustparadox_u.aggregate import locate_episode_results
        from experiments.trustparadox_u.paths import LEGACY_EPISODE_RESULTS_FILENAME

        legacy = tmp_path / LEGACY_EPISODE_RESULTS_FILENAME
        legacy.touch()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = locate_episode_results(tmp_path)
            assert result == legacy
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()

    def test_canonical_preferred_over_legacy(self, tmp_path: Path) -> None:
        """Canonical filename is preferred when both exist."""
        from experiments.trustparadox_u.aggregate import locate_episode_results
        from experiments.trustparadox_u.paths import (
            EPISODE_RESULTS_FILENAME,
            LEGACY_EPISODE_RESULTS_FILENAME,
        )

        canonical = tmp_path / EPISODE_RESULTS_FILENAME
        canonical.touch()
        legacy = tmp_path / LEGACY_EPISODE_RESULTS_FILENAME
        legacy.touch()

        result = locate_episode_results(tmp_path)
        assert result == canonical

    def test_missing_results_raises(self, tmp_path: Path) -> None:
        """Missing results file raises FileNotFoundError."""
        from experiments.trustparadox_u.aggregate import locate_episode_results

        with pytest.raises(FileNotFoundError, match="No episode results found"):
            locate_episode_results(tmp_path)


class TestCommitProvenanceValidation:
    """Section 5: Smoke artifact provenance validation."""

    def _make_manifest(self, commit: str = "a" * 40):
        from experiments.trustparadox_u.manifest import SmokeManifest

        return SmokeManifest(
            repository_commit=commit,
            generated_at_utc="2024-01-01T00:00:00Z",
            run_mode="test",
            config_hashes=("a" * 64,),
            provider="fixed",
            model=None,
            dimension=None,
            semantic_threshold=0.8,
            api_base_sanitized=None,
            episode_ids=("ep1",),
            seeds=(42,),
            result_count=1,
            audit_valid=True,
            audit_error_count=0,
            metric_counts={},
        )

    def test_matching_commit_passes(self) -> None:
        """Manifest commit matches expected commit -> release_certifying."""
        from experiments.trustparadox_u.aggregate import validate_commit_provenance

        full_sha = "a" * 40
        manifest = self._make_manifest(full_sha)
        result = validate_commit_provenance(manifest, expected_commit=full_sha)
        assert result["release_certifying"] is True
        assert result["historical"] is False

    def test_stale_commit_raises(self) -> None:
        """Manifest commit differs from expected -> StaleArtifactError."""
        from experiments.trustparadox_u.aggregate import (
            StaleArtifactError,
            validate_commit_provenance,
        )

        manifest = self._make_manifest("a" * 40)
        with pytest.raises(StaleArtifactError, match="Artifact commit mismatch"):
            validate_commit_provenance(manifest, expected_commit="b" * 40)

    def test_historical_override_returns_historical(self) -> None:
        """Historical mode returns historical metadata."""
        from experiments.trustparadox_u.aggregate import validate_commit_provenance

        manifest = self._make_manifest("a" * 40)
        result = validate_commit_provenance(
            manifest,
            expected_commit="b" * 40,
            allow_historical=True,
        )
        assert result["historical"] is True
        assert result["validation_mode"] == "historical_override"

    def test_dirty_artifact_raises(self) -> None:
        """Dirty artifact cannot certify a clean commit."""
        from experiments.trustparadox_u.aggregate import (
            StaleArtifactError,
            validate_commit_provenance,
        )

        full_sha = "a" * 40
        manifest = self._make_manifest(f"{full_sha}-dirty")
        with pytest.raises(StaleArtifactError, match="Artifact commit mismatch"):
            validate_commit_provenance(manifest, expected_commit=full_sha)

    def test_skip_check_returns_skipped(self) -> None:
        """skip_check returns without validation."""
        from experiments.trustparadox_u.aggregate import validate_commit_provenance

        manifest = self._make_manifest("a" * 40)
        result = validate_commit_provenance(manifest, skip_check=True)
        assert result["validation_mode"] == "diagnostic_skipped"

    def test_require_current_commit_checks_head(self) -> None:
        """--require-current-commit compares against current HEAD."""
        from experiments.trustparadox_u.aggregate import (
            StaleArtifactError,
            validate_commit_provenance,
        )

        manifest = self._make_manifest("aaaaaaa")
        with pytest.raises(StaleArtifactError):
            validate_commit_provenance(manifest, require_current_commit=True)

    def test_diagnostic_flag_in_provenance(self) -> None:
        """skip_check sets diagnostic=True."""
        from experiments.trustparadox_u.aggregate import validate_commit_provenance

        manifest = self._make_manifest("a" * 40)
        result = validate_commit_provenance(manifest, skip_check=True)
        assert result["diagnostic"] is True
        assert result["release_certifying"] is False

    def test_strict_mode_no_diagnostic_flag(self) -> None:
        """Matching commit has diagnostic=False."""
        from experiments.trustparadox_u.aggregate import validate_commit_provenance

        full_sha = "a" * 40
        manifest = self._make_manifest(full_sha)
        result = validate_commit_provenance(manifest, expected_commit=full_sha)
        assert result["diagnostic"] is False
        assert result["release_certifying"] is True


class TestSchemaCompatibility:
    """Section 4: Schema compatibility enforcement."""

    def test_legacy_schema_rejected_in_strict_mode(self, tmp_path: Path) -> None:
        """Schema 1.0 should fail in strict mode."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        # Write episode with schema 1.0 in proper envelope format
        episode_data = {
            "run_id": "r1",
            "episode_id": "ep1",
            "scenario_id": "s1",
            "trust_level": "default",
            "seed": 42,
            "turns": [],
            "contamination_states": {},
            "audit_entries": [],
            "metadata": {"forbidden_strings": ["x"], "config_hash": "a" * 64, "seed": 42},
        }
        data = {
            "schema_version": "1.0",
            "episode": episode_data,
        }
        with open(input_dir / "episodes.jsonl", "w") as f:
            f.write(json.dumps(data) + "\n")
        _write_valid_manifest(input_dir / "smoke_manifest.json")

        with patch.object(
            sys,
            "argv",
            [
                "aggregate",
                "--input",
                str(input_dir),
                "--output",
                str(output_dir),
                "--skip-commit-check",
            ],
        ):
            exit_code = main()

        assert exit_code == 7  # SCHEMA

    def test_legacy_schema_allowed_in_historical_mode(self, tmp_path: Path) -> None:
        """Schema 1.0 should pass with --allow-historical-artifacts."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        # Use the valid episode helper but override schema_version to 1.0
        result = _valid_result()
        result.metadata["run_mode"] = "test"
        result.metadata["semantic_threshold"] = 0.8
        episode_data = {
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
        data = {
            "schema_version": "1.0",
            "episode": episode_data,
        }
        with open(input_dir / "episodes.jsonl", "w") as f:
            f.write(json.dumps(data) + "\n")
        _write_valid_manifest(input_dir / "smoke_manifest.json")

        with patch.object(
            sys,
            "argv",
            [
                "aggregate",
                "--input",
                str(input_dir),
                "--output",
                str(output_dir),
                "--skip-commit-check",
                "--allow-historical-artifacts",
            ],
        ):
            exit_code = main()

        assert exit_code == 0


class TestAggregationManifest:
    """Section 9: Aggregation manifest output."""

    def test_aggregation_manifest_written(self, tmp_path: Path) -> None:
        """aggregation_manifest.json should be written with provenance."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        _write_valid_episodes(input_dir / "episodes.jsonl")
        _write_valid_manifest(input_dir / "smoke_manifest.json")

        with patch.object(
            sys,
            "argv",
            [
                "aggregate",
                "--input",
                str(input_dir),
                "--output",
                str(output_dir),
                "--skip-commit-check",
            ],
        ):
            exit_code = main()

        assert exit_code == 0
        agg_manifest = json.loads((output_dir / "aggregation_manifest.json").read_text())
        assert "artifact_provenance" in agg_manifest
        assert "result_schema_versions" in agg_manifest
        assert "outputs" in agg_manifest
        assert agg_manifest["artifact_provenance"]["diagnostic"] is True

    def test_diagnostic_warning_in_markdown(self, tmp_path: Path) -> None:
        """Skipped commit check should produce diagnostic markdown warning."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        _write_valid_episodes(input_dir / "episodes.jsonl")
        _write_valid_manifest(input_dir / "smoke_manifest.json")

        with patch.object(
            sys,
            "argv",
            [
                "aggregate",
                "--input",
                str(input_dir),
                "--output",
                str(output_dir),
                "--skip-commit-check",
            ],
        ):
            exit_code = main()

        assert exit_code == 0
        md = (output_dir / "summary.md").read_text()
        assert "Diagnostic artifact analysis" in md


class TestShortShaResolution:
    """Section 2: Short SHA resolution in provenance comparison."""

    def _make_manifest(self, commit: str = "a" * 40):
        from experiments.trustparadox_u.manifest import SmokeManifest

        return SmokeManifest(
            repository_commit=commit,
            generated_at_utc="2024-01-01T00:00:00Z",
            run_mode="test",
            config_hashes=("a" * 64,),
            provider="fixed",
            model=None,
            dimension=None,
            semantic_threshold=0.8,
            api_base_sanitized=None,
            episode_ids=("ep1",),
            seeds=(42,),
            result_count=1,
            audit_valid=True,
            audit_error_count=0,
            metric_counts={},
        )

    def test_short_expected_sha_resolves_to_full(self) -> None:
        """Short expected SHA resolves to full SHA before comparison."""
        from experiments.trustparadox_u.aggregate import validate_commit_provenance

        full_sha = "a" * 40
        manifest = self._make_manifest(full_sha)
        # Mock resolve_commit_sha to return the full SHA
        from unittest.mock import patch

        with patch(
            "experiments.trustparadox_u.aggregate.resolve_commit_sha",
            return_value=full_sha,
        ):
            result = validate_commit_provenance(
                manifest, expected_commit="aaaaaaa"
            )
        assert result["release_certifying"] is True
        assert result["validation_mode"] == "expected_commit"

    def test_full_expected_sha_matches_directly(self) -> None:
        """Full expected SHA matches without resolution."""
        from experiments.trustparadox_u.aggregate import validate_commit_provenance

        full_sha = "a" * 40
        manifest = self._make_manifest(full_sha)
        result = validate_commit_provenance(manifest, expected_commit=full_sha)
        assert result["release_certifying"] is True

    def test_unresolvable_short_sha_raises(self) -> None:
        """Unresolvable short SHA raises StaleArtifactError."""
        from unittest.mock import patch

        from experiments.trustparadox_u.aggregate import (
            StaleArtifactError,
            validate_commit_provenance,
        )

        manifest = self._make_manifest("a" * 40)
        with patch(
            "experiments.trustparadox_u.aggregate.resolve_commit_sha",
            side_effect=ValueError("Short SHA 'zzzzzzz' could not be resolved"),
        ):
            with pytest.raises(StaleArtifactError, match="Could not resolve"):
                validate_commit_provenance(manifest, expected_commit="zzzzzzz")


class TestFullShaRequirement:
    """Section 3: Full manifest SHAs for release certification."""

    def _make_manifest(self, commit: str):
        from experiments.trustparadox_u.manifest import SmokeManifest

        return SmokeManifest(
            repository_commit=commit,
            generated_at_utc="2024-01-01T00:00:00Z",
            run_mode="test",
            config_hashes=("a" * 64,),
            provider="fixed",
            model=None,
            dimension=None,
            semantic_threshold=0.8,
            api_base_sanitized=None,
            episode_ids=("ep1",),
            seeds=(42,),
            result_count=1,
            audit_valid=True,
            audit_error_count=0,
            metric_counts={},
        )

    def test_full_sha_certifies_release(self) -> None:
        """Full 40-char SHA certifies release."""
        from experiments.trustparadox_u.aggregate import validate_commit_provenance

        full_sha = "a" * 40
        manifest = self._make_manifest(full_sha)
        result = validate_commit_provenance(manifest, expected_commit=full_sha)
        assert result["release_certifying"] is True

    def test_short_sha_rejected_for_certification(self) -> None:
        """Short SHA is rejected for release certification."""
        from experiments.trustparadox_u.aggregate import (
            StaleArtifactError,
            validate_commit_provenance,
        )

        short_sha = "a" * 7
        manifest = self._make_manifest(short_sha)
        with pytest.raises(StaleArtifactError, match="full 40-character"):
            validate_commit_provenance(manifest, expected_commit=short_sha)

    def test_short_sha_allowed_in_historical_mode(self) -> None:
        """Short SHA is allowed when historical mode is active."""
        from experiments.trustparadox_u.aggregate import validate_commit_provenance

        short_sha = "a" * 7
        manifest = self._make_manifest(short_sha)
        result = validate_commit_provenance(
            manifest,
            expected_commit="b" * 40,
            allow_historical=True,
        )
        assert result["historical"] is True


class TestFutureSchemaRejection:
    """Section 4: Future schemas rejected during compatibility preflight."""

    def test_schema_1_2_rejected(self, tmp_path: Path) -> None:
        """Schema 1.2 is rejected with exit code 7."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        episode_data = {
            "run_id": "r1", "episode_id": "ep1", "scenario_id": "s1",
            "trust_level": "default", "seed": 42, "turns": [],
            "contamination_states": {}, "audit_entries": [], "metadata": {},
        }
        data = {"schema_version": "1.2", "episode": episode_data}
        with open(input_dir / "episodes.jsonl", "w") as f:
            f.write(json.dumps(data) + "\n")
        _write_valid_manifest(input_dir / "smoke_manifest.json")

        with patch.object(
            sys, "argv",
            ["aggregate", "--input", str(input_dir), "--output", str(output_dir),
             "--skip-commit-check"],
        ):
            exit_code = main()

        assert exit_code == 7  # SCHEMA

    def test_schema_2_0_rejected(self, tmp_path: Path) -> None:
        """Schema 2.0 is rejected with exit code 7."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        episode_data = {
            "run_id": "r1", "episode_id": "ep1", "scenario_id": "s1",
            "trust_level": "default", "seed": 42, "turns": [],
            "contamination_states": {}, "audit_entries": [], "metadata": {},
        }
        data = {"schema_version": "2.0", "episode": episode_data}
        with open(input_dir / "episodes.jsonl", "w") as f:
            f.write(json.dumps(data) + "\n")
        _write_valid_manifest(input_dir / "smoke_manifest.json")

        with patch.object(
            sys, "argv",
            ["aggregate", "--input", str(input_dir), "--output", str(output_dir),
             "--skip-commit-check"],
        ):
            exit_code = main()

        assert exit_code == 7

    def test_schema_99_0_rejected(self, tmp_path: Path) -> None:
        """Schema 99.0 is rejected with exit code 7."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        episode_data = {
            "run_id": "r1", "episode_id": "ep1", "scenario_id": "s1",
            "trust_level": "default", "seed": 42, "turns": [],
            "contamination_states": {}, "audit_entries": [], "metadata": {},
        }
        data = {"schema_version": "99.0", "episode": episode_data}
        with open(input_dir / "episodes.jsonl", "w") as f:
            f.write(json.dumps(data) + "\n")
        _write_valid_manifest(input_dir / "smoke_manifest.json")

        with patch.object(
            sys, "argv",
            ["aggregate", "--input", str(input_dir), "--output", str(output_dir),
             "--skip-commit-check"],
        ):
            exit_code = main()

        assert exit_code == 7

    def test_mixed_1_1_and_1_2_rejected(self, tmp_path: Path) -> None:
        """Mixed schemas 1.1 and 1.2 are rejected."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        ep1 = {
            "schema_version": "1.1",
            "episode": {
                "run_id": "r1", "episode_id": "ep1", "scenario_id": "s1",
                "trust_level": "default", "seed": 42, "turns": [],
                "contamination_states": {}, "audit_entries": [], "metadata": {},
            },
        }
        ep2 = {
            "schema_version": "1.2",
            "episode": {
                "run_id": "r2", "episode_id": "ep2", "scenario_id": "s1",
                "trust_level": "default", "seed": 43, "turns": [],
                "contamination_states": {}, "audit_entries": [], "metadata": {},
            },
        }
        with open(input_dir / "episodes.jsonl", "w") as f:
            f.write(json.dumps(ep1) + "\n" + json.dumps(ep2) + "\n")
        _write_valid_manifest(input_dir / "smoke_manifest.json")

        with patch.object(
            sys, "argv",
            ["aggregate", "--input", str(input_dir), "--output", str(output_dir),
             "--skip-commit-check"],
        ):
            exit_code = main()

        assert exit_code == 7


class TestProvenanceOnAllOutputs:
    """Section 6: Provenance embedded in every standalone output."""

    def test_unmatched_pairs_has_provenance(self, tmp_path: Path) -> None:
        """unmatched_pairs.json carries artifact_provenance."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        _write_valid_episodes(input_dir / "episodes.jsonl")
        _write_valid_manifest(input_dir / "smoke_manifest.json")

        with patch.object(
            sys, "argv",
            ["aggregate", "--input", str(input_dir), "--output", str(output_dir),
             "--skip-commit-check"],
        ):
            exit_code = main()

        assert exit_code == 0
        unmatched = json.loads((output_dir / "unmatched_pairs.json").read_text())
        assert "artifact_provenance" in unmatched
        prov = unmatched["artifact_provenance"]
        assert "diagnostic" in prov
        assert "release_certifying" in prov

    def test_all_outputs_agree_on_provenance(self, tmp_path: Path) -> None:
        """All output files agree on commit and mode."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        _write_valid_episodes(input_dir / "episodes.jsonl")
        _write_valid_manifest(input_dir / "smoke_manifest.json")

        with patch.object(
            sys, "argv",
            ["aggregate", "--input", str(input_dir), "--output", str(output_dir),
             "--skip-commit-check"],
        ):
            exit_code = main()

        assert exit_code == 0
        # Check provenance consistency across outputs
        metrics = json.loads((output_dir / "metrics.json").read_text())
        counts = json.loads((output_dir / "metric_counts.json").read_text())
        audit = json.loads((output_dir / "audit_report.json").read_text())
        utility = json.loads((output_dir / "utility_pairing.json").read_text())
        unmatched = json.loads((output_dir / "unmatched_pairs.json").read_text())

        for output in [metrics, counts, audit, utility, unmatched]:
            prov = output["artifact_provenance"]
            assert prov["diagnostic"] is True
            assert prov["release_certifying"] is False


class TestHistoricalDiagnosticOverlap:
    """Section 8: Historical and diagnostic mode overlap."""

    def test_legacy_schema_plus_skip_check(self, tmp_path: Path) -> None:
        """Legacy schema + skip-commit-check marks historical and diagnostic."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        result = _valid_result()
        result.metadata["run_mode"] = "test"
        result.metadata["semantic_threshold"] = 0.8
        episode_data = {
            "run_id": result.run_id, "episode_id": result.episode_id,
            "scenario_id": result.scenario_id, "trust_level": result.trust_level,
            "seed": result.seed, "turns": [], "contamination_states": {},
            "audit_entries": [], "task_success": result.task_success,
            "task_label": result.task_label,
            "cleaned_agents_exposed": result.cleaned_agents_exposed,
            "recontaminated_agents": result.recontaminated_agents,
            "metadata": result.metadata,
        }
        data = {"schema_version": "1.0", "episode": episode_data}
        with open(input_dir / "episodes.jsonl", "w") as f:
            f.write(json.dumps(data) + "\n")
        _write_valid_manifest(input_dir / "smoke_manifest.json")

        with patch.object(
            sys, "argv",
            ["aggregate", "--input", str(input_dir), "--output", str(output_dir),
             "--skip-commit-check", "--allow-historical-artifacts"],
        ):
            exit_code = main()

        assert exit_code == 0
        agg = json.loads((output_dir / "aggregation_manifest.json").read_text())
        prov = agg["artifact_provenance"]
        assert prov["historical"] is True
        assert prov["diagnostic"] is True
        assert prov["release_certifying"] is False


class TestAuditErrorCounting:
    """Section 10: Audit error counting excludes warnings."""

    def test_error_count_excludes_warnings(self, tmp_path: Path) -> None:
        """Audit error message reports errors and warnings separately."""
        from experiments.trustparadox_u.audit_results import AuditFinding, AuditReport

        report = AuditReport(
            findings=[
                AuditFinding(level="error", code="E1", message="err1"),
                AuditFinding(level="warning", code="W1", message="warn1"),
                AuditFinding(level="warning", code="W2", message="warn2"),
            ],
            episodes_audited=1,
            episodes_with_errors=1,
        )
        assert len(report.errors()) == 1
        assert len(report.warnings()) == 2
        assert report.has_errors is True
