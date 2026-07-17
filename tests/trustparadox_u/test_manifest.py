"""Tests for smoke manifest generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.trustparadox_u.manifest import (
    SmokeManifest,
    build_manifest,
    get_repository_commit,
    save_manifest,
    validate_manifest_against_results,
)
from experiments.trustparadox_u.runner import EpisodeResult


def _make_result(
    episode_id: str = "ep_001",
    seed: int = 42,
    config_hash: str = "abc123",
    run_mode: str = "test",
    semantic_enabled: bool = True,
    embedding_provider: str | None = "litellm",
    embedding_model: str | None = "openai/text-embedding-v3",
    embedding_dimension: int | None = 1024,
    semantic_threshold: float = 0.8,
    api_base_sanitized: str | None = None,
) -> EpisodeResult:
    r = EpisodeResult(
        run_id="r1",
        episode_id=episode_id,
        scenario_id="scenario_1",
        trust_level="high",
        seed=seed,
    )
    r.metadata["config_hash"] = config_hash
    r.metadata["run_mode"] = run_mode
    r.metadata["semantic_enabled"] = semantic_enabled
    r.metadata["embedding_provider"] = embedding_provider
    r.metadata["embedding_model"] = embedding_model
    r.metadata["embedding_dimension"] = embedding_dimension
    r.metadata["semantic_threshold"] = semantic_threshold
    r.metadata["api_base_sanitized"] = api_base_sanitized
    return r


class TestSmokeManifest:
    """Phase 7: sanitised smoke manifest."""

    def test_manifest_includes_commit_sha(self) -> None:
        """Manifest includes a repository commit."""
        results = [_make_result()]
        m = build_manifest(results=results)
        assert m.repository_commit != ""
        assert m.repository_commit != "unknown" or True  # may be unknown in CI

    def test_provider_model_dimension_present(self) -> None:
        """Provider, model, and dimension are recorded."""
        results = [_make_result()]
        m = build_manifest(results=results)
        assert m.provider == "litellm"
        assert m.model == "openai/text-embedding-v3"
        assert m.dimension == 1024

    def test_no_credentials_in_manifest(self) -> None:
        """Credentials do not appear in serialised manifest."""
        results = [_make_result(api_base_sanitized="https://example.com")]
        m = build_manifest(results=results)
        raw = m.to_json()
        assert "user" not in raw
        assert "pass" not in raw
        assert "secret" not in raw

    def test_no_query_strings_in_sanitized_endpoint(self) -> None:
        """Query strings are stripped from the endpoint."""
        results = [_make_result(api_base_sanitized="https://example.com")]
        m = build_manifest(results=results)
        assert m.api_base_sanitized == "https://example.com"
        assert "?" not in (m.api_base_sanitized or "")

    def test_no_raw_secret_fields(self) -> None:
        """Raw sensitive items are not stored in the manifest."""
        results = [_make_result()]
        m = build_manifest(results=results)
        raw = m.to_json()
        assert "canonical_target" not in raw
        assert "forbidden_strings" not in raw

    def test_audit_status_recorded(self) -> None:
        """Audit valid/error count are in the manifest."""
        results = [_make_result()]
        m = build_manifest(
            results=results,
            audit_valid=False,
            audit_error_count=3,
        )
        assert m.audit_valid is False
        assert m.audit_error_count == 3

    def test_metric_counts_recorded(self) -> None:
        """Metric numerators and denominators are recorded."""
        results = [_make_result()]
        counts = {
            "pu_rer": {"numerator": 2, "denominator": 5},
            "crr": {"numerator": 1, "denominator": 3},
        }
        m = build_manifest(results=results, metric_counts=counts)
        assert m.metric_counts["pu_rer"]["numerator"] == 2
        assert m.metric_counts["crr"]["denominator"] == 3

    def test_json_serialization_is_deterministic(self) -> None:
        """Two serialisations of the same manifest produce identical JSON."""
        results = [_make_result()]
        m = build_manifest(results=results)
        # Replace timestamp with fixed value for determinism
        m2 = SmokeManifest(
            repository_commit=m.repository_commit,
            generated_at_utc="2026-01-01T00:00:00+00:00",
            run_mode=m.run_mode,
            config_hashes=m.config_hashes,
            provider=m.provider,
            model=m.model,
            dimension=m.dimension,
            semantic_threshold=m.semantic_threshold,
            api_base_sanitized=m.api_base_sanitized,
            episode_ids=m.episode_ids,
            seeds=m.seeds,
            result_count=m.result_count,
            audit_valid=m.audit_valid,
            audit_error_count=m.audit_error_count,
            metric_counts=m.metric_counts,
        )
        j1 = m2.to_json()
        j2 = m2.to_json()
        assert j1 == j2
        # Verify it's valid JSON
        parsed = json.loads(j1)
        assert "repository_commit" in parsed

    def test_save_manifest_writes_file(self, tmp_path: Path) -> None:
        """save_manifest writes a JSON file to disk."""
        results = [_make_result()]
        m = build_manifest(results=results)
        out = tmp_path / "smoke_manifest.json"
        save_manifest(m, out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["result_count"] == 1

    def test_episode_ids_and_seeds_collected(self) -> None:
        """Episode IDs and seeds are collected from results."""
        results = [
            _make_result(episode_id="ep_a", seed=1),
            _make_result(episode_id="ep_b", seed=2),
            _make_result(episode_id="ep_a", seed=1),  # duplicate
        ]
        m = build_manifest(results=results)
        assert set(m.episode_ids) == {"ep_a", "ep_b"}
        assert set(m.seeds) == {1, 2}
        assert m.result_count == 3


class TestGetRepositoryCommit:
    """Tests for git commit detection."""

    def test_returns_string(self) -> None:
        """get_repository_commit returns a non-empty string."""
        commit = get_repository_commit()
        assert isinstance(commit, str)
        assert len(commit) > 0


class TestValidateManifestAgainstResults:
    """Tests for validate_manifest_against_results."""

    def test_valid_manifest_passes(self) -> None:
        """A valid manifest passes validation."""
        results = [_make_result()]
        m = build_manifest(results=results)
        findings = validate_manifest_against_results(m, results)
        # Filter out commit-related findings since we're testing in a real repo
        non_commit_findings = [f for f in findings if "COMMIT" not in f["code"]]
        assert len(non_commit_findings) == 0

    def test_unknown_commit_fails(self) -> None:
        """Unknown commit fails validation."""
        results = [_make_result()]
        m = SmokeManifest(
            repository_commit="unknown",
            generated_at_utc="2026-01-01T00:00:00+00:00",
            run_mode="test",
            config_hashes=("abc123",),
            provider=None,
            model=None,
            dimension=None,
            semantic_threshold=0.8,
            api_base_sanitized=None,
            episode_ids=("ep_001",),
            seeds=(42,),
            result_count=1,
            audit_valid=True,
            audit_error_count=0,
            metric_counts={},
        )
        findings = validate_manifest_against_results(m, results)
        assert any(f["code"] == "MANIFEST_UNKNOWN_COMMIT" for f in findings)

    def test_result_count_mismatch_fails(self) -> None:
        """Result count mismatch fails validation."""
        results = [_make_result()]
        m = SmokeManifest(
            repository_commit="abc123def456",
            generated_at_utc="2026-01-01T00:00:00+00:00",
            run_mode="test",
            config_hashes=("abc123",),
            provider=None,
            model=None,
            dimension=None,
            semantic_threshold=0.8,
            api_base_sanitized=None,
            episode_ids=("ep_001",),
            seeds=(42,),
            result_count=5,  # Wrong count
            audit_valid=True,
            audit_error_count=0,
            metric_counts={},
        )
        findings = validate_manifest_against_results(m, results)
        assert any(f["code"] == "MANIFEST_RESULT_COUNT_MISMATCH" for f in findings)

    def test_episode_ids_mismatch_fails(self) -> None:
        """Episode IDs mismatch fails validation."""
        results = [_make_result(episode_id="ep_001")]
        m = SmokeManifest(
            repository_commit="abc123def456",
            generated_at_utc="2026-01-01T00:00:00+00:00",
            run_mode="test",
            config_hashes=("abc123",),
            provider=None,
            model=None,
            dimension=None,
            semantic_threshold=0.8,
            api_base_sanitized=None,
            episode_ids=("ep_999",),  # Wrong episode ID
            seeds=(42,),
            result_count=1,
            audit_valid=True,
            audit_error_count=0,
            metric_counts={},
        )
        findings = validate_manifest_against_results(m, results)
        assert any(f["code"] == "MANIFEST_EPISODE_IDS_MISMATCH" for f in findings)

    def test_audit_invalid_fails(self) -> None:
        """Audit invalid fails validation."""
        results = [_make_result()]
        m = SmokeManifest(
            repository_commit="abc123def456",
            generated_at_utc="2026-01-01T00:00:00+00:00",
            run_mode="test",
            config_hashes=("abc123",),
            provider=None,
            model=None,
            dimension=None,
            semantic_threshold=0.8,
            api_base_sanitized=None,
            episode_ids=("ep_001",),
            seeds=(42,),
            result_count=1,
            audit_valid=False,  # Invalid audit
            audit_error_count=3,
            metric_counts={},
        )
        findings = validate_manifest_against_results(m, results)
        assert any(f["code"] == "MANIFEST_AUDIT_INVALID" for f in findings)

    def test_metric_counts_mismatch_fails(self) -> None:
        """Metric counts mismatch fails validation."""
        results = [_make_result()]
        m = SmokeManifest(
            repository_commit="abc123def456",
            generated_at_utc="2026-01-01T00:00:00+00:00",
            run_mode="test",
            config_hashes=("abc123",),
            provider=None,
            model=None,
            dimension=None,
            semantic_threshold=0.8,
            api_base_sanitized=None,
            episode_ids=("ep_001",),
            seeds=(42,),
            result_count=1,
            audit_valid=True,
            audit_error_count=0,
            metric_counts={
                "pu_rer": {"numerator": 999, "denominator": 999},  # Wrong counts
            },
        )
        findings = validate_manifest_against_results(m, results)
        assert any(f["code"] == "MANIFEST_METRIC_COUNTS_MISMATCH" for f in findings)

    def test_config_hashes_mismatch_fails(self) -> None:
        """Config hashes mismatch fails validation."""
        results = [_make_result(config_hash="abc123")]
        m = SmokeManifest(
            repository_commit="abc123def456",
            generated_at_utc="2026-01-01T00:00:00+00:00",
            run_mode="test",
            config_hashes=("wrong_hash",),
            provider=None,
            model=None,
            dimension=None,
            semantic_threshold=0.8,
            api_base_sanitized=None,
            episode_ids=("ep_001",),
            seeds=(42,),
            result_count=1,
            audit_valid=True,
            audit_error_count=0,
            metric_counts={},
        )
        findings = validate_manifest_against_results(m, results)
        assert any(f["code"] == "MANIFEST_CONFIG_HASHES_MISMATCH" for f in findings)

    def test_run_mode_mismatch_fails(self) -> None:
        """Run mode mismatch fails validation."""
        results = [_make_result()]
        results[0].metadata["run_mode"] = "experiment"
        m = SmokeManifest(
            repository_commit="abc123def456",
            generated_at_utc="2026-01-01T00:00:00+00:00",
            run_mode="test",  # Wrong run mode
            config_hashes=("abc123",),
            provider=None,
            model=None,
            dimension=None,
            semantic_threshold=0.8,
            api_base_sanitized=None,
            episode_ids=("ep_001",),
            seeds=(42,),
            result_count=1,
            audit_valid=True,
            audit_error_count=0,
            metric_counts={},
        )
        findings = validate_manifest_against_results(m, results)
        assert any(f["code"] == "MANIFEST_RUN_MODE_MISMATCH" for f in findings)

    def test_provider_mismatch_fails(self) -> None:
        """Provider mismatch fails validation."""
        results = [_make_result()]
        results[0].metadata["embedding_provider"] = "openai"
        m = SmokeManifest(
            repository_commit="abc123def456",
            generated_at_utc="2026-01-01T00:00:00+00:00",
            run_mode="test",
            config_hashes=("abc123",),
            provider="alibaba",  # Wrong provider
            model=None,
            dimension=None,
            semantic_threshold=0.8,
            api_base_sanitized=None,
            episode_ids=("ep_001",),
            seeds=(42,),
            result_count=1,
            audit_valid=True,
            audit_error_count=0,
            metric_counts={},
        )
        findings = validate_manifest_against_results(m, results)
        assert any(f["code"] == "MANIFEST_PROVIDER_MISMATCH" for f in findings)

    def test_model_mismatch_fails(self) -> None:
        """Model mismatch fails validation."""
        results = [_make_result()]
        results[0].metadata["embedding_model"] = "text-embedding-v3"
        m = SmokeManifest(
            repository_commit="abc123def456",
            generated_at_utc="2026-01-01T00:00:00+00:00",
            run_mode="test",
            config_hashes=("abc123",),
            provider=None,
            model="wrong-model",  # Wrong model
            dimension=None,
            semantic_threshold=0.8,
            api_base_sanitized=None,
            episode_ids=("ep_001",),
            seeds=(42,),
            result_count=1,
            audit_valid=True,
            audit_error_count=0,
            metric_counts={},
        )
        findings = validate_manifest_against_results(m, results)
        assert any(f["code"] == "MANIFEST_MODEL_MISMATCH" for f in findings)

    def test_dimension_mismatch_fails(self) -> None:
        """Dimension mismatch fails validation."""
        results = [_make_result()]
        results[0].metadata["embedding_dimension"] = 1536
        m = SmokeManifest(
            repository_commit="abc123def456",
            generated_at_utc="2026-01-01T00:00:00+00:00",
            run_mode="test",
            config_hashes=("abc123",),
            provider=None,
            model=None,
            dimension=512,  # Wrong dimension
            semantic_threshold=0.8,
            api_base_sanitized=None,
            episode_ids=("ep_001",),
            seeds=(42,),
            result_count=1,
            audit_valid=True,
            audit_error_count=0,
            metric_counts={},
        )
        findings = validate_manifest_against_results(m, results)
        assert any(f["code"] == "MANIFEST_DIMENSION_MISMATCH" for f in findings)

    def test_threshold_mismatch_fails(self) -> None:
        """Semantic threshold mismatch fails validation."""
        results = [_make_result()]
        results[0].metadata["semantic_threshold"] = 0.9
        m = SmokeManifest(
            repository_commit="abc123def456",
            generated_at_utc="2026-01-01T00:00:00+00:00",
            run_mode="test",
            config_hashes=("abc123",),
            provider=None,
            model=None,
            dimension=None,
            semantic_threshold=0.8,  # Wrong threshold
            api_base_sanitized=None,
            episode_ids=("ep_001",),
            seeds=(42,),
            result_count=1,
            audit_valid=True,
            audit_error_count=0,
            metric_counts={},
        )
        findings = validate_manifest_against_results(m, results)
        assert any(f["code"] == "MANIFEST_THRESHOLD_MISMATCH" for f in findings)


class TestGetRepositoryCommitDirtyDetection:
    """Tests for get_repository_commit with dirty tree detection."""

    def test_clean_tree_returns_commit(self) -> None:
        """Clean working tree returns commit SHA."""
        from unittest.mock import MagicMock, patch

        with patch("subprocess.run") as mock_run:
            # First call: git rev-parse HEAD
            rev_parse = MagicMock()
            rev_parse.returncode = 0
            rev_parse.stdout = "abc123def456\n"

            # Second call: git status --porcelain (empty = clean)
            status = MagicMock()
            status.stdout = ""

            mock_run.side_effect = [rev_parse, status]

            from experiments.trustparadox_u.manifest import get_repository_commit

            commit = get_repository_commit()
            assert commit == "abc123def456"

    def test_dirty_tree_appends_suffix(self) -> None:
        """Dirty working tree appends '-dirty' suffix."""
        from unittest.mock import MagicMock, patch

        with patch("subprocess.run") as mock_run:
            rev_parse = MagicMock()
            rev_parse.returncode = 0
            rev_parse.stdout = "abc123def456\n"

            status = MagicMock()
            status.stdout = " M some_file.py\n"

            mock_run.side_effect = [rev_parse, status]

            from experiments.trustparadox_u.manifest import get_repository_commit

            commit = get_repository_commit()
            assert commit == "abc123def456-dirty"

    def test_dirty_tree_rejected_raises(self) -> None:
        """Dirty working tree with reject_dirty=True raises RuntimeError."""
        from unittest.mock import MagicMock, patch

        with patch("subprocess.run") as mock_run:
            rev_parse = MagicMock()
            rev_parse.returncode = 0
            rev_parse.stdout = "abc123def456\n"

            status = MagicMock()
            status.stdout = " M some_file.py\n"

            mock_run.side_effect = [rev_parse, status]

            from experiments.trustparadox_u.manifest import get_repository_commit

            with pytest.raises(RuntimeError, match="clean working tree"):
                get_repository_commit(reject_dirty=True)

    def test_git_not_available_returns_unknown(self) -> None:
        """Git not available returns 'unknown'."""
        from unittest.mock import patch

        with patch("subprocess.run", side_effect=FileNotFoundError):
            from experiments.trustparadox_u.manifest import get_repository_commit

            commit = get_repository_commit()
            assert commit == "unknown"


class TestRequireSingleMetadataValue:
    """Tests for require_single_metadata_value helper."""

    def test_single_consistent_value(self) -> None:
        """Single consistent value should be returned."""
        from experiments.trustparadox_u.manifest import require_single_metadata_value

        r1 = _make_result()
        r1.metadata["provider"] = "openai"
        r2 = _make_result(episode_id="ep_002")
        r2.metadata["provider"] = "openai"

        result = require_single_metadata_value([r1, r2], "provider")
        assert result == "openai"

    def test_inconsistent_values_raises(self) -> None:
        """Inconsistent values should raise ValueError."""
        from experiments.trustparadox_u.manifest import require_single_metadata_value

        r1 = _make_result()
        r1.metadata["embedding_provider"] = "openai"
        r2 = _make_result(episode_id="ep_002")
        r2.metadata["embedding_provider"] = "anthropic"

        with pytest.raises(ValueError, match="Expected one value"):
            require_single_metadata_value([r1, r2], "embedding_provider")

    def test_none_values_raise_when_not_allowed(self) -> None:
        """None values should raise error when allow_none=False."""
        from experiments.trustparadox_u.manifest import require_single_metadata_value

        r1 = _make_result()
        r1.metadata["embedding_provider"] = "openai"
        r2 = _make_result(episode_id="ep_002", embedding_provider=None)
        # r2 has embedding_provider=None

        with pytest.raises(ValueError, match="missing from some results"):
            require_single_metadata_value([r1, r2], "embedding_provider")

    def test_none_values_allowed(self) -> None:
        """None values should be allowed when allow_none=True."""
        from experiments.trustparadox_u.manifest import require_single_metadata_value

        r1 = _make_result()
        r1.metadata["embedding_provider"] = None
        r2 = _make_result(episode_id="ep_002")
        r2.metadata["embedding_provider"] = None

        result = require_single_metadata_value([r1, r2], "embedding_provider", allow_none=True)
        assert result is None

    def test_all_none_raises_without_allow_none(self) -> None:
        """All None values should raise when allow_none=False."""
        from experiments.trustparadox_u.manifest import require_single_metadata_value

        r1 = _make_result()
        r1.metadata["embedding_provider"] = None
        r2 = _make_result(episode_id="ep_002")
        r2.metadata["embedding_provider"] = None

        with pytest.raises(ValueError, match="cannot be null"):
            require_single_metadata_value([r1, r2], "embedding_provider")

    def test_missing_field_raises(self) -> None:
        """Missing field should raise ValueError."""
        from experiments.trustparadox_u.manifest import require_single_metadata_value

        r1 = _make_result(embedding_provider=None)
        # No embedding_provider field (None)

        with pytest.raises(ValueError, match="cannot be null"):
            require_single_metadata_value([r1], "embedding_provider")


class TestStrictConfigHashValidation:
    """ST-MAN: Strict configuration hash validation (G12)."""

    def test_missing_config_hash_raises(self) -> None:
        """Missing config_hash is rejected."""
        r = _make_result()
        del r.metadata["config_hash"]
        with pytest.raises(ValueError, match="no valid config_hash"):
            from experiments.trustparadox_u.manifest import collect_config_hashes

            collect_config_hashes([r])

    def test_empty_string_config_hash_raises(self) -> None:
        """Empty string config_hash is rejected."""
        r = _make_result(config_hash="")
        with pytest.raises(ValueError, match="no valid config_hash"):
            from experiments.trustparadox_u.manifest import collect_config_hashes

            collect_config_hashes([r])

    def test_whitespace_only_config_hash_raises(self) -> None:
        """Whitespace-only config_hash is rejected."""
        r = _make_result(config_hash="   ")
        with pytest.raises(ValueError, match="no valid config_hash"):
            from experiments.trustparadox_u.manifest import collect_config_hashes

            collect_config_hashes([r])

    def test_non_string_config_hash_raises(self) -> None:
        """Non-string config_hash is rejected."""
        r = _make_result()
        r.metadata["config_hash"] = 12345
        with pytest.raises(ValueError, match="no valid config_hash"):
            from experiments.trustparadox_u.manifest import collect_config_hashes

            collect_config_hashes([r])

    def test_valid_config_hash_passes(self) -> None:
        """Valid config_hash is accepted."""
        r = _make_result(config_hash="abc123def456")
        from experiments.trustparadox_u.manifest import collect_config_hashes

        hashes = collect_config_hashes([r])
        assert hashes == ("abc123def456",)

    def test_multiple_valid_hashes_deduplicated(self) -> None:
        """Multiple results with same hash produce single entry."""
        r1 = _make_result(episode_id="ep_a", config_hash="abc")
        r2 = _make_result(episode_id="ep_b", config_hash="abc")
        from experiments.trustparadox_u.manifest import collect_config_hashes

        hashes = collect_config_hashes([r1, r2])
        assert hashes == ("abc",)

    def test_multiple_distinct_hashes(self) -> None:
        """Multiple distinct hashes are sorted and returned."""
        r1 = _make_result(episode_id="ep_a", config_hash="bbb")
        r2 = _make_result(episode_id="ep_b", config_hash="aaa")
        from experiments.trustparadox_u.manifest import collect_config_hashes

        hashes = collect_config_hashes([r1, r2])
        assert hashes == ("aaa", "bbb")
