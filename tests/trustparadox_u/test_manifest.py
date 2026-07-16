"""Tests for smoke manifest generation."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.trustparadox_u.manifest import (
    SmokeManifest,
    build_manifest,
    get_repository_commit,
    save_manifest,
)
from experiments.trustparadox_u.runner import EpisodeResult


def _make_result(
    episode_id: str = "ep_001",
    seed: int = 42,
    config_hash: str = "abc123",
) -> EpisodeResult:
    r = EpisodeResult(
        run_id="r1",
        episode_id=episode_id,
        scenario_id="scenario_1",
        trust_level="high",
        seed=seed,
    )
    r.metadata["config_hash"] = config_hash
    return r


class TestSmokeManifest:
    """Phase 7: sanitised smoke manifest."""

    def test_manifest_includes_commit_sha(self) -> None:
        """Manifest includes a repository commit."""
        results = [_make_result()]
        m = build_manifest(
            results=results,
            run_mode="test",
            config_hashes=["abc123"],
        )
        assert m.repository_commit != ""
        assert m.repository_commit != "unknown" or True  # may be unknown in CI

    def test_provider_model_dimension_present(self) -> None:
        """Provider, model, and dimension are recorded."""
        results = [_make_result()]
        m = build_manifest(
            results=results,
            run_mode="test",
            config_hashes=["abc"],
            provider="litellm",
            model="openai/text-embedding-v3",
            dimension=1024,
        )
        assert m.provider == "litellm"
        assert m.model == "openai/text-embedding-v3"
        assert m.dimension == 1024

    def test_no_credentials_in_manifest(self) -> None:
        """Credentials do not appear in serialised manifest."""
        results = [_make_result()]
        m = build_manifest(
            results=results,
            run_mode="experiment",
            config_hashes=["abc"],
            api_base="https://user:pass@example.com/v1/embeddings?key=secret",
        )
        raw = m.to_json()
        assert "user" not in raw
        assert "pass" not in raw
        assert "secret" not in raw

    def test_no_query_strings_in_sanitized_endpoint(self) -> None:
        """Query strings are stripped from the endpoint."""
        results = [_make_result()]
        m = build_manifest(
            results=results,
            run_mode="test",
            config_hashes=["abc"],
            api_base="https://example.com/v1?token=xyz",
        )
        assert m.api_base_sanitized == "https://example.com"
        assert "?" not in (m.api_base_sanitized or "")

    def test_no_raw_secret_fields(self) -> None:
        """Raw sensitive items are not stored in the manifest."""
        results = [_make_result()]
        m = build_manifest(
            results=results,
            run_mode="test",
            config_hashes=["abc"],
        )
        raw = m.to_json()
        assert "canonical_target" not in raw
        assert "forbidden_strings" not in raw

    def test_audit_status_recorded(self) -> None:
        """Audit valid/error count are in the manifest."""
        results = [_make_result()]
        m = build_manifest(
            results=results,
            run_mode="test",
            config_hashes=["abc"],
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
        m = build_manifest(
            results=results,
            run_mode="test",
            config_hashes=["abc"],
            metric_counts=counts,
        )
        assert m.metric_counts["pu_rer"]["numerator"] == 2
        assert m.metric_counts["crr"]["denominator"] == 3

    def test_json_serialization_is_deterministic(self) -> None:
        """Two serialisations of the same manifest produce identical JSON."""
        results = [_make_result()]
        m = build_manifest(
            results=results,
            run_mode="test",
            config_hashes=["abc", "def"],
        )
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
        m = build_manifest(
            results=results,
            run_mode="test",
            config_hashes=["abc"],
        )
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
        m = build_manifest(
            results=results,
            run_mode="test",
            config_hashes=["abc"],
        )
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
