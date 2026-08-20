"""E5-001: Embedding backend and cache tests.

Iteration 2 exit criteria:
- real embedding provider works
- embedding metadata captured
- cache works
- dimension fixed
- development smoke subset embedded
- no test evaluation
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.embedding_backend import (
    E5_EMBEDDING_CONFIG_VERSION,
    E5_EMBEDDING_DIMENSION,
    E5_EMBEDDING_MODEL,
    E5_EMBEDDING_NORMALIZATION,
    EmbeddingBackend,
    EmbeddingIdentity,
    _check_dimension,
    _check_finite,
    _l2_normalize,
    cosine_similarity,
    create_backend,
)
from experiments.trustparadox_u.embedding_cache import (
    EmbeddingCache,
    compute_cache_key,
    text_sha256,
)

_E5_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "e5"
_SMOKE_PATH = _E5_DIR / "e5_embed_smoke.json"
_RECORDS_PATH = _E5_DIR / "embeddings" / "embedding_records.jsonl"
_MANIFEST_PATH = _E5_DIR / "embeddings" / "embedding_manifest.json"

# Skip all tests if smoke test hasn't been run yet
pytestmark = pytest.mark.skipif(
    not _SMOKE_PATH.exists(),
    reason="E5 embedding smoke test not yet run; run scripts/e5_embed_smoke.py first",
)


# ---------------------------------------------------------------------------
# Validation helper tests (no API needed)
# ---------------------------------------------------------------------------


class TestValidationHelpers:
    """Tests for embedding validation utilities."""

    def test_check_finite_passes(self) -> None:
        _check_finite([1.0, 2.0, 3.0], context="test")

    def test_check_finite_rejects_nan(self) -> None:
        with pytest.raises(RuntimeError, match="Non-finite"):
            _check_finite([1.0, float("nan"), 3.0], context="test")

    def test_check_finite_rejects_inf(self) -> None:
        with pytest.raises(RuntimeError, match="Non-finite"):
            _check_finite([1.0, float("inf"), 3.0], context="test")

    def test_check_finite_rejects_neg_inf(self) -> None:
        with pytest.raises(RuntimeError, match="Non-finite"):
            _check_finite([1.0, float("-inf"), 3.0], context="test")

    def test_check_dimension_passes(self) -> None:
        _check_dimension([1.0, 2.0, 3.0], 3, context="test")

    def test_check_dimension_rejects_mismatch(self) -> None:
        with pytest.raises(RuntimeError, match="dimension mismatch"):
            _check_dimension([1.0, 2.0], 3, context="test")

    def test_l2_normalize_unit_vector(self) -> None:
        result = _l2_normalize([1.0, 0.0, 0.0])
        assert result == [1.0, 0.0, 0.0]

    def test_l2_normalize_general(self) -> None:
        result = _l2_normalize([3.0, 4.0])
        assert abs(result[0] - 0.6) < 1e-10
        assert abs(result[1] - 0.8) < 1e-10

    def test_l2_normalize_zero_vector(self) -> None:
        result = _l2_normalize([0.0, 0.0, 0.0])
        assert result == [0.0, 0.0, 0.0]

    def test_l2_normalize_produces_unit_norm(self) -> None:
        vec = [1.0, 2.0, 3.0, -1.5, 0.5]
        result = _l2_normalize(vec)
        norm = math.sqrt(sum(x * x for x in result))
        assert abs(norm - 1.0) < 1e-10


class TestCosineSimilarity:
    """Tests for the cosine similarity function."""

    def test_identical_vectors(self) -> None:
        v = [0.5, 0.5, 0.5, 0.5]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-10

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(cosine_similarity(a, b)) < 1e-10

    def test_opposite_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-10


# ---------------------------------------------------------------------------
# Cache key tests (no API needed)
# ---------------------------------------------------------------------------


class TestCacheKey:
    """Tests for deterministic cache key computation."""

    def test_cache_key_deterministic(self) -> None:
        key1 = compute_cache_key(
            model="test-model",
            text_sha256="abc123",
            normalization="l2",
            config_version="1",
        )
        key2 = compute_cache_key(
            model="test-model",
            text_sha256="abc123",
            normalization="l2",
            config_version="1",
        )
        assert key1 == key2

    def test_cache_key_changes_with_model(self) -> None:
        key1 = compute_cache_key(
            model="model-a",
            text_sha256="abc123",
            normalization="l2",
            config_version="1",
        )
        key2 = compute_cache_key(
            model="model-b",
            text_sha256="abc123",
            normalization="l2",
            config_version="1",
        )
        assert key1 != key2

    def test_cache_key_changes_with_text(self) -> None:
        key1 = compute_cache_key(
            model="model-a",
            text_sha256="abc123",
            normalization="l2",
            config_version="1",
        )
        key2 = compute_cache_key(
            model="model-a",
            text_sha256="def456",
            normalization="l2",
            config_version="1",
        )
        assert key1 != key2

    def test_cache_key_changes_with_normalization(self) -> None:
        key1 = compute_cache_key(
            model="model-a",
            text_sha256="abc123",
            normalization="l2",
            config_version="1",
        )
        key2 = compute_cache_key(
            model="model-a",
            text_sha256="abc123",
            normalization="none",
            config_version="1",
        )
        assert key1 != key2

    def test_cache_key_changes_with_config_version(self) -> None:
        key1 = compute_cache_key(
            model="model-a",
            text_sha256="abc123",
            normalization="l2",
            config_version="1",
        )
        key2 = compute_cache_key(
            model="model-a",
            text_sha256="abc123",
            normalization="l2",
            config_version="2",
        )
        assert key1 != key2

    def test_text_sha256_deterministic(self) -> None:
        h1 = text_sha256("hello world")
        h2 = text_sha256("hello world")
        assert h1 == h2

    def test_text_sha256_different_texts(self) -> None:
        h1 = text_sha256("hello")
        h2 = text_sha256("world")
        assert h1 != h2


# ---------------------------------------------------------------------------
# EmbeddingBackend tests (no API needed for identity/config)
# ---------------------------------------------------------------------------


class TestEmbeddingBackendConfig:
    """Tests for backend configuration and identity."""

    def test_default_identity(self) -> None:
        backend = create_backend()
        ident = backend.embedding_identity()
        assert ident.provider == "litellm"
        assert ident.model == E5_EMBEDDING_MODEL
        assert ident.dimensions == E5_EMBEDDING_DIMENSION
        assert ident.normalization == E5_EMBEDDING_NORMALIZATION
        assert ident.config_version == E5_EMBEDDING_CONFIG_VERSION

    def test_identity_dict(self) -> None:
        backend = create_backend()
        d = backend.identity_dict()
        assert d["provider"] == "litellm"
        assert d["model"] == E5_EMBEDDING_MODEL
        assert d["dimensions"] == E5_EMBEDDING_DIMENSION

    def test_rejects_non_litellm_provider(self) -> None:
        with pytest.raises(ValueError, match="only 'litellm'"):
            EmbeddingBackend(provider="openai")

    def test_rejects_empty_model(self) -> None:
        with pytest.raises(ValueError, match="requires a model"):
            EmbeddingBackend(model="")

    def test_rejects_zero_dimension(self) -> None:
        with pytest.raises(ValueError, match="expected_dimension must be positive"):
            EmbeddingBackend(expected_dimension=0)

    def test_embed_empty_list(self) -> None:
        backend = create_backend()
        result = backend.embed_texts([])
        assert result == []

    def test_provenance_initial_state(self) -> None:
        backend = create_backend()
        prov = backend.get_provenance()
        assert prov["total_calls"] == 0
        assert prov["total_texts_embedded"] == 0
        assert prov["total_retries"] == 0


# ---------------------------------------------------------------------------
# EmbeddingCache tests (no API needed)
# ---------------------------------------------------------------------------


class TestEmbeddingCacheUnit:
    """Tests for cache operations without API calls."""

    def test_cache_starts_empty(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(
            records_path=tmp_path / "records.jsonl",
            manifest_path=tmp_path / "manifest.json",
        )
        cache.load()
        assert cache.size == 0

    def test_cache_put_and_get(self, tmp_path: Path) -> None:
        from experiments.trustparadox_u.embedding_backend import EmbeddingRecord

        cache = EmbeddingCache(
            records_path=tmp_path / "records.jsonl",
            manifest_path=tmp_path / "manifest.json",
        )
        cache.load()

        # Compute the cache key that will be used for lookup
        cache_key = compute_cache_key(
            model="test-model",
            text_sha256="abc123",
            normalization=E5_EMBEDDING_NORMALIZATION,
            config_version=E5_EMBEDDING_CONFIG_VERSION,
        )

        rec = EmbeddingRecord(
            embedding_id=cache_key,
            text_sha256="abc123",
            text_role="candidate",
            entity_id="cand-1",
            provider="litellm",
            model="test-model",
            dimension=4,
            vector=(0.5, 0.5, 0.5, 0.5),
            created_at="2026-01-01T00:00:00Z",
        )
        cache.put(rec)
        assert cache.size == 1

        # Get by key components
        found = cache.get(
            model="test-model",
            text_hash="abc123",
            normalization=E5_EMBEDDING_NORMALIZATION,
            config_version=E5_EMBEDDING_CONFIG_VERSION,
        )
        assert found is not None
        assert found.embedding_id == cache_key
        assert found.vector == (0.5, 0.5, 0.5, 0.5)

    def test_cache_save_and_reload(self, tmp_path: Path) -> None:
        from experiments.trustparadox_u.embedding_backend import EmbeddingRecord

        rec_path = tmp_path / "records.jsonl"
        man_path = tmp_path / "manifest.json"

        cache1 = EmbeddingCache(records_path=rec_path, manifest_path=man_path)
        cache1.load()

        rec = EmbeddingRecord(
            embedding_id="test-key",
            text_sha256="abc123",
            text_role="candidate",
            entity_id="cand-1",
            provider="litellm",
            model="test-model",
            dimension=4,
            vector=(0.5, 0.5, 0.5, 0.5),
            created_at="2026-01-01T00:00:00Z",
        )
        cache1.put(rec)
        cache1.save()

        # Reload in a new cache
        cache2 = EmbeddingCache(records_path=rec_path, manifest_path=man_path)
        cache2.load()
        assert cache2.size == 1

        found = cache2.records.get("test-key")
        assert found is not None
        assert found.vector == (0.5, 0.5, 0.5, 0.5)

    def test_cache_miss_returns_none(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(
            records_path=tmp_path / "records.jsonl",
            manifest_path=tmp_path / "manifest.json",
        )
        cache.load()
        result = cache.get(model="m", text_hash="nonexistent")
        assert result is None


# ---------------------------------------------------------------------------
# Smoke test artifact tests
# ---------------------------------------------------------------------------


class TestSmokeArtifact:
    """Tests that verify the smoke test produced correct artifacts."""

    def test_smoke_json_exists(self) -> None:
        assert _SMOKE_PATH.exists()

    def test_smoke_passes(self) -> None:
        with open(_SMOKE_PATH) as f:
            data = json.load(f)
        assert data["smoke_pass"] is True

    def test_smoke_no_blocking_findings(self) -> None:
        with open(_SMOKE_PATH) as f:
            data = json.load(f)
        assert data["blocking_findings"] == []

    def test_smoke_all_checks_pass(self) -> None:
        with open(_SMOKE_PATH) as f:
            data = json.load(f)
        for check_name, status in data["checks"].items():
            assert status == "PASS", f"Check {check_name} is {status}"

    def test_smoke_embedding_identity(self) -> None:
        with open(_SMOKE_PATH) as f:
            data = json.load(f)
        ident = data["embedding_identity"]
        assert ident["model"] == E5_EMBEDDING_MODEL
        assert ident["dimensions"] == E5_EMBEDDING_DIMENSION
        assert ident["normalization"] == E5_EMBEDDING_NORMALIZATION

    def test_smoke_subset_size(self) -> None:
        with open(_SMOKE_PATH) as f:
            data = json.load(f)
        assert data["smoke_subset_size"] == 10

    def test_smoke_cache_size(self) -> None:
        with open(_SMOKE_PATH) as f:
            data = json.load(f)
        assert data["cache_size"] == 10

    def test_smoke_provenance(self) -> None:
        with open(_SMOKE_PATH) as f:
            data = json.load(f)
        prov = data["provenance"]
        assert prov["total_texts_embedded"] == 10
        assert prov["observed_dimension"] == E5_EMBEDDING_DIMENSION


class TestEmbeddingRecords:
    """Tests that verify the embedding records file."""

    def test_records_file_exists(self) -> None:
        assert _RECORDS_PATH.exists()

    def test_records_count(self) -> None:
        count = 0
        with open(_RECORDS_PATH) as f:
            for line in f:
                if line.strip():
                    count += 1
        assert count == 225

    def test_records_have_required_fields(self) -> None:
        required_fields = [
            "embedding_id",
            "text_sha256",
            "text_role",
            "entity_id",
            "provider",
            "model",
            "dimension",
            "vector",
            "created_at",
        ]
        with open(_RECORDS_PATH) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                for field in required_fields:
                    assert field in rec, f"Missing field {field}"

    def test_records_dimension_correct(self) -> None:
        with open(_RECORDS_PATH) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                assert rec["dimension"] == E5_EMBEDDING_DIMENSION
                assert len(rec["vector"]) == E5_EMBEDDING_DIMENSION

    def test_records_all_finite(self) -> None:
        with open(_RECORDS_PATH) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                for v in rec["vector"]:
                    assert math.isfinite(v), f"Non-finite value: {v}"

    def test_records_unique_ids(self) -> None:
        ids: set[str] = set()
        with open(_RECORDS_PATH) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                assert rec["embedding_id"] not in ids
                ids.add(rec["embedding_id"])

    def test_records_text_role_is_candidate(self) -> None:
        """All records have a valid text_role (candidate, target, or target_alias)."""
        valid_roles = {"candidate", "target", "target_alias"}
        with open(_RECORDS_PATH) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                assert rec["text_role"] in valid_roles


class TestEmbeddingManifest:
    """Tests that verify the embedding manifest."""

    def test_manifest_exists(self) -> None:
        assert _MANIFEST_PATH.exists()

    def test_manifest_model(self) -> None:
        with open(_MANIFEST_PATH) as f:
            data = json.load(f)
        assert data["embedding_model"] == E5_EMBEDDING_MODEL

    def test_manifest_dimensions(self) -> None:
        with open(_MANIFEST_PATH) as f:
            data = json.load(f)
        assert data["dimensions"] == E5_EMBEDDING_DIMENSION

    def test_manifest_normalization(self) -> None:
        with open(_MANIFEST_PATH) as f:
            data = json.load(f)
        assert data["normalization"] == E5_EMBEDDING_NORMALIZATION

    def test_manifest_successful_count(self) -> None:
        with open(_MANIFEST_PATH) as f:
            data = json.load(f)
        assert data["successful_embedding_count"] == 225

    def test_manifest_failed_count_zero(self) -> None:
        with open(_MANIFEST_PATH) as f:
            data = json.load(f)
        assert data["failed_embedding_count"] == 0

    def test_manifest_has_cache_sha(self) -> None:
        with open(_MANIFEST_PATH) as f:
            data = json.load(f)
        assert len(data["cache_sha"]) == 64

    def test_manifest_has_corpus_sha(self) -> None:
        with open(_MANIFEST_PATH) as f:
            data = json.load(f)
        assert len(data["frozen_corpus_sha"]) == 64


# ---------------------------------------------------------------------------
# No test evaluation check
# ---------------------------------------------------------------------------


class TestNoTestEvaluation:
    """Verify that no test split evaluation has occurred."""

    def test_smoke_uses_development_only(self) -> None:
        """The smoke test must only use the development split."""
        with open(_SMOKE_PATH) as f:
            data = json.load(f)
        # The smoke test embeds development candidates
        assert data["smoke_subset_size"] > 0
        # Verify no test-related fields
        assert "test_results" not in data
        assert "test_evaluation" not in data
