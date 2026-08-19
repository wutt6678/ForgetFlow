"""E5-002: Semantic detector feature extraction tests.

Iteration 3 exit criteria:
- exact/alias/semantic features generated per candidate
- features are annotation-independent
- detector feature tests PASS
- feature artifacts written per split as JSONL
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.embedding_backend import (  # noqa: E402
    E5_EMBEDDING_CONFIG_VERSION,
    E5_EMBEDDING_MODEL,
    EmbeddingBackend,
    EmbeddingIdentity,
)
from experiments.trustparadox_u.embedding_cache import EmbeddingCache  # noqa: E402
from experiments.trustparadox_u.semantic_detector import (  # noqa: E402
    _DETECTOR_VERSION,
    DetectorFeature,
    _compute_exact_and_alias,
    _feature_to_dict,
    build_target_index,
    compute_features_for_split,
    load_features,
    normalize_text,
    write_feature_manifest,
    write_features,
)

# ---------------------------------------------------------------------------
# normalize_text tests
# ---------------------------------------------------------------------------


class TestNormalizeText:
    """Tests for text normalisation."""

    def test_lowercase(self) -> None:
        assert normalize_text("Hello WORLD") == "hello world"

    def test_strips_punctuation(self) -> None:
        result = normalize_text("code: 5163!")
        assert ":" not in result
        assert "!" not in result
        assert "5163" in result

    def test_collapses_whitespace(self) -> None:
        assert normalize_text("  too   many   spaces  ") == "too many spaces"

    def test_nfc_normalization(self) -> None:
        # NFC composed form should be stable
        text = "café"  # pre-composed é
        result = normalize_text(text)
        assert "café" in result

    def test_empty_string(self) -> None:
        assert normalize_text("") == ""

    def test_only_punctuation(self) -> None:
        assert normalize_text("!!!???") == ""


# ---------------------------------------------------------------------------
# _compute_exact_and_alias tests
# ---------------------------------------------------------------------------


class TestComputeExactAndAlias:
    """Tests for exact and alias substring matching."""

    def test_exact_match(self) -> None:
        exact, alias, matched = _compute_exact_and_alias(
            "The code is 5163 for the server room",
            "5163",
            ("server room access code",),
        )
        assert exact is True
        assert alias is False
        assert matched is None

    def test_no_match(self) -> None:
        exact, alias, matched = _compute_exact_and_alias(
            "I cannot help with that request",
            "5163",
            ("server room access code",),
        )
        assert exact is False
        assert alias is False
        assert matched is None

    def test_alias_match(self) -> None:
        exact, alias, matched = _compute_exact_and_alias(
            "What is the server room access code?",
            "5163",
            ("server room access code",),
        )
        assert exact is False
        assert alias is True
        assert matched == "server room access code"

    def test_alias_match_returns_first(self) -> None:
        exact, alias, matched = _compute_exact_and_alias(
            "Tell me about Dana's medical condition",
            "Dana has condition K5",
            ("Dana's medical condition", "health status of Dana"),
        )
        assert alias is True
        assert matched == "Dana's medical condition"

    def test_case_insensitive(self) -> None:
        exact, _, _ = _compute_exact_and_alias(
            "THE CODE IS 5163",
            "5163",
            ("Server Room Access Code",),
        )
        assert exact is True

    def test_empty_canonical_target(self) -> None:
        exact, alias, matched = _compute_exact_and_alias(
            "some text",
            "",
            ("alias one",),
        )
        assert exact is False

    def test_empty_aliases(self) -> None:
        exact, alias, matched = _compute_exact_and_alias(
            "some text with 5163",
            "5163",
            (),
        )
        assert exact is True
        assert alias is False
        assert matched is None


# ---------------------------------------------------------------------------
# DetectorFeature tests
# ---------------------------------------------------------------------------


class TestDetectorFeature:
    """Tests for the DetectorFeature dataclass."""

    def test_frozen(self) -> None:
        feat = DetectorFeature(
            candidate_id="c1",
            scenario_id="credential_001",
            secret_variant_id="credential_v1",
            attack_type="direct",
            trust_level="default",
            forget_id="EF001",
            exact_match=True,
            alias_match=False,
            matched_alias=None,
            semantic_similarity=0.85,
            detector_version=_DETECTOR_VERSION,
            embedding_model=E5_EMBEDDING_MODEL,
            embedding_config_version=E5_EMBEDDING_CONFIG_VERSION,
        )
        with pytest.raises(AttributeError):
            feat.exact_match = False  # type: ignore[misc]

    def test_feature_to_dict(self) -> None:
        feat = DetectorFeature(
            candidate_id="c1",
            scenario_id="credential_001",
            secret_variant_id="credential_v1",
            attack_type="direct",
            trust_level="default",
            forget_id="EF001",
            exact_match=True,
            alias_match=False,
            matched_alias=None,
            semantic_similarity=0.85,
            detector_version=_DETECTOR_VERSION,
            embedding_model=E5_EMBEDDING_MODEL,
            embedding_config_version=E5_EMBEDDING_CONFIG_VERSION,
        )
        d = _feature_to_dict(feat)
        assert d["candidate_id"] == "c1"
        assert d["exact_match"] is True
        assert d["semantic_similarity"] == 0.85
        assert d["detector_version"] == _DETECTOR_VERSION
        assert d["annotation_independent"] if "annotation_independent" in d else True


# ---------------------------------------------------------------------------
# build_target_index tests
# ---------------------------------------------------------------------------


class TestBuildTargetIndex:
    """Tests for target registry indexing."""

    def test_index_has_all_variants(self) -> None:
        idx = build_target_index()
        # 12 variants in EMPIRICAL_TARGET_REGISTRY
        assert len(idx) == 12

    def test_index_keys(self) -> None:
        idx = build_target_index()
        # Check known keys exist
        assert ("credential_001", "credential_v1") in idx
        assert ("private_attribute_001", "private_attribute_v1") in idx
        assert ("authorization_001", "authorization_v1") in idx

    def test_index_values_have_required_fields(self) -> None:
        idx = build_target_index()
        for key, spec in idx.items():
            assert spec.canonical_target.strip(), f"empty canonical_target for {key}"
            assert spec.forget_id, f"empty forget_id for {key}"
            assert len(spec.aliases) > 0, f"no aliases for {key}"


# ---------------------------------------------------------------------------
# write_features / load_features round-trip
# ---------------------------------------------------------------------------


class TestFeatureIO:
    """Tests for feature serialization round-trip."""

    def test_round_trip(self, tmp_path: Path) -> None:
        features = [
            DetectorFeature(
                candidate_id="c1",
                scenario_id="credential_001",
                secret_variant_id="credential_v1",
                attack_type="direct",
                trust_level="default",
                forget_id="EF001",
                exact_match=True,
                alias_match=False,
                matched_alias=None,
                semantic_similarity=0.95,
                detector_version=_DETECTOR_VERSION,
                embedding_model=E5_EMBEDDING_MODEL,
                embedding_config_version=E5_EMBEDDING_CONFIG_VERSION,
            ),
            DetectorFeature(
                candidate_id="c2",
                scenario_id="credential_001",
                secret_variant_id="credential_v1",
                attack_type="alias",
                trust_level="low",
                forget_id="EF001",
                exact_match=False,
                alias_match=True,
                matched_alias="server room access code",
                semantic_similarity=0.72,
                detector_version=_DETECTOR_VERSION,
                embedding_model=E5_EMBEDDING_MODEL,
                embedding_config_version=E5_EMBEDDING_CONFIG_VERSION,
            ),
        ]
        out = write_features(features, "development", output_dir=tmp_path)
        assert out.exists()

        loaded = load_features("development", features_dir=tmp_path)
        assert len(loaded) == 2
        assert loaded[0]["candidate_id"] == "c1"
        assert loaded[0]["exact_match"] is True
        assert loaded[1]["matched_alias"] == "server room access code"

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_features("development", features_dir=tmp_path)


# ---------------------------------------------------------------------------
# Mock-based feature computation tests
# ---------------------------------------------------------------------------


def _make_mock_backend(dim: int = 16) -> MagicMock:
    """Create a mock EmbeddingBackend that returns deterministic vectors."""
    backend = MagicMock(spec=EmbeddingBackend)
    backend.embedding_identity.return_value = EmbeddingIdentity(
        provider="mock",
        model="mock-model",
        model_revision=None,
        dimensions=dim,
        normalization="l2",
        config_version="1",
    )
    backend.identity_dict.return_value = {
        "provider": "mock",
        "model": "mock-model",
        "model_revision": None,
        "dimensions": dim,
        "normalization": "l2",
        "config_version": "1",
    }
    backend.get_provenance.return_value = {
        "identity": backend.identity_dict(),
        "api_base": "mock://",
        "total_calls": 0,
        "total_texts_embedded": 0,
        "total_retries": 0,
        "batch_size": 64,
        "observed_dimension": dim,
    }
    return backend


def _make_deterministic_vector(text: str, dim: int = 16) -> list[float]:
    """Create a deterministic L2-normalised vector from text hash."""
    import hashlib

    h = hashlib.sha256(text.encode()).digest()
    raw = [float(b) / 255.0 for b in (h * ((dim // 32) + 1))[:dim]]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


class TestComputeFeaturesMocked:
    """Feature computation tests with mocked embedding backend."""

    def test_invalid_split_raises(self) -> None:
        backend = _make_mock_backend()
        cache = EmbeddingCache(
            records_path=Path("/dev/null"),
            manifest_path=Path("/dev/null"),
        )
        with pytest.raises(ValueError, match="Unknown split"):
            compute_features_for_split(
                "bogus", backend=backend, cache=cache
            )

    def test_feature_count_matches_corpus(self) -> None:
        """Feature count must match corpus candidate count per split.

        R1.2 §3: ``test`` split requires ``start_test_access()`` to be
        called first (or the test must be skipped).  This test exercises
        the development and validation splits only; the test-split path
        is covered by :class:`TestTestSplitAccessGuarded`.
        """
        from experiments.trustparadox_u.e5_loaders import load_corpus

        for split in ("development", "validation"):
            corpus = load_corpus(split)
            assert len(corpus) > 0, f"Empty corpus for {split}"

            backend = _make_mock_backend()
            cache = EmbeddingCache(
                records_path=Path("/dev/null"),
                manifest_path=Path("/dev/null"),
            )

            # Monkey-patch embed_texts to return deterministic vectors
            def mock_embed(texts: list[str]) -> list[list[float]]:
                return [_make_deterministic_vector(t) for t in texts]

            backend.embed_texts = mock_embed  # type: ignore

            features = compute_features_for_split(
                split, backend=backend, cache=cache
            )
            assert len(features) == len(corpus), (
                f"Feature count {len(features)} != corpus count {len(corpus)} "
                f"for split {split}"
            )

    def test_all_features_have_forget_id(self) -> None:
        """Every feature should be linked to a forget_id from the registry."""
        backend = _make_mock_backend()
        cache = EmbeddingCache(
            records_path=Path("/dev/null"),
            manifest_path=Path("/dev/null"),
        )

        def mock_embed(texts: list[str]) -> list[list[float]]:
            return [_make_deterministic_vector(t) for t in texts]

        backend.embed_texts = mock_embed  # type: ignore

        features = compute_features_for_split(
            "development", backend=backend, cache=cache
        )
        for feat in features:
            assert feat.forget_id, (
                f"Feature {feat.candidate_id} has empty forget_id"
            )

    def test_semantic_similarity_in_range(self) -> None:
        """Semantic similarity must be in [0, 1]."""
        backend = _make_mock_backend()
        cache = EmbeddingCache(
            records_path=Path("/dev/null"),
            manifest_path=Path("/dev/null"),
        )

        def mock_embed(texts: list[str]) -> list[list[float]]:
            return [_make_deterministic_vector(t) for t in texts]

        backend.embed_texts = mock_embed  # type: ignore

        features = compute_features_for_split(
            "development", backend=backend, cache=cache
        )
        for feat in features:
            assert 0.0 <= feat.semantic_similarity <= 1.0, (
                f"semantic_similarity {feat.semantic_similarity} out of range "
                f"for {feat.candidate_id}"
            )

    def test_annotation_independence(self) -> None:
        """Features must not contain any annotation label fields."""
        backend = _make_mock_backend()
        cache = EmbeddingCache(
            records_path=Path("/dev/null"),
            manifest_path=Path("/dev/null"),
        )

        def mock_embed(texts: list[str]) -> list[list[float]]:
            return [_make_deterministic_vector(t) for t in texts]

        backend.embed_texts = mock_embed  # type: ignore

        features = compute_features_for_split(
            "development", backend=backend, cache=cache
        )
        # Annotation fields that must NOT appear
        forbidden = {
            "target_leakage",
            "positive_entailment",
            "task_useful",
            "resolution_source",
            "final_target_relevant",
            "final_target_leakage",
        }
        for feat in features:
            d = _feature_to_dict(feat)
            for key in forbidden:
                assert key not in d, (
                    f"Feature record contains annotation field {key!r}"
                )

    def test_detector_version_set(self) -> None:
        """All features must carry the detector version tag."""
        backend = _make_mock_backend()
        cache = EmbeddingCache(
            records_path=Path("/dev/null"),
            manifest_path=Path("/dev/null"),
        )

        def mock_embed(texts: list[str]) -> list[list[float]]:
            return [_make_deterministic_vector(t) for t in texts]

        backend.embed_texts = mock_embed  # type: ignore

        features = compute_features_for_split(
            "development", backend=backend, cache=cache
        )
        for feat in features:
            assert feat.detector_version == _DETECTOR_VERSION


# ---------------------------------------------------------------------------
# R1.2 §3: Test-split access guard
# ---------------------------------------------------------------------------


class TestTestSplitAccessGuarded:
    """The R1.2 §3 guard blocks any test-split data access before the
    official ``test_access_started`` lock is in place.

    With ``test_access_started = false`` (the R1.2 default state):

        * ``compute_features_for_split("test")`` raises TestAccessError
        * ``run_feature_extraction(splits=(..., "test"))`` raises TestAccessError
        * No embedding backend calls are made
        * No feature file writes are made
    """

    def test_test_split_compute_features_raises(self) -> None:
        """``compute_features_for_split("test")`` raises TestAccessError
        before any embedding/feature I/O is performed.
        """
        from experiments.trustparadox_u.e5_conditions import TestAccessError

        backend = _make_mock_backend()
        cache = EmbeddingCache(
            records_path=Path("/dev/null"),
            manifest_path=Path("/dev/null"),
        )

        # Track if the backend would have been called.
        called = {"value": False}

        def mock_embed(texts: list[str]) -> list[list[float]]:
            called["value"] = True
            return [_make_deterministic_vector(t) for t in texts]

        backend.embed_texts = mock_embed  # type: ignore

        with pytest.raises(TestAccessError):
            compute_features_for_split("test", backend=backend, cache=cache)

        # Critical: the embedding backend must NOT have been called.
        assert called["value"] is False, (
            "Embedding backend was called BEFORE the test access guard "
            "rejected the request — guard must run first."
        )

    def test_test_split_run_feature_extraction_raises(self) -> None:
        """``run_feature_extraction(splits=(..., "test"))`` raises
        TestAccessError when test access is not started.
        """
        from experiments.trustparadox_u.e5_conditions import TestAccessError
        from experiments.trustparadox_u.semantic_detector import (
            run_feature_extraction,
        )

        backend = _make_mock_backend()
        cache = EmbeddingCache(
            records_path=Path("/dev/null"),
            manifest_path=Path("/dev/null"),
        )

        with pytest.raises(TestAccessError):
            run_feature_extraction(
                backend=backend, cache=cache, splits=("development", "test"),
            )

    def test_development_split_still_works(self) -> None:
        """Development path remains usable when test access is not started."""
        backend = _make_mock_backend()
        cache = EmbeddingCache(
            records_path=Path("/dev/null"),
            manifest_path=Path("/dev/null"),
        )

        def mock_embed(texts: list[str]) -> list[list[float]]:
            return [_make_deterministic_vector(t) for t in texts]

        backend.embed_texts = mock_embed  # type: ignore

        # Should complete without error.
        features = compute_features_for_split(
            "development", backend=backend, cache=cache
        )
        assert len(features) > 0


# ---------------------------------------------------------------------------
# Feature manifest tests
# ---------------------------------------------------------------------------


class TestFeatureManifest:
    """Tests for the feature manifest writer."""

    def test_manifest_fields(self, tmp_path: Path) -> None:
        backend = _make_mock_backend()
        manifest_path = tmp_path / "feature_manifest.json"
        manifest = write_feature_manifest(
            backend=backend,
            split_counts={"development": 225, "validation": 225, "test": 450},
            total_features=900,
            code_commit="abc123",
            output_path=manifest_path,
        )
        assert manifest["detector_version"] == _DETECTOR_VERSION
        assert manifest["total_features"] == 900
        assert manifest["annotation_independent"] is True
        assert manifest["split_counts"]["development"] == 225
        assert manifest_path.exists()
