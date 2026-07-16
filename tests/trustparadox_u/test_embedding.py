"""Tests for embedding providers."""

from experiments.trustparadox_u.embedding import (
    FixedEmbeddingProvider,
    RealEmbeddingProvider,
    cosine_similarity,
)


class TestFixedEmbeddingProvider:
    """Tests for FixedEmbeddingProvider."""

    def test_known_text_returns_vector(self) -> None:
        vectors = {"hello world": [1.0, 0.0, 0.0]}
        provider = FixedEmbeddingProvider(vectors)
        result = provider.embed(["hello world"])
        assert result == [[1.0, 0.0, 0.0]]

    def test_unknown_text_returns_default(self) -> None:
        vectors = {"hello": [1.0, 0.0]}
        provider = FixedEmbeddingProvider(vectors, default_vector=[0.0, 1.0])
        result = provider.embed(["unknown text"])
        assert result == [[0.0, 1.0]]

    def test_case_insensitive_lookup(self) -> None:
        vectors = {"Hello World": [1.0, 0.0]}
        provider = FixedEmbeddingProvider(vectors)
        result = provider.embed(["hello world"])
        assert result == [[1.0, 0.0]]

    def test_dimension_property(self) -> None:
        vectors = {"test": [1.0, 0.0, 0.0]}
        provider = FixedEmbeddingProvider(vectors)
        assert provider.dimension == 3

    def test_multiple_texts(self) -> None:
        vectors = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
        provider = FixedEmbeddingProvider(vectors)
        result = provider.embed(["a", "b", "c"])
        assert len(result) == 3
        assert result[0] == [1.0, 0.0]
        assert result[1] == [0.0, 1.0]
        # "c" is unknown, gets default (zero vector)
        assert result[2] == [0.0, 0.0]


class TestRealEmbeddingProvider:
    """Tests for RealEmbeddingProvider."""

    def test_raises_not_implemented(self) -> None:
        provider = RealEmbeddingProvider("test-model")
        import pytest

        with pytest.raises(NotImplementedError, match="requires a real embedding model"):
            provider.embed(["test"])


class TestCosineSimilarity:
    """Tests for cosine_similarity."""

    def test_identical_vectors(self) -> None:
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0

    def test_orthogonal_vectors(self) -> None:
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_opposite_vectors(self) -> None:
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0
