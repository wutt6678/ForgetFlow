"""Tests for embedding providers."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from experiments.trustparadox_u.embedding import (
    FixedEmbeddingProvider,
    RealEmbeddingProvider,
    cosine_similarity,
    extract_embedding_vectors,
    validate_embedding_batch,
)


def _mock_litellm(response: dict) -> MagicMock:
    """Create a mock litellm module with the given embedding response."""
    mock_module = MagicMock()
    mock_module.embedding.return_value = response
    return mock_module


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

    def test_provider_name(self) -> None:
        provider = FixedEmbeddingProvider({"a": [1.0]})
        assert provider.provider_name == "fixed"

    def test_model_name(self) -> None:
        provider = FixedEmbeddingProvider({"a": [1.0]})
        assert provider.model_name == "fixed"


class TestRealEmbeddingProvider:
    """Tests for RealEmbeddingProvider."""

    def test_requires_model_name(self) -> None:
        with pytest.raises(ValueError, match="model_name"):
            RealEmbeddingProvider(model_name="")

    def test_rejects_unsupported_provider(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            RealEmbeddingProvider(provider_name="openai", model_name="x")

    def test_rejects_nonpositive_dimension(self) -> None:
        with pytest.raises(ValueError, match="expected_dimension"):
            RealEmbeddingProvider(model_name="x", expected_dimension=0)

    def test_properties(self) -> None:
        p = RealEmbeddingProvider(model_name="text-embedding-3-small", expected_dimension=1536)
        assert p.provider_name == "litellm"
        assert p.model_name == "text-embedding-3-small"
        assert p.dimension == 1536

    def test_empty_input_returns_empty(self) -> None:
        p = RealEmbeddingProvider(model_name="text-embedding-3-small")
        assert p.embed([]) == []

    def test_embed_with_mocked_backend(self) -> None:
        p = RealEmbeddingProvider(model_name="text-embedding-3-small", expected_dimension=3)
        mock_response = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        mock_mod = _mock_litellm(mock_response)
        with patch.dict(sys.modules, {"litellm": mock_mod}):
            result = p.embed(["hello"])
        assert result == [[0.1, 0.2, 0.3]]
        assert p.dimension == 3
        mock_mod.embedding.assert_called_once_with(model="text-embedding-3-small", input=["hello"])

    def test_embed_multi_text(self) -> None:
        p = RealEmbeddingProvider(model_name="m", expected_dimension=2)
        mock_response = {"data": [{"embedding": [1.0, 0.0]}, {"embedding": [0.0, 1.0]}]}
        mock_mod = _mock_litellm(mock_response)
        with patch.dict(sys.modules, {"litellm": mock_mod}):
            result = p.embed(["a", "b"])
        assert len(result) == 2
        assert p.dimension == 2

    def test_backend_error_wrapped(self) -> None:
        p = RealEmbeddingProvider(model_name="m")
        mock_mod = MagicMock()
        mock_mod.embedding.side_effect = RuntimeError("API down")
        with patch.dict(sys.modules, {"litellm": mock_mod}):
            with pytest.raises(RuntimeError, match="Embedding request failed"):
                p.embed(["test"])

    def test_response_count_mismatch(self) -> None:
        p = RealEmbeddingProvider(model_name="m")
        mock_response = {"data": [{"embedding": [1.0]}]}
        mock_mod = _mock_litellm(mock_response)
        with patch.dict(sys.modules, {"litellm": mock_mod}):
            with pytest.raises(RuntimeError, match="response count"):
                p.embed(["a", "b"])

    def test_inconsistent_dimensions(self) -> None:
        p = RealEmbeddingProvider(model_name="m")
        mock_response = {"data": [{"embedding": [1.0, 0.0]}, {"embedding": [1.0]}]}
        mock_mod = _mock_litellm(mock_response)
        with patch.dict(sys.modules, {"litellm": mock_mod}):
            with pytest.raises(RuntimeError, match="inconsistent dimensions"):
                p.embed(["a", "b"])

    def test_empty_vector_rejected(self) -> None:
        p = RealEmbeddingProvider(model_name="m")
        mock_response = {"data": [{"embedding": []}]}
        mock_mod = _mock_litellm(mock_response)
        with patch.dict(sys.modules, {"litellm": mock_mod}):
            with pytest.raises(RuntimeError, match="empty vector"):
                p.embed(["a"])

    def test_dimension_mismatch(self) -> None:
        p = RealEmbeddingProvider(model_name="m", expected_dimension=5)
        mock_response = {"data": [{"embedding": [1.0, 0.0, 0.0]}]}
        mock_mod = _mock_litellm(mock_response)
        with patch.dict(sys.modules, {"litellm": mock_mod}):
            with pytest.raises(RuntimeError, match="dimension mismatch"):
                p.embed(["a"])

    def test_observed_dimension_recorded(self) -> None:
        p = RealEmbeddingProvider(model_name="m")
        assert p.dimension is None
        mock_response = {"data": [{"embedding": [1.0, 0.0, 0.0]}]}
        mock_mod = _mock_litellm(mock_response)
        with patch.dict(sys.modules, {"litellm": mock_mod}):
            p.embed(["a"])
        assert p.dimension == 3

    def test_no_fallback_on_backend_failure(self) -> None:
        """Backend failure must NOT silently produce fixed embeddings."""
        p = RealEmbeddingProvider(model_name="m")
        mock_mod = MagicMock()
        mock_mod.embedding.side_effect = ConnectionError("no network")
        with patch.dict(sys.modules, {"litellm": mock_mod}):
            with pytest.raises(RuntimeError, match="Embedding request failed"):
                p.embed(["test"])


class TestExtractEmbeddingVectors:
    def test_dict_response(self) -> None:
        resp = {"data": [{"embedding": [1.0, 2.0]}]}
        assert extract_embedding_vectors(resp) == [[1.0, 2.0]]

    def test_object_response(self) -> None:
        item = MagicMock()
        item.embedding = [3.0, 4.0]
        resp = MagicMock()
        resp.data = [item]
        assert extract_embedding_vectors(resp) == [[3.0, 4.0]]

    def test_empty_data(self) -> None:
        assert extract_embedding_vectors({"data": []}) == []


class TestValidateEmbeddingBatch:
    def test_valid_batch(self) -> None:
        validate_embedding_batch(
            texts=["a", "b"],
            vectors=[[1.0, 0.0], [0.0, 1.0]],
            expected_dimension=2,
        )

    def test_count_mismatch(self) -> None:
        with pytest.raises(RuntimeError, match="response count"):
            validate_embedding_batch(texts=["a", "b"], vectors=[[1.0]], expected_dimension=None)

    def test_empty_vectors(self) -> None:
        with pytest.raises(RuntimeError, match="no vectors"):
            validate_embedding_batch(texts=[], vectors=[], expected_dimension=None)

    def test_zero_length_vector(self) -> None:
        with pytest.raises(RuntimeError, match="empty vector"):
            validate_embedding_batch(texts=["a"], vectors=[[]], expected_dimension=None)

    def test_inconsistent_dimensions(self) -> None:
        with pytest.raises(RuntimeError, match="inconsistent"):
            validate_embedding_batch(
                texts=["a", "b"],
                vectors=[[1.0, 0.0], [1.0]],
                expected_dimension=None,
            )

    def test_expected_dimension_mismatch(self) -> None:
        with pytest.raises(RuntimeError, match="dimension mismatch"):
            validate_embedding_batch(texts=["a"], vectors=[[1.0, 0.0]], expected_dimension=5)


class TestCosineSimilarity:
    """Tests for cosine_similarity."""

    def test_identical_vectors(self) -> None:
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0

    def test_orthogonal_vectors(self) -> None:
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_opposite_vectors(self) -> None:
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0
