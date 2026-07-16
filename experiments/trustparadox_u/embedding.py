"""Embedding providers for semantic detection."""

from __future__ import annotations

import math
from typing import Any, Mapping, Protocol, Sequence


class EmbeddingProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str | None: ...

    @property
    def dimension(self) -> int | None: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class FixedEmbeddingProvider:
    """Deterministic test provider with predefined vectors.

    Uses normalized text as lookup key. Unknown text returns a configured
    default unrelated vector (does NOT hash arbitrary strings).
    """

    def __init__(
        self,
        vectors: Mapping[str, Sequence[float]],
        default_vector: Sequence[float] | None = None,
    ) -> None:
        self._vectors = {k.lower().strip(): list(v) for k, v in vectors.items()}
        if default_vector is not None:
            self._default = list(default_vector)
        else:
            # Zero vector as default (unrelated to everything)
            dim = len(next(iter(vectors.values()))) if vectors else 64
            self._default = [0.0] * dim
        self._dim = len(self._default)

    @property
    def provider_name(self) -> str:
        return "fixed"

    @property
    def model_name(self) -> str | None:
        return None

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        results = []
        for text in texts:
            key = text.lower().strip()
            if key in self._vectors:
                results.append(list(self._vectors[key]))
            else:
                results.append(list(self._default))
        return results


def extract_embedding_vectors(response: Any) -> list[list[float]]:
    """Extract embedding vectors from a LiteLLM response object."""
    data = response.get("data", []) if isinstance(response, dict) else getattr(response, "data", [])
    vectors: list[list[float]] = []
    for item in data:
        emb = (
            item.get("embedding", []) if isinstance(item, dict) else getattr(item, "embedding", [])
        )
        vectors.append(list(emb))
    return vectors


def validate_embedding_batch(
    *,
    texts: Sequence[str],
    vectors: Sequence[Sequence[float]],
    expected_dimension: int | None,
) -> None:
    """Validate an embedding batch for count, dimension, and non-emptiness."""
    if len(vectors) != len(texts):
        raise RuntimeError(
            f"Embedding response count ({len(vectors)}) "
            f"does not match input count ({len(texts)})"
        )

    if not vectors:
        raise RuntimeError("Embedding backend returned no vectors")

    dimensions = {len(vector) for vector in vectors}

    if 0 in dimensions:
        raise RuntimeError("Embedding backend returned an empty vector")

    if len(dimensions) != 1:
        raise RuntimeError(f"Embedding backend returned inconsistent dimensions: {dimensions}")

    observed_dimension = next(iter(dimensions))

    if expected_dimension is not None and observed_dimension != expected_dimension:
        raise RuntimeError(
            f"Embedding dimension mismatch: "
            f"expected={expected_dimension}, "
            f"observed={observed_dimension}"
        )


class RealEmbeddingProvider:
    """Experiment provider using a real embedding model via LiteLLM.

    Validates response count, dimension consistency, and non-empty vectors.
    """

    def __init__(
        self,
        *,
        provider_name: str = "litellm",
        model_name: str,
        expected_dimension: int | None = None,
        api_base: str | None = None,
    ) -> None:
        if provider_name != "litellm":
            raise ValueError(f"Unsupported embedding provider: {provider_name}")
        if not model_name:
            raise ValueError(
                "RealEmbeddingProvider requires a model_name. "
                "Set models.embedding_model in the experiment config."
            )
        if expected_dimension is not None and expected_dimension <= 0:
            raise ValueError("expected_dimension must be positive")

        self._provider_name = provider_name
        self._model_name = model_name
        self._expected_dimension = expected_dimension
        self._api_base = api_base
        self._observed_dimension: int | None = None

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int | None:
        return self._observed_dimension or self._expected_dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        try:
            import litellm  # type: ignore[import-untyped]

            kwargs: dict[str, Any] = {
                "model": self._model_name,
                "input": list(texts),
            }
            if self._api_base:
                kwargs["api_base"] = self._api_base

            response = litellm.embedding(**kwargs)
        except Exception as exc:
            from experiments.trustparadox_u.providers import sanitize_api_base

            host = sanitize_api_base(self._api_base)
            endpoint_info = f", endpoint={host}" if host else ""
            raise RuntimeError(
                f"Embedding request failed for "
                f"provider={self._provider_name}, "
                f"model={self._model_name}"
                f"{endpoint_info}"
            ) from exc

        vectors = extract_embedding_vectors(response)

        validate_embedding_batch(
            texts=texts,
            vectors=vectors,
            expected_dimension=self._expected_dimension,
        )

        self._observed_dimension = len(vectors[0])
        return vectors


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)
