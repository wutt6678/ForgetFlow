"""Embedding providers for semantic detection."""

from __future__ import annotations

import math
from typing import Mapping, Protocol, Sequence


class EmbeddingProvider(Protocol):
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


class RealEmbeddingProvider:
    """Experiment provider using a real embedding model.

    Wraps an embedding client that returns vectors for input texts.
    Validates response count, dimension consistency, and non-empty vectors.
    """

    def __init__(self, model_name: str | None) -> None:
        if not model_name:
            raise ValueError(
                "RealEmbeddingProvider requires a model_name. "
                "Set models.embedding_model in the experiment config."
            )
        self._model_name = model_name
        self._dim = 0

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError(
            f"RealEmbeddingProvider({self._model_name}) requires a real embedding model backend. "
            "Install and configure an embedding client to use experiment mode."
        )


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)
