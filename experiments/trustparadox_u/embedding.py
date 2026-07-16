"""Embedding providers for semantic detection."""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, Sequence


class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class StubEmbeddingProvider:
    """Deterministic stub for testing. Produces embeddings from text hashing."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        results = []
        for text in texts:
            vec = [0.0] * self._dim
            h = hashlib.sha256(text.lower().strip().encode()).digest()
            for i in range(self._dim):
                byte_idx = i % len(h)
                vec[i] = (h[byte_idx] - 128) / 128.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vec = [v / norm for v in vec]
            results.append(vec)
        return results


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)
