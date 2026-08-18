"""E5-001: Real embedding backend for ForgetFlow empirical evaluation.

This module provides the provider-independent embedding interface required
by the E5 plan.  It wraps the existing LiteLLM-based embedding provider
with the additional safety checks required for research-valid embedding:

- No hidden fallback model (fail closed on provider error).
- Fixed dimensionality across the entire campaign.
- Finite-value check (reject NaN / Inf).
- Consistent L2 normalization at embedding time.
- Full provenance metadata for every embedding call.

The primary frozen model for the E5 campaign is ``text-embedding-v3``
served via the Aliyun LiteLLM-compatible endpoint (OpenAI-compatible).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

# ---------------------------------------------------------------------------
# E5 frozen embedding configuration
# ---------------------------------------------------------------------------

E5_EMBEDDING_PROVIDER = "litellm"
E5_EMBEDDING_MODEL = "openai/text-embedding-v3"
E5_EMBEDDING_DIMENSION = 1024
E5_EMBEDDING_NORMALIZATION = "l2"
E5_EMBEDDING_BATCH_SIZE = 64
E5_EMBEDDING_CONFIG_VERSION = "1"

E5_API_BASE = (
    "https://llm-jhxtd03gjg0gd2o2.ap-southeast-1.maas.aliyuncs.com"
    "/compatible-mode/v1"
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddingIdentity:
    """Immutable record identifying the embedding configuration."""

    provider: str
    model: str
    model_revision: str | None
    dimensions: int
    normalization: str
    config_version: str


@dataclass(frozen=True)
class EmbeddingProvenance:
    """Provenance metadata for a single embedding batch call."""

    provider: str
    model: str
    api_base: str
    dimensions: int
    normalization: str
    batch_size: int
    timestamp: str
    retry_count: int
    provider_request_id: str | None = None


@dataclass(frozen=True)
class EmbeddingRecord:
    """A single cached embedding record."""

    embedding_id: str
    text_sha256: str
    text_role: str
    entity_id: str  # candidate_id / target_id / alias_id
    provider: str
    model: str
    dimension: int
    vector: tuple[float, ...]
    created_at: str


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _check_finite(vector: Sequence[float], *, context: str) -> None:
    """Raise if *vector* contains NaN or Inf values."""
    for i, v in enumerate(vector):
        if not math.isfinite(v):
            raise RuntimeError(
                f"Non-finite embedding value at index {i} in {context}: {v}"
            )


def _check_dimension(
    vector: Sequence[float],
    expected: int,
    *,
    context: str,
) -> None:
    """Raise if *vector* does not have *expected* length."""
    if len(vector) != expected:
        raise RuntimeError(
            f"Embedding dimension mismatch in {context}: "
            f"expected {expected}, got {len(vector)}"
        )


def _l2_normalize(vector: list[float]) -> list[float]:
    """Return a new L2-normalized copy of *vector*."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0:
        return list(vector)
    return [x / norm for x in vector]


# ---------------------------------------------------------------------------
# EmbeddingBackend
# ---------------------------------------------------------------------------


class EmbeddingBackend:
    """E5 frozen embedding backend.

    Wraps LiteLLM to provide a research-safe embedding interface with
    full provenance, dimension enforcement, finite-value checks, and
    consistent L2 normalization.
    """

    def __init__(
        self,
        *,
        provider: str = E5_EMBEDDING_PROVIDER,
        model: str = E5_EMBEDDING_MODEL,
        expected_dimension: int = E5_EMBEDDING_DIMENSION,
        api_base: str = E5_API_BASE,
        api_key: str | None = None,
        batch_size: int = E5_EMBEDDING_BATCH_SIZE,
        normalize: bool = True,
    ) -> None:
        if provider != "litellm":
            raise ValueError(
                f"E5 embedding backend supports only 'litellm', got {provider!r}"
            )
        if not model:
            raise ValueError("E5 embedding backend requires a model name")
        if expected_dimension <= 0:
            raise ValueError("expected_dimension must be positive")

        self._provider = provider
        self._model = model
        self._expected_dimension = expected_dimension
        self._api_base = api_base
        self._api_key = api_key
        self._batch_size = batch_size
        self._normalize = normalize
        self._total_calls = 0
        self._total_texts = 0
        self._total_retries = 0
        self._observed_dimension: int | None = None

    # -- identity ----------------------------------------------------------

    def embedding_identity(self) -> EmbeddingIdentity:
        """Return the frozen embedding identity for this backend."""
        return EmbeddingIdentity(
            provider=self._provider,
            model=self._model,
            model_revision=None,
            dimensions=self._expected_dimension,
            normalization=E5_EMBEDDING_NORMALIZATION if self._normalize else "none",
            config_version=E5_EMBEDDING_CONFIG_VERSION,
        )

    def identity_dict(self) -> dict[str, Any]:
        """Return embedding identity as a plain dict (for JSON serialization)."""
        ident = self.embedding_identity()
        return {
            "provider": ident.provider,
            "model": ident.model,
            "model_revision": ident.model_revision,
            "dimensions": ident.dimensions,
            "normalization": ident.normalization,
            "config_version": ident.config_version,
        }

    # -- core embedding ----------------------------------------------------

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts and return L2-normalized vectors.

        Texts are sent in batches of ``batch_size``.  Each batch is
        validated for count, dimension, and finite values.

        Raises:
            RuntimeError: On provider failure, dimension mismatch, or
                non-finite values.
        """
        if not texts:
            return []

        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors = self._embed_batch(batch)
            all_vectors.extend(vectors)

        return all_vectors

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """Embed a single batch with retry and validation."""
        retry_count = 0
        max_retries = 3

        while True:
            try:
                vectors = self._call_provider(batch)
                break
            except Exception:
                retry_count += 1
                if retry_count > max_retries:
                    raise RuntimeError(
                        f"Embedding failed after {max_retries} retries "
                        f"for model={self._model}"
                    )
                time.sleep(1.0 * retry_count)

        # Validate each vector
        for i, vec in enumerate(vectors):
            _check_dimension(
                vec, self._expected_dimension, context=f"batch item {i}"
            )
            _check_finite(vec, context=f"batch item {i}")

        # L2-normalize if configured
        if self._normalize:
            vectors = [_l2_normalize(v) for v in vectors]

        # Track observed dimension
        if self._observed_dimension is None:
            self._observed_dimension = len(vectors[0])
        elif self._observed_dimension != len(vectors[0]):
            raise RuntimeError(
                f"Inconsistent embedding dimension: "
                f"expected {self._observed_dimension}, "
                f"got {len(vectors[0])}"
            )

        self._total_calls += 1
        self._total_texts += len(batch)
        self._total_retries += retry_count

        return vectors

    def _call_provider(self, texts: list[str]) -> list[list[float]]:
        """Call the LiteLLM embedding provider.

        No hidden fallback: if the model is unavailable, this raises.
        """
        import litellm  # type: ignore[import-untyped]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "input": texts,
        }
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if self._api_key:
            kwargs["api_key"] = self._api_key

        response = litellm.embedding(**kwargs)

        # Extract vectors from response
        data = (
            response.get("data", [])
            if isinstance(response, dict)
            else getattr(response, "data", [])
        )
        if not data:
            raise RuntimeError("Embedding provider returned no data")

        vectors: list[list[float]] = []
        for item in data:
            emb = (
                item.get("embedding", [])
                if isinstance(item, dict)
                else getattr(item, "embedding", [])
            )
            vectors.append(list(emb))

        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Embedding count mismatch: "
                f"requested {len(texts)}, got {len(vectors)}"
            )

        return vectors

    # -- provenance --------------------------------------------------------

    def get_provenance(self) -> dict[str, Any]:
        """Return cumulative provenance for this backend instance."""
        return {
            "identity": self.identity_dict(),
            "api_base": self._api_base,
            "total_calls": self._total_calls,
            "total_texts_embedded": self._total_texts,
            "total_retries": self._total_retries,
            "batch_size": self._batch_size,
            "observed_dimension": self._observed_dimension,
        }


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def create_backend(
    *,
    api_key: str | None = None,
) -> EmbeddingBackend:
    """Create the frozen E5 embedding backend with default settings."""
    return EmbeddingBackend(api_key=api_key)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors.

    Assumes vectors are already L2-normalized (as produced by
    EmbeddingBackend with normalize=True).  Falls back to full
    computation for non-normalized inputs.
    """
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)
