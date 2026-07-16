"""Shared embedding provider construction.

Ensures the runner and preflight build identical providers from the
same ``ModelsConfig``.
"""

from __future__ import annotations

from urllib.parse import urlsplit

from experiments.trustparadox_u.config import ModelsConfig
from experiments.trustparadox_u.embedding import RealEmbeddingProvider


def build_real_embedding_provider(models: ModelsConfig) -> RealEmbeddingProvider:
    """Construct a ``RealEmbeddingProvider`` from a ``ModelsConfig``.

    Raises ``ValueError`` when required fields are missing.
    """
    if not models.embedding_provider:
        raise ValueError("embedding_provider is required")

    if not models.embedding_model:
        raise ValueError("embedding_model is required")

    return RealEmbeddingProvider(
        provider_name=models.embedding_provider,
        model_name=models.embedding_model,
        expected_dimension=models.embedding_dimension,
        api_base=models.api_base,
    )


def sanitize_api_base(api_base: str | None) -> str | None:
    """Return ``scheme://host`` from *api_base*, stripping credentials.

    Returns ``None`` when *api_base* is falsy.  Returns
    ``"<configured>"`` when the URL is malformed.
    """
    if not api_base:
        return None

    parsed = urlsplit(api_base)

    if not parsed.scheme or not parsed.hostname:
        return "<configured>"

    return f"{parsed.scheme}://{parsed.hostname}"
