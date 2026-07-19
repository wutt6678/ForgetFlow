"""Shared provider construction.

Ensures the runner and preflight build identical providers from the
same ``ModelsConfig``.
"""

from __future__ import annotations

from urllib.parse import urlsplit

from experiments.trustparadox_u.chat_provider import LiteLLMResponseProvider
from experiments.trustparadox_u.config import ModelsConfig
from experiments.trustparadox_u.embedding import RealEmbeddingProvider


def build_real_chat_provider(models: ModelsConfig) -> LiteLLMResponseProvider:
    """Construct a ``LiteLLMResponseProvider`` from a ``ModelsConfig``.

    Raises ``ValueError`` when required fields are missing.
    """
    if not models.chat_model:
        raise ValueError("chat_model is required")

    return LiteLLMResponseProvider(
        model_name=models.chat_model,
        temperature=models.chat_temperature,
        max_tokens=models.chat_max_tokens,
        api_base=models.api_base,
        api_key_env=models.api_key_env or None,
    )


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
    """Return ``scheme://host[:port]`` from *api_base*, stripping credentials.

    Returns ``None`` when *api_base* is falsy.  Returns
    ``"<configured>"`` when the URL is malformed.
    Preserves port numbers to distinguish between different local endpoints.
    """
    if not api_base:
        return None

    parsed = urlsplit(api_base)

    if not parsed.scheme or not parsed.hostname:
        return "<configured>"

    host = parsed.hostname
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"

    return f"{parsed.scheme}://{host}"
