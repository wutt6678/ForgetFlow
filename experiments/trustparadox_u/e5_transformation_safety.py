"""E5 shared transformation embedding safety check (R1.2b §8, §9).

This module provides a single shared helper used by both the E5
FirewallRunner and the marble FlowGate to verify that a transformed
output is semantically safe — i.e. its embedding similarity to every
active forget target canonical and alias is below the frozen threshold
τ_sem.

The helper reuses the frozen E5 embedding backend and cache.  It does
NOT implement a second embedding client.  Transformed-text embeddings
are computed on-demand (development-only), cached, and reproducible.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransformationEmbeddingSafetyResult:
    """Result of the transformation embedding safety recheck.

    Attributes:
        is_safe: True iff all similarities are strictly below tau_sem.
        max_similarity: Highest cosine similarity observed across all
            reference embeddings.
        n_references_checked: Number of reference embeddings compared.
        tau_sem: The threshold used for the check.
        cache_hit: Whether the transformed text was already cached.
    """

    is_safe: bool
    max_similarity: float
    n_references_checked: int
    tau_sem: float
    cache_hit: bool


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def check_transformation_embedding_safety(
    transformed_text: str,
    *,
    reference_texts: Sequence[str],
    backend: Any,
    cache: Any,
    tau_sem: float,
) -> TransformationEmbeddingSafetyResult:
    """Check whether a transformed output is semantically safe.

    Embeds *transformed_text* using the frozen E5 embedding backend
    (``openai/text-embedding-v3``) and compares it against every
    reference text (canonical targets + aliases from active forget
    records).  The check passes iff every cosine similarity is
    strictly below *tau_sem*.

    The transformed text is cached after the first embedding so that
    repeated calls are deterministic and free of additional provider
    calls.

    Args:
        transformed_text: The text produced by a transformation
            (redact / abstract).
        reference_texts: Canonical targets and aliases from the active
            forget records.  Empty strings are skipped.
        backend: An :class:`EmbeddingBackend` instance (frozen model).
        cache: An :class:`EmbeddingCache` instance.
        tau_sem: Semantic similarity threshold.  Same threshold used
            by the input semantic detector.

    Returns:
        A :class:`TransformationEmbeddingSafetyResult`.
    """
    from experiments.trustparadox_u.embedding_backend import (
        E5_EMBEDDING_CONFIG_VERSION,
        E5_EMBEDDING_MODEL,
        E5_EMBEDDING_NORMALIZATION,
        cosine_similarity,
    )
    from experiments.trustparadox_u.embedding_cache import (
        compute_cache_key,
        text_sha256,
    )

    # Filter out empty reference texts
    refs = [t for t in reference_texts if t]
    if not refs:
        # No references to check → trivially safe
        return TransformationEmbeddingSafetyResult(
            is_safe=True,
            max_similarity=0.0,
            n_references_checked=0,
            tau_sem=tau_sem,
            cache_hit=False,
        )

    # --- Embed the transformed text (cache-first) ---
    t_sha = text_sha256(transformed_text)
    cache_key = compute_cache_key(
        model=E5_EMBEDDING_MODEL,
        text_sha256=t_sha,
        normalization=E5_EMBEDDING_NORMALIZATION,
        config_version=E5_EMBEDDING_CONFIG_VERSION,
    )

    cached_record = cache.get(
        model=E5_EMBEDDING_MODEL,
        text_hash=t_sha,
    )
    cache_hit = cached_record is not None

    if cached_record is not None:
        transformed_vec = list(cached_record.vector)
    else:
        # On-demand provider call (development-only; logged + cached)
        logger.info(
            "R1.2b embedding recheck: on-demand embedding for "
            "transformed text (sha256=%s)", t_sha[:16],
        )
        vectors = backend.embed_texts([transformed_text])
        transformed_vec = vectors[0]

        # Store in cache for reproducibility
        from experiments.trustparadox_u.embedding_backend import EmbeddingRecord
        from datetime import datetime, timezone

        record = EmbeddingRecord(
            embedding_id=cache_key,
            text_sha256=t_sha,
            text_role="transformed_output",
            entity_id="recheck",
            provider=E5_EMBEDDING_MODEL,
            model=E5_EMBEDDING_MODEL,
            dimension=len(transformed_vec),
            vector=tuple(transformed_vec),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        cache.put(record)

    # --- Collect reference vectors ---
    max_sim = 0.0
    n_checked = 0

    for ref_text in refs:
        ref_sha = text_sha256(ref_text)
        ref_record = cache.get(
            model=E5_EMBEDDING_MODEL,
            text_hash=ref_sha,
        )
        if ref_record is None:
            # Reference not in cache — embed on demand
            logger.info(
                "R1.2b embedding recheck: on-demand reference "
                "embedding (sha256=%s)", ref_sha[:16],
            )
            ref_vectors = backend.embed_texts([ref_text])
            ref_vec = ref_vectors[0]

            from experiments.trustparadox_u.embedding_backend import EmbeddingRecord
            from datetime import datetime, timezone

            ref_key = compute_cache_key(
                model=E5_EMBEDDING_MODEL,
                text_sha256=ref_sha,
                normalization=E5_EMBEDDING_NORMALIZATION,
                config_version=E5_EMBEDDING_CONFIG_VERSION,
            )
            ref_record = EmbeddingRecord(
                embedding_id=ref_key,
                text_sha256=ref_sha,
                text_role="reference",
                entity_id="recheck_ref",
                provider=E5_EMBEDDING_MODEL,
                model=E5_EMBEDDING_MODEL,
                dimension=len(ref_vec),
                vector=tuple(ref_vec),
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            cache.put(ref_record)
        else:
            ref_vec = list(ref_record.vector)

        sim = cosine_similarity(transformed_vec, ref_vec)
        if sim > max_sim:
            max_sim = sim
        n_checked += 1

    is_safe = max_sim < tau_sem

    return TransformationEmbeddingSafetyResult(
        is_safe=is_safe,
        max_similarity=max_sim,
        n_references_checked=n_checked,
        tau_sem=tau_sem,
        cache_hit=cache_hit,
    )


def collect_reference_texts(active_records: Sequence[Any]) -> list[str]:
    """Collect canonical targets + aliases from active forget records.

    Used by both FlowGate and E5 FirewallRunner to build the reference
    text list for :func:`check_transformation_embedding_safety`.
    """
    refs: list[str] = []
    for rec in active_records:
        if rec.canonical_target:
            refs.append(rec.canonical_target)
        for alias in getattr(rec, "aliases", ()):
            if alias:
                refs.append(alias)
    return refs
