"""E5-001: Deterministic embedding cache.

Embeddings are computed once and reused.  The cache key is a SHA-256
digest of:

    embedding model identity + text SHA-256 + normalization policy +
    embedding-config version

This ensures that cache entries are uniquely tied to the exact embedding
configuration that produced them, preventing silent cross-config
contamination.

Cache artifacts are written to:

    results/empirical_v2/e5/embeddings/
        embedding_records.jsonl   — one JSON record per embedded text
        embedding_manifest.json   — campaign-level metadata
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from experiments.trustparadox_u.embedding_backend import (
    E5_EMBEDDING_CONFIG_VERSION,
    E5_EMBEDDING_DIMENSION,
    E5_EMBEDDING_MODEL,
    E5_EMBEDDING_NORMALIZATION,
    E5_EMBEDDING_PROVIDER,
    EmbeddingBackend,
    EmbeddingRecord,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_E5_DIR = Path(__file__).resolve().parents[2] / "results" / "empirical_v2" / "e5"
_EMBEDDINGS_DIR = _E5_DIR / "embeddings"
_RECORDS_PATH = _EMBEDDINGS_DIR / "embedding_records.jsonl"
_MANIFEST_PATH = _EMBEDDINGS_DIR / "embedding_manifest.json"


# ---------------------------------------------------------------------------
# Cache key computation
# ---------------------------------------------------------------------------


def compute_cache_key(
    *,
    model: str,
    text_sha256: str,
    normalization: str,
    config_version: str,
) -> str:
    """Compute a deterministic cache key for an embedding record.

    The key binds the embedding to the exact model, text content,
    normalization policy, and config version that produced it.
    """
    payload = (
        f"{model}|{text_sha256}|{normalization}|{config_version}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def text_sha256(text: str) -> str:
    """Return SHA-256 hex digest of a text string (UTF-8 encoded)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# EmbeddingCache
# ---------------------------------------------------------------------------


class EmbeddingCache:
    """Deterministic embedding cache with JSONL persistence.

    The cache stores embedding records keyed by (model, text_sha256,
    normalization, config_version).  On disk, records are stored as
    JSONL with the full vector inline.
    """

    def __init__(
        self,
        *,
        records_path: Path = _RECORDS_PATH,
        manifest_path: Path = _MANIFEST_PATH,
    ) -> None:
        self._records_path = records_path
        self._manifest_path = manifest_path
        self._records: dict[str, EmbeddingRecord] = {}
        self._loaded = False

    # -- load / save -------------------------------------------------------

    def load(self) -> None:
        """Load existing cache records from disk (if any)."""
        if not self._records_path.exists():
            self._loaded = True
            return

        with open(self._records_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                rec = _record_from_dict(raw)
                self._records[rec.embedding_id] = rec

        self._loaded = True

    def save(self) -> None:
        """Write all cache records to disk as JSONL."""
        self._records_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._records_path, "w") as f:
            for rec in self._records.values():
                f.write(json.dumps(_record_to_dict(rec)) + "\n")

    def save_manifest(
        self,
        *,
        backend: EmbeddingBackend,
        candidate_count: int = 0,
        target_count: int = 0,
        alias_count: int = 0,
        failed_count: int = 0,
        code_commit: str = "unknown",
        frozen_corpus_sha: str = "",
        global_annotation_freeze_sha: str = "",
    ) -> dict[str, Any]:
        """Write the embedding manifest to disk and return it."""
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)

        # Compute cache SHA
        cache_sha = ""
        if self._records_path.exists():
            h = hashlib.sha256()
            with open(self._records_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            cache_sha = h.hexdigest()

        identity = backend.identity_dict()
        provenance = backend.get_provenance()

        manifest: dict[str, Any] = {
            "schema_version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "embedding_provider": identity["provider"],
            "embedding_model": identity["model"],
            "embedding_model_revision": identity["model_revision"],
            "dimensions": identity["dimensions"],
            "normalization": identity["normalization"],
            "batch_size": provenance["batch_size"],
            "config_version": identity["config_version"],
            "code_commit": code_commit,
            "frozen_corpus_sha": frozen_corpus_sha,
            "global_annotation_freeze_sha": global_annotation_freeze_sha,
            "candidate_count": candidate_count,
            "target_count": target_count,
            "alias_count": alias_count,
            "successful_embedding_count": len(self._records),
            "failed_embedding_count": failed_count,
            "cache_sha": cache_sha,
            "total_api_calls": provenance["total_calls"],
            "total_texts_embedded": provenance["total_texts_embedded"],
            "total_retries": provenance["total_retries"],
        }

        with open(self._manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")

        return manifest

    # -- lookup / store ----------------------------------------------------

    def get(
        self,
        *,
        model: str,
        text_hash: str,
        normalization: str = E5_EMBEDDING_NORMALIZATION,
        config_version: str = E5_EMBEDDING_CONFIG_VERSION,
    ) -> EmbeddingRecord | None:
        """Look up a cached embedding by its cache key components."""
        key = compute_cache_key(
            model=model,
            text_sha256=text_hash,
            normalization=normalization,
            config_version=config_version,
        )
        return self._records.get(key)

    def put(self, record: EmbeddingRecord) -> None:
        """Store an embedding record in the cache."""
        self._records[record.embedding_id] = record

    @property
    def size(self) -> int:
        """Return the number of cached records."""
        return len(self._records)

    @property
    def records(self) -> dict[str, EmbeddingRecord]:
        """Return all cached records (read-only view)."""
        return dict(self._records)

    # -- bulk embed --------------------------------------------------------

    def embed_and_cache(
        self,
        *,
        backend: EmbeddingBackend,
        texts_with_ids: list[tuple[str, str, str]],
        # (text, text_role, entity_id)
    ) -> list[EmbeddingRecord]:
        """Embed texts using the backend, caching results.

        Args:
            backend: The embedding backend to use.
            texts_with_ids: List of (text, text_role, entity_id) tuples.

        Returns:
            List of newly created EmbeddingRecords (not cache hits).
        """
        if not self._loaded:
            self.load()

        identity = backend.embedding_identity()
        new_records: list[EmbeddingRecord] = []
        texts_to_embed: list[str] = []
        pending_meta: list[tuple[str, str, str]] = []  # (text_hash, role, eid)

        for text, text_role, entity_id in texts_with_ids:
            t_hash = text_sha256(text)
            existing = self.get(
                model=identity.model,
                text_hash=t_hash,
                normalization=identity.normalization,
                config_version=identity.config_version,
            )
            if existing is not None:
                continue  # cache hit

            texts_to_embed.append(text)
            pending_meta.append((t_hash, text_role, entity_id))

        if texts_to_embed:
            vectors = backend.embed_texts(texts_to_embed)
            now = datetime.now(timezone.utc).isoformat()

            for (t_hash, text_role, entity_id), vector in zip(
                pending_meta, vectors
            ):
                cache_key = compute_cache_key(
                    model=identity.model,
                    text_sha256=t_hash,
                    normalization=identity.normalization,
                    config_version=identity.config_version,
                )
                record = EmbeddingRecord(
                    embedding_id=cache_key,
                    text_sha256=t_hash,
                    text_role=text_role,
                    entity_id=entity_id,
                    provider=identity.provider,
                    model=identity.model,
                    dimension=identity.dimensions,
                    vector=tuple(vector),
                    created_at=now,
                )
                self.put(record)
                new_records.append(record)

        return new_records


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _record_to_dict(rec: EmbeddingRecord) -> dict[str, Any]:
    """Serialize an EmbeddingRecord to a JSON-compatible dict."""
    return {
        "embedding_id": rec.embedding_id,
        "text_sha256": rec.text_sha256,
        "text_role": rec.text_role,
        "entity_id": rec.entity_id,
        "provider": rec.provider,
        "model": rec.model,
        "dimension": rec.dimension,
        "vector": list(rec.vector),
        "created_at": rec.created_at,
    }


def _record_from_dict(raw: dict[str, Any]) -> EmbeddingRecord:
    """Deserialize an EmbeddingRecord from a dict."""
    return EmbeddingRecord(
        embedding_id=raw["embedding_id"],
        text_sha256=raw["text_sha256"],
        text_role=raw["text_role"],
        entity_id=raw["entity_id"],
        provider=raw["provider"],
        model=raw["model"],
        dimension=raw["dimension"],
        vector=tuple(raw["vector"]),
        created_at=raw["created_at"],
    )


def load_manifest() -> dict[str, Any]:
    """Load the embedding manifest from disk."""
    if not _MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Embedding manifest not found: {_MANIFEST_PATH}"
        )
    with open(_MANIFEST_PATH) as f:
        return json.load(f)  # type: ignore[no-any-return]
