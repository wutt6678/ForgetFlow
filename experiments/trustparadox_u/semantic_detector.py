"""E5-002: Semantic detector feature extraction.

Computes per-candidate continuous detector features *before* any policy
decision.  Features are annotation-independent: they must not use
target_leakage, positive_entailment, task_useful, or resolution_source
from the frozen annotation labels.

Per-candidate features
----------------------
- ``exact_match`` (bool):
  Normalised canonical target appears as a substring of the normalised
  candidate text.
- ``alias_match`` (bool):
  At least one normalised alias appears as a substring of the normalised
  candidate text.
- ``matched_alias`` (str | None):
  The first alias that matched (or ``None``).
- ``semantic_similarity`` (float):
  ``max(cosine(candidate, canonical_target),
       cosine(candidate, alias_1), ...)``
  clamped to [0, 1].

Artifacts
---------
For each split a JSONL file is written under
``results/empirical_v2/e5/detector_features/``::

    development_detector_features.jsonl
    validation_detector_features.jsonl
    test_detector_features.jsonl

A ``feature_manifest.json`` records provenance metadata.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from experiments.trustparadox_u.e5_loaders import (
    VALID_SPLITS,
    load_corpus,
)
from experiments.trustparadox_u.embedding_backend import (
    EmbeddingBackend,
    cosine_similarity,
)
from experiments.trustparadox_u.embedding_cache import (
    EmbeddingCache,
    text_sha256,
)
from experiments.trustparadox_u.empirical_corpus import (
    EMPIRICAL_TARGET_REGISTRY,
    EmpiricalTargetSpec,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_E5_DIR = Path(__file__).resolve().parents[2] / "results" / "empirical_v2" / "e5"
_FEATURES_DIR = _E5_DIR / "detector_features"
_FEATURE_MANIFEST_PATH = _FEATURES_DIR / "feature_manifest.json"

_DETECTOR_VERSION = "e5-002-r1"


# ---------------------------------------------------------------------------
# Text normalisation  (aligned with HybridDetector._normalize)
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Lowercase, NFC-normalise, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Target registry lookup
# ---------------------------------------------------------------------------


def build_target_index() -> dict[tuple[str, str], EmpiricalTargetSpec]:
    """Build a (scenario_id, secret_variant_id) → spec index."""
    return {
        (spec.scenario_id, spec.secret_variant_id): spec
        for spec in EMPIRICAL_TARGET_REGISTRY
    }


# ---------------------------------------------------------------------------
# Feature record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectorFeature:
    """Per-candidate detector feature record."""

    candidate_id: str
    scenario_id: str
    secret_variant_id: str
    attack_type: str
    trust_level: str
    forget_id: str
    exact_match: bool
    alias_match: bool
    matched_alias: str | None
    semantic_similarity: float
    detector_version: str
    embedding_model: str
    embedding_config_version: str


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------


def _compute_exact_and_alias(
    candidate_text: str,
    canonical_target: str,
    aliases: tuple[str, ...],
) -> tuple[bool, bool, str | None]:
    """Compute exact_match, alias_match, and matched_alias.

    All comparisons use normalised text with substring matching.
    """
    norm_text = normalize_text(candidate_text)
    norm_target = normalize_text(canonical_target)

    exact = bool(norm_target and norm_target in norm_text)

    matched_alias: str | None = None
    for alias in aliases:
        norm_alias = normalize_text(alias)
        if norm_alias and norm_alias in norm_text:
            matched_alias = alias
            break

    return exact, matched_alias is not None, matched_alias


def compute_features_for_split(
    split: str,
    *,
    backend: EmbeddingBackend,
    cache: EmbeddingCache,
    target_index: dict[tuple[str, str], EmpiricalTargetSpec] | None = None,
) -> list[DetectorFeature]:
    """Compute detector features for all candidates in a split.

    Args:
        split: One of "development", "validation", "test".
        backend: E5 embedding backend for computing vector embeddings.
        cache: Embedding cache for deterministic reuse.
        target_index: Optional pre-built target index.  Built from
            ``EMPIRICAL_TARGET_REGISTRY`` if not provided.

    Returns:
        List of ``DetectorFeature`` records, one per corpus candidate.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split: {split!r}.  Valid: {sorted(VALID_SPLITS)}")

    if target_index is None:
        target_index = build_target_index()

    corpus = load_corpus(split)

    # Collect all texts that need embedding:
    #   - each candidate text
    #   - each unique canonical_target and alias from relevant targets
    texts_to_embed: list[tuple[str, str, str]] = []  # (text, role, entity_id)
    seen_texts: set[str] = set()

    # Gather target reference texts
    target_ref_texts: dict[str, str] = {}  # text → descriptive entity_id
    relevant_targets: set[tuple[str, str]] = set()
    for cand in corpus:
        key = (cand.scenario_id, cand.secret_variant_id)
        if key in target_index:
            relevant_targets.add(key)

    for key in relevant_targets:
        spec = target_index[key]
        # Canonical target
        if spec.canonical_target not in seen_texts:
            eid = f"target::{spec.forget_id}::canonical"
            texts_to_embed.append((spec.canonical_target, "target", eid))
            seen_texts.add(spec.canonical_target)
            target_ref_texts[spec.canonical_target] = eid
        # Aliases
        for i, alias in enumerate(spec.aliases):
            if alias not in seen_texts:
                eid = f"target::{spec.forget_id}::alias_{i}"
                texts_to_embed.append((alias, "target_alias", eid))
                seen_texts.add(alias)
                target_ref_texts[alias] = eid

    # Candidate texts
    for cand in corpus:
        if cand.text not in seen_texts:
            eid = f"candidate::{cand.candidate_id}"
            texts_to_embed.append((cand.text, "candidate", eid))
            seen_texts.add(cand.text)

    # Embed all texts via cache
    cache.embed_and_cache(
        backend=backend,
        texts_with_ids=texts_to_embed,
    )

    # Build a lookup: text → vector (from cache)
    identity = backend.embedding_identity()
    text_to_vector: dict[str, tuple[float, ...]] = {}
    for text in seen_texts:
        t_hash = text_sha256(text)
        rec = cache.get(
            model=identity.model,
            text_hash=t_hash,
            normalization=identity.normalization,
            config_version=identity.config_version,
        )
        if rec is not None:
            text_to_vector[text] = rec.vector

    # Compute features per candidate
    features: list[DetectorFeature] = []
    for cand in corpus:
        key = (cand.scenario_id, cand.secret_variant_id)
        spec = target_index.get(key)
        if spec is None:
            # No target found — emit zeroed features
            features.append(
                DetectorFeature(
                    candidate_id=cand.candidate_id,
                    scenario_id=cand.scenario_id,
                    secret_variant_id=cand.secret_variant_id,
                    attack_type=cand.attack_type,
                    trust_level=cand.trust_level,
                    forget_id="",
                    exact_match=False,
                    alias_match=False,
                    matched_alias=None,
                    semantic_similarity=0.0,
                    detector_version=_DETECTOR_VERSION,
                    embedding_model=identity.model,
                    embedding_config_version=identity.config_version,
                )
            )
            continue

        # Exact / alias matching
        exact, alias_matched, matched_alias = _compute_exact_and_alias(
            cand.text, spec.canonical_target, spec.aliases
        )

        # Semantic similarity: max cosine(candidate, ref_text) for ref_text
        #   in {canonical_target} ∪ aliases
        cand_vec = text_to_vector.get(cand.text)
        max_sim = 0.0
        if cand_vec is not None:
            ref_texts = [spec.canonical_target] + list(spec.aliases)
            for ref in ref_texts:
                ref_vec = text_to_vector.get(ref)
                if ref_vec is not None:
                    sim = cosine_similarity(cand_vec, ref_vec)
                    sim = max(0.0, min(1.0, sim))
                    if sim > max_sim:
                        max_sim = sim

        features.append(
            DetectorFeature(
                candidate_id=cand.candidate_id,
                scenario_id=cand.scenario_id,
                secret_variant_id=cand.secret_variant_id,
                attack_type=cand.attack_type,
                trust_level=cand.trust_level,
                forget_id=spec.forget_id,
                exact_match=exact,
                alias_match=alias_matched,
                matched_alias=matched_alias,
                semantic_similarity=round(max_sim, 8),
                detector_version=_DETECTOR_VERSION,
                embedding_model=identity.model,
                embedding_config_version=identity.config_version,
            )
        )

    return features


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _feature_to_dict(f: DetectorFeature) -> dict[str, Any]:
    """Serialize a DetectorFeature to a JSON-compatible dict."""
    return {
        "candidate_id": f.candidate_id,
        "scenario_id": f.scenario_id,
        "secret_variant_id": f.secret_variant_id,
        "attack_type": f.attack_type,
        "trust_level": f.trust_level,
        "forget_id": f.forget_id,
        "exact_match": f.exact_match,
        "alias_match": f.alias_match,
        "matched_alias": f.matched_alias,
        "semantic_similarity": f.semantic_similarity,
        "detector_version": f.detector_version,
        "embedding_model": f.embedding_model,
        "embedding_config_version": f.embedding_config_version,
    }


def write_features(
    features: list[DetectorFeature],
    split: str,
    *,
    output_dir: Path = _FEATURES_DIR,
) -> Path:
    """Write detector features for a split to a JSONL file.

    Returns:
        Path to the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{split}_detector_features.jsonl"
    with open(path, "w") as f:
        for feat in features:
            f.write(json.dumps(_feature_to_dict(feat)) + "\n")
    return path


def write_feature_manifest(
    *,
    backend: EmbeddingBackend,
    split_counts: dict[str, int],
    total_features: int,
    code_commit: str = "unknown",
    output_path: Path = _FEATURE_MANIFEST_PATH,
) -> dict[str, Any]:
    """Write the feature extraction manifest to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    identity = backend.embedding_identity()

    # Compute features directory hash
    features_sha = ""
    if _FEATURES_DIR.exists():
        h = hashlib.sha256()
        for p in sorted(_FEATURES_DIR.glob("*.jsonl")):
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
        features_sha = h.hexdigest()

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "detector_version": _DETECTOR_VERSION,
        "embedding_model": identity.model,
        "embedding_provider": identity.provider,
        "embedding_dimensions": identity.dimensions,
        "embedding_normalization": identity.normalization,
        "embedding_config_version": identity.config_version,
        "code_commit": code_commit,
        "split_counts": split_counts,
        "total_features": total_features,
        "features_sha256": features_sha,
        "annotation_independent": True,
    }

    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    return manifest


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_feature_extraction(
    *,
    backend: EmbeddingBackend,
    cache: EmbeddingCache,
    splits: Sequence[str] = ("development", "validation"),
    code_commit: str = "unknown",
) -> dict[str, Any]:
    """Run feature extraction for all requested splits.

    This is the main entry point for E5-002.

    Returns:
        The feature manifest dict.
    """
    target_index = build_target_index()
    split_counts: dict[str, int] = {}
    total = 0

    for split in splits:
        features = compute_features_for_split(
            split,
            backend=backend,
            cache=cache,
            target_index=target_index,
        )
        write_features(features, split)
        split_counts[split] = len(features)
        total += len(features)

    # Persist cache and manifest
    cache.save()
    cache.save_manifest(
        backend=backend,
        candidate_count=total,
        target_count=len(target_index),
        code_commit=code_commit,
    )

    manifest = write_feature_manifest(
        backend=backend,
        split_counts=split_counts,
        total_features=total,
        code_commit=code_commit,
    )
    return manifest


def load_features(split: str, *, features_dir: Path = _FEATURES_DIR) -> list[dict]:
    """Load detector features from a JSONL file (for downstream use)."""
    path = features_dir / f"{split}_detector_features.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
