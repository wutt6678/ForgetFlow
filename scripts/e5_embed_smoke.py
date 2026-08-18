"""E5-001: Embedding smoke test on development subset.

This script embeds a small deterministic subset of the development split
using the real embedding backend, caches the results, and verifies:

1. Same text → same cached vector
2. Dimension is stable (1024)
3. cosine(text, text) ≈ 1
4. All values are finite
5. Cache reload reproduces the same vector
6. Model metadata is preserved

Exit codes:
    0 → GO — all smoke checks pass
    1 → FAIL — one or more checks failed

Produces:
    results/empirical_v2/e5/embeddings/embedding_records.jsonl
    results/empirical_v2/e5/embeddings/embedding_manifest.json
    results/empirical_v2/e5/e5_embed_smoke.json
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.e5_loaders import (
    load_global_freeze_manifest,
    load_split,
)
from experiments.trustparadox_u.embedding_backend import (
    E5_EMBEDDING_DIMENSION,
    EmbeddingBackend,
    cosine_similarity,
    create_backend,
)
from experiments.trustparadox_u.embedding_cache import (
    EmbeddingCache,
    text_sha256,
)

_E5_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "e5"
_SMOKE_PATH = _E5_DIR / "e5_embed_smoke.json"

# Number of development candidates to embed for the smoke test
SMOKE_SUBSET_SIZE = 10


def _get_code_commit() -> str:
    """Return the current git commit hash, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def run_smoke(*, api_key: str | None = None) -> dict:
    """Execute the embedding smoke test.

    Returns:
        Dict with all smoke test results suitable for JSON serialization.
    """
    checks: dict[str, str] = {}
    findings: list[str] = []

    # ------------------------------------------------------------------
    # 1. Load development split
    # ------------------------------------------------------------------
    dev_data = load_split("development")
    corpus_subset = dev_data.corpus[:SMOKE_SUBSET_SIZE]
    texts = [c.text for c in corpus_subset]
    entity_ids = [c.candidate_id for c in corpus_subset]

    print(f"E5-001: Embedding smoke test on {len(texts)} development texts")

    # ------------------------------------------------------------------
    # 2. Create backend and cache
    # ------------------------------------------------------------------
    backend = create_backend(api_key=api_key)
    cache = EmbeddingCache()
    cache.load()

    identity = backend.identity_dict()
    print(f"  Model: {identity['model']}")
    print(f"  Dimension: {identity['dimensions']}")
    print(f"  Normalization: {identity['normalization']}")

    # ------------------------------------------------------------------
    # 3. Embed the subset
    # ------------------------------------------------------------------
    texts_with_ids = [
        (text, "candidate", eid)
        for text, eid in zip(texts, entity_ids)
    ]
    new_records = cache.embed_and_cache(
        backend=backend,
        texts_with_ids=texts_with_ids,
    )
    print(f"  Embedded {len(new_records)} new records "
          f"(cache size: {cache.size})")

    # ------------------------------------------------------------------
    # 4. Check: dimension is correct
    # ------------------------------------------------------------------
    all_correct_dim = all(
        len(rec.vector) == E5_EMBEDDING_DIMENSION
        for rec in cache.records.values()
    )
    if not all_correct_dim:
        findings.append("Not all vectors have expected dimension")
        checks["dimension_correct"] = "FAIL"
    else:
        checks["dimension_correct"] = "PASS"

    # ------------------------------------------------------------------
    # 5. Check: all values finite
    # ------------------------------------------------------------------
    all_finite = True
    for rec in cache.records.values():
        for v in rec.vector:
            if not math.isfinite(v):
                all_finite = False
                break
        if not all_finite:
            break

    if not all_finite:
        findings.append("Non-finite values detected in embeddings")
        checks["all_finite"] = "FAIL"
    else:
        checks["all_finite"] = "PASS"

    # ------------------------------------------------------------------
    # 6. Check: cosine(text, text) ≈ 1
    # ------------------------------------------------------------------
    self_sim_ok = True
    for rec in cache.records.values():
        sim = cosine_similarity(rec.vector, rec.vector)
        if abs(sim - 1.0) > 1e-6:
            self_sim_ok = False
            findings.append(
                f"cosine(text,text) = {sim} for {rec.entity_id}, "
                f"expected ≈ 1.0"
            )
            break

    if self_sim_ok:
        checks["self_cosine_approx_1"] = "PASS"
    else:
        checks["self_cosine_approx_1"] = "FAIL"

    # ------------------------------------------------------------------
    # 7. Check: same text → same cached vector (re-embed)
    # ------------------------------------------------------------------
    # Re-embed the same texts — should all be cache hits
    re_records = cache.embed_and_cache(
        backend=backend,
        texts_with_ids=texts_with_ids,
    )
    if len(re_records) == 0:
        checks["cache_deterministic"] = "PASS"
    else:
        findings.append(
            f"Re-embedding produced {len(re_records)} new records "
            f"(expected 0 cache hits)"
        )
        checks["cache_deterministic"] = "FAIL"

    # ------------------------------------------------------------------
    # 8. Check: cache reload reproduces same vector
    # ------------------------------------------------------------------
    # Save and reload
    cache.save()
    cache2 = EmbeddingCache()
    cache2.load()

    reload_match = True
    for eid, rec1 in cache.records.items():
        rec2 = cache2.records.get(eid)
        if rec2 is None:
            reload_match = False
            findings.append(f"Record {eid} missing after reload")
            break
        if rec1.vector != rec2.vector:
            reload_match = False
            findings.append(f"Vector mismatch after reload for {eid}")
            break

    if reload_match:
        checks["cache_reload_match"] = "PASS"
    else:
        checks["cache_reload_match"] = "FAIL"

    # ------------------------------------------------------------------
    # 9. Check: model metadata preserved
    # ------------------------------------------------------------------
    all_same_model = all(
        rec.model == identity["model"]
        for rec in cache.records.values()
    )
    all_same_provider = all(
        rec.provider == identity["provider"]
        for rec in cache.records.values()
    )
    all_same_dim = all(
        rec.dimension == identity["dimensions"]
        for rec in cache.records.values()
    )

    if all_same_model and all_same_provider and all_same_dim:
        checks["metadata_preserved"] = "PASS"
    else:
        findings.append("Model metadata mismatch in cached records")
        checks["metadata_preserved"] = "FAIL"

    # ------------------------------------------------------------------
    # 10. Save manifest
    # ------------------------------------------------------------------
    cache.save()
    freeze_manifest = load_global_freeze_manifest()

    manifest = cache.save_manifest(
        backend=backend,
        candidate_count=len(texts),
        target_count=0,
        alias_count=0,
        failed_count=0,
        code_commit=_get_code_commit(),
        frozen_corpus_sha=freeze_manifest.get(
            "frozen_corpus_manifest_sha256", ""
        ),
        global_annotation_freeze_sha=freeze_manifest.get(
            "global_annotation_freeze_sha256", ""
        ),
    )

    # ------------------------------------------------------------------
    # 11. Build smoke result
    # ------------------------------------------------------------------
    smoke_pass = all(v == "PASS" for v in checks.values())

    result: dict = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": _get_code_commit(),
        "embedding_identity": identity,
        "smoke_subset_size": len(texts),
        "cache_size": cache.size,
        "checks": checks,
        "smoke_pass": smoke_pass,
        "blocking_findings": findings,
        "provenance": backend.get_provenance(),
    }

    _SMOKE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_SMOKE_PATH, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    n_pass = sum(1 for v in checks.values() if v == "PASS")
    n_fail = sum(1 for v in checks.values() if v == "FAIL")
    print(f"\nChecks: {n_pass} PASS, {n_fail} FAIL")

    if smoke_pass:
        print("E5-001 SMOKE: GO")
    else:
        print("E5-001 SMOKE: FAIL")
        for finding in findings:
            print(f"  - {finding}")

    return result


def _load_api_key() -> str | None:
    """Load the embedding API key from env or litellm_config.yaml."""
    import os

    key = os.environ.get("EMBEDDING_API_KEY")
    if key:
        return key

    # Try reading from litellm_config.yaml
    config_path = _PROJECT_ROOT / "litellm_config.yaml"
    if config_path.exists():
        import yaml

        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        model_list = cfg.get("model_list", [])
        for entry in model_list:
            params = entry.get("litellm_params", {})
            api_key = params.get("api_key")
            if api_key:
                return api_key

    return None


def main() -> None:
    """Entry point."""
    api_key = _load_api_key()

    result = run_smoke(api_key=api_key)

    if result["smoke_pass"]:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
