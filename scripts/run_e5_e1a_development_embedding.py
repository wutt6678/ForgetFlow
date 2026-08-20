#!/usr/bin/env python3
"""E5-E1a: Development embedding + detector feature generation campaign.

Strict invariants:
    - 225/225 development candidates embedded + features computed
    - All required canonical targets and aliases embedded
    - Real frozen embedding model (text-embedding-v3)
    - Complete embedding provenance
    - 0 test embeddings
    - 0 test features
    - test_access_started remains false
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.e5_loaders import load_corpus
from experiments.trustparadox_u.e5_conditions import require_test_access_started, TestAccessError
from experiments.trustparadox_u.embedding_backend import EmbeddingBackend, create_backend
from experiments.trustparadox_u.embedding_cache import EmbeddingCache
from experiments.trustparadox_u.semantic_detector import (
    build_target_index,
    run_feature_extraction,
    load_features,
)


def _get_api_key() -> str:
    """Read API key from litellm_config.yaml."""
    import yaml  # type: ignore[import-untyped]

    config_path = _PROJECT_ROOT / "litellm_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"LiteLLM config not found: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    for model_entry in config.get("model_list", []):
        params = model_entry.get("litellm_params", {})
        if "api_key" in params:
            return params["api_key"]
    raise ValueError("No api_key found in litellm_config.yaml")


def _get_commit_sha() -> str:
    """Get current HEAD commit SHA."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _verify_test_access_not_started() -> None:
    """Verify test_access_started is still false."""
    try:
        require_test_access_started()
        # If we get here, test access HAS been started — that's a violation
        raise RuntimeError(
            "E5-E1a INVARIANT VIOLATION: test_access_started is true. "
            "Development embedding campaign must not run after test access."
        )
    except TestAccessError:
        # This is the expected outcome — test access has NOT started
        pass


def main() -> None:
    print("=" * 70)
    print("E5-E1a: Development Embedding + Detector Feature Campaign")
    print("=" * 70)

    # --- Pre-flight: verify test access not started ---
    print("\n[1/7] Verifying test_access_started = false ...")
    _verify_test_access_not_started()
    print("  OK: test_access_started is false")

    # --- Load API key ---
    print("\n[2/7] Loading API key from litellm_config.yaml ...")
    api_key = _get_api_key()
    print(f"  OK: API key loaded ({len(api_key)} chars)")

    # --- Get commit SHA ---
    commit_sha = _get_commit_sha()
    print(f"  Commit: {commit_sha}")

    # --- Verify corpus ---
    print("\n[3/7] Verifying development corpus ...")
    dev_corpus = load_corpus("development")
    n_dev = len(dev_corpus)
    print(f"  Development candidates: {n_dev}")
    assert n_dev == 225, f"Expected 225 development candidates, got {n_dev}"

    # --- Verify target resolution ---
    print("\n[4/7] Verifying target resolution ...")
    target_index = build_target_index()
    dev_targets = {
        key: spec for key, spec in target_index.items()
        if spec.split == "development"
    }
    print(f"  Development targets: {len(dev_targets)}")
    for key, spec in sorted(dev_targets.items()):
        n_aliases = len(spec.aliases)
        print(
            f"    {spec.forget_id}: {spec.scenario_id}/{spec.secret_variant_id} "
            f"(canonical={spec.canonical_target!r}, aliases={n_aliases})"
        )

    # Verify all corpus candidates resolve
    for cand in dev_corpus:
        key = (cand.scenario_id, cand.secret_variant_id)
        assert key in target_index, (
            f"Corpus candidate {cand.candidate_id} does not resolve to a target spec"
        )
    print("  OK: all 225 candidates resolve to development targets")

    # --- Create backend and cache ---
    print("\n[5/7] Creating embedding backend + loading cache ...")
    backend = create_backend(api_key=api_key)
    # The Aliyun endpoint limits batch size to 10; override default 64
    backend._batch_size = 10
    cache = EmbeddingCache()
    cache.load()
    print(f"  Existing cache records: {cache.size}")
    identity = backend.embedding_identity()
    print(f"  Model: {identity.model}")
    print(f"  Dimensions: {identity.dimensions}")
    print(f"  Normalization: {identity.normalization}")

    # --- Run feature extraction (development only) ---
    print("\n[6/7] Running development feature extraction ...")
    print("  This will embed all candidate texts + target canonicals + aliases")
    print("  via the real frozen embedding model.")

    manifest = run_feature_extraction(
        backend=backend,
        cache=cache,
        splits=("development",),
        code_commit=commit_sha,
    )

    print(f"\n  Feature extraction complete!")
    print(f"  Total features: {manifest['total_features']}")
    print(f"  Split counts: {manifest['split_counts']}")
    print(f"  Embedding model: {manifest['embedding_model']}")
    print(f"  Total API calls: {manifest.get('total_api_calls', 'N/A')}")

    # --- Post-flight verification ---
    print("\n[7/7] Post-flight verification ...")

    # Verify features were written
    features = load_features("development")
    n_features = len(features)
    print(f"  Development features on disk: {n_features}")
    assert n_features == 225, f"Expected 225 features, got {n_features}"

    # Verify test access still not started
    _verify_test_access_not_started()
    print("  test_access_started: false (verified)")

    # Verify no test embeddings or features
    test_features_path = (
        _PROJECT_ROOT / "results" / "empirical_v2" / "e5"
        / "detector_features" / "test_detector_features.jsonl"
    )
    assert not test_features_path.exists(), "Test feature file exists — invariant violated!"
    print("  Test features: 0 (verified)")

    # --- Update phase state ---
    print("\nUpdating phase state ...")
    phase_path = _PROJECT_ROOT / "results" / "empirical_v2" / "e5" / "e5_phase.json"
    with open(phase_path) as f:
        phase = json.load(f)
    phase["development_embedding_complete"] = True
    phase["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(phase_path, "w") as f:
        json.dump(phase, f, indent=2)
        f.write("\n")
    print("  development_embedding_complete: true")

    # --- Write campaign report ---
    report = {
        "campaign": "E5-E1a",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit": commit_sha,
        "embedding_model": identity.model,
        "embedding_provider": identity.provider,
        "embedding_dimensions": identity.dimensions,
        "embedding_normalization": identity.normalization,
        "config_version": identity.config_version,
        "development_candidates": n_dev,
        "development_features": n_features,
        "development_targets": len(dev_targets),
        "test_embeddings": 0,
        "test_features": 0,
        "test_access_started": False,
        "total_api_calls": manifest.get("total_api_calls"),
        "total_texts_embedded": manifest.get("total_texts_embedded"),
        "cache_sha": manifest.get("cache_sha", ""),
        "features_sha256": manifest.get("features_sha256", ""),
        "status": "COMPLETE",
    }

    report_path = (
        _PROJECT_ROOT / "results" / "empirical_v2" / "e5"
        / "e5_e1a_campaign_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    print(f"\n{'=' * 70}")
    print("E5-E1a CAMPAIGN COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Development: {n_features}/225 candidates embedded + featured")
    print(f"  Test: 0 embeddings, 0 features")
    print(f"  test_access_started: false")
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
