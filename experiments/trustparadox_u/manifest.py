"""Smoke manifest generation for reproducible experiment provenance."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.trustparadox_u.providers import sanitize_api_base


@dataclass(frozen=True)
class SmokeManifest:
    """Sanitized provenance record for a smoke run."""

    repository_commit: str
    generated_at_utc: str
    run_mode: str
    config_hashes: tuple[str, ...]
    provider: str | None
    model: str | None
    dimension: int | None
    semantic_threshold: float
    api_base_sanitized: str | None
    episode_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    result_count: int
    audit_valid: bool
    audit_error_count: int
    metric_counts: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serialisable dict."""
        d = asdict(self)
        return d

    def to_json(self, *, indent: int = 2) -> str:
        """Deterministic JSON serialisation."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def get_repository_commit() -> str:
    """Return the current HEAD commit SHA, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "unknown"


def build_manifest(
    *,
    results: list[Any],
    run_mode: str,
    config_hashes: list[str],
    provider: str | None = None,
    model: str | None = None,
    dimension: int | None = None,
    semantic_threshold: float = 0.8,
    api_base: str | None = None,
    audit_valid: bool = True,
    audit_error_count: int = 0,
    metric_counts: dict[str, dict[str, int]] | None = None,
) -> SmokeManifest:
    """Build a sanitised smoke manifest from run results."""
    episode_ids = tuple(sorted({r.episode_id for r in results}))
    seeds = tuple(sorted({r.seed for r in results}))

    return SmokeManifest(
        repository_commit=get_repository_commit(),
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        run_mode=run_mode,
        config_hashes=tuple(sorted(set(config_hashes))),
        provider=provider,
        model=model,
        dimension=dimension,
        semantic_threshold=semantic_threshold,
        api_base_sanitized=sanitize_api_base(api_base),
        episode_ids=episode_ids,
        seeds=seeds,
        result_count=len(results),
        audit_valid=audit_valid,
        audit_error_count=audit_error_count,
        metric_counts=metric_counts or {},
    )


def save_manifest(manifest: SmokeManifest, output_path: str | Path) -> Path:
    """Write manifest to disk and return the path."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(manifest.to_json())
    return p
