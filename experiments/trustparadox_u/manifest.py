"""Smoke manifest generation for reproducible experiment provenance."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.trustparadox_u.providers import sanitize_api_base

# Regex for valid git commit SHA (7-40 hex characters)
COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")


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


def validate_manifest_against_results(
    manifest: SmokeManifest,
    results: list[Any],
) -> list[dict[str, str]]:
    """Validate that manifest matches the actual result set.

    Returns a list of findings (dicts with 'code' and 'message' keys).
    Empty list means validation passed.
    """
    findings: list[dict[str, str]] = []

    # 1. Check result_count
    if manifest.result_count != len(results):
        findings.append({
            "code": "MANIFEST_RESULT_COUNT_MISMATCH",
            "message": f"Manifest says {manifest.result_count} results, got {len(results)}",
        })

    # 2. Check episode IDs match
    actual_episode_ids = sorted({r.episode_id for r in results})
    if list(manifest.episode_ids) != actual_episode_ids:
        findings.append({
            "code": "MANIFEST_EPISODE_IDS_MISMATCH",
            "message": f"Episode IDs don't match: manifest={manifest.episode_ids}, actual={actual_episode_ids}",
        })

    # 3. Check seeds match
    actual_seeds = sorted({r.seed for r in results})
    if list(manifest.seeds) != actual_seeds:
        findings.append({
            "code": "MANIFEST_SEEDS_MISMATCH",
            "message": f"Seeds don't match: manifest={manifest.seeds}, actual={actual_seeds}",
        })

    # 4. Check audit status is true
    if not manifest.audit_valid:
        findings.append({
            "code": "MANIFEST_AUDIT_INVALID",
            "message": f"Manifest reports audit invalid with {manifest.audit_error_count} errors",
        })

    # 5. Check repository commit is valid
    if manifest.repository_commit == "unknown":
        findings.append({
            "code": "MANIFEST_UNKNOWN_COMMIT",
            "message": "Repository commit is 'unknown'",
        })
    elif not COMMIT_RE.match(manifest.repository_commit.replace("-dirty", "")):
        findings.append({
            "code": "MANIFEST_INVALID_COMMIT",
            "message": f"Repository commit '{manifest.repository_commit}' is not a valid SHA",
        })

    # 6. Check metric counts match evaluator output
    if manifest.metric_counts:
        from experiments.trustparadox_u.evaluator import evaluate_all

        evaluation = evaluate_all(results)
        expected_counts = {
            "pu_rer": {
                "numerator": evaluation.pu_rer.numerator,
                "denominator": evaluation.pu_rer.denominator,
            },
            "crr": {
                "numerator": evaluation.crr.numerator,
                "denominator": evaluation.crr.denominator,
            },
            "rr": {
                "numerator": evaluation.rr.numerator,
                "denominator": evaluation.rr.denominator,
            },
            "fbr": {
                "numerator": evaluation.fbr.numerator,
                "denominator": evaluation.fbr.denominator,
            },
        }
        if manifest.metric_counts != expected_counts:
            findings.append({
                "code": "MANIFEST_METRIC_COUNTS_MISMATCH",
                "message": "Metric counts don't match evaluator output",
            })

    return findings
