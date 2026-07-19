"""Smoke manifest generation for reproducible experiment provenance."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Regex for valid git commit SHA (7-40 hex characters)
COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")

# Full 40-character SHA for portable artifact certification
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class RepositoryProvenance:
    """Parsed repository commit with dirty-tree flag."""

    commit: str
    dirty: bool

    @property
    def raw(self) -> str:
        """Return the raw string representation."""
        return f"{self.commit}-dirty" if self.dirty else self.commit


def parse_repository_provenance(value: str) -> RepositoryProvenance:
    """Parse a repository commit string into a RepositoryProvenance."""
    dirty = value.endswith("-dirty")
    commit = value.removesuffix("-dirty") if dirty else value

    if not COMMIT_RE.match(commit):
        raise ValueError(f"Invalid repository commit: {value!r}")

    return RepositoryProvenance(commit=commit, dirty=dirty)


def resolve_commit_sha(value: str) -> str:
    """Resolve a commit string to a full 40-character SHA.

    If the value is already a full SHA, return it.
    If it's a short SHA, resolve via git rev-parse.
    Raises ValueError for malformed or ambiguous values.
    """
    # Strip dirty suffix
    commit = value.removesuffix("-dirty")

    if FULL_SHA_RE.match(commit):
        return commit

    if COMMIT_RE.match(commit):
        # Short SHA — resolve via git
        try:
            result = subprocess.run(
                ["git", "rev-parse", commit + "^{commit}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                resolved = result.stdout.strip()
                if FULL_SHA_RE.match(resolved):
                    return resolved
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        raise ValueError(
            f"Short SHA {commit!r} could not be resolved to a full 40-character SHA. "
            f"Use --expected-commit with a full SHA for portable certification."
        )

    raise ValueError(f"Invalid commit SHA: {value!r}")


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
    embedding_threshold: float
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


def get_repository_commit(*, reject_dirty: bool = False) -> str:
    """Return the current HEAD commit SHA, or 'unknown'.

    Args:
        reject_dirty: If True, raise RuntimeError when working tree is dirty.
                      If False, append '-dirty' suffix to the commit SHA.

    Returns:
        The commit SHA, optionally with '-dirty' suffix.

    Raises:
        RuntimeError: If reject_dirty=True and working tree is dirty.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return "unknown"
        commit = result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "unknown"

    # Check for dirty working tree
    try:
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        dirty = bool(status_result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        dirty = False

    if dirty:
        if reject_dirty:
            raise RuntimeError("Publication experiment requires a clean working tree")
        return f"{commit}-dirty"

    return commit


def require_single_metadata_value(
    results: list[Any],
    field: str,
    *,
    allow_none: bool = False,
) -> Any:
    """Extract a single consistent metadata value from all results.

    Args:
        results: List of EpisodeResult objects.
        field: Metadata field name to extract.
        allow_none: If True, None values are allowed when all results have None.
                   If False, None values cause an error.

    Returns:
        The single consistent value.

    Raises:
        ValueError: If results is empty, multiple inconsistent values are found,
                   or None is present when allow_none=False.
    """
    if not results:
        raise ValueError("Cannot derive manifest metadata from an empty result set")

    values = {result.metadata.get(field) for result in results}

    # Handle all-None case
    if values == {None}:
        if allow_none:
            return None
        raise ValueError(f"Metadata field {field!r} cannot be null")

    # Check for mixed None and non-None values
    if None in values:
        raise ValueError(f"Metadata field {field!r} is missing from some results")

    # Check for multiple different non-None values
    if len(values) != 1:
        raise ValueError(f"Expected one value for {field!r}, found {values!r}")

    return next(iter(values))


def collect_config_hashes(results: list[Any]) -> tuple[str, ...]:
    """Collect and validate config hashes from results.

    Rejects missing, empty, or non-string config_hash values.

    Args:
        results: List of EpisodeResult objects.

    Returns:
        Sorted tuple of unique config hash strings.

    Raises:
        ValueError: If any result has a missing, empty, or non-string config_hash.
    """
    hashes: set[str] = set()

    for result in results:
        value = result.metadata.get("config_hash")

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Episode {result.episode_id} has no valid config_hash")

        hashes.add(value)

    return tuple(sorted(hashes))


def build_manifest(
    *,
    results: list[Any],
    audit_valid: bool = True,
    audit_error_count: int = 0,
    metric_counts: dict[str, dict[str, int]] | None = None,
    reject_dirty: bool = False,
    repository_commit: str | None = None,
) -> SmokeManifest:
    """Build a sanitised smoke manifest from run results.

    All provenance metadata is derived from the completed results to ensure
    the manifest describes what actually executed, not what was configured.

    Args:
        results: List of EpisodeResult objects from the run.
        audit_valid: Whether the audit passed validation.
        audit_error_count: Number of audit errors found.
        metric_counts: Metric numerator/denominator counts.
        reject_dirty: If True, raise RuntimeError when working tree is dirty.
        repository_commit: Pre-resolved repository commit. If None, will be resolved.

    Returns:
        A SmokeManifest derived from the results.

    Raises:
        ValueError: If results are empty or have inconsistent metadata.
        RuntimeError: If reject_dirty=True and working tree is dirty.
    """
    if not results:
        raise ValueError("Cannot build manifest from empty result set")

    # Derive all metadata from results
    run_mode = str(require_single_metadata_value(results, "run_mode"))
    embedding_enabled = bool(require_single_metadata_value(results, "embedding_enabled"))

    # For semantic-disabled runs, allow None for embedding fields
    provider = require_single_metadata_value(
        results, "embedding_provider", allow_none=not embedding_enabled
    )
    model = require_single_metadata_value(
        results, "embedding_model", allow_none=not embedding_enabled
    )
    dimension = require_single_metadata_value(
        results, "embedding_dimension", allow_none=not embedding_enabled
    )

    # Semantic threshold is always present in metadata but may be None when semantic is disabled
    embedding_threshold_raw = require_single_metadata_value(
        results, "embedding_threshold", allow_none=not embedding_enabled
    )
    embedding_threshold = (
        float(embedding_threshold_raw) if embedding_threshold_raw is not None else 0.0
    )

    # API base is optional
    api_base_sanitized = require_single_metadata_value(
        results, "api_base_sanitized", allow_none=True
    )

    # Derive config hashes from results (with validation)
    config_hashes = collect_config_hashes(results)

    # Derive episode IDs and seeds from results
    episode_ids = tuple(sorted({r.episode_id for r in results}))
    seeds = tuple(sorted({r.seed for r in results}))

    # Validate semantic experiment requirements
    if run_mode == "experiment" and embedding_enabled:
        if provider is None or provider == "fixed":
            raise ValueError(
                f"Semantic experiment requires a real embedding provider, got {provider!r}"
            )
        if model is None:
            raise ValueError("Semantic experiment requires an embedding model")
        if dimension is None or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(
                f"Semantic experiment requires a positive integer dimension, got {dimension!r}"
            )
        if not isinstance(embedding_threshold, (int, float)):
            raise ValueError(
                f"Semantic experiment requires a numeric threshold, got {embedding_threshold!r}"
            )

    return SmokeManifest(
        repository_commit=repository_commit
        if repository_commit is not None
        else get_repository_commit(reject_dirty=reject_dirty),
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        run_mode=run_mode,
        config_hashes=config_hashes,
        provider=provider,
        model=model,
        dimension=dimension,
        embedding_threshold=embedding_threshold,
        api_base_sanitized=api_base_sanitized,
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
        findings.append(
            {
                "code": "MANIFEST_RESULT_COUNT_MISMATCH",
                "message": f"Manifest says {manifest.result_count} results, got {len(results)}",
            }
        )

    # 2. Check episode IDs match
    actual_episode_ids = sorted({r.episode_id for r in results})
    if list(manifest.episode_ids) != actual_episode_ids:
        findings.append(
            {
                "code": "MANIFEST_EPISODE_IDS_MISMATCH",
                "message": f"Episode IDs don't match: manifest={manifest.episode_ids}, actual={actual_episode_ids}",
            }
        )

    # 3. Check seeds match
    actual_seeds = sorted({r.seed for r in results})
    if list(manifest.seeds) != actual_seeds:
        findings.append(
            {
                "code": "MANIFEST_SEEDS_MISMATCH",
                "message": f"Seeds don't match: manifest={manifest.seeds}, actual={actual_seeds}",
            }
        )

    # 4. Check audit status is true
    if not manifest.audit_valid:
        findings.append(
            {
                "code": "MANIFEST_AUDIT_INVALID",
                "message": f"Manifest reports audit invalid with {manifest.audit_error_count} errors",
            }
        )

    # 5. Check repository commit is valid
    if manifest.repository_commit == "unknown":
        findings.append(
            {
                "code": "MANIFEST_UNKNOWN_COMMIT",
                "message": "Repository commit is 'unknown'",
            }
        )
    elif not COMMIT_RE.match(manifest.repository_commit.replace("-dirty", "")):
        findings.append(
            {
                "code": "MANIFEST_INVALID_COMMIT",
                "message": f"Repository commit '{manifest.repository_commit}' is not a valid SHA",
            }
        )

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
            "rr_clean": {
                "numerator": evaluation.rr_clean.numerator,
                "denominator": evaluation.rr_clean.denominator,
            },
            "rr_at_risk": {
                "numerator": evaluation.rr_at_risk.numerator,
                "denominator": evaluation.rr_at_risk.denominator,
            },
            "fbr": {
                "numerator": evaluation.fbr.numerator,
                "denominator": evaluation.fbr.denominator,
            },
        }
        if manifest.metric_counts != expected_counts:
            findings.append(
                {
                    "code": "MANIFEST_METRIC_COUNTS_MISMATCH",
                    "message": "Metric counts don't match evaluator output",
                }
            )

    # 7. Check config hashes match (using strict validation)
    try:
        actual_config_hashes = list(collect_config_hashes(results))
    except ValueError as exc:
        findings.append(
            {
                "code": "MANIFEST_CONFIG_HASHES_INVALID",
                "message": str(exc),
            }
        )
        actual_config_hashes = None
    if actual_config_hashes is not None and list(manifest.config_hashes) != actual_config_hashes:
        findings.append(
            {
                "code": "MANIFEST_CONFIG_HASHES_MISMATCH",
                "message": f"Config hashes don't match: manifest={manifest.config_hashes}, actual={actual_config_hashes}",
            }
        )

    # 8. Check run mode matches
    actual_run_modes = {r.metadata.get("run_mode") for r in results}
    if len(actual_run_modes) > 1:
        findings.append(
            {
                "code": "MANIFEST_MULTIPLE_RUN_MODES",
                "message": f"Results have multiple run modes: {actual_run_modes}",
            }
        )
    elif actual_run_modes and manifest.run_mode not in actual_run_modes:
        findings.append(
            {
                "code": "MANIFEST_RUN_MODE_MISMATCH",
                "message": f"Run mode doesn't match: manifest={manifest.run_mode}, actual={actual_run_modes.pop()}",
            }
        )

    # 9. Check provider matches
    actual_providers = {r.metadata.get("embedding_provider") for r in results}
    if len(actual_providers) > 1:
        findings.append(
            {
                "code": "MANIFEST_MULTIPLE_PROVIDERS",
                "message": f"Results have multiple providers: {actual_providers}",
            }
        )
    elif actual_providers and manifest.provider not in actual_providers:
        findings.append(
            {
                "code": "MANIFEST_PROVIDER_MISMATCH",
                "message": f"Provider doesn't match: manifest={manifest.provider}, actual={actual_providers.pop()}",
            }
        )

    # 10. Check model matches
    actual_models = {r.metadata.get("embedding_model") for r in results}
    if len(actual_models) > 1:
        findings.append(
            {
                "code": "MANIFEST_MULTIPLE_MODELS",
                "message": f"Results have multiple models: {actual_models}",
            }
        )
    elif actual_models and manifest.model not in actual_models:
        findings.append(
            {
                "code": "MANIFEST_MODEL_MISMATCH",
                "message": f"Model doesn't match: manifest={manifest.model}, actual={actual_models.pop()}",
            }
        )

    # 11. Check dimension matches
    actual_dimensions = {r.metadata.get("embedding_dimension") for r in results}
    if len(actual_dimensions) > 1:
        findings.append(
            {
                "code": "MANIFEST_MULTIPLE_DIMENSIONS",
                "message": f"Results have multiple dimensions: {actual_dimensions}",
            }
        )
    elif actual_dimensions and manifest.dimension not in actual_dimensions:
        findings.append(
            {
                "code": "MANIFEST_DIMENSION_MISMATCH",
                "message": f"Dimension doesn't match: manifest={manifest.dimension}, actual={actual_dimensions.pop()}",
            }
        )

    # 12. Check semantic threshold matches
    actual_thresholds = {r.metadata.get("embedding_threshold") for r in results}
    # Skip threshold check when semantic is disabled (all thresholds are None)
    semantic_disabled = actual_thresholds == {None}
    if len(actual_thresholds) > 1:
        findings.append(
            {
                "code": "MANIFEST_MULTIPLE_THRESHOLDS",
                "message": f"Results have multiple thresholds: {actual_thresholds}",
            }
        )
    elif (
        not semantic_disabled
        and actual_thresholds
        and manifest.embedding_threshold not in actual_thresholds
    ):
        findings.append(
            {
                "code": "MANIFEST_THRESHOLD_MISMATCH",
                "message": f"Threshold doesn't match: manifest={manifest.embedding_threshold}, actual={actual_thresholds.pop()}",
            }
        )

    # 13. Check endpoint provenance
    actual_endpoints = {r.metadata.get("api_base_sanitized") for r in results}
    if len(actual_endpoints) > 1:
        findings.append(
            {
                "code": "MANIFEST_MULTIPLE_ENDPOINTS",
                "message": f"Results have multiple endpoints: {actual_endpoints}",
            }
        )
    elif actual_endpoints and manifest.api_base_sanitized not in actual_endpoints:
        findings.append(
            {
                "code": "MANIFEST_ENDPOINT_MISMATCH",
                "message": f"Endpoint doesn't match: manifest={manifest.api_base_sanitized}, actual={actual_endpoints.pop()}",
            }
        )

    return findings


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class StudyManifest:
    """Whole-study manifest covering all conditions and outputs.

    References per-condition child manifests and hashes all output files.
    """

    repository_commit: str
    artifact_dirty: bool
    result_schema_versions: tuple[str, ...]
    result_count: int
    condition_manifests: dict[str, str]
    output_files: dict[str, str]
    output_hashes: dict[str, str]
    audit_valid: bool
    manifest_valid: bool
    release_certifying: bool
    generated_at_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serialisable dict."""
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        """Deterministic JSON serialisation."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def build_study_manifest(
    *,
    repository_commit: str,
    artifact_dirty: bool,
    result_schema_versions: tuple[str, ...],
    result_count: int,
    condition_manifests: dict[str, str],
    output_dir: Path,
    audit_valid: bool,
    manifest_valid: bool,
    release_certifying: bool,
) -> StudyManifest:
    """Build a study-level manifest from completed run outputs.

    Computes SHA-256 hashes for all output files in the directory.
    """
    output_files: dict[str, str] = {}
    output_hashes: dict[str, str] = {}

    # Exclude study manifest files from self-hashing (circular dependency)
    excluded_files = {"study_manifest.json", "study_manifest_validation.json"}

    for fpath in sorted(output_dir.iterdir()):
        if fpath.is_file() and fpath.name not in excluded_files:
            rel = fpath.name
            output_files[rel] = rel
            output_hashes[rel] = sha256_file(fpath)

    # Also hash files in manifests/ subdirectory
    manifests_dir = output_dir / "manifests"
    if manifests_dir.is_dir():
        for fpath in sorted(manifests_dir.iterdir()):
            if fpath.is_file():
                rel = f"manifests/{fpath.name}"
                output_files[rel] = rel
                output_hashes[rel] = sha256_file(fpath)

    return StudyManifest(
        repository_commit=repository_commit,
        artifact_dirty=artifact_dirty,
        result_schema_versions=result_schema_versions,
        result_count=result_count,
        condition_manifests=condition_manifests,
        output_files=output_files,
        output_hashes=output_hashes,
        audit_valid=audit_valid,
        manifest_valid=manifest_valid,
        release_certifying=release_certifying,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def validate_study_manifest(
    manifest: StudyManifest,
    output_dir: Path,
) -> list[dict[str, str]]:
    """Validate a study manifest against the output directory.

    Returns a list of findings. Empty list means validation passed.
    """
    findings: list[dict[str, str]] = []

    # 1. Check repository commit
    commit_clean = manifest.repository_commit.replace("-dirty", "")
    if not COMMIT_RE.match(commit_clean):
        findings.append(
            {
                "code": "STUDY_INVALID_COMMIT",
                "message": f"Invalid commit: {manifest.repository_commit!r}",
            }
        )

    # 2. Check dirty flag for certification
    if manifest.release_certifying and manifest.artifact_dirty:
        findings.append(
            {
                "code": "STUDY_DIRTY_ARTIFACT",
                "message": "Release-certifying manifest has dirty artifact flag",
            }
        )

    # 3. Check result count
    if manifest.result_count <= 0:
        findings.append(
            {
                "code": "STUDY_NO_RESULTS",
                "message": f"Result count is {manifest.result_count}",
            }
        )

    # 4. Check schema versions
    if not manifest.result_schema_versions:
        findings.append(
            {
                "code": "STUDY_NO_SCHEMA",
                "message": "No schema versions recorded",
            }
        )

    # 5. Check condition manifests exist
    for cond_name, cond_path in manifest.condition_manifests.items():
        full_path = output_dir / cond_path
        if not full_path.exists():
            findings.append(
                {
                    "code": "STUDY_MISSING_CONDITION_MANIFEST",
                    "message": f"Missing condition manifest: {cond_name} at {cond_path}",
                }
            )

    # 6. Check output files exist and hashes match
    for file_key, file_rel in manifest.output_files.items():
        full_path = output_dir / file_rel
        if not full_path.exists():
            findings.append(
                {
                    "code": "STUDY_MISSING_OUTPUT",
                    "message": f"Missing output file: {file_key}",
                }
            )
        elif file_key in manifest.output_hashes:
            actual_hash = sha256_file(full_path)
            expected_hash = manifest.output_hashes[file_key]
            if actual_hash != expected_hash:
                findings.append(
                    {
                        "code": "STUDY_HASH_MISMATCH",
                        "message": f"Hash mismatch for {file_key}: expected {expected_hash[:16]}..., got {actual_hash[:16]}...",
                    }
                )

    # 7. Check audit valid
    if not manifest.audit_valid:
        findings.append(
            {
                "code": "STUDY_AUDIT_INVALID",
                "message": "Study manifest records audit as invalid",
            }
        )

    # 8. Check manifest valid
    if not manifest.manifest_valid:
        findings.append(
            {
                "code": "STUDY_MANIFEST_INVALID",
                "message": "Study manifest records child manifest validation as invalid",
            }
        )

    return findings
