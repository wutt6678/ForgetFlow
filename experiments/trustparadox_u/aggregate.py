"""Aggregate experiment results into summary tables."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from experiments.trustparadox_u.audit_results import (
    AuditReport,
    audit_results,
    validate_for_aggregation,
)
from experiments.trustparadox_u.evaluator import (
    EvalMetrics,
    compute_utility_retention,
    evaluate_all,
)
from experiments.trustparadox_u.manifest import (
    FULL_SHA_RE,
    SmokeManifest,
    resolve_commit_sha,
    validate_manifest_against_results,
)
from experiments.trustparadox_u.paths import (
    EPISODE_RESULTS_FILENAME,
    LEGACY_EPISODE_RESULTS_FILENAME,
)
from experiments.trustparadox_u.runner import EpisodeResult
from experiments.trustparadox_u.serialization import (
    RESULT_SCHEMA_VERSION,
    inspect_result_schema_versions,
    load_episode_results,
    load_smoke_manifest,
    parse_schema_version,
)


class AggregationError(Exception):
    """Base exception for aggregation errors."""


class ResultLoadError(AggregationError):
    """Error loading episode results."""


class ManifestValidationError(AggregationError):
    """Error validating manifest against results."""


class StaleArtifactError(AggregationError):
    """Error when artifact commit does not match expected commit."""


class SchemaCompatibilityError(AggregationError):
    """Error when result schema is incompatible with requested mode."""


class ResultAuditError(AggregationError):
    """Error auditing results."""


def _dummy_manifest() -> SmokeManifest:
    """Create a placeholder manifest for diagnostic no-manifest mode."""
    return SmokeManifest(
        repository_commit="unknown",
        generated_at_utc="",
        run_mode="diagnostic",
        config_hashes=(),
        provider=None,
        model=None,
        dimension=None,
        semantic_threshold=0.0,
        api_base_sanitized=None,
        episode_ids=(),
        seeds=(),
        result_count=0,
        audit_valid=True,
        audit_error_count=0,
        metric_counts={},
    )


def format_metric(metric: dict[str, Any]) -> str:
    """Format a structured metric dict for display.

    Returns 'N/A' when value is None, otherwise a 3-decimal string.
    """
    value = metric["value"]
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def metric_value(metric: dict[str, Any]) -> float | None:
    """Extract the scalar value from a metric dict."""
    v = metric["value"]
    if v is None:
        return None
    return float(v)


def load_results(input_dir: str | Path) -> list[dict[str, Any]]:
    p = Path(input_dir)
    results = []
    for fp in sorted(p.glob("*.json")):
        with open(fp) as f:
            results.append(json.load(f))
    return results


def aggregate_summary(
    variant_results: dict[str, list[EpisodeResult]],
    allow_errors: bool = False,
) -> dict[str, dict[str, Any]]:
    """Compute per-variant evaluation metrics.

    Each value is a dict with keys like 'pu_rer', 'crr', etc., where
    each value is itself a dict with 'value', 'numerator', 'denominator',
    and 'reason' (from MetricValue.to_dict()).

    Validates results before aggregation. Raises InvalidExperimentResults
    if validation fails and allow_errors is False.
    """
    summary: dict[str, dict[str, Any]] = {}
    for variant, results in variant_results.items():
        # Validate results before aggregation
        validate_for_aggregation(results, allow_errors=allow_errors)
        metrics = evaluate_all(results)
        summary[variant] = metrics.to_dict()
    return summary


def format_table(summary: dict[str, dict[str, Any]], title: str = "Results") -> str:
    """Format a Markdown table from aggregated summary.

    Handles both structured metric dicts (with 'value' key) and plain floats.
    """
    lines = [f"\n## {title}\n"]
    lines.append("| Variant | PU-RER | CRR | RR | FBR |")
    lines.append("|---|---:|---:|---:|---:|")
    for variant, metrics in sorted(summary.items()):
        pu = _format_metric_field(metrics, "pu_rer")
        crr = _format_metric_field(metrics, "crr")
        rr = _format_metric_field(metrics, "rr")
        fbr = _format_metric_field(metrics, "fbr")
        lines.append(f"| {variant} | {pu} | {crr} | {rr} | {fbr} |")
    return "\n".join(lines)


def _format_metric_field(metrics: dict[str, Any], key: str) -> str:
    """Format a single metric from the summary, handling structured and plain forms."""
    m = metrics[key]
    if isinstance(m, dict):
        return format_metric(m)
    # Legacy plain float
    if m is None:
        return "N/A"
    return f"{m:.3f}"


def format_extended_table(summary: dict[str, dict[str, Any]], title: str = "Results") -> str:
    """Format a Markdown table with numerator/denominator columns."""
    lines = [f"\n## {title}\n"]
    header = "| Variant | PU-RER | PU-RER num | PU-RER den | CRR | CRR num | CRR den |"
    lines.append(header)
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for variant, metrics in sorted(summary.items()):
        pu = _get_metric_dict(metrics, "pu_rer")
        crr = _get_metric_dict(metrics, "crr")
        pu_str = format_metric(pu)
        crr_str = format_metric(crr)
        lines.append(
            f"| {variant} | {pu_str} | {pu['numerator']} | {pu['denominator']} "
            f"| {crr_str} | {crr['numerator']} | {crr['denominator']} |"
        )
    return "\n".join(lines)


def _get_metric_dict(metrics: dict[str, Any], key: str) -> dict[str, Any]:
    """Get a metric as a dict, handling both structured and plain forms."""
    m = metrics[key]
    if isinstance(m, dict):
        return m
    # Legacy plain float — wrap in dict
    return {"value": m, "numerator": 0, "denominator": 0, "reason": None}


def write_aggregation_outputs(
    output_dir: Path,
    results: list[EpisodeResult],
    evaluation: EvalMetrics,
    manifest: SmokeManifest,
    audit_report: AuditReport,
    provenance_meta: dict[str, Any] | None = None,
) -> None:
    """Write all aggregation output files to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    prov = provenance_meta or {}

    # Minimal provenance block for embedding in standalone outputs
    provenance_block = {
        "artifact_commit": prov.get("artifact_commit", ""),
        "artifact_dirty": prov.get("artifact_dirty", False),
        "expected_commit": prov.get("expected_commit", ""),
        "historical": prov.get("historical", False),
        "diagnostic": prov.get("diagnostic", False),
        "validation_mode": prov.get("validation_mode", "unknown"),
        "release_certifying": prov.get("release_certifying", False),
    }

    # metrics.json
    metrics_dict = evaluation.to_dict()
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {"artifact_provenance": provenance_block, "metrics": metrics_dict},
            indent=2,
            sort_keys=True,
        )
    )

    # metric_counts.json
    metric_counts = {
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
    (output_dir / "metric_counts.json").write_text(
        json.dumps(
            {"artifact_provenance": provenance_block, "counts": metric_counts},
            indent=2,
            sort_keys=True,
        )
    )

    # summary.json
    summary_dict = {
        "metrics": metrics_dict,
        "result_count": len(results),
        "episode_ids": sorted({r.episode_id for r in results}),
        "manifest": manifest.to_dict(),
        "artifact_provenance": prov,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_dict, indent=2, sort_keys=True))

    # audit_report.json
    (output_dir / "audit_report.json").write_text(
        json.dumps(
            {"artifact_provenance": provenance_block, **audit_report.to_dict()},
            indent=2,
            sort_keys=True,
        )
    )

    # utility_pairing.json and unmatched_pairs.json
    fw_results = [r for r in results if r.metadata.get("firewall_enabled", True)]
    no_fw_results = [r for r in results if not r.metadata.get("firewall_enabled", True)]

    utility_pairing: dict[str, Any]
    unmatched: dict[str, Any]

    if fw_results and no_fw_results:
        utility_result = compute_utility_retention(fw_results, no_fw_results)
        utility_pairing = {
            "matched_keys": [list(k) for k in utility_result.matched_keys],
            "metric": utility_result.metric.to_dict(),
        }
        unmatched = {
            "unmatched_firewall_keys": [list(k) for k in utility_result.unmatched_firewall_keys],
            "unmatched_baseline_keys": [list(k) for k in utility_result.unmatched_baseline_keys],
        }
    else:
        utility_pairing = {"matched_keys": [], "metric": None}
        unmatched = {"unmatched_firewall_keys": [], "unmatched_baseline_keys": []}

    (output_dir / "utility_pairing.json").write_text(
        json.dumps(
            {"artifact_provenance": provenance_block, **utility_pairing},
            indent=2,
            sort_keys=True,
        )
    )
    (output_dir / "unmatched_pairs.json").write_text(
        json.dumps(
            {"artifact_provenance": provenance_block, **unmatched},
            indent=2,
            sort_keys=True,
        )
    )

    # aggregation_manifest.json
    schema_versions = sorted({r.schema_version for r in results})
    agg_manifest = {
        "artifact_provenance": prov,
        "result_schema_versions": schema_versions,
        "outputs": {
            "metrics": "metrics.json",
            "metric_counts": "metric_counts.json",
            "audit_report": "audit_report.json",
            "summary": "summary.json",
            "utility_pairing": "utility_pairing.json",
            "unmatched_pairs": "unmatched_pairs.json",
        },
    }
    (output_dir / "aggregation_manifest.json").write_text(
        json.dumps(agg_manifest, indent=2, sort_keys=True)
    )

    # summary.md
    summary_data = {"default": metrics_dict}
    md_lines: list[str] = []
    if prov.get("historical"):
        md_lines.append(
            f"> **Historical artifact analysis.** "
            f"> These results were produced by commit "
            f"> `{prov.get('artifact_commit', 'unknown')}` and do not "
            f"> validate the current implementation.\n"
        )
    if prov.get("diagnostic"):
        md_lines.append(
            "> **Diagnostic artifact analysis.** "
            "> Repository commit validation was skipped. "
            "> These results cannot certify a release or experiment SHA.\n"
        )
    if prov.get("validation_mode") == "missing_manifest_diagnostic":
        md_lines.append(
            "> **Diagnostic artifact analysis.** "
            "> No authoritative smoke manifest was available. "
            "> These outputs cannot certify a release or experiment commit.\n"
        )
    md_lines.append(format_table(summary_data, title="Aggregation Summary"))
    (output_dir / "summary.md").write_text("\n".join(md_lines))


def locate_episode_results(input_dir: Path) -> Path:
    """Locate episode results file, supporting canonical and legacy filenames."""
    canonical = input_dir / EPISODE_RESULTS_FILENAME
    if canonical.exists():
        return canonical

    legacy = input_dir / LEGACY_EPISODE_RESULTS_FILENAME
    if legacy.exists():
        import warnings

        warnings.warn(
            f"{LEGACY_EPISODE_RESULTS_FILENAME} is deprecated; "
            f"rename it to {EPISODE_RESULTS_FILENAME}",
            DeprecationWarning,
            stacklevel=2,
        )
        return legacy

    raise FileNotFoundError(
        f"No episode results found in {input_dir}. "
        f"Expected {EPISODE_RESULTS_FILENAME} or {LEGACY_EPISODE_RESULTS_FILENAME}"
    )


def validate_manifest_or_raise(
    manifest: SmokeManifest,
    results: list[EpisodeResult],
) -> None:
    """Validate manifest against results, raising on any mismatch."""
    findings = validate_manifest_against_results(manifest, results)
    if findings:
        codes = [f["code"] for f in findings]
        raise ManifestValidationError(f"Manifest validation failed: {', '.join(codes)}")


def validate_commit_provenance(
    manifest: SmokeManifest,
    *,
    expected_commit: str | None = None,
    require_current_commit: bool = False,
    allow_historical: bool = False,
    skip_check: bool = False,
) -> dict[str, Any]:
    """Validate that the manifest's commit matches the expected or current commit.

    Returns a provenance metadata dict. Raises StaleArtifactError if the commit
    doesn't match and historical mode is off.

    Default behavior: require current clean commit match.
    """
    from experiments.trustparadox_u.manifest import (
        get_repository_commit,
        parse_repository_provenance,
    )

    artifact_prov = parse_repository_provenance(manifest.repository_commit)

    provenance_meta: dict[str, Any] = {
        "artifact_commit": artifact_prov.commit,
        "artifact_dirty": artifact_prov.dirty,
        "historical": False,
        "diagnostic": False,
        "validation_mode": "strict",
        "release_certifying": False,
    }

    if skip_check:
        provenance_meta["validation_mode"] = "diagnostic_skipped"
        provenance_meta["diagnostic"] = True
        return provenance_meta

    # Determine expected commit
    if require_current_commit or (expected_commit is None and not allow_historical):
        current_raw = get_repository_commit()
        current_prov = parse_repository_provenance(current_raw)
        expected_prov = current_prov
        provenance_meta["expected_commit"] = current_prov.commit
        provenance_meta["expected_dirty"] = current_prov.dirty
    elif expected_commit is not None:
        # Resolve short SHA to full 40-char SHA before comparison
        try:
            expected_full_sha = resolve_commit_sha(expected_commit)
        except ValueError as exc:
            raise StaleArtifactError(
                f"Could not resolve expected commit {expected_commit!r}: {exc}"
            ) from exc
        expected_prov = parse_repository_provenance(expected_full_sha)
        provenance_meta["expected_commit"] = expected_prov.commit
        provenance_meta["expected_dirty"] = expected_prov.dirty
        provenance_meta["validation_mode"] = "expected_commit"
    else:
        # allow_historical without expected
        provenance_meta["validation_mode"] = "historical_override"
        provenance_meta["historical"] = True
        return provenance_meta

    # Compare
    if artifact_prov.commit == expected_prov.commit and not artifact_prov.dirty:
        # Clean match
        if expected_prov.dirty:
            # Current is dirty, artifact is clean -> can't certify
            provenance_meta["validation_mode"] = "dirty_checkout"
            provenance_meta["release_certifying"] = False
            if not allow_historical:
                raise StaleArtifactError(
                    f"Current checkout is dirty ({expected_prov.commit}-dirty). "
                    f"Cannot certify release. Use --allow-historical-artifacts."
                )
            provenance_meta["historical"] = True
        else:
            # Require full 40-char SHA for release certification
            if not FULL_SHA_RE.match(artifact_prov.commit):
                raise StaleArtifactError(
                    f"Release-certifying manifests must contain a full 40-character "
                    f"Git commit SHA, got {artifact_prov.commit!r} "
                    f"({len(artifact_prov.commit)} chars)."
                )
            provenance_meta["release_certifying"] = True
        return provenance_meta

    # Mismatch or dirty artifact
    if allow_historical:
        provenance_meta["validation_mode"] = "historical_override"
        provenance_meta["historical"] = True
        return provenance_meta

    raise StaleArtifactError(
        f"Artifact commit mismatch: manifest has {manifest.repository_commit!r}, "
        f"expected {expected_prov.raw!r}. "
        f"Use --allow-historical-artifacts to analyze older artifacts."
    )


def main() -> int:
    """Run the aggregation CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate TrustParadox-U experiment results")
    parser.add_argument("--input", required=True, help="Input directory with episodes.jsonl")
    parser.add_argument("--output", required=True, help="Output directory for aggregation results")
    parser.add_argument(
        "--allow-missing-manifest",
        action="store_true",
        help="Allow aggregation without a manifest (diagnostic only)",
    )
    parser.add_argument(
        "--expected-commit",
        default=None,
        help="Require manifest commit to match this SHA",
    )
    parser.add_argument(
        "--require-current-commit",
        action="store_true",
        help="Require manifest commit to match the current HEAD",
    )
    parser.add_argument(
        "--allow-historical-artifacts",
        action="store_true",
        help="Allow artifacts from a different commit (marked historical)",
    )
    parser.add_argument(
        "--skip-commit-check",
        action="store_true",
        help="Skip commit provenance check (diagnostic only)",
    )
    args = parser.parse_args()

    # Default: require current commit unless explicitly overridden
    if (
        args.expected_commit is None
        and not args.allow_historical_artifacts
        and not args.skip_commit_check
    ):
        args.require_current_commit = True

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    try:
        # 1. Locate result and manifest files
        episodes_path = locate_episode_results(input_dir)

        # 2. Load manifest
        manifest_path = input_dir / "smoke_manifest.json"
        diagnostic_only = False
        if manifest_path.exists():
            manifest_data = load_smoke_manifest(manifest_path)
            manifest = SmokeManifest(**manifest_data)
        elif args.allow_missing_manifest:
            print(
                "WARNING: No manifest found. Running in diagnostic-only mode.",
                file=sys.stderr,
            )
            diagnostic_only = True
            manifest = None
        else:
            raise ManifestValidationError(
                f"Manifest not found: {manifest_path}. "
                "Use --allow-missing-manifest for diagnostic-only mode."
            )

        # 3. Validate commit provenance (before auditing)
        provenance_meta: dict[str, Any] = {}
        if manifest is not None:
            provenance_meta = validate_commit_provenance(
                manifest,
                expected_commit=args.expected_commit,
                require_current_commit=args.require_current_commit,
                allow_historical=args.allow_historical_artifacts,
                skip_check=args.skip_commit_check,
            )
            if provenance_meta.get("historical"):
                print(
                    f"WARNING: HISTORICAL artifact analysis. "
                    f"Manifest commit {provenance_meta.get('artifact_commit')} "
                    f"does not validate the current implementation.",
                    file=sys.stderr,
                )

        # 4. Inspect result envelope schema versions
        try:
            schema_versions = inspect_result_schema_versions(episodes_path)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ResultLoadError(f"Failed to inspect {episodes_path}: {exc}") from exc
        is_historical_mode = bool(
            args.allow_historical_artifacts or provenance_meta.get("historical")
        )

        # 5. Validate schema compatibility
        current = parse_schema_version(RESULT_SCHEMA_VERSION)
        for sv in schema_versions:
            sv_parsed = parse_schema_version(sv)
            if sv_parsed == current:
                # Exact current schema — supported
                pass
            elif sv_parsed > current:
                # Future schema — not implemented, reject
                raise SchemaCompatibilityError(
                    f"Future result schema {sv!r} is unsupported; "
                    f"current schema is {RESULT_SCHEMA_VERSION!r}"
                )
            elif is_historical_mode or diagnostic_only:
                # Legacy schema in historical/diagnostic mode — allowed
                pass
            else:
                raise SchemaCompatibilityError(
                    f"Result schema {sv!r} is not supported for release certification. "
                    f"Required: {RESULT_SCHEMA_VERSION}. "
                    f"Use --allow-historical-artifacts for legacy analysis."
                )

        # 5b. Update provenance for legacy schema content
        if schema_versions:
            contains_legacy_schema = any(
                parse_schema_version(version) < current for version in schema_versions
            )
            if contains_legacy_schema:
                provenance_meta["historical"] = True
                provenance_meta["release_certifying"] = False
                if provenance_meta.get("validation_mode") not in (
                    "historical_override",
                    "diagnostic_skipped",
                    "missing_manifest_diagnostic",
                ):
                    provenance_meta["validation_mode"] = "historical_diagnostic"

        # 6. Deserialize results
        try:
            results = load_episode_results(episodes_path)
        except (TypeError, ValueError) as exc:
            raise ResultLoadError(f"Failed to load {episodes_path}: {exc}") from exc

        # 7. Audit results
        audit_report = audit_results(results)
        if audit_report.has_errors:
            error_count = len(audit_report.errors())
            warning_count = len(audit_report.warnings())
            raise ResultAuditError(
                f"Audit failed with {error_count} error(s) and {warning_count} warning(s)"
            )

        # 8. Validate manifest against results
        if manifest is not None:
            validate_manifest_or_raise(manifest, results)

        # 9. Validate for aggregation
        validate_for_aggregation(results)

        # 10. Aggregate
        evaluation = evaluate_all(results)

        # 11. Write outputs
        if diagnostic_only:
            # Build diagnostic provenance for no-manifest mode
            diagnostic_provenance: dict[str, Any] = {
                "artifact_commit": None,
                "artifact_dirty": None,
                "expected_commit": None,
                "historical": False,
                "diagnostic": True,
                "validation_mode": "missing_manifest_diagnostic",
                "release_certifying": False,
            }
            provenance_meta = diagnostic_provenance
            write_aggregation_outputs(
                output_dir=output_dir,
                results=results,
                evaluation=evaluation,
                manifest=manifest or _dummy_manifest(),
                audit_report=audit_report,
                provenance_meta=provenance_meta,
            )
        else:
            assert manifest is not None, "manifest must be set in non-diagnostic mode"
            write_aggregation_outputs(
                output_dir=output_dir,
                results=results,
                evaluation=evaluation,
                manifest=manifest,
                audit_report=audit_report,
                provenance_meta=provenance_meta,
            )

    except StaleArtifactError as exc:
        print(f"Aggregation failed [STALE_ARTIFACT]: {exc}", file=sys.stderr)
        return 6
    except SchemaCompatibilityError as exc:
        print(f"Aggregation failed [SCHEMA]: {exc}", file=sys.stderr)
        return 7
    except FileNotFoundError as exc:
        print(f"Aggregation failed [INPUT_MISSING]: {exc}", file=sys.stderr)
        return 2
    except ResultLoadError as exc:
        print(f"Aggregation failed [RESULT_LOAD]: {exc}", file=sys.stderr)
        return 3
    except ResultAuditError as exc:
        print(f"Aggregation failed [AUDIT]: {exc}", file=sys.stderr)
        return 4
    except ManifestValidationError as exc:
        print(f"Aggregation failed [MANIFEST]: {exc}", file=sys.stderr)
        return 5
    except AggregationError as exc:
        print(f"Aggregation failed: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Aggregation failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
