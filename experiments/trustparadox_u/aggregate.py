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
    SmokeManifest,
    validate_manifest_against_results,
)
from experiments.trustparadox_u.paths import (
    EPISODE_RESULTS_FILENAME,
    LEGACY_EPISODE_RESULTS_FILENAME,
)
from experiments.trustparadox_u.runner import EpisodeResult
from experiments.trustparadox_u.serialization import (
    load_episode_results,
    load_smoke_manifest,
)


class AggregationError(Exception):
    """Base exception for aggregation errors."""


class ResultLoadError(AggregationError):
    """Error loading episode results."""


class ManifestValidationError(AggregationError):
    """Error validating manifest against results."""


class ResultAuditError(AggregationError):
    """Error auditing results."""


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
) -> None:
    """Write all aggregation output files to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # metrics.json
    metrics_dict = evaluation.to_dict()
    (output_dir / "metrics.json").write_text(json.dumps(metrics_dict, indent=2, sort_keys=True))

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
        json.dumps(metric_counts, indent=2, sort_keys=True)
    )

    # summary.json
    summary_dict = {
        "metrics": metrics_dict,
        "result_count": len(results),
        "episode_ids": sorted({r.episode_id for r in results}),
        "manifest": manifest.to_dict(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_dict, indent=2, sort_keys=True))

    # audit_report.json
    (output_dir / "audit_report.json").write_text(
        json.dumps(audit_report.to_dict(), indent=2, sort_keys=True)
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
        json.dumps(utility_pairing, indent=2, sort_keys=True)
    )
    (output_dir / "unmatched_pairs.json").write_text(
        json.dumps(unmatched, indent=2, sort_keys=True)
    )

    # summary.md
    summary_data = {"default": metrics_dict}
    md = format_table(summary_data, title="Aggregation Summary")
    (output_dir / "summary.md").write_text(md)


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
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    try:
        # 1. Load episode results
        episodes_path = locate_episode_results(input_dir)
        try:
            results = load_episode_results(episodes_path)
        except (TypeError, ValueError) as exc:
            raise ResultLoadError(f"Failed to load {episodes_path}: {exc}") from exc

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

        # 3. Audit results
        audit_report = audit_results(results)
        if audit_report.has_errors:
            error_count = len(audit_report.findings)
            raise ResultAuditError(f"Audit failed with {error_count} errors")

        # 4. Validate manifest against results
        if manifest is not None:
            validate_manifest_or_raise(manifest, results)

        # 5. Validate for aggregation
        validate_for_aggregation(results)

        # 6. Aggregate
        evaluation = evaluate_all(results)

        # 7. Write outputs
        if diagnostic_only:
            # Write limited diagnostic outputs
            output_dir.mkdir(parents=True, exist_ok=True)
            metrics_dict = evaluation.to_dict()
            (output_dir / "metrics.json").write_text(
                json.dumps(metrics_dict, indent=2, sort_keys=True)
            )
            (output_dir / "diagnostic_warning.txt").write_text(
                "DIAGNOSTIC ONLY: No manifest was provided. "
                "These results are not publication-ready.\n"
            )
        else:
            assert manifest is not None, "manifest must be set in non-diagnostic mode"
            write_aggregation_outputs(
                output_dir=output_dir,
                results=results,
                evaluation=evaluation,
                manifest=manifest,
                audit_report=audit_report,
            )

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
