"""Aggregate experiment results into summary tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.trustparadox_u.audit_results import validate_for_aggregation
from experiments.trustparadox_u.evaluator import evaluate_all
from experiments.trustparadox_u.runner import EpisodeResult


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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(f"Aggregate results from {args.input} to {args.output}")
