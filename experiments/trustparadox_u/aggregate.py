"""Aggregate experiment results into summary tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.trustparadox_u.evaluator import EvalMetrics, evaluate_all
from experiments.trustparadox_u.runner import EpisodeResult


def load_results(input_dir: str | Path) -> list[dict[str, Any]]:
    p = Path(input_dir)
    results = []
    for fp in sorted(p.glob("*.json")):
        with open(fp) as f:
            results.append(json.load(f))
    return results


def aggregate_summary(
    variant_results: dict[str, list[EpisodeResult]],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for variant, results in variant_results.items():
        metrics = evaluate_all(results)
        summary[variant] = metrics.to_dict()
    return summary


def format_table(summary: dict[str, dict[str, Any]], title: str = "Results") -> str:
    lines = [f"\n## {title}\n"]
    lines.append("| Variant | PU-RER | CRR | RR | FBR |")
    lines.append("|---|---:|---:|---:|---:|")
    for variant, metrics in sorted(summary.items()):
        pu = f"{metrics['pu_rer']:.3f}" if metrics['pu_rer'] is not None else "N/A"
        crr = f"{metrics['crr']:.3f}" if metrics['crr'] is not None else "N/A"
        rr = f"{metrics['rr']:.3f}" if metrics['rr'] is not None else "N/A"
        fbr = f"{metrics['fbr']:.3f}" if metrics['fbr'] is not None else "N/A"
        lines.append(f"| {variant} | {pu} | {crr} | {rr} | {fbr} |")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(f"Aggregate results from {args.input} to {args.output}")
