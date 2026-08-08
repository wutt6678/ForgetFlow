"""E2-F: pilot analysis pipeline.

Analyses the labeled pilot attempts to answer the core research question:
does trust-level framing affect the generating agent's disclosure behaviour?

The pilot has real LLM generation (not deterministic templates), so unlike
Panel B of the frozen-corpus analysis, we CAN evaluate whether trust
manipulation changes behaviour.

Inputs:
  - labeled_pilot_attempts.jsonl (from E2-E)
  - labeling_report.json (from E2-E)

Outputs:
  - pilot_analysis_report.json: per-scenario and overall trust-manipulation
    analysis with exposure rates, rate differences, and directional checks.
  - pilot_analysis_summary.md: human-readable summary.

Checklist coverage:
- E2-014: trust-manipulation analysis on labeled pilot
- E2-015: per-scenario breakdown (never pooled)
- E2-016: directional expectation checks
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.empirical_corpus import (  # noqa: E402
    EMPIRICAL_PROTOCOL_VERSION,
    EMPIRICAL_STUDY_VERSION,
)

TRUST_LEVELS: tuple[str, ...] = ("low", "default", "high")


def _load_labeled_attempts(path: Path) -> list[dict[str, Any]]:
    """Load labeled pilot attempts from JSONL."""
    attempts: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                attempts.append(json.loads(line))
    return attempts


def _compute_rates(
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute exposure rates grouped by scenario and trust level.

    Returns a dict with:
    - overall: total, positive, rate
    - by_scenario: {scenario: {total, positive, rate}}
    - by_trust: {trust: {total, positive, rate}}
    - by_scenario_trust: {scenario: {trust: {total, positive, rate}}}
    """
    total = len(attempts)
    total_pos = sum(1 for a in attempts if a["is_positive_exposure"])

    by_scenario: dict[str, dict[str, Any]] = {}
    by_trust: dict[str, dict[str, Any]] = {}
    by_scenario_trust: dict[str, dict[str, dict[str, Any]]] = {}

    # Group by scenario.
    scenario_groups: dict[str, list[dict[str, Any]]] = {}
    for a in attempts:
        sc = a["scenario_id"]
        scenario_groups.setdefault(sc, []).append(a)

    for sc in sorted(scenario_groups):
        group = scenario_groups[sc]
        pos = sum(1 for a in group if a["is_positive_exposure"])
        by_scenario[sc] = {
            "total": len(group),
            "positive": pos,
            "rate": round(pos / len(group), 4) if group else 0.0,
        }

        # Sub-group by trust within scenario.
        trust_groups: dict[str, list[dict[str, Any]]] = {}
        for a in group:
            tr = a.get("trust_level", "unknown")
            trust_groups.setdefault(tr, []).append(a)

        by_scenario_trust[sc] = {}
        for tr in sorted(trust_groups):
            tg = trust_groups[tr]
            tp = sum(1 for a in tg if a["is_positive_exposure"])
            by_scenario_trust[sc][tr] = {
                "total": len(tg),
                "positive": tp,
                "rate": round(tp / len(tg), 4) if tg else 0.0,
            }

    # Group by trust (across scenarios).
    trust_groups_all: dict[str, list[dict[str, Any]]] = {}
    for a in attempts:
        tr = a.get("trust_level", "unknown")
        trust_groups_all.setdefault(tr, []).append(a)

    for tr in sorted(trust_groups_all):
        tg = trust_groups_all[tr]
        pos = sum(1 for a in tg if a["is_positive_exposure"])
        by_trust[tr] = {
            "total": len(tg),
            "positive": pos,
            "rate": round(pos / len(tg), 4) if tg else 0.0,
        }

    return {
        "overall": {
            "total": total,
            "positive": total_pos,
            "rate": round(total_pos / total, 4) if total else 0.0,
        },
        "by_scenario": by_scenario,
        "by_trust": by_trust,
        "by_scenario_trust": by_scenario_trust,
    }


def _directional_checks(
    by_scenario_trust: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Check directional expectations for trust manipulation.

    For each scenario, compare exposure rates across trust levels.
    Report:
    - rate_spread: max(rate) - min(rate) across trust levels
    - low_vs_high: rate(low) - rate(high)
    - monotonic: whether rates are monotonically ordered
    - direction: "increasing" (low < high), "decreasing" (low > high),
      or "non-monotonic"
    """
    checks: dict[str, dict[str, Any]] = {}

    for sc in sorted(by_scenario_trust):
        rates = by_scenario_trust[sc]
        low_rate = rates.get("low", {}).get("rate", 0.0)
        default_rate = rates.get("default", {}).get("rate", 0.0)
        high_rate = rates.get("high", {}).get("rate", 0.0)

        all_rates = [low_rate, default_rate, high_rate]
        spread = max(all_rates) - min(all_rates)

        diff_low_high = low_rate - high_rate

        # Determine direction.
        if low_rate < high_rate:
            direction = "increasing"
        elif low_rate > high_rate:
            direction = "decreasing"
        else:
            direction = "flat"

        # Check monotonicity.
        is_monotonic_inc = low_rate <= default_rate <= high_rate
        is_monotonic_dec = low_rate >= default_rate >= high_rate
        monotonic = is_monotonic_inc or is_monotonic_dec

        checks[sc] = {
            "low_rate": low_rate,
            "default_rate": default_rate,
            "high_rate": high_rate,
            "rate_spread": round(spread, 4),
            "low_minus_high": round(diff_low_high, 4),
            "direction": direction,
            "monotonic": monotonic,
        }

    return checks


def _label_distribution(
    attempts: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """Compute label distribution by scenario."""
    dist: dict[str, dict[str, int]] = {}
    for a in attempts:
        sc = a["scenario_id"]
        label = a.get("exposure_label", "none")
        if sc not in dist:
            dist[sc] = {}
        dist[sc][label] = dist[sc].get(label, 0) + 1
    return {sc: dict(sorted(labels.items())) for sc, labels in sorted(dist.items())}


def run_analysis(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Run the pilot analysis pipeline.

    Args:
        input_dir: Directory containing labeled_pilot_attempts.jsonl.
        output_dir: Directory to write analysis outputs.

    Returns:
        Analysis report dictionary.
    """
    labeled_path = input_dir / "labeled_pilot_attempts.jsonl"
    if not labeled_path.exists():
        raise FileNotFoundError(f"Labeled attempts not found: {labeled_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    attempts = _load_labeled_attempts(labeled_path)
    rates = _compute_rates(attempts)
    directional = _directional_checks(rates["by_scenario_trust"])
    label_dist = _label_distribution(attempts)

    # Count refusal and provider_error.
    refusal_count = sum(1 for a in attempts if a.get("refusal", False))
    error_count = sum(1 for a in attempts if a.get("generation_status") == "provider_error")

    report: dict[str, Any] = {
        "analysis_type": "pilot_trust_manipulation",
        "protocol_version": EMPIRICAL_PROTOCOL_VERSION,
        "study_version": EMPIRICAL_STUDY_VERSION,
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        "input_file": str(labeled_path),
        "total_attempts": len(attempts),
        "refusal_count": refusal_count,
        "provider_error_count": error_count,
        "exposure_rates": rates,
        "directional_checks": directional,
        "label_distribution_by_scenario": label_dist,
    }

    # Write report.
    report_path = output_dir / "pilot_analysis_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Write human-readable summary.
    summary_path = output_dir / "pilot_analysis_summary.md"
    _write_summary(report, summary_path)

    return report


def _write_summary(report: dict[str, Any], path: Path) -> None:
    """Write a human-readable Markdown summary."""
    lines: list[str] = []
    lines.append("# E2-F: Pilot Trust-Manipulation Analysis")
    lines.append("")
    lines.append(f"**Protocol version:** {report['protocol_version']}")
    lines.append(f"**Analysis timestamp:** {report['analysis_timestamp']}")
    lines.append(f"**Total attempts:** {report['total_attempts']}")
    lines.append(f"**Refusals:** {report['refusal_count']}")
    lines.append(f"**Provider errors:** {report['provider_error_count']}")
    lines.append("")

    # Overall rate.
    overall = report["exposure_rates"]["overall"]
    lines.append("## Overall Exposure Rate")
    lines.append("")
    lines.append(
        f"- **{overall['positive']}/{overall['total']}** "
        f"({overall['rate']:.1%}) positive exposures"
    )
    lines.append("")

    # By scenario.
    lines.append("## By Scenario")
    lines.append("")
    lines.append("| Scenario | Total | Positive | Rate |")
    lines.append("|----------|-------|----------|------|")
    for sc, stats in sorted(report["exposure_rates"]["by_scenario"].items()):
        lines.append(f"| {sc} | {stats['total']} | {stats['positive']} " f"| {stats['rate']:.1%} |")
    lines.append("")

    # By scenario × trust.
    lines.append("## By Scenario × Trust Level")
    lines.append("")
    for sc in sorted(report["exposure_rates"]["by_scenario_trust"]):
        lines.append(f"### {sc}")
        lines.append("")
        lines.append("| Trust | Total | Positive | Rate |")
        lines.append("|-------|-------|----------|------|")
        for tr, stats in sorted(report["exposure_rates"]["by_scenario_trust"][sc].items()):
            lines.append(
                f"| {tr} | {stats['total']} | {stats['positive']} " f"| {stats['rate']:.1%} |"
            )
        lines.append("")

    # Directional checks.
    lines.append("## Directional Checks")
    lines.append("")
    for sc, check in sorted(report["directional_checks"].items()):
        lines.append(f"### {sc}")
        lines.append("")
        lines.append(f"- Low: {check['low_rate']:.1%}")
        lines.append(f"- Default: {check['default_rate']:.1%}")
        lines.append(f"- High: {check['high_rate']:.1%}")
        lines.append(f"- Spread: {check['rate_spread']:.1%}")
        lines.append(f"- Direction: **{check['direction']}**")
        lines.append(f"- Monotonic: {check['monotonic']}")
        lines.append("")

    # Label distribution.
    lines.append("## Label Distribution")
    lines.append("")
    for sc, dist in sorted(report["label_distribution_by_scenario"].items()):
        lines.append(f"### {sc}")
        lines.append("")
        for label, count in sorted(dist.items()):
            lines.append(f"- {label}: {count}")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    """CLI entry point for E2-F analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="E2-F: Pilot trust-manipulation analysis",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing labeled_pilot_attempts.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write analysis outputs",
    )
    args = parser.parse_args()

    print("E2-F: Pilot Analysis Pipeline")
    print(f"  Input:  {args.input_dir}")
    print(f"  Output: {args.output_dir}")

    report = run_analysis(args.input_dir, args.output_dir)

    print(f"\n  Total attempts: {report['total_attempts']}")
    overall = report["exposure_rates"]["overall"]
    print(
        f"  Overall exposure rate: "
        f"{overall['positive']}/{overall['total']} ({overall['rate']:.1%})"
    )
    print("\n  Directional checks:")
    for sc, check in sorted(report["directional_checks"].items()):
        print(
            f"    {sc}: low={check['low_rate']:.1%}, "
            f"default={check['default_rate']:.1%}, "
            f"high={check['high_rate']:.1%} "
            f"→ {check['direction']}"
        )
    print(f"\n  Report: {args.output_dir / 'pilot_analysis_report.json'}")
    print(f"  Summary: {args.output_dir / 'pilot_analysis_summary.md'}")


if __name__ == "__main__":
    main()
