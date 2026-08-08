"""E2-E: pilot labeling pipeline.

Applies the frozen independent oracle (``classify_released_target_exposure``)
to each of the 90 raw pilot generation attempts.  The labeling is
deterministic and scenario-aware:

1. Load ``raw_generation_attempts.jsonl`` from the pilot output directory.
2. For each attempt, look up the ``EmpiricalTargetSpec`` by
   ``secret_variant_id``.
3. Derive ``target_type`` from ``scenario_id``.
4. Run the frozen oracle classifier on ``candidate_text``.
5. Record the exposure label and whether it is a positive disclosure.

Outputs:
- ``labeled_pilot_attempts.jsonl``: one JSON line per attempt with the
  original fields plus ``exposure_label``, ``is_positive_exposure``,
  and ``target_type``.
- ``labeling_report.json``: summary statistics (counts by label, by
  scenario, by trust level, positive exposure rates).

Checklist coverage:
- E2-011: frozen independent oracle labeling of all pilot attempts
- E2-012: deterministic, reproducible labels (no LLM calls)
- E2-013: provenance chain from raw attempts → labeled attempts
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.empirical_corpus import (  # noqa: E402
    EMPIRICAL_PROTOCOL_VERSION,
    EMPIRICAL_STUDY_VERSION,
    EmpiricalTargetSpec,
    get_target_spec,
)
from experiments.trustparadox_u.exposure_ontology import (  # noqa: E402
    POSITIVE_DISCLOSURE_LABELS,
)
from experiments.trustparadox_u.runner import (  # noqa: E402
    classify_released_target_exposure,
)

# Map scenario_id → target_type for the classifier.
_SCENARIO_TARGET_TYPE: dict[str, str] = {
    "credential_001": "credential",
    "private_attribute_001": "private_attribute",
    "authorization_001": "authorization",
}


def _derive_target_type(scenario_id: str) -> str:
    """Derive target_type from scenario_id."""
    if scenario_id in _SCENARIO_TARGET_TYPE:
        return _SCENARIO_TARGET_TYPE[scenario_id]
    # Fallback: strip trailing _NNN suffix.
    prefix = scenario_id.rsplit("_", 1)[0]
    if prefix in _SCENARIO_TARGET_TYPE.values():
        return prefix
    raise ValueError(f"Unknown scenario_id: {scenario_id!r}")


def label_pilot_attempt(
    attempt: dict[str, Any],
    spec: EmpiricalTargetSpec,
    target_type: str,
) -> dict[str, Any]:
    """Label a single pilot attempt using the frozen oracle.

    Returns a new dict with all original fields plus labeling fields.
    """
    candidate_text = attempt.get("candidate_text", "")
    attack_type = attempt.get("attack_type", "direct_disclosure")

    # Skip classification for failed attempts — label as "none".
    if attempt.get("generation_status") != "success":
        exposure_label = "none"
    elif attempt.get("refusal", False):
        exposure_label = "none"
    else:
        exposure_label = classify_released_target_exposure(
            released_text=candidate_text,
            canonical_target=spec.canonical_target,
            aliases=list(spec.aliases),
            semantic_variants=list(spec.semantic_descriptions),
            permitted_residuals=list(spec.permitted_residuals),
            target_type=target_type,
            attack_type=attack_type,
        )

    is_positive = exposure_label in POSITIVE_DISCLOSURE_LABELS

    labeled = dict(attempt)
    labeled["exposure_label"] = exposure_label
    labeled["is_positive_exposure"] = is_positive
    labeled["target_type"] = target_type
    labeled["labeling_oracle"] = "classify_released_target_exposure"
    labeled["labeling_timestamp"] = datetime.now(timezone.utc).isoformat()
    return labeled


def run_labeling(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Run the labeling pipeline on pilot attempts.

    Args:
        input_dir: Directory containing raw_generation_attempts.jsonl.
        output_dir: Directory to write labeled outputs.

    Returns:
        Labeling report dictionary.
    """
    raw_path = input_dir / "raw_generation_attempts.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw attempts not found: {raw_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load raw attempts.
    attempts: list[dict[str, Any]] = []
    with open(raw_path) as f:
        for line in f:
            line = line.strip()
            if line:
                attempts.append(json.loads(line))

    # Label each attempt.
    labeled: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    positive_by_scenario: Counter[str] = Counter()
    total_by_scenario: Counter[str] = Counter()
    positive_by_trust: Counter[str] = Counter()
    total_by_trust: Counter[str] = Counter()

    for attempt in attempts:
        variant_id = attempt["secret_variant_id"]
        scenario_id = attempt["scenario_id"]
        trust_level = attempt.get("trust_level", "unknown")
        target_type = _derive_target_type(scenario_id)
        spec = get_target_spec(variant_id)

        labeled_attempt = label_pilot_attempt(attempt, spec, target_type)
        labeled.append(labeled_attempt)

        label = labeled_attempt["exposure_label"]
        is_pos = labeled_attempt["is_positive_exposure"]
        label_counts[label] += 1
        total_by_scenario[scenario_id] += 1
        total_by_trust[trust_level] += 1
        if is_pos:
            positive_by_scenario[scenario_id] += 1
            positive_by_trust[trust_level] += 1

    # Write labeled attempts.
    labeled_path = output_dir / "labeled_pilot_attempts.jsonl"
    with open(labeled_path, "w") as f:
        for item in labeled:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Compute summary statistics.
    total = len(labeled)
    total_positive = sum(1 for item in labeled if item["is_positive_exposure"])
    exposure_rate = total_positive / total if total > 0 else 0.0

    # Per-scenario rates.
    scenario_rates: dict[str, dict[str, Any]] = {}
    for scenario in sorted(total_by_scenario):
        n = total_by_scenario[scenario]
        pos = positive_by_scenario.get(scenario, 0)
        scenario_rates[scenario] = {
            "total": n,
            "positive": pos,
            "rate": round(pos / n, 4) if n > 0 else 0.0,
        }

    # Per-trust-level rates.
    trust_rates: dict[str, dict[str, Any]] = {}
    for trust in sorted(total_by_trust):
        n = total_by_trust[trust]
        pos = positive_by_trust.get(trust, 0)
        trust_rates[trust] = {
            "total": n,
            "positive": pos,
            "rate": round(pos / n, 4) if n > 0 else 0.0,
        }

    # Cross-tabulation: scenario × trust.
    cross_tab: dict[str, dict[str, dict[str, Any]]] = {}
    for attempt in labeled:
        sc = attempt["scenario_id"]
        tr = attempt.get("trust_level", "unknown")
        if sc not in cross_tab:
            cross_tab[sc] = {}
        if tr not in cross_tab[sc]:
            cross_tab[sc][tr] = {"total": 0, "positive": 0}
        cross_tab[sc][tr]["total"] += 1
        if attempt["is_positive_exposure"]:
            cross_tab[sc][tr]["positive"] += 1

    # Finalize cross-tab rates.
    cross_tab_rates: dict[str, dict[str, Any]] = {}
    for sc in sorted(cross_tab):
        cross_tab_rates[sc] = {}
        for tr in sorted(cross_tab[sc]):
            entry = cross_tab[sc][tr]
            n = entry["total"]
            pos = entry["positive"]
            cross_tab_rates[sc][tr] = {
                "total": n,
                "positive": pos,
                "rate": round(pos / n, 4) if n > 0 else 0.0,
            }

    report: dict[str, Any] = {
        "labeling_oracle": "classify_released_target_exposure",
        "protocol_version": EMPIRICAL_PROTOCOL_VERSION,
        "study_version": EMPIRICAL_STUDY_VERSION,
        "labeling_timestamp": datetime.now(timezone.utc).isoformat(),
        "input_file": str(raw_path),
        "output_file": str(labeled_path),
        "total_attempts": total,
        "total_positive_exposures": total_positive,
        "overall_exposure_rate": round(exposure_rate, 4),
        "label_distribution": dict(sorted(label_counts.items())),
        "positive_exposure_labels": sorted(POSITIVE_DISCLOSURE_LABELS),
        "by_scenario": scenario_rates,
        "by_trust_level": trust_rates,
        "by_scenario_and_trust": cross_tab_rates,
    }

    # Write report.
    report_path = output_dir / "labeling_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def main() -> None:
    """CLI entry point for E2-E labeling pipeline."""
    parser = argparse.ArgumentParser(
        description="E2-E: Label pilot generation attempts",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing raw_generation_attempts.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write labeled outputs",
    )
    args = parser.parse_args()

    print("E2-E: Pilot Labeling Pipeline")
    print(f"  Input:  {args.input_dir}")
    print(f"  Output: {args.output_dir}")

    report = run_labeling(args.input_dir, args.output_dir)

    print(f"\n  Total attempts: {report['total_attempts']}")
    print(f"  Positive exposures: {report['total_positive_exposures']}")
    print(f"  Overall exposure rate: {report['overall_exposure_rate']:.4f}")
    print("\n  Label distribution:")
    for label, count in sorted(report["label_distribution"].items()):
        print(f"    {label}: {count}")
    print("\n  By scenario:")
    for scenario, stats in sorted(report["by_scenario"].items()):
        print(f"    {scenario}: {stats['positive']}/{stats['total']} " f"({stats['rate']:.4f})")
    print("\n  By trust level:")
    for trust, stats in sorted(report["by_trust_level"].items()):
        print(f"    {trust}: {stats['positive']}/{stats['total']} " f"({stats['rate']:.4f})")
    print(f"\n  Report written to {args.output_dir / 'labeling_report.json'}")
    print(f"  Labeled attempts written to " f"{args.output_dir / 'labeled_pilot_attempts.jsonl'}")


if __name__ == "__main__":
    main()
