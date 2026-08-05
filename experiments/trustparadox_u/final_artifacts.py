"""Iteration 14: Final artifacts and paper tables.

Aggregates all experiment results into final summary artifacts:
1. Main results table (Table 1): metrics by condition
2. Leakage breakdown table (Table 2): by attack category
3. Parameter sensitivity table (Table 3): from sweep
4. Statistical comparison table (Table 4): paired tests
5. Target-type results table (Table 5): by target type and scenario (§36)
6. Corpus manifest summary
7. Final study manifest

Remediation §35: every table is self-describing — it carries the study
class, condition, split, attack population, target type, metric
definition, numerator, denominator, pairing unit, confidence-interval
source, and sample counts, so a reader can interpret it without opening
the implementation.

Remediation §36: results are reported per target type and scenario
before pooling; pooled rates appear only as secondary summaries.

Exit criterion:
  All tables are generated and internally consistent.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.conditions import REPLAY_CONDITIONS  # noqa: E402
from experiments.trustparadox_u.research_protocol import (  # noqa: E402
    COMPARISONS,
    POPULATION,
    PROTOCOL_VERSION,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parents[2] / "results" / "frozen_replay"
SWEEP_DIR = Path(__file__).parents[2] / "results" / "parameter_sweep"
LEAKAGE_DIR = Path(__file__).parents[2] / "results" / "leakage_analysis"
STATS_DIR = Path(__file__).parents[2] / "results" / "paired_statistics"
CORPUS_DIR = Path(__file__).parents[2] / "data" / "trustparadox_u" / "frozen_corpus"
FINAL_DIR = Path(__file__).parents[2] / "results" / "final_artifacts"

BASELINE_CONDITION = "no_firewall"

# §36: disclosure semantics differ per target type, so results are
# reported per type before any pooling.
TARGET_TYPES: tuple[str, ...] = ("credential", "private_attribute", "authorization")
TARGET_TYPE_METRICS: tuple[str, ...] = ("pu_rer", "crr", "rr", "fbr")

# §35: protocol token that anchors each reported metric to the declared
# comparison whose outcome defines it (word-boundary match).
_PROTOCOL_METRIC_TOKENS: dict[str, str] = {
    "pu_rer": "pu_rer",
    "exposure": "pu_rer",
    "crr": "crr",
    "reconstruction": "crr",
    "rr": "rr",
    "recontamination": "rr",
    "fbr": "false-block",
    "false_block": "false-block",
    "utility_false_block": "false-block",
    "paired_policy_utility_retention": "utility retention",
    "utility": "utility retention",
}


# ---------------------------------------------------------------------------
# §35: self-describing table context and metric definitions
# ---------------------------------------------------------------------------


def build_table_context(**overrides: Any) -> dict[str, Any]:
    """Shared self-describing context for every final table (§35).

    States the study class, split, attack population, conditions and
    protocol version that declared the metrics, so a reader can
    interpret the table without opening the implementation.
    """
    from experiments.trustparadox_u.status import STUDY_CLASS_DIAGNOSTIC

    study_class = STUDY_CLASS_DIAGNOSTIC
    run_manifest_path = RESULTS_DIR / "run_manifest.json"
    if run_manifest_path.exists():
        study_class = str(json.loads(run_manifest_path.read_text()).get("study_class", study_class))
    context: dict[str, Any] = {
        "study_class": study_class,
        "protocol_version": PROTOCOL_VERSION,
        "population": POPULATION,
        "split": (
            "full frozen corpus: deterministic replay runs every split "
            "(development, validation, test) under every condition"
        ),
        "attack_population": (
            "frozen attack candidates: direct_probe, semantic_paraphrase, "
            "multi_step_reconstruction, recontamination_probe, plus "
            "legitimate_task and benign_control utility probes"
        ),
        "conditions": list(REPLAY_CONDITIONS),
        "baseline_condition": BASELINE_CONDITION,
        "pairing_unit": "candidate_id (single-turn) / sequence_id (multi-step)",
        "confidence_intervals": (
            "per-comparison 95% CIs (rate, bootstrap, cluster bootstrap) " "are reported in Table 4"
        ),
    }
    context.update(overrides)
    return context


def metric_definitions(*metrics: str) -> dict[str, dict[str, str]]:
    """§35: definition, numerator, denominator, and units per metric.

    Each definition is anchored to the first declared protocol
    comparison whose outcome names the metric, so the table states
    exactly how each number is computed.
    """
    definitions: dict[str, dict[str, str]] = {}
    for metric in metrics:
        token = _PROTOCOL_METRIC_TOKENS.get(metric, metric)
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])")
        spec = next((c for c in COMPARISONS if pattern.search(c.outcome)), None)
        definitions[metric] = {
            "definition": spec.outcome if spec else "not declared in protocol",
            "numerator": spec.numerator if spec else "",
            "denominator": spec.denominator if spec else "",
            "unit_of_analysis": spec.unit_of_analysis if spec else "",
            "pairing_unit": spec.pairing_unit if spec else "",
            "aggregation_level": spec.aggregation_level if spec else "",
        }
    return definitions


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------


def build_table1_main_results() -> dict[str, Any]:
    """Table 1: Main results by condition.

    §35: pooled rows carry value/numerator/denominator per metric plus
    the self-describing context. §36: this pooled view is the secondary
    summary; per-target-type results in Table 5 are primary.
    """
    metrics_path = RESULTS_DIR / "metrics_by_condition.json"
    if not metrics_path.exists():
        return {"error": "metrics_by_condition.json not found"}

    data = json.loads(metrics_path.read_text())
    rows = []
    for condition, metrics in sorted(data.items()):
        row = {"condition": condition}
        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict):
                row[f"{metric_name}_value"] = metric_data.get("value")
                row[f"{metric_name}_numerator"] = metric_data.get("numerator")
                row[f"{metric_name}_denominator"] = metric_data.get("denominator")
        rows.append(row)

    return {
        "table": "Table 1: Main Results by Condition",
        "context": build_table_context(
            target_type="all pooled",
            trust_level="all pooled",
            aggregation=(
                "Pooled over target types, scenarios, and trust levels: "
                "secondary pooled summary; per-target-type results are "
                "primary (Table 5, §36)"
            ),
        ),
        "metric_definitions": metric_definitions(
            "pu_rer", "crr", "rr", "fbr", "paired_policy_utility_retention"
        ),
        "rows": rows,
    }


def build_table2_leakage_breakdown() -> dict[str, Any]:
    """Table 2: Leakage breakdown by attack type under the no-firewall baseline.

    Reads the FF92-016 leakage analysis: one row per attack type with the
    evaluable exposure (pu_rer), reconstruction (crr), and recontamination
    (rr) rates, so the baseline surface forms and multi-step channels are
    each visible.
    """
    analysis_path = LEAKAGE_DIR / "leakage_analysis.json"
    if not analysis_path.exists():
        return {"error": "leakage_analysis.json not found"}

    data = json.loads(analysis_path.read_text())
    baseline = data.get("by_condition_and_attack", {}).get(BASELINE_CONDITION, {})

    def _rate(breakdown: dict[str, Any], metric: str) -> float:
        value = breakdown.get(metric, {}).get("value")
        return round(value, 4) if value is not None else 0.0

    rows = []
    for attack_type, breakdown in sorted(baseline.items()):
        rows.append(
            {
                "category": attack_type,
                "n_episodes": breakdown.get("candidate_trials", 0),
                "exposure_rate": _rate(breakdown, "pu_rer"),
                "reconstruction_rate": _rate(breakdown, "crr"),
                "recontamination_rate": _rate(breakdown, "rr"),
            }
        )

    return {
        "table": "Table 2: Leakage by Attack Category",
        "context": build_table_context(
            condition=BASELINE_CONDITION,
            target_type="all pooled",
            trust_level="all pooled",
            aggregation="One row per attack population category",
        ),
        "metric_definitions": metric_definitions("pu_rer", "crr", "rr"),
        "rows": rows,
    }


def build_table3_parameter_sensitivity() -> dict[str, Any]:
    """Table 3: Parameter sensitivity from the FF92-018 sweep.

    One row per swept value of each hyperparameter, showing the metrics
    belonging to that parameter's function plus the frozen selection.
    Remediation §30: every row carries the sweep purpose label
    (selection vs sensitivity) and the evaluation split it used.
    """
    summary_path = SWEEP_DIR / "sweep_summary.json"
    if not summary_path.exists():
        return {"error": "sweep_summary.json not found"}

    data = json.loads(summary_path.read_text())

    def _value(metric: dict[str, Any] | None) -> float | None:
        if not metric:
            return None
        value = metric.get("value")
        return round(value, 4) if value is not None else None

    rows = []
    for parameter, sweep in sorted(data.get("sweeps", {}).items()):
        selected = sweep.get("selected_value")
        for point in sweep.get("points", []):
            row: dict[str, Any] = {
                "parameter": parameter,
                "value": point.get("value"),
                "sweep_purpose": sweep.get("sweep_purpose", "selection"),
                "split": sweep.get("split", ""),
                "selected": point.get("value") == selected,
            }
            for metric_name, metric in point.get("metrics", {}).items():
                row[f"{metric_name}_value"] = _value(metric)
            rows.append(row)

    return {
        "table": "Table 3: Parameter Sensitivity",
        "context": build_table_context(
            target_type="all pooled",
            trust_level="all pooled",
            aggregation="One row per swept parameter value",
        ),
        "metric_definitions": metric_definitions(
            "pu_rer", "crr", "fbr", "paired_policy_utility_retention"
        ),
        "rows": rows,
        "num_parameters": len(data.get("sweeps", {})),
    }


def build_table4_statistical_comparisons() -> dict[str, Any]:
    """Table 4: Paired statistical comparisons.

    §35: each row states its pairing unit, paired sample size, rate and
    cluster-bootstrap confidence intervals, so inference is readable
    without opening the statistics implementation.
    """
    stats_path = STATS_DIR / "paired_statistics.json"
    if not stats_path.exists():
        return {"error": "paired_statistics.json not found"}

    data = json.loads(stats_path.read_text())
    comparisons = data.get("comparisons", [])

    rows = []
    seen_metrics: set[str] = set()
    for comp in comparisons:
        metric = str(comp.get("metric", ""))
        seen_metrics.add(metric)
        ci = comp.get("cluster_bootstrap_ci_95")
        rows.append(
            {
                "condition_a": comp.get("condition_a", ""),
                "condition_b": comp.get("condition_b", ""),
                "metric": metric,
                "rate_a": comp.get("rate_a", 0.0),
                "rate_a_ci_95": comp.get("rate_ci_95_a"),
                "rate_b": comp.get("rate_b", 0.0),
                "rate_b_ci_95": comp.get("rate_ci_95_b"),
                "pairing_unit": comp.get("pairing_unit", ""),
                "n_pairs": comp.get("n_pairs"),
                "cluster_bootstrap_ci_95": ci,
                "cohens_h": comp.get("cohens_h", 0.0),
                "p_value": comp.get("mcnemar", {}).get("p_value", 1.0),
                "significant": comp.get("mcnemar", {}).get("p_value", 1.0) < 0.05,
            }
        )

    return {
        "table": "Table 4: Paired Statistical Comparisons",
        "context": build_table_context(
            target_type="all pooled",
            trust_level="all pooled",
            aggregation="One row per paired condition comparison",
        ),
        "metric_definitions": metric_definitions(*sorted(seen_metrics)),
        "rows": rows,
        "num_comparisons": len(rows),
    }


def build_table5_target_type_results() -> dict[str, Any]:
    """Table 5: Results by target type and scenario (§36).

    Per-target-type and per-scenario breakdowns are primary; pooled
    by-condition rows appear only as secondary summaries. A macro-
    average across target types is reported for each condition.
    """
    analysis_path = LEAKAGE_DIR / "leakage_analysis.json"
    if not analysis_path.exists():
        return {"error": "leakage_analysis.json not found"}

    data = json.loads(analysis_path.read_text())
    by_target = data.get("by_condition_and_target_type", {})
    by_scenario = data.get("by_condition_and_scenario", {})
    by_condition = data.get("by_condition", {})

    def _breakdown_row(row_key: dict[str, Any], breakdown: dict[str, Any]) -> dict[str, Any]:
        row: dict[str, Any] = dict(row_key)
        row["sample_count"] = breakdown.get("candidate_trials", 0)
        row["failed_count"] = breakdown.get("failed_trials", 0)
        for metric in TARGET_TYPE_METRICS:
            entry = breakdown.get(metric) or {}
            for field in ("value", "numerator", "denominator", "evaluable"):
                row[f"{metric}_{field}"] = entry.get(field)
        return row

    target_rows = []
    for condition in sorted(by_target):
        for target_type, breakdown in sorted(by_target[condition].items()):
            target_rows.append(
                _breakdown_row({"condition": condition, "target_type": target_type}, breakdown)
            )

    scenario_rows = []
    for condition in sorted(by_scenario):
        for scenario_id, breakdown in sorted(by_scenario[condition].items()):
            scenario_rows.append(
                _breakdown_row({"condition": condition, "scenario_id": scenario_id}, breakdown)
            )

    # §36: macro-average across target types within each condition.
    macro_rows = []
    for condition in sorted(by_target):
        row: dict[str, Any] = {"condition": condition}
        for metric in TARGET_TYPE_METRICS:
            values = [
                breakdown[metric]["value"]
                for breakdown in by_target[condition].values()
                if breakdown.get(metric, {}).get("value") is not None
            ]
            row[f"{metric}_value"] = round(sum(values) / len(values), 4) if values else None
            row[f"{metric}_target_types_averaged"] = len(values)
        macro_rows.append(row)

    # Pooled rates are reported, but only as a secondary summary.
    pooled_rows = []
    for condition, breakdown in sorted(by_condition.items()):
        row = _breakdown_row(
            {
                "condition": condition,
                "role": "secondary_summary",
                "target_type": "all pooled",
            },
            breakdown,
        )
        pooled_rows.append(row)

    def _pu_rer_spread(condition: str) -> tuple[float, float] | None:
        values = [
            breakdown.get("pu_rer", {}).get("value")
            for breakdown in by_target.get(condition, {}).values()
            if breakdown.get("pu_rer", {}).get("value") is not None
        ]
        return (min(values), max(values)) if values else None

    heterogeneity: dict[str, Any] = {}
    for condition in (BASELINE_CONDITION, "full_mvp"):
        spread = _pu_rer_spread(condition)
        if spread is not None:
            heterogeneity[condition] = {
                "pu_rer_min": spread[0],
                "pu_rer_max": spread[1],
            }

    return {
        "table": "Table 5: Results by Target Type and Scenario",
        "context": build_table_context(
            condition="all replay conditions",
            target_type="reported per target type before pooling",
            trust_level="all pooled",
            aggregation=(
                "Per-target-type and per-scenario rows are primary; pooled "
                "by-condition rows carry role=secondary_summary (§36)"
            ),
        ),
        "metric_definitions": metric_definitions(*TARGET_TYPE_METRICS),
        "by_target_type": target_rows,
        "by_scenario": scenario_rows,
        "macro_average_by_target_type": macro_rows,
        "pooled_secondary": pooled_rows,
        "heterogeneity_note": (
            "Exposure (pu_rer) min/max spread across target types per "
            "condition; a wide spread means pooling hides target-type "
            "differences, so no primary conclusion may rest on the pooled "
            "rate alone (§36)."
        ),
        "heterogeneity": heterogeneity,
        "rows": target_rows,
    }


def build_corpus_summary() -> dict[str, Any]:
    """Corpus manifest summary."""
    manifest_path = CORPUS_DIR / "corpus_manifest.json"
    if not manifest_path.exists():
        return {"error": "corpus_manifest.json not found"}

    data = json.loads(manifest_path.read_text())
    return {
        "corpus_hash": data.get("corpus_sha256", ""),
        "total_candidates": data.get("candidate_count", 0),
        "scenarios": data.get("scenarios", {}),
        "attack_types": data.get("attack_type_counts", {}),
        "splits": data.get("split_counts", {}),
    }


def build_annotation_summary() -> dict[str, Any]:
    """Annotation manifest summary."""
    ann_path = CORPUS_DIR / "annotation_manifest.json"
    if not ann_path.exists():
        return {"error": "annotation_manifest.json not found"}

    data = json.loads(ann_path.read_text())
    return {
        "annotation_hash": data.get("annotation_hash", ""),
        "total_annotations": data.get("annotation_count", 0),
        "review_status": data.get("review_status_counts", {}),
        "attack_type_counts": data.get("attack_type_counts", {}),
    }


# ---------------------------------------------------------------------------
# Final study manifest
# ---------------------------------------------------------------------------


def build_study_manifest(
    tables: dict[str, Any],
    corpus_summary: dict[str, Any],
    annotation_summary: dict[str, Any],
) -> dict[str, Any]:
    """Build the final study manifest.

    FF92-023: provenance is snapshotted via the certification helper so the
    manifest carries tested/artifact commits and repository cleanliness.
    FF92-021: exit criteria are derived from the artifacts actually built,
    never hardcoded; formal certification lives in research_valid_gate.json.
    """
    from experiments.trustparadox_u.artifact_provenance import (
        build_certification_provenance,
        code_tree_is_clean,
    )
    from experiments.trustparadox_u.research_protocol import (
        PROTOCOL_VERSION,
        TABLE_QUESTION_MAP,
    )
    from experiments.trustparadox_u.status import (
        STUDY_CLASS_DIAGNOSTIC,
        validate_study_class,
    )

    provenance = build_certification_provenance(repository_clean=code_tree_is_clean())

    # Remediation §4: every artifact records the study class that produced
    # it; the replay run manifest is the source of truth for this study.
    study_class = STUDY_CLASS_DIAGNOSTIC
    replay_manifest = RESULTS_DIR / "run_manifest.json"
    if replay_manifest.exists():
        study_class = str(json.loads(replay_manifest.read_text()).get("study_class", study_class))
    validate_study_class(study_class)

    def _table(prefix: str) -> dict[str, Any]:
        return next(
            (d for d in tables.values() if str(d.get("table", "")).startswith(prefix)),
            {},
        )

    table1 = _table("Table 1")
    table2 = _table("Table 2")
    table3 = _table("Table 3")
    table4 = _table("Table 4")
    table5 = _table("Table 5")

    # §36: per-target-type results must exist before pooling claims.
    target_types_covered = {row.get("target_type") for row in table5.get("rows", [])}
    table5_rows_present = {
        str(row.get("target_type")) for row in table5.get("by_target_type", [])
    } == set(TARGET_TYPES)
    pooled_secondary_only = all(
        row.get("role") == "secondary_summary" for row in table5.get("pooled_secondary", [])
    )

    return {
        "schema_version": "2.0.0",
        "study_name": "TrustParadox-U Primary Study",
        "study_class": study_class,
        # Remediation §2: artifacts are interpretable against the protocol
        # version that declared their questions and table mappings.
        "protocol_version": PROTOCOL_VERSION,
        "table_question_map": {k: list(v) for k, v in TABLE_QUESTION_MAP.items()},
        "repository_commit": provenance["artifact_generation_commit"],
        "provenance": provenance,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus": corpus_summary,
        "annotations": annotation_summary,
        "tables": {
            name: {
                "title": data.get("table", name),
                "num_rows": len(data.get("rows", [])),
            }
            for name, data in tables.items()
        },
        "conditions": list(REPLAY_CONDITIONS),
        "exit_criteria": {
            "all_tables_built": not any("error" in data for data in tables.values()),
            "all_conditions_run": len(table1.get("rows", [])) >= len(REPLAY_CONDITIONS),
            "all_metrics_computed": "error" not in table1 and bool(table1.get("rows")),
            "paired_statistics_available": table4.get("num_comparisons", 0) > 0,
            "leakage_breakdown_available": bool(table2.get("rows")),
            "parameter_sweep_complete": table3.get("num_parameters", 0) > 0,
            "target_type_results_reported": (
                bool(table5.get("rows"))
                and table5_rows_present
                and table5.get("macro_average_by_target_type")
                and pooled_secondary_only
                and len(target_types_covered) >= len(TARGET_TYPES)
            ),
        },
        "certification": "see research_valid_gate.json for the research-valid verdict",
    }


# ---------------------------------------------------------------------------
# Markdown table formatter
# ---------------------------------------------------------------------------


def format_table_as_markdown(title: str, rows: list[dict], headers: list[str] | None = None) -> str:
    """Format a table as markdown."""
    if not rows:
        return f"### {title}\n\nNo data.\n"

    if headers is None:
        headers = list(rows[0].keys())

    lines = [f"### {title}", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")

    for row in rows:
        values = []
        for h in headers:
            v = row.get(h, "")
            if isinstance(v, float):
                v = f"{v:.4f}"
            values.append(str(v))
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Produce final artifacts and paper tables."""
    print("Iteration 14: Final Artifacts and Paper Tables")
    print("=" * 50)

    # FF92-022: refuse to build tables from invalidated inputs.
    from experiments.trustparadox_u.invalidation import reject_invalidated_inputs

    reject_invalidated_inputs([RESULTS_DIR, SWEEP_DIR, LEAKAGE_DIR, STATS_DIR])

    # Build all tables
    tables = {
        "table1_main_results": build_table1_main_results(),
        "table2_leakage_breakdown": build_table2_leakage_breakdown(),
        "table3_parameter_sensitivity": build_table3_parameter_sensitivity(),
        "table4_statistical_comparisons": build_table4_statistical_comparisons(),
        "table5_target_type_results": build_table5_target_type_results(),
    }

    # Build summaries
    corpus_summary = build_corpus_summary()
    annotation_summary = build_annotation_summary()

    # Build study manifest
    study_manifest = build_study_manifest(tables, corpus_summary, annotation_summary)

    # Write results
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    # Write each table as JSON
    for name, data in tables.items():
        (FINAL_DIR / f"{name}.json").write_text(json.dumps(data, indent=2))

    # Write study manifest
    (FINAL_DIR / "study_manifest.json").write_text(json.dumps(study_manifest, indent=2))

    # Write markdown summary
    md_lines = ["# TrustParadox-U Primary Study Results", ""]
    md_lines.append(f"**Generated:** {study_manifest['generated_at']}")
    md_lines.append(f"**Commit:** {study_manifest['repository_commit']}")
    md_lines.append("")

    # Table 1
    t1 = tables["table1_main_results"]
    if "rows" in t1:
        md_lines.append(
            format_table_as_markdown(
                t1["table"],
                t1["rows"],
                [
                    "condition",
                    "crr_value",
                    "rr_value",
                    "fbr_value",
                    "paired_policy_utility_retention_value",
                ],
            )
        )
        md_lines.append("")
        md_lines.append("*Pooled summary (§36): per-target-type results in Table 5 are primary.*")
        md_lines.append("")

    # Table 2
    t2 = tables["table2_leakage_breakdown"]
    if "rows" in t2:
        md_lines.append(
            format_table_as_markdown(
                t2["table"],
                t2["rows"],
                [
                    "category",
                    "n_episodes",
                    "exposure_rate",
                    "reconstruction_rate",
                    "recontamination_rate",
                ],
            )
        )
        md_lines.append("")

    # Table 4
    t4 = tables["table4_statistical_comparisons"]
    if "rows" in t4:
        md_lines.append(
            format_table_as_markdown(
                t4["table"],
                t4["rows"],
                [
                    "condition_a",
                    "condition_b",
                    "metric",
                    "rate_a",
                    "rate_b",
                    "cohens_h",
                    "p_value",
                    "significant",
                ],
            )
        )
        md_lines.append("")

    # Table 5 (§36)
    t5 = tables["table5_target_type_results"]
    if "rows" in t5:
        md_lines.append(
            format_table_as_markdown(
                t5["table"],
                t5["by_target_type"],
                [
                    "condition",
                    "target_type",
                    "sample_count",
                    "pu_rer_value",
                    "crr_value",
                    "rr_value",
                    "fbr_value",
                ],
            )
        )
        md_lines.append("")
        md_lines.append(
            format_table_as_markdown(
                "Table 5 (scenario-level)",
                t5["by_scenario"],
                [
                    "condition",
                    "scenario_id",
                    "sample_count",
                    "pu_rer_value",
                    "crr_value",
                    "rr_value",
                    "fbr_value",
                ],
            )
        )
        md_lines.append("")

    # Exit criteria
    md_lines.append("## Exit Criteria")
    for criterion, passed in study_manifest["exit_criteria"].items():
        status = "PASS" if passed else "FAIL"
        md_lines.append(f"- {criterion}: {status}")
    md_lines.append("")
    md_lines.append(
        "Research-valid certification is decided by `research_valid_gate.json`, "
        "not by this manifest."
    )

    (FINAL_DIR / "study_summary.md").write_text("\n".join(md_lines))

    # Print summary
    print(f"\nStudy manifest: {FINAL_DIR / 'study_manifest.json'}")
    print(f"Study summary: {FINAL_DIR / 'study_summary.md'}")
    print(f"\nTables generated: {len(tables)}")
    for name, data in tables.items():
        n_rows = len(data.get("rows", []))
        print(f"  {name}: {n_rows} rows")

    all_passed = all(study_manifest["exit_criteria"].values())
    status = "PASSED" if all_passed else "FAILED"
    print(
        f"\nExit criterion: {status} (tables built: "
        f"{sum(not any(k == 'error' for k in d) for d in tables.values())}/{len(tables)})"
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
