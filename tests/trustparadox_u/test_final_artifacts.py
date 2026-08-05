"""Tests for Iteration 14: Final artifacts and paper tables."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.conditions import REPLAY_CONDITIONS  # noqa: E402
from experiments.trustparadox_u.final_artifacts import (  # noqa: E402
    STUDY_LIMITATIONS,
    TARGET_TYPES,
    build_annotation_summary,
    build_corpus_summary,
    build_study_manifest,
    build_table1_main_results,
    build_table2_leakage_breakdown,
    build_table3_parameter_sensitivity,
    build_table4_statistical_comparisons,
    build_table5_target_type_results,
    build_table6_trust_analysis,
    format_table_as_markdown,
)


class TestTable1MainResults:
    """Tests for Table 1."""

    def test_has_rows(self) -> None:
        t1 = build_table1_main_results()
        assert "rows" in t1
        # Remediation §3: one row per canonical replay condition.
        assert len(t1["rows"]) == len(REPLAY_CONDITIONS)

    def test_all_conditions_present(self) -> None:
        t1 = build_table1_main_results()
        conditions = {r["condition"] for r in t1["rows"]}
        assert conditions == set(REPLAY_CONDITIONS)


class TestTable2LeakageBreakdown:
    """Tests for Table 2."""

    def test_has_rows(self) -> None:
        t2 = build_table2_leakage_breakdown()
        assert "rows" in t2
        assert len(t2["rows"]) > 0

    def test_has_required_fields(self) -> None:
        t2 = build_table2_leakage_breakdown()
        for row in t2["rows"]:
            assert "category" in row
            assert "exposure_rate" in row


class TestTable3ParameterSensitivity:
    """Tests for Table 3 (FF92-018 one-at-a-time sweep)."""

    def test_has_rows(self) -> None:
        t3 = build_table3_parameter_sensitivity()
        assert "rows" in t3
        # One row per swept value: 6 + 4 + 5 + 4 across the four parameters.
        assert len(t3["rows"]) == 19

    def test_has_all_parameters(self) -> None:
        t3 = build_table3_parameter_sensitivity()
        params = {r["parameter"] for r in t3["rows"]}
        assert params == {
            "embedding_threshold",
            "claim_confidence_threshold",
            "history.window_size",
            "monitoring.duration_rounds",
        }

    def test_each_parameter_has_selection(self) -> None:
        t3 = build_table3_parameter_sensitivity()
        params = {r["parameter"] for r in t3["rows"]}
        for parameter in params:
            selected = [r for r in t3["rows"] if r["parameter"] == parameter and r["selected"]]
            assert len(selected) == 1


class TestTable4StatisticalComparisons:
    """Tests for Table 4."""

    def test_has_rows(self) -> None:
        t4 = build_table4_statistical_comparisons()
        assert "rows" in t4
        assert len(t4["rows"]) > 0

    def test_has_significance_flags(self) -> None:
        t4 = build_table4_statistical_comparisons()
        for row in t4["rows"]:
            assert "significant" in row
            assert "p_value" in row


class TestTable5TargetTypeResults:
    """Tests for Table 5 (§36: per-target-type results before pooling)."""

    def test_has_rows(self) -> None:
        t5 = build_table5_target_type_results()
        assert "rows" in t5
        assert len(t5["rows"]) > 0

    def test_all_target_types_covered(self) -> None:
        t5 = build_table5_target_type_results()
        covered = {row["target_type"] for row in t5["by_target_type"]}
        assert set(TARGET_TYPES) <= covered

    def test_scenario_rows_present(self) -> None:
        t5 = build_table5_target_type_results()
        assert t5["by_scenario"]
        for row in t5["by_scenario"]:
            assert "scenario_id" in row
            assert "condition" in row

    def test_macro_average_per_condition(self) -> None:
        t5 = build_table5_target_type_results()
        conditions = {row["condition"] for row in t5["by_target_type"]}
        macro_conditions = {row["condition"] for row in t5["macro_average_by_target_type"]}
        assert conditions == macro_conditions

    def test_pooled_rows_are_secondary(self) -> None:
        # §36 acceptance: no primary conclusion relies only on pooled rate.
        t5 = build_table5_target_type_results()
        assert t5["pooled_secondary"]
        for row in t5["pooled_secondary"]:
            assert row["role"] == "secondary_summary"

    def test_heterogeneity_reported(self) -> None:
        t5 = build_table5_target_type_results()
        assert t5["heterogeneity_note"]
        assert t5["heterogeneity"]


class TestSelfDescribingContext:
    """§35: every table states study class, split, population, and metric
    definitions so a reader can interpret it without opening the code."""

    CONTEXT_KEYS = {
        "study_class",
        "protocol_version",
        "population",
        "split",
        "attack_population",
        "conditions",
        "baseline_condition",
        "pairing_unit",
        "confidence_intervals",
    }

    def _tables(self) -> dict:
        return {
            "t1": build_table1_main_results(),
            "t2": build_table2_leakage_breakdown(),
            "t3": build_table3_parameter_sensitivity(),
            "t4": build_table4_statistical_comparisons(),
            "t5": build_table5_target_type_results(),
        }

    def test_every_table_has_context(self) -> None:
        for name, table in self._tables().items():
            context = table.get("context")
            assert context, f"{name} missing context"
            assert self.CONTEXT_KEYS <= set(context), f"{name} context incomplete"

    def test_every_table_has_metric_definitions(self) -> None:
        for name, table in self._tables().items():
            definitions = table.get("metric_definitions")
            assert definitions, f"{name} missing metric_definitions"
            for metric, spec in definitions.items():
                assert spec["definition"], f"{name}/{metric} has no definition"
                assert spec["numerator"], f"{name}/{metric} has no numerator"
                assert spec["denominator"], f"{name}/{metric} has no denominator"

    def test_table1_rows_carry_numerator_and_denominator(self) -> None:
        t1 = build_table1_main_results()
        row = t1["rows"][0]
        value_cols = [key for key in row if key.endswith("_value")]
        assert value_cols
        for key in value_cols:
            metric = key[: -len("_value")]
            assert f"{metric}_numerator" in row
            assert f"{metric}_denominator" in row

    def test_table4_rows_carry_pairing_and_cis(self) -> None:
        t4 = build_table4_statistical_comparisons()
        for row in t4["rows"]:
            assert row["pairing_unit"]
            assert row["n_pairs"] is not None
            assert "rate_a_ci_95" in row
            assert "cluster_bootstrap_ci_95" in row


class TestCorpusSummary:
    """Tests for corpus summary."""

    def test_has_hash(self) -> None:
        summary = build_corpus_summary()
        assert "corpus_hash" in summary
        assert len(summary["corpus_hash"]) > 0


class TestAnnotationSummary:
    """Tests for annotation summary."""

    def test_has_hash(self) -> None:
        summary = build_annotation_summary()
        assert "annotation_hash" in summary


class TestStudyManifest:
    """Tests for study manifest."""

    def test_has_exit_criteria(self) -> None:
        tables = {
            "t1": build_table1_main_results(),
            "t2": build_table2_leakage_breakdown(),
            "t3": build_table3_parameter_sensitivity(),
            "t4": build_table4_statistical_comparisons(),
            "t5": build_table5_target_type_results(),
            "t6": build_table6_trust_analysis(),
        }
        corpus = build_corpus_summary()
        ann = build_annotation_summary()
        manifest = build_study_manifest(tables, corpus, ann)
        assert "exit_criteria" in manifest
        assert all(manifest["exit_criteria"].values())

    def test_records_study_class(self) -> None:
        # Remediation §4: every artifact records its study class.
        from experiments.trustparadox_u.status import STUDY_CLASSES

        tables = {"t1": build_table1_main_results()}
        manifest = build_study_manifest(tables, build_corpus_summary(), build_annotation_summary())
        assert manifest["study_class"] in STUDY_CLASSES


class TestStudyLimitationsRemediation38:
    """§38: the study states its limits explicitly and precisely."""

    REQUIRED_LIMITATIONS = (
        "parameter-level machine unlearning",
        "deletion of information from model weights",
        "deletion from external provider logs",
        "deletion from hidden model state outside the experimental harness",
        "resistance to all adaptive adversaries",
        "generalization beyond the tested agent architectures and models",
    )

    def test_all_six_limitations_declared(self) -> None:
        for limit in self.REQUIRED_LIMITATIONS:
            assert limit in STUDY_LIMITATIONS["not_demonstrated"]

    def test_precise_terminology(self) -> None:
        scope = STUDY_LIMITATIONS["scope"]
        assert "enforced forgetting" in scope
        assert "release control" in scope

    def test_no_internal_unlearning_claim(self) -> None:
        terminology = STUDY_LIMITATIONS["terminology"]
        assert "does not" in terminology or "not used" in terminology

    def test_manifest_carries_limitations(self) -> None:
        tables = {"t1": build_table1_main_results()}
        manifest = build_study_manifest(tables, build_corpus_summary(), build_annotation_summary())
        assert manifest["limitations"] == STUDY_LIMITATIONS

    def test_summary_markdown_renders_limitations(self) -> None:
        from experiments.trustparadox_u.final_artifacts import FINAL_DIR

        summary_path = FINAL_DIR / "study_summary.md"
        if not summary_path.exists():
            return
        text = summary_path.read_text()
        assert "Study Limitations" in text
        for limit in self.REQUIRED_LIMITATIONS:
            assert limit in text


class TestMarkdownFormatter:
    """Tests for markdown table formatting."""

    def test_basic_table(self) -> None:
        rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        md = format_table_as_markdown("Test", rows)
        assert "### Test" in md
        assert "| a | b |" in md

    def test_empty_table(self) -> None:
        md = format_table_as_markdown("Empty", [])
        assert "No data" in md
