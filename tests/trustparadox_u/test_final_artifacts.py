"""Tests for Iteration 14: Final artifacts and paper tables."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.final_artifacts import (  # noqa: E402
    build_annotation_summary,
    build_corpus_summary,
    build_study_manifest,
    build_table1_main_results,
    build_table2_leakage_breakdown,
    build_table3_parameter_sensitivity,
    build_table4_statistical_comparisons,
    format_table_as_markdown,
)


class TestTable1MainResults:
    """Tests for Table 1."""

    def test_has_rows(self) -> None:
        t1 = build_table1_main_results()
        assert "rows" in t1
        assert len(t1["rows"]) == 5  # 5 conditions

    def test_all_conditions_present(self) -> None:
        t1 = build_table1_main_results()
        conditions = {r["condition"] for r in t1["rows"]}
        assert conditions == {
            "full_mvp",
            "no_monitoring",
            "no_claim_detection",
            "binary_policy",
            "one_time_monitoring",
        }


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
    """Tests for Table 3."""

    def test_has_rows(self) -> None:
        t3 = build_table3_parameter_sensitivity()
        assert "rows" in t3
        assert len(t3["rows"]) == 9  # 3+3+3

    def test_has_all_parameters(self) -> None:
        t3 = build_table3_parameter_sensitivity()
        params = {r["parameter"] for r in t3["rows"]}
        assert params == {"detector", "monitoring_duration", "policy"}


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
        }
        corpus = build_corpus_summary()
        ann = build_annotation_summary()
        manifest = build_study_manifest(tables, corpus, ann)
        assert "exit_criteria" in manifest
        assert all(manifest["exit_criteria"].values())


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
