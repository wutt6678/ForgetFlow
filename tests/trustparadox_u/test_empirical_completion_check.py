"""Tests for E2-I: completion check pipeline.

Verifies the completion check logic including artifact existence,
detailed validation, and overall pass/fail determination.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.trustparadox_u.run_e2_completion_check import (
    _check_analysis_breakdown,
    _check_file_exists,
    _check_frozen_status,
    _check_pilot_attempt_count,
    _check_provenance_completeness,
    _count_jsonl_lines,
    run_completion_check,
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestCountJsonlLines:
    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        assert _count_jsonl_lines(path) == 0

    def test_blank_lines_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "blank.jsonl"
        path.write_text("\n\n\n")
        assert _count_jsonl_lines(path) == 0

    def test_counts_non_empty_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        path.write_text('{"a":1}\n{"b":2}\n\n{"c":3}\n')
        assert _count_jsonl_lines(path) == 3


class TestCheckFileExists:
    def test_file_exists(self, tmp_path: Path) -> None:
        (tmp_path / "test.json").write_text("{}")
        assert _check_file_exists(tmp_path, "test.json") is True

    def test_file_missing(self, tmp_path: Path) -> None:
        assert _check_file_exists(tmp_path, "missing.json") is False


# ---------------------------------------------------------------------------
# Pilot attempt count
# ---------------------------------------------------------------------------


class TestCheckPilotAttemptCount:
    def test_correct_count(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "raw_generation_attempts.jsonl"
        lines = [f'{{"attempt_id":{i}}}' for i in range(90)]
        raw_path.write_text("\n".join(lines))

        result = _check_pilot_attempt_count(tmp_path)
        assert result["passed"] is True
        assert result["attempt_count"] == 90

    def test_wrong_count(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "raw_generation_attempts.jsonl"
        lines = [f'{{"attempt_id":{i}}}' for i in range(80)]
        raw_path.write_text("\n".join(lines))

        result = _check_pilot_attempt_count(tmp_path)
        assert result["passed"] is False
        assert "expected 90 attempts, got 80" in result["reason"]

    def test_missing_file(self, tmp_path: Path) -> None:
        result = _check_pilot_attempt_count(tmp_path)
        assert result["passed"] is False
        assert "not found" in result["reason"]


# ---------------------------------------------------------------------------
# Frozen status
# ---------------------------------------------------------------------------


class TestCheckFrozenStatus:
    def test_correct_status(self, tmp_path: Path) -> None:
        manifest = {"status": "frozen_post_pilot"}
        (tmp_path / "frozen_prompt_manifest.json").write_text(json.dumps(manifest))

        result = _check_frozen_status(tmp_path)
        assert result["passed"] is True
        assert result["status"] == "frozen_post_pilot"

    def test_wrong_status(self, tmp_path: Path) -> None:
        manifest = {"status": "pre_trust_pilot"}
        (tmp_path / "frozen_prompt_manifest.json").write_text(json.dumps(manifest))

        result = _check_frozen_status(tmp_path)
        assert result["passed"] is False
        assert "pre_trust_pilot" in result["reason"]

    def test_missing_file(self, tmp_path: Path) -> None:
        result = _check_frozen_status(tmp_path)
        assert result["passed"] is False
        assert "not found" in result["reason"]


# ---------------------------------------------------------------------------
# Analysis breakdown
# ---------------------------------------------------------------------------


class TestCheckAnalysisBreakdown:
    def test_valid_breakdown(self, tmp_path: Path) -> None:
        report = {
            "exposure_rates": {"by_scenario": {"auth": {}, "cred": {}}},
            "directional_checks": {"auth": {}},
        }
        (tmp_path / "pilot_analysis_report.json").write_text(json.dumps(report))

        result = _check_analysis_breakdown(tmp_path)
        assert result["passed"] is True
        assert result["num_scenarios"] == 2
        assert result["num_directional"] == 1

    def test_missing_by_scenario(self, tmp_path: Path) -> None:
        report = {"exposure_rates": {}, "directional_checks": {"auth": {}}}
        (tmp_path / "pilot_analysis_report.json").write_text(json.dumps(report))

        result = _check_analysis_breakdown(tmp_path)
        assert result["passed"] is False
        assert "by_scenario" in result["reason"]

    def test_missing_directional(self, tmp_path: Path) -> None:
        report = {
            "exposure_rates": {"by_scenario": {"auth": {}}},
            "directional_checks": {},
        }
        (tmp_path / "pilot_analysis_report.json").write_text(json.dumps(report))

        result = _check_analysis_breakdown(tmp_path)
        assert result["passed"] is False
        assert "directional_checks" in result["reason"]

    def test_missing_file(self, tmp_path: Path) -> None:
        result = _check_analysis_breakdown(tmp_path)
        assert result["passed"] is False
        assert "not found" in result["reason"]


# ---------------------------------------------------------------------------
# Provenance completeness
# ---------------------------------------------------------------------------


class TestCheckProvenanceCompleteness:
    def test_complete_provenance(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "raw_generation_attempts.jsonl"
        attempts = [
            {
                "generator_provider": "openai",
                "generator_model_requested": "gpt-4",
                "transport": "api",
                "generation_mode": "real",
            }
        ]
        raw_path.write_text("\n".join(json.dumps(a) for a in attempts))

        result = _check_provenance_completeness(tmp_path)
        assert result["passed"] is True

    def test_missing_field(self, tmp_path: Path) -> None:
        raw_path = tmp_path / "raw_generation_attempts.jsonl"
        attempts = [{"generator_provider": "openai"}]
        raw_path.write_text("\n".join(json.dumps(a) for a in attempts))

        result = _check_provenance_completeness(tmp_path)
        assert result["passed"] is False
        assert "problems" in result
        assert len(result["problems"]) > 0

    def test_missing_file(self, tmp_path: Path) -> None:
        result = _check_provenance_completeness(tmp_path)
        assert result["passed"] is False
        assert "not found" in result["reason"]


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class TestRunCompletionCheck:
    def _make_artifacts(self, root: Path) -> None:
        """Create minimal artifact structure for completion check."""
        # Connectivity smoke
        conn_dir = root / "results/empirical_v2/e2_connectivity_smoke"
        conn_dir.mkdir(parents=True)
        (conn_dir / "validation_report.json").write_text("{}")

        # Trust pilot
        pilot_dir = root / "results/empirical_v2/e2_trust_pilot"
        pilot_dir.mkdir(parents=True)
        attempts = [
            {
                "generator_provider": "openai",
                "generator_model_requested": "gpt-4",
                "transport": "api",
                "generation_mode": "real",
            }
            for _ in range(90)
        ]
        (pilot_dir / "raw_generation_attempts.jsonl").write_text(
            "\n".join(json.dumps(a) for a in attempts)
        )
        (pilot_dir / "validation_report.json").write_text("{}")

        # Pilot labels
        labels_dir = root / "results/empirical_v2/e2_pilot_labels"
        labels_dir.mkdir(parents=True)
        (labels_dir / "labeled_pilot_attempts.jsonl").write_text("")
        (labels_dir / "labeling_report.json").write_text("{}")

        # Pilot analysis
        analysis_dir = root / "results/empirical_v2/e2_pilot_analysis"
        analysis_dir.mkdir(parents=True)
        report = {
            "exposure_rates": {"by_scenario": {"auth": {}}},
            "directional_checks": {"auth": {}},
        }
        (analysis_dir / "pilot_analysis_report.json").write_text(json.dumps(report))

        # Bounded revision
        revision_dir = root / "results/empirical_v2/e2_bounded_revision"
        revision_dir.mkdir(parents=True)
        (revision_dir / "bounded_revision_report.json").write_text("{}")

        # Prompt freeze
        freeze_dir = root / "results/empirical_v2/e2_prompt_freeze"
        freeze_dir.mkdir(parents=True)
        manifest = {"status": "frozen_post_pilot"}
        (freeze_dir / "frozen_prompt_manifest.json").write_text(json.dumps(manifest))
        (freeze_dir / "frozen_freeze_report.json").write_text("{}")

    def test_all_passed(self, tmp_path: Path) -> None:
        self._make_artifacts(tmp_path)
        output_dir = tmp_path / "output"

        report = run_completion_check(tmp_path, output_dir)

        assert report["all_passed"] is True
        assert report["num_passed"] == report["num_total"]
        assert (output_dir / "e2_completion_report.json").exists()

    def test_missing_artifacts(self, tmp_path: Path) -> None:
        # Empty project root - all artifact checks should fail
        report = run_completion_check(tmp_path)

        assert report["all_passed"] is False
        assert report["num_passed"] < report["num_total"]

    def test_code_level_checks(self, tmp_path: Path) -> None:
        self._make_artifacts(tmp_path)
        report = run_completion_check(tmp_path)

        # E2-001: phase-lock enum
        assert report["checks"]["E2-001_phase_lock"]["passed"] is True

        # E2-004: clean-tree gate
        assert report["checks"]["E2-004_clean_tree"]["passed"] is True
