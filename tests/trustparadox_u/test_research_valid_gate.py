"""Tests for Iterations 15-16: Closed-loop validation and research-valid gate."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.closed_loop_validation import (  # noqa: E402
    _compare_metrics,
    run_closed_loop_validation,
)
from experiments.trustparadox_u.research_valid_gate import (  # noqa: E402
    check_all_conditions_run,
    check_annotations_valid,
    check_corpus_valid,
    check_final_artifacts,
    check_leakage_analysis_available,
    check_paired_statistics_available,
    check_parameter_sweep_complete,
    run_research_valid_gate,
)


class TestCompareMetrics:
    """Tests for metric comparison."""

    def test_identical_metrics_match(self) -> None:
        m = {"full_mvp": {"crr": {"value": 0.5}}}
        diffs = _compare_metrics(m, m)
        assert len(diffs) == 0

    def test_different_metrics_differ(self) -> None:
        m1 = {"full_mvp": {"crr": {"value": 0.5}}}
        m2 = {"full_mvp": {"crr": {"value": 0.9}}}
        diffs = _compare_metrics(m1, m2, tolerance=0.01)
        assert len(diffs) == 1

    def test_none_values_skipped(self) -> None:
        m1 = {"full_mvp": {"crr": {"value": None}}}
        m2 = {"full_mvp": {"crr": {"value": 0.5}}}
        diffs = _compare_metrics(m1, m2)
        assert len(diffs) == 0

    def test_within_tolerance(self) -> None:
        m1 = {"full_mvp": {"crr": {"value": 0.5}}}
        m2 = {"full_mvp": {"crr": {"value": 0.505}}}
        diffs = _compare_metrics(m1, m2, tolerance=0.01)
        assert len(diffs) == 0


class TestClosedLoopValidation:
    """Tests for closed-loop validation."""

    def test_deterministic_reproducibility(self) -> None:
        result = run_closed_loop_validation(max_candidates=10)
        assert result["passed"] is True
        assert result["num_mismatches"] == 0


class TestGateChecks:
    """Tests for individual gate checks."""

    def test_all_conditions_run(self) -> None:
        result = check_all_conditions_run()
        assert result["passed"] is True

    def test_corpus_valid(self) -> None:
        result = check_corpus_valid()
        assert result["passed"] is True
        assert result["candidate_count"] > 0

    def test_annotations_valid(self) -> None:
        result = check_annotations_valid()
        assert result["passed"] is True
        assert result["unresolved"] == 0

    def test_leakage_analysis_available(self) -> None:
        result = check_leakage_analysis_available()
        assert result["passed"] is True

    def test_paired_statistics_available(self) -> None:
        result = check_paired_statistics_available()
        assert result["passed"] is True

    def test_parameter_sweep_complete(self) -> None:
        result = check_parameter_sweep_complete()
        assert result["passed"] is True
        assert result["grid_size"] == 27

    def test_final_artifacts(self) -> None:
        result = check_final_artifacts()
        assert result["passed"] is True


class TestResearchValidGate:
    """Tests for the final gate."""

    def test_all_gates_pass(self) -> None:
        result = run_research_valid_gate()
        assert result["verdict"] == "research_valid"
        assert result["all_passed"] is True

    def test_has_all_gate_names(self) -> None:
        result = run_research_valid_gate()
        expected_gates = {
            "all_conditions_run",
            "corpus_valid",
            "annotations_valid",
            "leakage_analysis_available",
            "paired_statistics_available",
            "parameter_sweep_complete",
            "closed_loop_validation",
            "final_artifacts",
            "all_tests_pass",
        }
        assert set(result["gates"].keys()) == expected_gates
