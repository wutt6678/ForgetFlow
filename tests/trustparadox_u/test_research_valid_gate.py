"""Tests for FF92-020 reproducibility validation and the research-valid gate."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.deterministic_reproducibility_validation import (  # noqa: E402
    run_deterministic_reproducibility_validation,
    trial_hash,
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


class TestDeterministicReproducibilityValidation:
    """FF92-020: reproducibility validation compares at every layer."""

    def test_reproducible_across_all_layers(self) -> None:
        result = run_deterministic_reproducibility_validation(max_candidates=10)
        assert result["passed"] is True
        assert result["num_mismatches"] == 0
        assert set(result["checks"]) == {
            "candidate_level",
            "trial_level",
            "metric_counts",
            "hashes",
        }
        for check in result["checks"].values():
            assert check["passed"] is True

    def test_no_closed_loop_naming(self) -> None:
        result = run_deterministic_reproducibility_validation(max_candidates=5)
        assert result["validation"] == "deterministic_reproducibility"

    def test_trial_hashes_match_across_reruns(self) -> None:
        from experiments.trustparadox_u.frozen_replay import run_frozen_replay

        run1 = run_frozen_replay(max_candidates_per_condition=5, run_id="th1")
        run2 = run_frozen_replay(max_candidates_per_condition=5, run_id="th2")
        for condition in run1:
            hashes1 = {
                r.candidate_sample_id: trial_hash(r)
                for r in run1[condition].episode_results
            }
            hashes2 = {
                r.candidate_sample_id: trial_hash(r)
                for r in run2[condition].episode_results
            }
            assert hashes1 == hashes2


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
        assert result["num_sweeps"] == 4
        assert result["final_test_split"] == "test"

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
            "deterministic_reproducibility_validation",
            "final_artifacts",
            "all_tests_pass",
        }
        assert set(result["gates"].keys()) == expected_gates
