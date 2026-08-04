"""Tests for FF92-020 reproducibility validation and the FF92-021 gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u import research_valid_gate as gate_module  # noqa: E402
from experiments.trustparadox_u.deterministic_reproducibility_validation import (  # noqa: E402
    run_deterministic_reproducibility_validation,
    trial_hash,
)
from experiments.trustparadox_u.research_valid_gate import (  # noqa: E402
    SUBSTANTIVE_GATES,
    check_annotations_valid,
    check_conditions_valid,
    check_corpus_valid,
    check_metrics_recompute,
    check_no_invalidated_artifacts,
    check_replay_complete,
    check_repository_provenance,
    check_statistical_analysis_valid,
    check_tests_pass,
    run_research_valid_gate,
    verdict_for,
)

_TEST_COMMIT = "a" * 40


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
                r.candidate_sample_id: trial_hash(r) for r in run1[condition].episode_results
            }
            hashes2 = {
                r.candidate_sample_id: trial_hash(r) for r in run2[condition].episode_results
            }
            assert hashes1 == hashes2


class TestCorpusAndAnnotationGates:
    """Content-based corpus/annotation certification on the real data."""

    def test_corpus_valid(self) -> None:
        result = check_corpus_valid()
        assert result["passed"] is True, result
        assert result["candidate_count"] > 0

    def test_annotations_valid(self) -> None:
        result = check_annotations_valid()
        assert result["passed"] is True, result
        assert result["annotation_count"] > 0


class TestRealArtifactGates:
    """Content gates on the committed trial/statistics artifacts."""

    def test_conditions_match_frozen_builders(self) -> None:
        result = check_conditions_valid()
        assert result["passed"] is True, result

    def test_replay_covers_every_candidate_via_trial_units(self) -> None:
        result = check_replay_complete()
        assert result["passed"] is True, result
        assert result["trial_count"] == result["expected_count"]

    def test_statistical_comparisons_pair_consistently(self) -> None:
        result = check_statistical_analysis_valid()
        assert result["passed"] is True, result
        assert result["num_comparisons"] > 0


class TestFileExistenceIsNotCertification:
    """FF92-021 acceptance: file existence alone cannot yield research_valid."""

    def test_empty_artifacts_cannot_pass_substantive_gates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gate_module, "REPLAY_DIR", tmp_path / "frozen_replay")
        monkeypatch.setattr(gate_module, "RESULTS_DIR", tmp_path)
        monkeypatch.setattr(gate_module, "FINAL_DIR", tmp_path / "final_artifacts")
        monkeypatch.setattr(gate_module, "CORPUS_DIR", tmp_path / "frozen_corpus")
        monkeypatch.setattr(
            gate_module,
            "PROVENANCE_ARTIFACTS",
            (tmp_path / "frozen_replay" / "run_manifest.json",),
        )
        for rel in (
            "frozen_replay/run_manifest.json",
            "frozen_replay/candidate_trials.jsonl",
            "frozen_replay/metrics_by_condition.json",
            "frozen_replay/resolved_conditions.json",
            "leakage_analysis/leakage_analysis.json",
            "paired_statistics/paired_statistics.json",
            "parameter_sweep/sweep_summary.json",
            "deterministic_reproducibility_validation/validation_result.json",
            "final_artifacts/study_manifest.json",
        ):
            target = tmp_path / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("{}")

        assert check_corpus_valid()["passed"] is False
        assert check_annotations_valid()["passed"] is False
        assert check_metrics_recompute()["passed"] is False
        # No markers and no claims yet: invalidation state itself is clean.
        assert check_no_invalidated_artifacts()["passed"] is True
        verdict = verdict_for(
            {name: {"passed": False} for name in SUBSTANTIVE_GATES}
            | {"tests_pass": {"passed": True}, "static_checks": {"passed": True}}
        )
        assert verdict == "diagnostic"


class TestVerdictLogic:
    """FF92-021: research_valid only when every gate passes."""

    def _gates(self, **overrides: dict[str, object]) -> dict[str, dict[str, object]]:
        gates: dict[str, dict[str, object]] = {name: {"passed": True} for name in SUBSTANTIVE_GATES}
        gates["tests_pass"] = {"passed": True}
        gates["static_checks"] = {"passed": True}
        for name, update in overrides.items():
            gates[name].update(update)
        return gates

    def test_all_pass_is_research_valid(self) -> None:
        assert verdict_for(self._gates()) == "research_valid"

    def test_not_run_checks_cap_verdict_at_release_candidate(self) -> None:
        gates = self._gates(
            tests_pass={"passed": False, "not_run": True},
            static_checks={"passed": False, "not_run": True},
        )
        assert verdict_for(gates) == "release_candidate"

    def test_failed_directional_gate_is_diagnostic(self) -> None:
        gates = self._gates(metrics_recompute={"passed": False})
        assert verdict_for(gates) == "diagnostic"

    def test_not_run_substantive_gate_is_diagnostic(self) -> None:
        gates = self._gates(corpus_valid={"passed": False, "not_run": True})
        assert verdict_for(gates) == "diagnostic"


class TestTestsGateNeverAutoPasses:
    """FF92-021: no gate auto-passes via PYTEST_CURRENT_TEST."""

    def test_tests_gate_fails_not_run_under_pytest(self) -> None:
        # The enclosing pytest run sets PYTEST_CURRENT_TEST.
        result = check_tests_pass()
        assert result["passed"] is False
        assert result.get("not_run") is True

    def test_full_gate_never_research_valid_under_pytest(self) -> None:
        result = run_research_valid_gate()
        assert result["gates"]["tests_pass"]["passed"] is False
        assert result["gates"]["static_checks"]["passed"] is False
        assert result["verdict"] != "research_valid"


class TestProvenanceGate:
    """FF92-021/023: wrong artifact commit or dirty tree fails certification."""

    def _patch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manifest: dict[str, object]
    ) -> Path:
        manifest_path = tmp_path / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        monkeypatch.setattr(gate_module, "PROVENANCE_ARTIFACTS", (manifest_path,))
        monkeypatch.setattr(gate_module, "_current_commit", lambda: _TEST_COMMIT)
        return manifest_path

    def _provenance(self, **overrides: object) -> dict[str, object]:
        record: dict[str, object] = {
            "tested_code_commit": _TEST_COMMIT,
            "artifact_generation_commit": _TEST_COMMIT,
            "repository_clean": True,
            "workflow_run_id": "1",
            "workflow_attempt": "1",
        }
        record.update(overrides)
        return record

    def test_matching_clean_provenance_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "experiments.trustparadox_u.artifact_provenance.code_tree_is_clean", lambda: True
        )
        self._patch(tmp_path, monkeypatch, {"provenance": self._provenance()})
        result = check_repository_provenance()
        assert result["passed"] is True, result

    def test_wrong_artifact_commit_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch(
            tmp_path,
            monkeypatch,
            {"provenance": self._provenance(artifact_generation_commit="b" * 40)},
        )
        result = check_repository_provenance()
        assert result["passed"] is False
        assert any("stale_result_commit" in f or "commit_mismatch" in f for f in result["findings"])

    def test_dirty_artifact_fails(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch(tmp_path, monkeypatch, {"provenance": self._provenance(repository_clean=False)})
        result = check_repository_provenance()
        assert result["passed"] is False
        assert any("dirty_artifact" in f for f in result["findings"])

    def test_missing_provenance_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch(tmp_path, monkeypatch, {"git_commit": _TEST_COMMIT})
        result = check_repository_provenance()
        assert result["passed"] is False
        assert any("missing_provenance_fields" in f for f in result["findings"])

    def test_dirty_code_tree_fails(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "experiments.trustparadox_u.artifact_provenance.code_tree_is_clean", lambda: False
        )
        self._patch(tmp_path, monkeypatch, {"provenance": self._provenance()})
        result = check_repository_provenance()
        assert result["passed"] is False
        assert any("repository_tree_dirty" in f for f in result["findings"])


class TestInvalidationGate:
    """FF92-022: invalidated inputs block certification and final tables."""

    def test_marker_outside_archive_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from experiments.trustparadox_u.invalidation import INVALIDATION_MARKER

        stale = tmp_path / "frozen_replay"
        stale.mkdir(parents=True)
        (stale / "INVALIDATION_MARKER.json").write_text(json.dumps(INVALIDATION_MARKER))
        monkeypatch.setattr(gate_module, "RESULTS_DIR", tmp_path)
        result = check_no_invalidated_artifacts()
        assert result["passed"] is False
        assert result["invalidation_markers"]

    def test_marker_inside_archive_is_fine(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from experiments.trustparadox_u.invalidation import INVALIDATION_MARKER

        archived = tmp_path / "archive" / "92bc12e_diagnostic_invalid" / "frozen_replay"
        archived.mkdir(parents=True)
        (archived / "INVALIDATION_MARKER.json").write_text(json.dumps(INVALIDATION_MARKER))
        monkeypatch.setattr(gate_module, "RESULTS_DIR", tmp_path)
        result = check_no_invalidated_artifacts()
        assert result["passed"] is True

    def test_stale_research_valid_claim_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stale = tmp_path / "final_artifacts"
        stale.mkdir(parents=True)
        (stale / "study_manifest.json").write_text(json.dumps({"research_valid": True}))
        monkeypatch.setattr(gate_module, "RESULTS_DIR", tmp_path)
        monkeypatch.setattr(gate_module, "FINAL_DIR", stale)
        result = check_no_invalidated_artifacts()
        assert result["passed"] is False
        assert result["stale_research_valid_claims"]

    def test_gate_excludes_its_own_previous_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        final_dir = tmp_path / "final_artifacts"
        final_dir.mkdir(parents=True)
        (final_dir / "research_valid_gate.json").write_text(
            json.dumps({"verdict": "research_valid"})
        )
        monkeypatch.setattr(gate_module, "RESULTS_DIR", tmp_path)
        monkeypatch.setattr(gate_module, "FINAL_DIR", final_dir)
        result = check_no_invalidated_artifacts()
        assert result["passed"] is True


class TestMetricsGateBehavior:
    """FF92-021: undefined utility or None-valued 'computed' metric fails."""

    def test_none_metric_called_computed_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from experiments.trustparadox_u.frozen_replay import BASELINE_CONDITION

        replay_dir = tmp_path / "frozen_replay"
        replay_dir.mkdir(parents=True)
        for name in (
            "candidate_trials.jsonl",
            "reconstruction_trials.jsonl",
            "recontamination_trials.jsonl",
            "utility_trials.jsonl",
        ):
            (replay_dir / name).write_text("")
        stored = {
            BASELINE_CONDITION: {
                "pu_rer": {"value": 0.5, "numerator": 1, "denominator": 2},
                "paired_policy_utility_retention": {
                    "value": None,
                    "numerator": 0,
                    "denominator": 0,
                    "evaluable": False,
                    "reason": "baseline_condition",
                },
            },
            "full_mvp": {
                "pu_rer": {"value": None, "numerator": 0, "denominator": 0, "evaluable": True},
                "paired_policy_utility_retention": {
                    "value": None,
                    "numerator": 0,
                    "denominator": 0,
                    "evaluable": False,
                },
            },
        }
        (replay_dir / "metrics_by_condition.json").write_text(json.dumps(stored))
        monkeypatch.setattr(gate_module, "REPLAY_DIR", replay_dir)

        result = check_metrics_recompute()
        assert result["passed"] is False
        assert any("none_metric_called_computed" in f for f in result["findings"])
        assert any("utility_not_evaluable" in f for f in result["findings"])
