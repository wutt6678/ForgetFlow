"""Tests for Iteration 9: Frozen replay runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.candidates import (  # noqa: E402
    load_frozen_corpus,
)
from experiments.trustparadox_u.dataset import load_episode  # noqa: E402
from experiments.trustparadox_u.frozen_replay import (  # noqa: E402
    CONDITIONS,
    ConditionResult,
    build_config_for_condition,
    run_condition,
    run_frozen_replay,
    write_results,
)

SCENARIOS_DIR = Path(__file__).parents[2] / "data" / "trustparadox_u" / "scenarios"
CORPUS_DIR = Path(__file__).parents[2] / "data" / "trustparadox_u" / "frozen_corpus"


class TestConditions:
    """Tests for condition definitions."""

    def test_all_five_conditions_defined(self) -> None:
        assert "full_mvp" in CONDITIONS
        assert "no_monitoring" in CONDITIONS
        assert "no_claim_detection" in CONDITIONS
        assert "binary_policy" in CONDITIONS
        assert "one_time_monitoring" in CONDITIONS

    def test_all_conditions_have_firewall(self) -> None:
        for name, params in CONDITIONS.items():
            assert params["firewall_enabled"] is True, f"{name} missing firewall"


class TestBuildConfig:
    """Tests for config building per condition."""

    def test_build_all_conditions(self) -> None:
        for name in CONDITIONS:
            config = build_config_for_condition(name)
            assert config.firewall_enabled is True

    def test_unknown_condition_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown condition"):
            build_config_for_condition("nonexistent")

    def test_no_monitoring_has_no_continuous(self) -> None:
        config = build_config_for_condition("no_monitoring")
        assert config.monitoring.continuous is False

    def test_full_mvp_has_continuous_monitoring(self) -> None:
        config = build_config_for_condition("full_mvp")
        assert config.monitoring.continuous is True

    def test_no_claim_detection_has_claim_matching_disabled(self) -> None:
        config = build_config_for_condition("no_claim_detection")
        assert config.detector.claim_matching_enabled is False


class TestRunCondition:
    """Tests for running a single condition."""

    @pytest.fixture
    def scenario_episodes(self):
        eps = {}
        for sid, fname in [
            ("credential_001", "pilot_credential.yaml"),
            ("attribute_001", "pilot_private_attribute.yaml"),
            ("auth_001", "pilot_authorization.yaml"),
        ]:
            eps[sid] = load_episode(SCENARIOS_DIR / fname)
        return eps

    @pytest.fixture
    def small_candidates(self):
        index = load_frozen_corpus(CORPUS_DIR / "frozen_corpus.jsonl")
        return list(index.candidates)[:6]

    def test_run_condition_produces_results(self, scenario_episodes, small_candidates) -> None:
        result = run_condition("full_mvp", small_candidates, scenario_episodes, seed=42)
        assert isinstance(result, ConditionResult)
        assert len(result.episode_results) == len(small_candidates)

    def test_run_condition_has_metrics(self, scenario_episodes, small_candidates) -> None:
        result = run_condition("full_mvp", small_candidates, scenario_episodes, seed=42)
        assert "crr" in result.metrics
        assert "rr" in result.metrics
        assert "paired_policy_utility_retention" in result.metrics


class TestRunFrozenReplay:
    """Tests for the full frozen replay."""

    def test_all_conditions_produce_results(self) -> None:
        results = run_frozen_replay(max_candidates_per_condition=3)
        assert len(results) == len(CONDITIONS)
        for name in CONDITIONS:
            assert name in results
            assert len(results[name].episode_results) > 0

    def test_candidate_sample_id_set(self) -> None:
        results = run_frozen_replay(max_candidates_per_condition=3)
        for name, cr in results.items():
            for er in cr.episode_results:
                assert er.candidate_sample_id != ""


class TestWriteResults:
    """Tests for writing results."""

    def test_write_creates_files(self, tmp_path: Path) -> None:
        results = run_frozen_replay(max_candidates_per_condition=3)
        write_results(results, tmp_path, run_id="test")

        assert (tmp_path / "metrics_by_condition.json").exists()
        assert (tmp_path / "summary.json").exists()
        assert (tmp_path / "episodes.jsonl").exists()

    def test_summary_has_all_conditions(self, tmp_path: Path) -> None:
        results = run_frozen_replay(max_candidates_per_condition=3)
        write_results(results, tmp_path, run_id="test")

        summary = json.loads((tmp_path / "summary.json").read_text())
        assert "conditions" in summary
        for name in CONDITIONS:
            assert name in summary["conditions"]
