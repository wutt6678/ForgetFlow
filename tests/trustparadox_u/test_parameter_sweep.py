"""Tests for Iteration 11: Parameter sweep."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.candidates import load_frozen_corpus  # noqa: E402
from experiments.trustparadox_u.parameter_sweep import (  # noqa: E402
    DETECTOR_LABELS,
    MONITORING_DURATIONS,
    POLICY_LABELS,
    DetectorConfig,
    PolicyConfig,
    _load_scenarios,
    run_full_sweep,
    run_sweep_point,
    write_sweep_results,
)

CORPUS_DIR = Path(__file__).parents[2] / "data" / "trustparadox_u" / "frozen_corpus"


class TestSweepGrid:
    """Tests for sweep grid definition."""

    def test_three_detector_levels(self) -> None:
        assert len(DETECTOR_LABELS) == 3

    def test_three_monitoring_levels(self) -> None:
        assert len(MONITORING_DURATIONS) == 3

    def test_three_policy_levels(self) -> None:
        assert len(POLICY_LABELS) == 3

    def test_total_grid_is_27(self) -> None:
        assert len(DETECTOR_LABELS) * len(MONITORING_DURATIONS) * len(POLICY_LABELS) == 27


class TestSweepPoint:
    """Tests for running individual sweep points."""

    @pytest.fixture
    def small_candidates(self):
        index = load_frozen_corpus(CORPUS_DIR / "frozen_corpus.jsonl")
        return list(index.candidates)[:9]

    @pytest.fixture
    def scenarios(self):
        return _load_scenarios()

    def test_single_sweep_point(self, small_candidates, scenarios) -> None:
        dc = DetectorConfig(
            exact_enabled=True,
            entity_enabled=True,
            embedding_enabled=False,
            claim_matching_enabled=True,
        )
        pc = PolicyConfig()
        sp = run_sweep_point(
            "high",
            dc,
            5,
            "rich",
            pc,
            small_candidates,
            scenarios,
            seed=42,
        )
        assert sp.num_episodes == len(small_candidates)
        assert "crr" in sp.metrics


class TestFullSweep:
    """Tests for the full sweep."""

    def test_full_sweep_produces_27_points(self) -> None:
        results = run_full_sweep(max_candidates=3)
        assert len(results) == 27

    def test_all_points_have_metrics(self) -> None:
        results = run_full_sweep(max_candidates=3)
        for sp in results:
            assert "crr" in sp.metrics
            assert "rr" in sp.metrics


class TestWriteSweep:
    """Tests for writing sweep results."""

    def test_write_creates_files(self, tmp_path: Path) -> None:
        results = run_full_sweep(max_candidates=3)
        write_sweep_results(results, tmp_path)
        assert (tmp_path / "sweep_grid.json").exists()
        assert (tmp_path / "sweep_summary.json").exists()

    def test_summary_has_marginals(self, tmp_path: Path) -> None:
        results = run_full_sweep(max_candidates=3)
        write_sweep_results(results, tmp_path)
        summary = json.loads((tmp_path / "sweep_summary.json").read_text())
        assert "crr_by_detector" in summary
        assert "crr_by_monitoring_duration" in summary
        assert "crr_by_policy" in summary
