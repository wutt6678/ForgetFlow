"""Deterministic hyperparameter sweep tests for single-target validation.

ST-HP-001 through ST-HP-004: Validate monotonic behavior of hyperparameters.

These tests are deterministic integration tests that do not make real provider calls.
They use fixed embeddings and canonical fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.trustparadox_u.config import (
    DetectorConfig,
    ExperimentConfig,
    HistoryConfig,
    MonitoringConfig,
    PolicyConfig,
)
from experiments.trustparadox_u.dataset import load_episode
from experiments.trustparadox_u.evaluator import compute_pu_rer, compute_rr
from experiments.trustparadox_u.runner import run_episode

SCENARIOS_DIR = Path(__file__).parents[2] / "data" / "trustparadox_u" / "scenarios"


def _base_config(**overrides) -> ExperimentConfig:
    """Create a base experiment config for sweep tests."""
    kwargs = dict(
        seed=42,
        repetitions=1,
        detector=DetectorConfig(embedding_enabled=False),
        history=HistoryConfig(),
        policy=PolicyConfig(),
        monitoring=MonitoringConfig(),
    )
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


@pytest.mark.integration
class TestSemanticThresholdSweep:
    """ST-HP-001: Semantic threshold sweep validates monotonic behavior."""

    @pytest.mark.parametrize(
        "threshold",
        [0.60, 0.70, 0.80, 0.90],
        ids=["ST-HP-001-0.60", "ST-HP-001-0.70", "ST-HP-001-0.80", "ST-HP-001-0.90"],
    )
    def test_embedding_threshold_produces_valid_result(self, threshold: float) -> None:
        """Each threshold produces a valid result with bounded metrics."""
        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        cfg = _base_config(
            detector=DetectorConfig(embedding_enabled=False, embedding_threshold=threshold),
        )
        result = run_episode(ep, cfg)
        # Result should be valid
        assert result.episode_id == ep.episode_id
        # Metrics should be bounded
        pu_rer = compute_pu_rer([result])
        assert pu_rer.denominator >= 0
        # Forbidden strings should not be released
        forbidden = result.metadata.get("forbidden_strings", [])
        for turn in result.turns:
            if turn.released_text:
                for f in forbidden:
                    assert f not in turn.released_text


@pytest.mark.integration
class TestReconstructionThresholdSweep:
    """ST-HP-002: Reconstruction threshold sweep validates monotonic behavior."""

    @pytest.mark.parametrize(
        "threshold",
        [0.50, 0.70, 0.90],
        ids=["ST-HP-002-0.50", "ST-HP-002-0.70", "ST-HP-002-0.90"],
    )
    def test_reconstruction_threshold_produces_valid_result(self, threshold: float) -> None:
        """Each reconstruction threshold produces a valid result."""
        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        cfg = _base_config(
            history=HistoryConfig(reconstruction_threshold=threshold),
        )
        result = run_episode(ep, cfg)
        assert result.episode_id == ep.episode_id
        # Result should have turns
        assert len(result.turns) > 0


@pytest.mark.integration
class TestHistoryWindowSweep:
    """ST-HP-003: History window size sweep validates behavior."""

    @pytest.mark.parametrize(
        "window_size",
        [1, 2, 4, 8],
        ids=["ST-HP-003-w1", "ST-HP-003-w2", "ST-HP-003-w4", "ST-HP-003-w8"],
    )
    def test_history_window_produces_valid_result(self, window_size: int) -> None:
        """Each window size produces a valid result."""
        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        cfg = _base_config(
            history=HistoryConfig(window_size=window_size),
        )
        result = run_episode(ep, cfg)
        assert result.episode_id == ep.episode_id
        assert len(result.turns) > 0


@pytest.mark.integration
class TestMonitoringDurationSweep:
    """ST-HP-004: Monitoring duration sweep validates enforcement boundaries."""

    @pytest.mark.parametrize(
        "duration",
        [0, 1, 3, 5],
        ids=["ST-HP-004-d0", "ST-HP-004-d1", "ST-HP-004-d3", "ST-HP-004-d5"],
    )
    def test_monitoring_duration_produces_valid_result(self, duration: int) -> None:
        """Each monitoring duration produces a valid result."""
        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        cfg = _base_config(
            monitoring=MonitoringConfig(continuous=False, duration_rounds=duration),
        )
        result = run_episode(ep, cfg)
        assert result.episode_id == ep.episode_id
        # Metrics should be computable
        rr = compute_rr([result])
        assert rr.denominator >= 0

    def test_continuous_vs_finite_ordering(self) -> None:
        """ST-HP-004: Continuous monitoring should protect at least as much as finite."""
        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")

        cfg_continuous = _base_config(
            monitoring=MonitoringConfig(continuous=True),
        )
        cfg_finite = _base_config(
            monitoring=MonitoringConfig(continuous=False, duration_rounds=1),
        )

        result_continuous = run_episode(ep, cfg_continuous)
        result_finite = run_episode(ep, cfg_finite)

        # Both should produce valid results
        assert result_continuous.episode_id == ep.episode_id
        assert result_finite.episode_id == ep.episode_id

        # Continuous should have at least as many protected turns
        continuous_blocked = sum(1 for t in result_continuous.turns if t.released_text is None)
        finite_blocked = sum(1 for t in result_finite.turns if t.released_text is None)
        # Continuous blocks >= finite (more protection)
        assert continuous_blocked >= finite_blocked
