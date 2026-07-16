"""Tests for experiment configuration loading."""

from pathlib import Path

import pytest

from experiments.trustparadox_u.config import (
    DetectorConfig,
    ExperimentConfig,
    HistoryConfig,
    MonitoringConfig,
    PolicyConfig,
    load_config,
)

SMOKE_YAML = Path(__file__).parents[2] / "experiments" / "trustparadox_u" / "configs" / "smoke.yaml"


class TestDetectorConfig:
    def test_defaults(self) -> None:
        c = DetectorConfig()
        assert c.semantic_threshold == 0.80

    def test_invalid_threshold(self) -> None:
        with pytest.raises(ValueError, match="semantic_threshold"):
            DetectorConfig(semantic_threshold=1.5)


class TestHistoryConfig:
    def test_defaults(self) -> None:
        c = HistoryConfig()
        assert c.window_size == 5

    def test_invalid_window(self) -> None:
        with pytest.raises(ValueError, match="window_size"):
            HistoryConfig(window_size=0)

    def test_invalid_recon_threshold(self) -> None:
        with pytest.raises(ValueError, match="reconstruction_threshold"):
            HistoryConfig(reconstruction_threshold=-0.1)


class TestPolicyConfig:
    def test_negative_weight(self) -> None:
        with pytest.raises(ValueError, match="privacy_utility_weight"):
            PolicyConfig(privacy_utility_weight=-1.0)


class TestMonitoringConfig:
    def test_negative_duration(self) -> None:
        with pytest.raises(ValueError, match="duration_rounds"):
            MonitoringConfig(duration_rounds=-1)


class TestExperimentConfig:
    def test_invalid_repetitions(self) -> None:
        with pytest.raises(ValueError, match="repetitions"):
            ExperimentConfig(
                seed=42,
                repetitions=0,
                detector=DetectorConfig(),
                history=HistoryConfig(),
                policy=PolicyConfig(),
                monitoring=MonitoringConfig(),
            )

    def test_to_dict(self) -> None:
        cfg = ExperimentConfig(
            seed=42,
            repetitions=3,
            detector=DetectorConfig(),
            history=HistoryConfig(),
            policy=PolicyConfig(),
            monitoring=MonitoringConfig(),
        )
        d = cfg.to_dict()
        assert d["seed"] == 42
        assert d["detector"]["semantic_threshold"] == 0.80


class TestLoadConfig:
    def test_load_smoke(self) -> None:
        cfg = load_config(SMOKE_YAML)
        assert cfg.seed == 42
        assert cfg.repetitions == 1
        assert cfg.detector.exact_enabled is True
        assert cfg.history.window_size == 5

    def test_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")
