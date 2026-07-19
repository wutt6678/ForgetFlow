"""Tests for experiment configuration loading."""

from pathlib import Path

import pytest

from experiments.trustparadox_u.config import (
    DetectorConfig,
    ExperimentConfig,
    HistoryConfig,
    ModelsConfig,
    MonitoringConfig,
    PolicyConfig,
    RunConfig,
    load_config,
    validate_embedding_config,
)

SMOKE_YAML = Path(__file__).parents[2] / "experiments" / "trustparadox_u" / "configs" / "smoke.yaml"


class TestDetectorConfig:
    def test_defaults(self) -> None:
        c = DetectorConfig()
        assert c.embedding_threshold == 0.80

    def test_invalid_threshold(self) -> None:
        with pytest.raises(ValueError, match="embedding_threshold"):
            DetectorConfig(embedding_threshold=1.5)


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
        assert d["detector"]["embedding_threshold"] == 0.80


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


def _make_config(
    *,
    embedding_enabled: bool = True,
    mode: str = "test",
    provider: str | None = None,
    model: str | None = None,
    dimension: int | None = None,
) -> ExperimentConfig:
    return ExperimentConfig(
        seed=42,
        repetitions=1,
        detector=DetectorConfig(embedding_enabled=embedding_enabled),
        history=HistoryConfig(),
        policy=PolicyConfig(),
        monitoring=MonitoringConfig(),
        run=RunConfig(mode=mode),
        models=ModelsConfig(
            embedding_provider=provider,
            embedding_model=model,
            embedding_dimension=dimension,
        ),
    )


class TestModelsConfig:
    def test_defaults(self) -> None:
        m = ModelsConfig()
        assert m.embedding_provider is None
        assert m.embedding_model is None
        assert m.embedding_dimension is None

    def test_full_construction(self) -> None:
        m = ModelsConfig(
            embedding_provider="litellm",
            embedding_model="text-embedding-3-small",
            embedding_dimension=1536,
        )
        assert m.embedding_provider == "litellm"
        assert m.embedding_model == "text-embedding-3-small"
        assert m.embedding_dimension == 1536


class TestValidateEmbeddingConfig:
    def test_semantic_disabled_no_validation(self) -> None:
        cfg = _make_config(embedding_enabled=False)
        validate_embedding_config(cfg)  # should not raise

    def test_valid_fixed_test(self) -> None:
        cfg = _make_config(mode="test", provider="fixed", dimension=3)
        validate_embedding_config(cfg)  # should not raise

    def test_valid_null_provider_test(self) -> None:
        cfg = _make_config(mode="test")
        validate_embedding_config(cfg)  # should not raise

    def test_valid_litellm_experiment(self) -> None:
        cfg = _make_config(
            mode="experiment",
            provider="litellm",
            model="text-embedding-3-small",
            dimension=1536,
        )
        validate_embedding_config(cfg)  # should not raise

    def test_experiment_without_model_fails(self) -> None:
        with pytest.raises(ValueError, match="embedding_model"):
            _make_config(mode="experiment", provider="litellm")

    def test_experiment_without_provider_fails(self) -> None:
        with pytest.raises(ValueError, match="embedding_provider"):
            _make_config(mode="experiment", model="text-embedding-3-small")

    def test_experiment_with_wrong_provider_fails(self) -> None:
        with pytest.raises(ValueError, match="embedding_provider"):
            _make_config(mode="experiment", provider="fixed", model="x")

    def test_test_mode_with_real_provider_fails(self) -> None:
        with pytest.raises(ValueError, match="embedding_provider"):
            _make_config(mode="test", provider="litellm")

    def test_zero_dimension_fails(self) -> None:
        with pytest.raises(ValueError, match="embedding_dimension"):
            _make_config(mode="test", provider="fixed", dimension=0)

    def test_negative_dimension_fails(self) -> None:
        with pytest.raises(ValueError, match="embedding_dimension"):
            _make_config(mode="test", provider="fixed", dimension=-1)

    def test_unsupported_provider_fails(self) -> None:
        with pytest.raises(ValueError, match="embedding_provider"):
            _make_config(mode="test", provider="huggingface")

    def test_invalid_config_via_yaml_loading(self) -> None:
        """Invalid config loaded from YAML fails during construction."""
        import os
        import tempfile

        bad_yaml = """
run:
  mode: experiment
  seed: 42
  repetitions: 1
models:
  embedding_provider: fixed
  embedding_model: null
  embedding_dimension: 3
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(bad_yaml)
            f.flush()
            try:
                with pytest.raises(ValueError):
                    load_config(f.name)
            finally:
                os.unlink(f.name)
