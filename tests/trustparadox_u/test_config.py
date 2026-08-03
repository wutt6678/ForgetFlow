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
CONFIGS_DIR = Path(__file__).parents[2] / "experiments" / "trustparadox_u" / "configs"


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


class TestConditionTrialHashes:
    """P1 #13: condition and trial hash semantics.

    The condition hash captures all behavioural configuration (firewall flag,
    detector/history/policy/monitoring, thresholds, model identifiers, rule
    versions) but EXCLUDES trial identity (seed, scenario, secret variant).
    The trial hash additionally includes scenario, seed, and secret variant.
    """

    @staticmethod
    def _cfg(**overrides: object) -> ExperimentConfig:
        kwargs: dict = dict(
            seed=42,
            repetitions=1,
            detector=DetectorConfig(embedding_enabled=False),
            history=HistoryConfig(),
            policy=PolicyConfig(),
            monitoring=MonitoringConfig(),
        )
        kwargs.update(overrides)
        return ExperimentConfig(**kwargs)

    def test_firewall_enabled_changes_condition_hash(self) -> None:
        """firewall_enabled is part of condition identity."""
        cfg = self._cfg()
        assert cfg.condition_hash(firewall_enabled=True) != cfg.condition_hash(
            firewall_enabled=False
        )

    def test_monitoring_duration_changes_condition_hash(self) -> None:
        """Monitoring configuration is part of condition identity."""
        a = self._cfg(monitoring=MonitoringConfig(continuous=False, duration_rounds=0))
        b = self._cfg(monitoring=MonitoringConfig(continuous=False, duration_rounds=5))
        c = self._cfg(monitoring=MonitoringConfig(continuous=True))
        hashes = {
            a.condition_hash(firewall_enabled=True),
            b.condition_hash(firewall_enabled=True),
            c.condition_hash(firewall_enabled=True),
        }
        assert len(hashes) == 3

    def test_detector_history_policy_change_condition_hash(self) -> None:
        """Detector/history/policy configuration is part of condition identity."""
        base = self._cfg()
        ref = base.condition_hash(firewall_enabled=True)
        variants = [
            self._cfg(detector=DetectorConfig(embedding_enabled=False, entity_enabled=False)),
            self._cfg(history=HistoryConfig(window_size=10)),
            self._cfg(policy=PolicyConfig(rich_actions_enabled=False)),
        ]
        for variant in variants:
            assert variant.condition_hash(firewall_enabled=True) != ref

    def test_condition_hash_excludes_seed(self) -> None:
        """Seed is trial identity, not condition identity."""
        a = self._cfg(seed=42)
        b = self._cfg(seed=99)
        assert a.condition_hash(firewall_enabled=True) == b.condition_hash(firewall_enabled=True)

    def test_seed_changes_trial_hash_only(self) -> None:
        """Changing seed changes trial hash but NOT condition hash."""
        a = self._cfg(seed=42)
        b = self._cfg(seed=99)
        # Condition hash is seed-independent.
        assert a.condition_hash(firewall_enabled=True) == b.condition_hash(firewall_enabled=True)
        # Trial hash is seed-dependent.
        th_a = a.trial_hash(
            firewall_enabled=True, scenario_id="pilot_credential", secret_variant_id="sv1"
        )
        th_b = b.trial_hash(
            firewall_enabled=True, scenario_id="pilot_credential", secret_variant_id="sv1"
        )
        assert th_a != th_b

    def test_trial_hash_includes_scenario_and_variant(self) -> None:
        """Trial hash changes with scenario and secret variant."""
        cfg = self._cfg()
        base = cfg.trial_hash(
            firewall_enabled=True, scenario_id="pilot_credential", secret_variant_id="sv1"
        )
        diff_scenario = cfg.trial_hash(
            firewall_enabled=True, scenario_id="pilot_authorization", secret_variant_id="sv1"
        )
        diff_variant = cfg.trial_hash(
            firewall_enabled=True, scenario_id="pilot_credential", secret_variant_id="sv2"
        )
        assert base != diff_scenario
        assert base != diff_variant

    def test_hashes_are_sha256(self) -> None:
        """Both hashes are valid 64-char SHA-256 hex digests."""
        cfg = self._cfg()
        cond = cfg.condition_hash(firewall_enabled=True)
        trial = cfg.trial_hash(
            firewall_enabled=True, scenario_id="pilot_credential", secret_variant_id="sv1"
        )
        for h in (cond, trial):
            assert len(h) == 64
            assert all(c in "0123456789abcdef" for c in h)


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


class TestAllYamlConfigsLoad:
    """Phase 0.2: Every YAML config in the configs directory must load."""

    _yaml_files = sorted(CONFIGS_DIR.glob("*.yaml"))

    @pytest.mark.parametrize("yaml_path", _yaml_files, ids=lambda p: p.name)
    def test_config_loads(self, yaml_path: Path) -> None:
        cfg = load_config(yaml_path)
        assert isinstance(cfg, ExperimentConfig)
        assert cfg.seed is not None
        assert cfg.repetitions >= 1

    @pytest.mark.parametrize("yaml_path", _yaml_files, ids=lambda p: p.name)
    def test_config_has_firewall_enabled(self, yaml_path: Path) -> None:
        cfg = load_config(yaml_path)
        assert isinstance(cfg.firewall_enabled, bool)

    @pytest.mark.parametrize("yaml_path", _yaml_files, ids=lambda p: p.name)
    def test_config_detector_has_claim_fields(self, yaml_path: Path) -> None:
        cfg = load_config(yaml_path)
        assert isinstance(cfg.detector.claim_matching_enabled, bool)
        assert 0.0 <= cfg.detector.claim_confidence_threshold <= 1.0

    @pytest.mark.parametrize("yaml_path", _yaml_files, ids=lambda p: p.name)
    def test_config_detector_has_embedding_fields(self, yaml_path: Path) -> None:
        cfg = load_config(yaml_path)
        assert isinstance(cfg.detector.embedding_enabled, bool)
        assert 0.0 <= cfg.detector.embedding_threshold <= 1.0
