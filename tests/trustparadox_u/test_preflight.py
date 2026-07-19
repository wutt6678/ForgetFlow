"""Tests for shared provider builder and preflight module."""

from experiments.trustparadox_u.config import ModelsConfig
from experiments.trustparadox_u.providers import (
    build_real_embedding_provider,
    sanitize_api_base,
)


class TestBuildRealEmbeddingProvider:
    """Shared provider builder tests."""

    def test_forwards_provider_name(self) -> None:
        models = ModelsConfig(
            embedding_provider="litellm",
            embedding_model="openai/text-embedding-v3",
        )
        provider = build_real_embedding_provider(models)
        assert provider.provider_name == "litellm"

    def test_forwards_model_name(self) -> None:
        models = ModelsConfig(
            embedding_provider="litellm",
            embedding_model="openai/text-embedding-v3",
        )
        provider = build_real_embedding_provider(models)
        assert provider.model_name == "openai/text-embedding-v3"

    def test_forwards_expected_dimension(self) -> None:
        models = ModelsConfig(
            embedding_provider="litellm",
            embedding_model="openai/text-embedding-v3",
            embedding_dimension=1024,
        )
        provider = build_real_embedding_provider(models)
        assert provider.dimension == 1024

    def test_forwards_api_base(self) -> None:
        models = ModelsConfig(
            embedding_provider="litellm",
            embedding_model="openai/text-embedding-v3",
            api_base="https://example.test/v1",
        )
        provider = build_real_embedding_provider(models)
        assert provider._api_base == "https://example.test/v1"

    def test_absent_api_base_is_none(self) -> None:
        models = ModelsConfig(
            embedding_provider="litellm",
            embedding_model="openai/text-embedding-v3",
        )
        provider = build_real_embedding_provider(models)
        assert provider._api_base is None

    def test_missing_provider_raises(self) -> None:
        models = ModelsConfig(embedding_model="openai/text-embedding-v3")
        import pytest

        with pytest.raises(ValueError, match="embedding_provider"):
            build_real_embedding_provider(models)

    def test_missing_model_raises(self) -> None:
        models = ModelsConfig(embedding_provider="litellm")
        import pytest

        with pytest.raises(ValueError, match="embedding_model"):
            build_real_embedding_provider(models)


class TestSanitizeApiBase:
    """Endpoint sanitization tests."""

    def test_strips_path_and_query(self) -> None:
        assert sanitize_api_base("https://example.test/v1?key=abc") == "https://example.test"

    def test_strips_credentials(self) -> None:
        assert sanitize_api_base("https://user:pass@example.test/v1") == "https://example.test"

    def test_none_returns_none(self) -> None:
        assert sanitize_api_base(None) is None

    def test_empty_returns_none(self) -> None:
        assert sanitize_api_base("") is None

    def test_malformed_returns_configured(self) -> None:
        assert sanitize_api_base("not-a-url") == "<configured>"

    def test_preserves_port(self) -> None:
        """Port numbers are preserved in sanitized endpoint."""
        assert sanitize_api_base("http://localhost:8000/v1") == "http://localhost:8000"
        assert sanitize_api_base("https://example.test:9000/v1") == "https://example.test:9000"

    def test_different_ports_are_distinct(self) -> None:
        """Different ports produce different sanitized endpoints."""
        ep1 = sanitize_api_base("http://localhost:8000/v1")
        ep2 = sanitize_api_base("http://localhost:9000/v1")
        assert ep1 != ep2

    def test_strips_fragment(self) -> None:
        """Fragment is stripped from sanitized endpoint."""
        assert sanitize_api_base("https://example.test/v1#section") == "https://example.test"


class TestPreflightIntegration:
    """Preflight uses the same provider builder as the runner."""

    def test_preflight_without_probe_makes_no_api_call(self) -> None:
        from experiments.trustparadox_u.config import (
            DetectorConfig,
            ExperimentConfig,
            HistoryConfig,
            MonitoringConfig,
            PolicyConfig,
        )
        from experiments.trustparadox_u.preflight import run_preflight

        config = ExperimentConfig(
            seed=42,
            repetitions=1,
            detector=DetectorConfig(embedding_enabled=False),
            history=HistoryConfig(),
            policy=PolicyConfig(),
            monitoring=MonitoringConfig(),
        )
        failures = run_preflight(config, probe_provider=False)
        assert failures == []

    def test_custom_endpoint_without_api_key_env_passes(self) -> None:
        """Custom endpoint without api_key_env doesn't require standard keys."""
        import os
        from unittest.mock import patch

        from experiments.trustparadox_u.config import (
            DetectorConfig,
            ExperimentConfig,
            HistoryConfig,
            ModelsConfig,
            MonitoringConfig,
            PolicyConfig,
            RunConfig,
        )
        from experiments.trustparadox_u.preflight import run_preflight

        config = ExperimentConfig(
            seed=42,
            repetitions=1,
            detector=DetectorConfig(embedding_enabled=True),
            history=HistoryConfig(),
            policy=PolicyConfig(),
            monitoring=MonitoringConfig(),
            run=RunConfig(mode="experiment"),
            models=ModelsConfig(
                embedding_provider="litellm",
                embedding_model="openai/text-embedding-v3",
                api_base="http://localhost:8000/v1",
            ),
        )

        # Clear any API keys from environment
        with patch.dict(os.environ, {}, clear=True):
            failures = run_preflight(config, probe_provider=False)
            # Should not fail due to missing standard API keys
            assert not any("API key" in f for f in failures)

    def test_custom_endpoint_with_missing_api_key_env_fails(self) -> None:
        """Custom endpoint with api_key_env requires that variable."""
        import os
        from unittest.mock import patch

        from experiments.trustparadox_u.config import (
            DetectorConfig,
            ExperimentConfig,
            HistoryConfig,
            ModelsConfig,
            MonitoringConfig,
            PolicyConfig,
            RunConfig,
        )
        from experiments.trustparadox_u.preflight import run_preflight

        config = ExperimentConfig(
            seed=42,
            repetitions=1,
            detector=DetectorConfig(embedding_enabled=True),
            history=HistoryConfig(),
            policy=PolicyConfig(),
            monitoring=MonitoringConfig(),
            run=RunConfig(mode="experiment"),
            models=ModelsConfig(
                embedding_provider="litellm",
                embedding_model="openai/text-embedding-v3",
                api_base="http://localhost:8000/v1",
                api_key_env="CUSTOM_API_KEY",
            ),
        )

        with patch.dict(os.environ, {}, clear=True):
            failures = run_preflight(config, probe_provider=False)
            assert any("CUSTOM_API_KEY" in f for f in failures)

    def test_custom_endpoint_with_api_key_env_set_passes(self) -> None:
        """Custom endpoint with api_key_env set passes."""
        import os
        from unittest.mock import patch

        from experiments.trustparadox_u.config import (
            DetectorConfig,
            ExperimentConfig,
            HistoryConfig,
            ModelsConfig,
            MonitoringConfig,
            PolicyConfig,
            RunConfig,
        )
        from experiments.trustparadox_u.preflight import run_preflight

        config = ExperimentConfig(
            seed=42,
            repetitions=1,
            detector=DetectorConfig(embedding_enabled=True),
            history=HistoryConfig(),
            policy=PolicyConfig(),
            monitoring=MonitoringConfig(),
            run=RunConfig(mode="experiment"),
            models=ModelsConfig(
                embedding_provider="litellm",
                embedding_model="openai/text-embedding-v3",
                api_base="http://localhost:8000/v1",
                api_key_env="CUSTOM_API_KEY",
            ),
        )

        with patch.dict(os.environ, {"CUSTOM_API_KEY": "test-key"}):
            failures = run_preflight(config, probe_provider=False)
            assert not any("CUSTOM_API_KEY" in f for f in failures)
