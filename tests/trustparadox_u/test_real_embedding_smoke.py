"""Real-embedding smoke study scaffold (ST-SEM-REAL).

This test validates that the real-embedding path executes correctly
without silent fallback. It requires a real embedding provider.

If no provider is configured, the tests are skipped.
"""

from __future__ import annotations

import os
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
)
from experiments.trustparadox_u.dataset import load_episode
from experiments.trustparadox_u.runner import run_episode

SCENARIOS_DIR = Path(__file__).parents[2] / "data" / "trustparadox_u" / "scenarios"


def _has_real_embedding_provider() -> bool:
    """Check if a real embedding provider is configured."""
    # Check for litellm provider configuration
    provider = os.environ.get("EMBEDDING_PROVIDER", "")
    api_key = os.environ.get("LITELLM_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    return provider == "litellm" and bool(api_key)


skip_no_provider = pytest.mark.skipif(
    not _has_real_embedding_provider(),
    reason="No real embedding provider configured",
)


@pytest.mark.integration
@skip_no_provider
class TestRealEmbeddingSmoke:
    """Real-embedding smoke study assertions."""

    def test_provider_is_not_fixed(self) -> None:
        """Provider is not 'fixed' in real-embedding mode."""
        cfg = ExperimentConfig(
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
                embedding_dimension=1024,
                api_base="https://llm-jhxtd03gjg0gd2o2.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
            ),
        )
        assert cfg.models.embedding_provider != "fixed"
        assert cfg.models.embedding_provider == "litellm"

    def test_real_embedding_produces_nonzero_semantic_scores(self) -> None:
        """Real embeddings produce non-constant semantic scores."""
        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        cfg = ExperimentConfig(
            seed=42,
            repetitions=1,
            detector=DetectorConfig(embedding_enabled=True, embedding_threshold=0.80),
            history=HistoryConfig(),
            policy=PolicyConfig(),
            monitoring=MonitoringConfig(),
            run=RunConfig(mode="experiment"),
            models=ModelsConfig(
                embedding_provider="litellm",
                embedding_model="openai/text-embedding-v3",
                embedding_dimension=1024,
                api_base="https://llm-jhxtd03gjg0gd2o2.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
            ),
        )
        result = run_episode(ep, cfg)
        # Verify semantic scores are present and not all zero
        semantic_scores = []
        for turn in result.turns:
            if turn.decision and turn.decision.detector_result:
                semantic_scores.append(turn.decision.detector_result.semantic_score)
        # At least some scores should be non-zero
        assert any(s > 0 for s in semantic_scores), "All semantic scores are zero"

    def test_audit_passes_with_real_embeddings(self) -> None:
        """Full audit passes with real embedding results (not just FORBIDDEN check)."""
        from experiments.trustparadox_u.audit_results import AuditReport, audit_episode_result

        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        cfg = ExperimentConfig(
            seed=42,
            repetitions=1,
            detector=DetectorConfig(embedding_enabled=True, embedding_threshold=0.80),
            history=HistoryConfig(),
            policy=PolicyConfig(),
            monitoring=MonitoringConfig(),
            run=RunConfig(mode="experiment"),
            models=ModelsConfig(
                embedding_provider="litellm",
                embedding_model="openai/text-embedding-v3",
                embedding_dimension=1024,
                api_base="https://llm-jhxtd03gjg0gd2o2.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
            ),
        )
        result = run_episode(ep, cfg)
        findings = audit_episode_result(result)
        report = AuditReport(findings=findings, episodes_audited=1)
        # Full audit must pass, not just FORBIDDEN check
        assert not report.has_errors, f"Audit has errors: {[f.message for f in report.errors()]}"

    def test_provider_provenance_recorded(self) -> None:
        """Provider, model, dimension, and endpoint provenance are recorded."""
        ep = load_episode(SCENARIOS_DIR / "pilot_credential.yaml")
        cfg = ExperimentConfig(
            seed=42,
            repetitions=1,
            detector=DetectorConfig(embedding_enabled=True, embedding_threshold=0.80),
            history=HistoryConfig(),
            policy=PolicyConfig(),
            monitoring=MonitoringConfig(),
            run=RunConfig(mode="experiment"),
            models=ModelsConfig(
                embedding_provider="litellm",
                embedding_model="openai/text-embedding-v3",
                embedding_dimension=1024,
                api_base="https://llm-jhxtd03gjg0gd2o2.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
            ),
        )
        result = run_episode(ep, cfg)
        # Verify provenance is recorded in metadata
        assert result.metadata.get("embedding_provider") == "litellm"
        assert result.metadata.get("embedding_model") == "openai/text-embedding-v3"
        assert result.metadata.get("embedding_dimension", 0) > 0
