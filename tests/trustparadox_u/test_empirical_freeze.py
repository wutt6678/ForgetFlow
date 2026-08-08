"""Tests for empirical freeze module (E2 repair §37-39)."""

from __future__ import annotations

from experiments.trustparadox_u.empirical_corpus import EmpiricalPhase
from experiments.trustparadox_u.empirical_freeze import (
    create_e2_iteration_manifest,
    create_frozen_prompt_manifest,
    create_phase_transition_manifest,
)


class TestFrozenPromptManifest:
    """Tests for frozen prompt manifest (E2 repair §37)."""

    def test_create_frozen_manifest(self) -> None:
        """Test creating a frozen prompt manifest."""
        manifest = create_frozen_prompt_manifest(
            selected_pilot_version="E2_PRIMARY_V1",
            generator_provider="openai",
            generator_model="openai/qwen3.7-plus",
            generator_temperature=0.3,
            generator_max_tokens=512,
            pilot_execution_seed=20260808,
            manipulation_strength="strong",
            manipulation_analysis_sha256="abc123",
            prompt_hashes={
                "system": "sys_hash",
                "low": "low_hash",
                "default": "default_hash",
                "high": "high_hash",
                "primary_task": "task_hash",
            },
        )
        assert manifest.status == "frozen_after_E2"
        assert manifest.selected_pilot_version == "E2_PRIMARY_V1"
        assert manifest.generator_provider == "openai"
        assert manifest.manipulation_strength == "strong"

    def test_frozen_manifest_to_dict(self) -> None:
        """Test conversion to dict."""
        manifest = create_frozen_prompt_manifest(
            selected_pilot_version="E2_PRIMARY_V1",
            generator_provider="openai",
            generator_model="openai/qwen3.7-plus",
            generator_temperature=0.3,
            generator_max_tokens=512,
            pilot_execution_seed=20260808,
            manipulation_strength="strong",
            manipulation_analysis_sha256="abc123",
            prompt_hashes={},
        )
        d = manifest.to_dict()
        assert d["schema_version"] == "1.0.0"
        assert d["protocol_version"] == "2.0.0"
        assert d["status"] == "frozen_after_E2"


class TestE2IterationManifest:
    """Tests for E2 iteration manifest (E2 repair §38)."""

    def test_create_e2_manifest(self) -> None:
        """Test creating an E2 iteration manifest."""
        manifest = create_e2_iteration_manifest(
            selected_pilot_version="E2_PRIMARY_V1",
            pilot_config_sha256="config_hash",
            request_schedule_sha256="schedule_hash",
            raw_generation_sha256="raw_hash",
            primary_labels_sha256="labels_hash",
            analysis_sha256="analysis_hash",
            frozen_prompt_manifest_sha256="frozen_hash",
            planned_requests=90,
            actual_attempts=90,
            successful_responses=85,
            eligible_responses=80,
            matched_family_count=27,
            low_rate=0.10,
            default_rate=0.15,
            high_rate=0.25,
            high_low_risk_difference=0.15,
            high_low_ci95=[0.05, 0.25],
            manipulation_strength="strong",
            connectivity_provider="openai",
            connectivity_model="openai/qwen3.7-plus",
            pilot_provider="openai",
            pilot_model="openai/qwen3.7-plus",
        )
        assert manifest.iteration == "E2"
        assert manifest.planned_requests == 90
        assert manifest.prompts_frozen is True
        assert manifest.e2_complete is True
        assert manifest.connectivity_pilot_model_match is True

    def test_e2_manifest_to_dict(self) -> None:
        """Test conversion to dict."""
        manifest = create_e2_iteration_manifest(
            selected_pilot_version="E2_PRIMARY_V1",
            pilot_config_sha256="config_hash",
            request_schedule_sha256="schedule_hash",
            raw_generation_sha256="raw_hash",
            primary_labels_sha256="labels_hash",
            analysis_sha256="analysis_hash",
            frozen_prompt_manifest_sha256="frozen_hash",
            planned_requests=90,
            actual_attempts=90,
            successful_responses=85,
            eligible_responses=80,
            matched_family_count=27,
            low_rate=0.10,
            default_rate=0.15,
            high_rate=0.25,
            high_low_risk_difference=0.15,
            high_low_ci95=[0.05, 0.25],
            manipulation_strength="strong",
            connectivity_provider="openai",
            connectivity_model="openai/qwen3.7-plus",
            pilot_provider="openai",
            pilot_model="openai/qwen3.7-plus",
        )
        d = manifest.to_dict()
        assert d["iteration"] == "E2"
        assert d["protocol_version"] == "2.0.0"
        assert d["prompts_frozen"] is True
        assert d["validation_generation_unlocked"] is False


class TestPhaseTransition:
    """Tests for phase transition (E2 repair §39)."""

    def test_create_phase_transition_manifest(self) -> None:
        """Test creating phase transition manifest."""
        manifest = create_phase_transition_manifest("prompt_manifest_sha256")
        assert manifest["phase"] == EmpiricalPhase.E2_PROMPTS_FROZEN.value
        assert manifest["trust_prompts_frozen"] is True
        assert manifest["full_corpus_generation_authorized"] is False
        assert manifest["prompt_manifest_sha256"] == "prompt_manifest_sha256"

    def test_phase_transition_has_required_fields(self) -> None:
        """Test that phase transition manifest has all required fields."""
        manifest = create_phase_transition_manifest("sha256")
        assert "schema_version" in manifest
        assert "protocol_version" in manifest
        assert "study_version" in manifest
        assert "phase" in manifest
        assert "timestamp" in manifest
