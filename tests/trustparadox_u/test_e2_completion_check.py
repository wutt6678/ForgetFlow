"""Tests for E2 completion checker (E2 repair §40-51)."""

from __future__ import annotations

from experiments.trustparadox_u.empirical_corpus import EmpiricalPhase
from experiments.trustparadox_u.run_e2_completion_check import (
    check_annotation_independence,
    check_bounded_revision,
    check_generator_freeze,
    check_model_consistency,
    check_phase_state,
    check_primary_pilot_task,
    check_protocol_consistency,
    check_schedule,
    check_statistics,
    run_completion_check,
)


class TestProtocolConsistency:
    """Tests for protocol consistency check (E2 repair §41)."""

    def test_passes_with_correct_versions(self) -> None:
        """Test that check passes with correct versions."""
        artifacts = {
            "manifest": {"protocol_version": "2.0.0", "study_version": "2.0.0"},
        }
        result = check_protocol_consistency(artifacts)
        assert result.passed is True

    def test_fails_with_wrong_protocol(self) -> None:
        """Test that check fails with wrong protocol version."""
        artifacts = {
            "manifest": {"protocol_version": "1.0.0", "study_version": "2.0.0"},
        }
        result = check_protocol_consistency(artifacts)
        assert result.passed is False
        assert result.failure_code == "empirical_protocol_version_mismatch"


class TestPhaseState:
    """Tests for phase state check (E2 repair §42)."""

    def test_passes_with_frozen_phase(self) -> None:
        """Test that check passes with E2_PROMPTS_FROZEN."""
        phase_file = {
            "phase": EmpiricalPhase.E2_PROMPTS_FROZEN.value,
            "trust_prompts_frozen": True,
            "full_corpus_generation_authorized": False,
        }
        result = check_phase_state(phase_file)
        assert result.passed is True

    def test_fails_with_missing_phase(self) -> None:
        """Test that check fails with missing phase file."""
        result = check_phase_state(None)
        assert result.passed is False
        assert result.failure_code == "empirical_phase_file_missing"

    def test_fails_with_wrong_phase(self) -> None:
        """Test that check fails with wrong phase."""
        phase_file = {
            "phase": EmpiricalPhase.E2_TRUST_PILOT.value,
            "trust_prompts_frozen": False,
        }
        result = check_phase_state(phase_file)
        assert result.passed is False
        assert result.failure_code == "empirical_phase_not_frozen"

    def test_fails_with_premature_authorization(self) -> None:
        """Test that check fails with premature corpus authorization."""
        phase_file = {
            "phase": EmpiricalPhase.E2_PROMPTS_FROZEN.value,
            "trust_prompts_frozen": True,
            "full_corpus_generation_authorized": True,
        }
        result = check_phase_state(phase_file)
        assert result.passed is False
        assert result.failure_code == "full_corpus_prematurely_authorized"


class TestModelConsistency:
    """Tests for model consistency check (E2 repair §43)."""

    def test_passes_with_matching_models(self) -> None:
        """Test that check passes with matching models."""
        connectivity = {"provider": "openai", "model": "qwen3.7-plus"}
        pilot = {"provider": "openai", "model": "qwen3.7-plus"}
        result = check_model_consistency(connectivity, pilot)
        assert result.passed is True

    def test_fails_with_mismatched_models(self) -> None:
        """Test that check fails with mismatched models."""
        connectivity = {"provider": "openai", "model": "qwen3.7-plus"}
        pilot = {"provider": "openai", "model": "different-model"}
        result = check_model_consistency(connectivity, pilot)
        assert result.passed is False
        assert result.failure_code == "connectivity_pilot_model_mismatch"


class TestPrimaryPilotTask:
    """Tests for primary pilot task check (E2 repair §44)."""

    def test_passes_with_trust_discretion(self) -> None:
        """Test that check passes with trust_discretion_task."""
        manifest = {"attack_type": "trust_discretion_task"}
        result = check_primary_pilot_task(manifest)
        assert result.passed is True

    def test_fails_with_direct_disclosure(self) -> None:
        """Test that check fails with direct_disclosure."""
        manifest = {"attack_type": "direct_disclosure"}
        result = check_primary_pilot_task(manifest)
        assert result.passed is False
        assert result.failure_code == "primary_pilot_uses_explicit_disclosure_attack"


class TestSchedule:
    """Tests for schedule check (E2 repair §45)."""

    def test_passes_with_complete_schedule(self) -> None:
        """Test that check passes with complete schedule."""
        requests = [
            {"generation_family_id": f"family_{i}", "trust_level": level}
            for i in range(30)
            for level in ["low", "default", "high"]
        ]
        schedule = {"requests": requests}
        result = check_schedule(schedule)
        assert result.passed is True

    def test_fails_with_missing_schedule(self) -> None:
        """Test that check fails with missing schedule."""
        result = check_schedule(None)
        assert result.passed is False
        assert result.failure_code == "pilot_schedule_missing"

    def test_fails_with_wrong_request_count(self) -> None:
        """Test that check fails with wrong request count."""
        schedule = {"requests": [{"generation_family_id": "f1", "trust_level": "low"}] * 50}
        result = check_schedule(schedule)
        assert result.passed is False


class TestAnnotationIndependence:
    """Tests for annotation independence check (E2 repair §46)."""

    def test_passes_with_independence_enforced(self) -> None:
        """Test that check passes with independence enforced."""
        labels_report = {
            "evaluator_independence": {
                "independence_enforced": True,
                "generator_evaluator_id": "generator_g",
                "labeling_evaluator_id": "independent_evaluator_j",
            },
        }
        result = check_annotation_independence(labels_report)
        assert result.passed is True

    def test_fails_without_independence(self) -> None:
        """Test that check fails without independence enforced."""
        labels_report = {"evaluator_independence": {"independence_enforced": False}}
        result = check_annotation_independence(labels_report)
        assert result.passed is False
        assert result.failure_code == "pilot_annotation_not_independent"


class TestStatistics:
    """Tests for statistics check (E2 repair §47)."""

    def test_passes_with_complete_statistics(self) -> None:
        """Test that check passes with complete statistics."""
        analysis = {
            "complete_families": 30,
            "high_minus_low_risk_difference": 0.15,
            "bootstrap_ci_lower": 0.05,
            "bootstrap_ci_upper": 0.25,
        }
        result = check_statistics(analysis)
        assert result.passed is True

    def test_fails_with_incomplete_families(self) -> None:
        """Test that check fails with incomplete families."""
        analysis = {"complete_families": 25}
        result = check_statistics(analysis)
        assert result.passed is False
        assert result.failure_code == "pilot_pairing_incomplete"


class TestBoundedRevision:
    """Tests for bounded revision check (E2 repair §48)."""

    def test_passes_with_valid_version(self) -> None:
        """Test that check passes with valid version."""
        manifest = {"selected_pilot_version": "E2_PRIMARY_V1"}
        result = check_bounded_revision(manifest)
        assert result.passed is True

    def test_fails_with_invalid_version(self) -> None:
        """Test that check fails with invalid version."""
        manifest = {"selected_pilot_version": "INVALID"}
        result = check_bounded_revision(manifest)
        assert result.passed is False


class TestGeneratorFreeze:
    """Tests for generator freeze check (E2 repair §49)."""

    def test_passes_with_complete_freeze(self) -> None:
        """Test that check passes with complete freeze."""
        manifest = {
            "generator_provider": "openai",
            "generator_model_requested": "qwen3.7-plus",
            "generator_temperature": 0.3,
            "generator_max_tokens": 512,
            "pilot_execution_seed": 20260808,
            "system_prompt_hash": "abc123",
            "status": "frozen_after_E2",
        }
        result = check_generator_freeze(manifest)
        assert result.passed is True

    def test_fails_with_missing_fields(self) -> None:
        """Test that check fails with missing fields."""
        manifest = {"generator_provider": "openai"}
        result = check_generator_freeze(manifest)
        assert result.passed is False
        assert result.failure_code == "generator_configuration_not_frozen"


class TestRunCompletionCheck:
    """Tests for complete completion check (E2 repair §40-51)."""

    def test_run_completion_check_all_pass(self) -> None:
        """Test that all checks pass with valid inputs."""
        report = run_completion_check(
            artifacts={"manifest": {"protocol_version": "2.0.0", "study_version": "2.0.0"}},
            phase_file={
                "phase": EmpiricalPhase.E2_PROMPTS_FROZEN.value,
                "trust_prompts_frozen": True,
                "full_corpus_generation_authorized": False,
            },
            connectivity_config={"provider": "openai", "model": "qwen3.7-plus"},
            pilot_config={"provider": "openai", "model": "qwen3.7-plus"},
            pilot_manifest={"attack_type": "trust_discretion_task"},
            schedule={
                "requests": [
                    {"generation_family_id": f"family_{i}", "trust_level": level}
                    for i in range(30)
                    for level in ["low", "default", "high"]
                ],
            },
            labels_report={
                "evaluator_independence": {"independence_enforced": True},
            },
            analysis={
                "complete_families": 30,
                "high_minus_low_risk_difference": 0.15,
                "bootstrap_ci_lower": 0.05,
                "bootstrap_ci_upper": 0.25,
            },
            freeze_manifest={
                "selected_pilot_version": "E2_PRIMARY_V1",
                "generator_provider": "openai",
                "generator_model_requested": "qwen3.7-plus",
                "generator_temperature": 0.3,
                "generator_max_tokens": 512,
                "pilot_execution_seed": 20260808,
                "system_prompt_hash": "abc123",
                "status": "frozen_after_E2",
            },
        )
        assert report.all_passed is True
        assert len(report.checks) == 9
