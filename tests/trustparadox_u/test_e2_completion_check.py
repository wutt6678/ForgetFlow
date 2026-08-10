"""Tests for E2 completion checker (E2 repair §40-51, E2R-021-025, E2R-034/035)."""

from __future__ import annotations

from experiments.trustparadox_u.empirical_corpus import EmpiricalPhase
from experiments.trustparadox_u.run_e2_completion_check import (
    check_annotation_independence,
    check_artifact_hash_binding,
    check_bounded_revision,
    check_evaluator_connectivity,
    check_evaluator_freeze,
    check_evaluator_model_identity,
    check_floor_effect_diagnostic,
    check_generator_evaluator_independence,
    check_generator_freeze,
    check_human_review_completion,
    check_model_consistency,
    check_pairing_audit,
    check_phase_state,
    check_primary_label_completeness,
    check_primary_pilot_task,
    check_protocol_consistency,
    check_schedule,
    check_statistics,
    check_synthetic_regression,
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
    """Tests for primary pilot task check (E2R-022)."""

    def test_passes_with_explicit_metadata(self) -> None:
        """Test that check passes with explicit pilot_type and attack_type."""
        manifest = {
            "pilot_type": "primary_trust_discretion",
            "attack_type": "trust_discretion_task",
        }
        result = check_primary_pilot_task(manifest)
        assert result.passed is True

    def test_fails_with_missing_metadata(self) -> None:
        """Test that check fails with missing pilot_type or attack_type."""
        manifest = {"attack_type": "trust_discretion_task"}
        result = check_primary_pilot_task(manifest)
        assert result.passed is False
        assert result.failure_code == "primary_pilot_metadata_missing"

    def test_fails_with_direct_disclosure(self) -> None:
        """Test that check fails with direct_disclosure."""
        manifest = {
            "pilot_type": "primary_trust_discretion",
            "attack_type": "direct_disclosure",
        }
        result = check_primary_pilot_task(manifest)
        assert result.passed is False
        assert result.failure_code == "primary_pilot_uses_explicit_disclosure_attack"

    def test_fails_with_artifact_class_inference(self) -> None:
        """Test that check fails when only artifact_class is present (no inference)."""
        manifest = {"artifact_class": "trust_pilot"}
        result = check_primary_pilot_task(manifest)
        assert result.passed is False


class TestSchedule:
    """Tests for schedule check (E2R-023)."""

    def test_passes_with_complete_schedule(self) -> None:
        """Test that check passes with complete schedule."""
        requests = [
            {
                "generation_family_id": f"family_{i}",
                "trust_level": level,
                "scenario_id": "credential_001",
                "secret_variant_id": "v1",
                "attack_type": "trust_discretion_task",
            }
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

    def test_fails_with_inconsistent_scenario_within_family(self) -> None:
        """Test that check fails with different scenario_ids within family."""
        requests = [
            {
                "generation_family_id": "family_0",
                "trust_level": "low",
                "scenario_id": "credential_001",
                "secret_variant_id": "v1",
                "attack_type": "trust_discretion_task",
            },
            {
                "generation_family_id": "family_0",
                "trust_level": "default",
                "scenario_id": "private_attribute_001",  # Different!
                "secret_variant_id": "v1",
                "attack_type": "trust_discretion_task",
            },
            {
                "generation_family_id": "family_0",
                "trust_level": "high",
                "scenario_id": "credential_001",
                "secret_variant_id": "v1",
                "attack_type": "trust_discretion_task",
            },
        ]
        # Add 29 more complete families
        for i in range(1, 30):
            for level in ["low", "default", "high"]:
                requests.append(
                    {
                        "generation_family_id": f"family_{i}",
                        "trust_level": level,
                        "scenario_id": "credential_001",
                        "secret_variant_id": "v1",
                        "attack_type": "trust_discretion_task",
                    }
                )
        schedule = {"requests": requests}
        result = check_schedule(schedule)
        assert result.passed is False
        assert result.failure_code == "pilot_family_inconsistent"


class TestAnnotationIndependence:
    """Tests for annotation independence check (E2R-021)."""

    def test_passes_with_complete_metadata(self) -> None:
        """Test that check passes with complete evaluator metadata."""
        labels_report = {
            "evaluator_provider": "aliyun",
            "evaluator_model_requested": "qwen3.8-max",
            "evaluator_model_returned": "qwen3.8-max",
            "evaluator_prompt_hash": "abc123",
            "generator_model": "qwen3.7-plus",
            "primary_label_sha256": "def456",
            "num_labeled_attempts": 90,
            "num_resolved_labels": 90,
        }
        result = check_annotation_independence(labels_report)
        assert result.passed is True

    def test_fails_with_missing_evaluator_model(self) -> None:
        """Test that check fails with missing evaluator model."""
        labels_report = {
            "evaluator_provider": "aliyun",
            "evaluator_model_requested": "",
            "evaluator_model_returned": "qwen3.8-max",
        }
        result = check_annotation_independence(labels_report)
        assert result.passed is False
        assert result.failure_code == "pilot_evaluator_model_missing"

    def test_fails_with_same_generator_evaluator(self) -> None:
        """Test that check fails when generator == evaluator."""
        labels_report = {
            "evaluator_provider": "aliyun",
            "evaluator_model_requested": "qwen3.7-plus",
            "evaluator_model_returned": "qwen3.7-plus",
            "evaluator_prompt_hash": "abc123",
            "generator_model": "qwen3.7-plus",
            "primary_label_sha256": "def456",
            "num_labeled_attempts": 90,
            "num_resolved_labels": 90,
        }
        result = check_annotation_independence(labels_report)
        assert result.passed is False
        assert result.failure_code == "pilot_evaluator_same_as_generator"

    def test_fails_with_incomplete_labels(self) -> None:
        """Test that check fails with incomplete labels."""
        labels_report = {
            "evaluator_provider": "aliyun",
            "evaluator_model_requested": "qwen3.8-max",
            "evaluator_model_returned": "qwen3.8-max",
            "evaluator_prompt_hash": "abc123",
            "generator_model": "qwen3.7-plus",
            "primary_label_sha256": "def456",
            "num_labeled_attempts": 85,
            "num_resolved_labels": 85,
        }
        result = check_annotation_independence(labels_report)
        assert result.passed is False
        assert result.failure_code == "pilot_primary_labels_incomplete"


class TestStatistics:
    """Tests for statistics check (E2R-024)."""

    def test_passes_with_complete_statistics(self) -> None:
        """Test that check passes with complete statistics."""
        analysis = {
            "matched_family_count": 30,
            "excluded_family_count": 0,
            "pairing_unit": "generation_family_id",
            "high_minus_low_risk_difference": 0.15,
            "high_minus_low_ci95": [0.05, 0.25],
            "behavioral_refusal_effect": 0.10,
            "task_compliance_effect": 0.12,
        }
        result = check_statistics(analysis)
        assert result.passed is True

    def test_fails_with_incomplete_families(self) -> None:
        """Test that check fails with incomplete families."""
        analysis = {"matched_family_count": 25}
        result = check_statistics(analysis)
        assert result.passed is False
        assert result.failure_code == "pilot_pairing_incomplete"

    def test_fails_with_missing_pairing_unit(self) -> None:
        """Test that check fails with missing pairing_unit."""
        analysis = {
            "matched_family_count": 30,
            "excluded_family_count": 0,
            "pairing_unit": None,
        }
        result = check_statistics(analysis)
        assert result.passed is False
        assert result.failure_code == "pilot_pairing_unit_invalid"


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


class TestSyntheticRegression:
    """Tests for synthetic regression check (E2R-025)."""

    def test_passes_with_complete_report(self) -> None:
        """Test that check passes with complete synthetic regression report."""
        report = {
            "synthetic_release_id": "synthetic_v1",
            "scientific_release_digest": "sha256:abc123",
            "table_1_sha256": "hash1",
            "table_2_sha256": "hash2",
            "table_3_sha256": "hash3",
            "table_4_sha256": "hash4",
            "table_5_sha256": "hash5",
            "table_6_sha256": "hash6",
            "synthetic_gate_status": "synthetic_benchmark_valid",
        }
        result = check_synthetic_regression(report)
        assert result.passed is True

    def test_fails_with_missing_report(self) -> None:
        """Test that check fails with missing report."""
        result = check_synthetic_regression(None)
        assert result.passed is False
        assert result.failure_code == "synthetic_regression_report_missing"

    def test_fails_with_invalid_gate_status(self) -> None:
        """Test that check fails with invalid gate status."""
        report = {
            "synthetic_release_id": "synthetic_v1",
            "scientific_release_digest": "sha256:abc123",
            "table_1_sha256": "hash1",
            "table_2_sha256": "hash2",
            "table_3_sha256": "hash3",
            "table_4_sha256": "hash4",
            "table_5_sha256": "hash5",
            "table_6_sha256": "hash6",
            "synthetic_gate_status": "invalid",
        }
        result = check_synthetic_regression(report)
        assert result.passed is False
        assert result.failure_code == "synthetic_gate_invalid"


class TestAdditionalChecks:
    """Tests for additional E2R-034 checks."""

    def test_evaluator_model_identity(self) -> None:
        """Test evaluator model identity check."""
        labels_report = {
            "evaluator_provider": "aliyun",
            "evaluator_model_requested": "qwen3.8-max",
        }
        result = check_evaluator_model_identity(labels_report)
        assert result.passed is True

    def test_generator_evaluator_independence(self) -> None:
        """Test generator-evaluator independence check."""
        labels_report = {
            "generator_model": "qwen3.7-plus",
            "evaluator_model_returned": "qwen3.8-max",
        }
        result = check_generator_evaluator_independence(labels_report)
        assert result.passed is True

    def test_evaluator_connectivity(self) -> None:
        """Test evaluator connectivity check."""
        labels_report = {
            "evaluator_provider": "aliyun",
            "evaluator_model_requested": "qwen3.8-max",
        }
        result = check_evaluator_connectivity(labels_report)
        assert result.passed is True

    def test_primary_label_completeness(self) -> None:
        """Test primary label completeness check."""
        labels_report = {
            "num_labeled_attempts": 90,
            "num_resolved_labels": 90,
        }
        result = check_primary_label_completeness(labels_report)
        assert result.passed is True

    def test_human_review_completion(self) -> None:
        """Test human review completion check."""
        labels_report = {
            "num_review_required": 5,
            "num_adjudicated": 5,
        }
        result = check_human_review_completion(labels_report)
        assert result.passed is True

    def test_pairing_audit(self) -> None:
        """Test pairing audit check."""
        analysis = {
            "pairing_audit": {"audit_status": "passed"},
        }
        result = check_pairing_audit(analysis)
        assert result.passed is True

    def test_floor_effect_diagnostic(self) -> None:
        """Test floor effect diagnostic check."""
        analysis = {
            "floor_effect_diagnostic": {"status": "floor_effect_detected"},
        }
        result = check_floor_effect_diagnostic(analysis)
        assert result.passed is True

    def test_evaluator_freeze(self) -> None:
        """Test evaluator freeze check."""
        labels_report = {
            "evaluator_prompt_hash": "abc123",
            "evaluator_model_revision": "v1",
        }
        result = check_evaluator_freeze(labels_report)
        assert result.passed is True

    def test_artifact_hash_binding(self) -> None:
        """Test artifact hash binding check."""
        artifact_hashes = {
            "raw_pilot_attempts": "hash1",
            "request_schedule": "hash2",
            "generator_prompt_manifest": "hash3",
            "evaluator_prompt_manifest": "hash4",
            "primary_labels": "hash5",
            "reference_labels": "hash6",
            "adjudication_log": "hash7",
            "pairing_audit": "hash8",
            "pilot_analysis": "hash9",
            "floor_effect_diagnostic": "hash10",
            "bounded_revision_report": "hash11",
            "frozen_prompt_manifest": "hash12",
            "synthetic_regression_report": "hash13",
        }
        result = check_artifact_hash_binding(artifact_hashes)
        assert result.passed is True


class TestRunCompletionCheck:
    """Tests for complete completion check (E2R-034)."""

    def test_run_completion_check_all_pass(self) -> None:
        """Test that all checks pass with valid inputs."""
        labels_report = {
            "evaluator_provider": "aliyun",
            "evaluator_model_requested": "qwen3.8-max",
            "evaluator_model_returned": "qwen3.8-max",
            "evaluator_prompt_hash": "abc123",
            "evaluator_model_revision": "v1",
            "generator_model": "qwen3.7-plus",
            "primary_label_sha256": "def456",
            "num_labeled_attempts": 90,
            "num_resolved_labels": 90,
            "num_review_required": 0,
            "num_adjudicated": 0,
        }
        analysis = {
            "matched_family_count": 30,
            "excluded_family_count": 0,
            "pairing_unit": "generation_family_id",
            "high_minus_low_risk_difference": 0.15,
            "high_minus_low_ci95": [0.05, 0.25],
            "behavioral_refusal_effect": 0.10,
            "task_compliance_effect": 0.12,
            "pairing_audit": {"audit_status": "passed"},
            "floor_effect_diagnostic": {"status": "floor_effect_detected"},
        }
        artifact_hashes = {
            "raw_pilot_attempts": "hash1",
            "request_schedule": "hash2",
            "generator_prompt_manifest": "hash3",
            "evaluator_prompt_manifest": "hash4",
            "primary_labels": "hash5",
            "reference_labels": "hash6",
            "adjudication_log": "hash7",
            "pairing_audit": "hash8",
            "pilot_analysis": "hash9",
            "floor_effect_diagnostic": "hash10",
            "bounded_revision_report": "hash11",
            "frozen_prompt_manifest": "hash12",
            "synthetic_regression_report": "hash13",
        }
        synthetic_report = {
            "synthetic_release_id": "synthetic_v1",
            "scientific_release_digest": "sha256:abc123",
            "table_1_sha256": "hash1",
            "table_2_sha256": "hash2",
            "table_3_sha256": "hash3",
            "table_4_sha256": "hash4",
            "table_5_sha256": "hash5",
            "table_6_sha256": "hash6",
            "synthetic_gate_status": "synthetic_benchmark_valid",
        }
        report = run_completion_check(
            artifacts={"manifest": {"protocol_version": "2.0.0", "study_version": "2.0.0"}},
            phase_file={
                "phase": EmpiricalPhase.E2_PROMPTS_FROZEN.value,
                "trust_prompts_frozen": True,
                "full_corpus_generation_authorized": False,
            },
            connectivity_config={"provider": "openai", "model": "qwen3.7-plus"},
            pilot_config={"provider": "openai", "model": "qwen3.7-plus"},
            pilot_manifest={
                "pilot_type": "primary_trust_discretion",
                "attack_type": "trust_discretion_task",
            },
            schedule={
                "requests": [
                    {
                        "generation_family_id": f"family_{i}",
                        "trust_level": level,
                        "scenario_id": "credential_001",
                        "secret_variant_id": "v1",
                        "attack_type": "trust_discretion_task",
                    }
                    for i in range(30)
                    for level in ["low", "default", "high"]
                ],
            },
            labels_report=labels_report,
            analysis=analysis,
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
            synthetic_regression_report=synthetic_report,
            artifact_hashes=artifact_hashes,
        )
        assert report.all_passed is True
        assert len(report.checks) == 20
        assert report.research_status == "empirical_pilot_complete"
        assert len(report.artifact_hashes) == 13
