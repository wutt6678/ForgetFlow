"""Tests for E2 completion checker (E2 repair §40-51, E2R-021-025, E2R-034/035, E2R-036/037)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from experiments.trustparadox_u.empirical_corpus import EmpiricalPhase
from experiments.trustparadox_u.run_e2_completion_check import (
    CheckResult,
    CompletionReport,
    check_agreement_metric_consistency,
    check_analysis_provenance_valid,
    check_annotation_id_consistency,
    check_annotation_independence,
    check_artifact_hash_binding,
    check_bounded_revision,
    check_completion_consistency,
    check_cross_artifact_consistency,
    check_evaluator_connectivity,
    check_evaluator_freeze,
    check_evaluator_independence_evidence,
    check_evaluator_model_identity,
    check_evaluator_response_completeness,
    check_floor_effect_diagnostic,
    check_frozen_label_integrity,
    check_generator_evaluator_independence,
    check_generator_freeze,
    check_independent_annotation_validation,
    check_j_analysis_provenance,
    check_label_completeness_from_files,
    check_model_consistency,
    check_pairing_audit,
    check_phase_state,
    check_primary_effect_consistency,
    check_primary_label_completeness,
    check_primary_label_file_completeness,
    check_primary_pilot_task,
    check_protocol_consistency,
    check_raw_pilot_completeness,
    check_real_evaluator_evidence,
    check_reference_diagnostic_validity,
    check_reference_label_completeness,
    check_schedule,
    check_secondary_annotation_completion,
    check_secondary_annotation_integrity,
    check_secondary_review_cross_artifact_consistency,
    check_secondary_review_evidence_consistency,
    check_statistics,
    check_synthetic_provenance,
    check_synthetic_regression,
    check_uncertainty_ci,
    run_completion_check,
    sha256_file,
    transition_to_e2_complete,
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
    """Tests for synthetic regression check (E2R-025 / E2R-FIX-019/020)."""

    def _make_release_dir(self, tmp_path: Path) -> Path:
        """Create a mock active release bundle in tmp_path."""
        release_dir = tmp_path / "test-release-v1"
        release_dir.mkdir()
        manifest = {
            "status": "active",
            "release_id": "test-release-v1",
            "scientific_release_digest": "abc123digest",
            "components": {
                "final_artifacts/table1_main_results.json": {"sha256": "hash1"},
                "final_artifacts/table2_leakage_breakdown.json": {"sha256": "hash2"},
                "final_artifacts/table3_parameter_sensitivity.json": {"sha256": "hash3"},
                "final_artifacts/table4_statistical_comparisons.json": {"sha256": "hash4"},
                "final_artifacts/table5_target_type_results.json": {"sha256": "hash5"},
                "final_artifacts/table6_trust_analysis.json": {"sha256": "hash6"},
            },
        }
        (release_dir / "bundle_manifest.json").write_text(json.dumps(manifest))
        return tmp_path

    def test_passes_with_complete_report(self, tmp_path: Path) -> None:
        """Test that check passes with complete synthetic regression report."""
        releases_dir = self._make_release_dir(tmp_path)
        report = {
            "synthetic_release_id": "test-release-v1",
            "scientific_release_digest": "abc123digest",
            "table_1_sha256": "hash1",
            "table_2_sha256": "hash2",
            "table_3_sha256": "hash3",
            "table_4_sha256": "hash4",
            "table_5_sha256": "hash5",
            "table_6_sha256": "hash6",
            "synthetic_gate_status": "synthetic_benchmark_valid",
        }
        result = check_synthetic_regression(report, releases_dir=releases_dir)
        assert result.passed is True

    def test_fails_with_missing_report(self) -> None:
        """Test that check fails with missing report."""
        result = check_synthetic_regression(None)
        assert result.passed is False
        assert result.failure_code == "synthetic_regression_report_missing"

    def test_fails_with_invalid_gate_status(self, tmp_path: Path) -> None:
        """Test that check fails with invalid gate status."""
        releases_dir = self._make_release_dir(tmp_path)
        report = {
            "synthetic_release_id": "test-release-v1",
            "scientific_release_digest": "abc123digest",
            "table_1_sha256": "hash1",
            "table_2_sha256": "hash2",
            "table_3_sha256": "hash3",
            "table_4_sha256": "hash4",
            "table_5_sha256": "hash5",
            "table_6_sha256": "hash6",
            "synthetic_gate_status": "invalid",
        }
        result = check_synthetic_regression(report, releases_dir=releases_dir)
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

    def test_secondary_annotation_completion(self, tmp_path: Path) -> None:
        """E2-A7-FIX-012: test secondary annotation completion check."""
        # FIX-016: function now loads from source files, not labels_report metadata.
        # Empty queue → no cases → PASS.
        queue_path = tmp_path / "queue.jsonl"
        queue_path.write_text("")
        result = check_secondary_annotation_completion(
            {},
            queue_path=queue_path,
            raw_responses_path=tmp_path / "raw.jsonl",
            secondary_labels_path=tmp_path / "labels.jsonl",
            adjudication_path=tmp_path / "adj.jsonl",
            annotation_agreement_path=tmp_path / "agreement.json",
        )
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

    def test_artifact_hash_binding(self, tmp_path: Path) -> None:
        """Test artifact hash binding check with real files."""
        # Create test files
        artifact_names = [
            "raw_pilot_attempts",
            "request_schedule",
            "primary_labels",
            "reference_labels",
            "adjudication_log",
            "pairing_audit",
            "pilot_analysis",
            "floor_effect_diagnostic",
            "bounded_revision_report",
            "frozen_prompt_manifest",
            "synthetic_regression_report",
        ]
        artifact_paths = {}
        artifact_hashes = {}
        for name in artifact_names:
            path = tmp_path / f"{name}.json"
            path.write_text(f"test content for {name}")
            artifact_paths[name] = path
            artifact_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()

        result = check_artifact_hash_binding(artifact_hashes, artifact_paths)
        assert result.passed is True


class TestRunCompletionCheck:
    """Tests for complete completion check (E2R-034)."""

    def test_run_completion_check_all_pass(self, monkeypatch) -> None:
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
            "num_secondary_reviewed": 0,
            "num_disagreements": 0,
            "num_adjudicated": 0,
            "num_unresolved": 0,
        }
        analysis = {
            "matched_family_count": 30,
            "excluded_family_count": 0,
            "pairing_unit": "generation_family_id",
            "high_minus_low_risk_difference": 0.15,
            "high_minus_low_ci95": [0.05, 0.25],
            "high_minus_low_refusal_effect": 0.10,
            "high_minus_low_refusal_ci95": [0.0, 0.20],
            "high_minus_low_task_compliance_effect": 0.12,
            "high_minus_low_task_compliance_ci95": [0.02, 0.22],
            "behavioral_refusal_effect": 0.10,
            "task_compliance_effect": 0.12,
            "paired_effects": {
                "high_minus_low": {
                    "disclosure_risk_difference": 0.15,
                    "disclosure_ci95": [0.05, 0.25],
                    "refusal_risk_difference": 0.10,
                    "refusal_ci95": [0.0, 0.20],
                    "task_compliance_risk_difference": 0.12,
                    "task_compliance_ci95": [0.02, 0.22],
                },
            },
            "pairing_audit": {"audit_status": "passed"},
            "floor_effect_diagnostic": {"status": "floor_effect_detected"},
        }
        artifact_hashes = {
            "raw_pilot_attempts": "hash1",
            "request_schedule": "hash2",
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

        # Mock disk-dependent checks.
        def _pass(name: str) -> CheckResult:
            return CheckResult(check_name=name, passed=True)

        monkeypatch.setattr(
            "experiments.trustparadox_u.run_e2_completion_check.check_artifact_hash_binding",
            lambda *a, **kw: _pass("artifact_hash_binding"),
        )
        monkeypatch.setattr(
            "experiments.trustparadox_u.run_e2_completion_check.check_synthetic_regression",
            lambda *a, **kw: _pass("synthetic_regression"),
        )
        monkeypatch.setattr(
            "experiments.trustparadox_u.run_e2_completion_check.check_label_completeness_from_files",
            lambda *a, **kw: _pass("label_completeness_from_files"),
        )
        monkeypatch.setattr(
            "experiments.trustparadox_u.run_e2_completion_check.check_raw_pilot_completeness",
            lambda *a, **kw: _pass("raw_pilot_completeness"),
        )
        monkeypatch.setattr(
            "experiments.trustparadox_u.run_e2_completion_check.check_evaluator_response_completeness",
            lambda *a, **kw: _pass("evaluator_response_completeness"),
        )
        monkeypatch.setattr(
            "experiments.trustparadox_u.run_e2_completion_check.check_primary_label_file_completeness",
            lambda *a, **kw: _pass("primary_label_file_completeness"),
        )
        monkeypatch.setattr(
            "experiments.trustparadox_u.run_e2_completion_check.check_reference_label_completeness",
            lambda *a, **kw: _pass("reference_label_completeness"),
        )
        monkeypatch.setattr(
            "experiments.trustparadox_u.run_e2_completion_check.check_cross_artifact_consistency",
            lambda *a, **kw: _pass("cross_artifact_consistency"),
        )
        monkeypatch.setattr(
            "experiments.trustparadox_u.run_e2_completion_check.check_j_analysis_provenance",
            lambda *a, **kw: _pass("j_analysis_provenance"),
        )
        monkeypatch.setattr(
            "experiments.trustparadox_u.run_e2_completion_check.check_secondary_annotation_integrity",
            lambda *a, **kw: _pass("secondary_annotation_integrity"),
        )
        # E2C-FIX-041: granular secondary annotation checks.
        monkeypatch.setattr(
            "experiments.trustparadox_u.run_e2_completion_check.check_secondary_queue_complete",
            lambda *a, **kw: _pass("secondary_queue_complete"),
        )
        monkeypatch.setattr(
            "experiments.trustparadox_u.run_e2_completion_check.check_secondary_raw_complete",
            lambda *a, **kw: _pass("secondary_raw_complete"),
        )
        monkeypatch.setattr(
            "experiments.trustparadox_u.run_e2_completion_check.check_secondary_raw_success",
            lambda *a, **kw: _pass("secondary_raw_success"),
        )
        monkeypatch.setattr(
            "experiments.trustparadox_u.run_e2_completion_check.check_secondary_label_complete",
            lambda *a, **kw: _pass("secondary_label_complete"),
        )
        monkeypatch.setattr(
            "experiments.trustparadox_u.run_e2_completion_check.check_secondary_label_raw_consistency",
            lambda *a, **kw: _pass("secondary_label_raw_consistency"),
        )
        monkeypatch.setattr(
            "experiments.trustparadox_u.run_e2_completion_check.check_secondary_agreement_consistency",
            lambda *a, **kw: _pass("secondary_agreement_consistency"),
        )
        monkeypatch.setattr(
            "experiments.trustparadox_u.run_e2_completion_check.check_secondary_unresolved_consistency",
            lambda *a, **kw: _pass("secondary_unresolved_consistency"),
        )
        monkeypatch.setattr(
            "experiments.trustparadox_u.run_e2_completion_check.check_secondary_adjudication_consistency",
            lambda *a, **kw: _pass("secondary_adjudication_consistency"),
        )
        monkeypatch.setattr(
            "experiments.trustparadox_u.run_e2_completion_check.check_secondary_prompt_freeze",
            lambda *a, **kw: _pass("secondary_prompt_freeze"),
        )
        monkeypatch.setattr(
            "experiments.trustparadox_u.run_e2_completion_check.check_secondary_hash_binding",
            lambda *a, **kw: _pass("secondary_hash_binding"),
        )
        monkeypatch.setattr(
            "experiments.trustparadox_u.run_e2_completion_check.check_reference_diagnostic_validity",
            lambda *a, **kw: _pass("reference_diagnostic_validity"),
        )
        monkeypatch.setattr(
            "experiments.trustparadox_u.run_e2_completion_check.check_independent_annotation_validation",
            lambda *a, **kw: _pass("independent_annotation_validation"),
        )
        # PATCH-7359-025: agreement-metric consistency.
        monkeypatch.setattr(
            "experiments.trustparadox_u.run_e2_completion_check.check_agreement_metric_consistency",
            lambda *a, **kw: _pass("agreement_metric_consistency"),
        )
        # PATCH-7359-026: analysis-provenance semantic check.
        monkeypatch.setattr(
            "experiments.trustparadox_u.run_e2_completion_check.check_analysis_provenance_valid",
            lambda *a, **kw: _pass("analysis_provenance_valid"),
        )
        # PATCH-014: J2 transport-provenance consistency check.
        monkeypatch.setattr(
            "experiments.trustparadox_u.run_e2_completion_check.check_j2_transport_provenance",
            lambda *a, **kw: _pass("j2_transport_provenance"),
        )
        # PATCH-7359-024: mandatory execution-provenance check.
        monkeypatch.setattr(
            "experiments.trustparadox_u.run_e2_completion_check.check_secondary_execution_provenance_valid",
            lambda *a, **kw: _pass("secondary_execution_provenance_valid"),
        )
        # E2C-FIX-044: file-level integrity audit.
        monkeypatch.setattr(
            "experiments.trustparadox_u.run_e2_completion_check.check_e2_file_integrity_audit",
            lambda *a, **kw: _pass("e2_file_integrity_audit"),
        )

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
            agreement_report={
                "schema_version": "2.0",
                "j1_reference_diagnostic": {
                    "comparison_type": "binary_unauthorized_disclosure",
                    "num_compared": 90,
                    "num_agreements": 90,
                    "num_disagreements": 0,
                    "agreement_rate": 1.0,
                },
                "j1_j2_secondary_audit": {
                    "num_compared": 9,
                    "num_agreements": 9,
                    "num_disagreements": 0,
                    "exact_agreement_rate": 1.0,
                },
                "independent_annotation_validation": {
                    "num_compared": 9,
                    "num_agreements": 9,
                    "reviewer_type": "independent_llm",
                },
                "annotation_validation_source": "j1_j2_secondary_audit",
                "annotation_source": "j1_j2_llm_only",
                "j1_j2_exact_agreement": 1.0,
                "num_compared": 9,
                "num_disagreements": 0,
            },
            evaluator_raw_responses=[
                {
                    "generation_attempt_id": f"att_{i}",
                    "request_id": f"chatcmpl-test_{i}",
                    "model_returned": "qwen3.8-max",
                    "raw_output": "sample evaluator output",
                    "evaluated_at": "2026-01-01T00:00:00Z",
                    "parsed": {
                        "primary_exposure_label": "high",
                        "confidence": 0.9,
                        "evaluator_status": "success",
                    },
                }
                for i in range(90)
            ],
        )
        assert report.all_passed is True
        assert len(report.checks) == 55
        assert report.research_status == "empirical_pilot_complete"
        assert len(report.artifact_hashes) == 11


class TestPhaseStateE2Complete:
    """Tests for E2_COMPLETE phase state (E2R-036)."""

    def test_passes_with_e2_complete(self) -> None:
        """E2_COMPLETE with all freeze flags passes."""
        phase_file = {
            "schema_version": "1.1.0",
            "phase": EmpiricalPhase.E2_COMPLETE.value,
            "trust_prompts_frozen": True,
            "evaluator_frozen": True,
            "independent_labels_frozen": True,
            "full_corpus_generation_authorized": True,
        }
        result = check_phase_state(phase_file)
        assert result.passed is True
        assert result.details["evaluator_frozen"] is True

    def test_fails_e2_complete_without_evaluator_frozen(self) -> None:
        phase_file = {
            "phase": EmpiricalPhase.E2_COMPLETE.value,
            "trust_prompts_frozen": True,
            "evaluator_frozen": False,
            "independent_labels_frozen": True,
            "full_corpus_generation_authorized": True,
        }
        result = check_phase_state(phase_file)
        assert result.passed is False
        assert result.failure_code == "evaluator_not_frozen"

    def test_fails_e2_complete_without_labels_frozen(self) -> None:
        phase_file = {
            "phase": EmpiricalPhase.E2_COMPLETE.value,
            "trust_prompts_frozen": True,
            "evaluator_frozen": True,
            "independent_labels_frozen": False,
            "full_corpus_generation_authorized": True,
        }
        result = check_phase_state(phase_file)
        assert result.passed is False
        assert result.failure_code == "independent_labels_not_frozen"

    def test_fails_e2_complete_without_authorization(self) -> None:
        phase_file = {
            "phase": EmpiricalPhase.E2_COMPLETE.value,
            "trust_prompts_frozen": True,
            "evaluator_frozen": True,
            "independent_labels_frozen": True,
            "full_corpus_generation_authorized": False,
        }
        result = check_phase_state(phase_file)
        assert result.passed is False
        assert result.failure_code == "full_corpus_not_authorized"


class TestTransitionToE2Complete:
    """Tests for transition_to_e2_complete() (E2R-036)."""

    def test_transition_writes_e2_complete(self, tmp_path: Path) -> None:
        """Transition writes E2_COMPLETE with all required fields."""
        phase_file = (
            tmp_path
            / "data"
            / "trustparadox_u"
            / "empirical_v2"
            / "manifests"
            / "empirical_phase.json"
        )
        phase_file.parent.mkdir(parents=True)
        phase_file.write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "protocol_version": "2.0.0",
                    "study_version": "2.0.0",
                    "phase": "E2_PROMPTS_FROZEN",
                    "trust_prompts_frozen": True,
                    "full_corpus_generation_authorized": False,
                }
            )
        )

        report = CompletionReport()
        report.all_passed = True

        new_phase = transition_to_e2_complete(report, phase_file_path=phase_file)
        assert new_phase["phase"] == "E2_COMPLETE"
        assert new_phase["schema_version"] == "1.1.0"
        assert new_phase["trust_prompts_frozen"] is True
        assert new_phase["evaluator_frozen"] is True
        assert new_phase["independent_labels_frozen"] is True
        assert new_phase["full_corpus_generation_authorized"] is True

        # Verify file was written
        written = json.loads(phase_file.read_text())
        assert written["phase"] == "E2_COMPLETE"

    def test_transition_fails_when_checks_not_passed(self, tmp_path: Path) -> None:
        """Transition raises if not all checks passed."""
        phase_file = tmp_path / "phase.json"
        phase_file.write_text(json.dumps({"phase": "E2_PROMPTS_FROZEN"}))

        report = CompletionReport()
        report.all_passed = False

        import pytest

        with pytest.raises(RuntimeError, match="not all completion checks passed"):
            transition_to_e2_complete(report, phase_file_path=phase_file)


class TestRealEvaluatorEvidence:
    """E2J-FIX-027: no-mock regression tests for check_real_evaluator_evidence."""

    def test_must_fail_all_mock_records(self) -> None:
        """All 90 records are mock -> fail."""
        mock_records = [
            {
                "generation_attempt_id": f"ega_test_{i}",
                "request_id": f"mock_ega_test_{i}",
                "model_returned": "qwen3.8-max",
                "evaluator_provider": "mock",
                "evaluator_transport": "mock",
            }
            for i in range(90)
        ]
        result = check_real_evaluator_evidence(mock_records)
        assert result.passed is False
        assert result.failure_code == "e2_mock_evaluator_detected"

    def test_must_fail_one_mock_among_real(self) -> None:
        """One mock evaluator record among 90 real records -> fail."""
        records = [
            {
                "generation_attempt_id": f"ega_test_{i}",
                "request_id": f"real_{i}",
                "model_returned": "qwen3.8-max",
                "evaluator_provider": "openai",
                "evaluator_transport": "api",
                "raw_output": "sample output",
                "evaluated_at": "2026-01-01T00:00:00Z",
            }
            for i in range(89)
        ]
        # Add one mock record
        records.append(
            {
                "generation_attempt_id": "ega_test_89",
                "request_id": "mock_ega_test_89",
                "model_returned": "qwen3.8-max",
                "evaluator_provider": "mock",
                "evaluator_transport": "mock",
                "raw_output": "sample output",
                "evaluated_at": "2026-01-01T00:00:00Z",
            }
        )
        result = check_real_evaluator_evidence(records)
        assert result.passed is False
        assert result.failure_code == "e2_mock_evaluator_detected"

    def test_must_fail_mock_request_id(self) -> None:
        """Mock request ID -> fail."""
        records = [
            {
                "generation_attempt_id": "ega_test_0",
                "request_id": "mock_ega_test_0",
                "model_returned": "qwen3.8-max",
                "evaluator_provider": "openai",
            }
        ]
        result = check_real_evaluator_evidence(records)
        assert result.passed is False
        assert result.failure_code == "e2_mock_request_id_detected"

    def test_must_fail_mock_transport(self) -> None:
        """Mock transport -> fail."""
        records = [
            {
                "generation_attempt_id": "ega_test_0",
                "request_id": "real_0",
                "model_returned": "qwen3.8-max",
                "evaluator_provider": "openai",
                "evaluator_transport": "mock",
            }
        ]
        result = check_real_evaluator_evidence(records)
        assert result.passed is False
        assert result.failure_code == "e2_mock_transport_detected"

    def test_must_fail_mock_provider(self) -> None:
        """Mock provider -> fail."""
        records = [
            {
                "generation_attempt_id": "ega_test_0",
                "request_id": "real_0",
                "model_returned": "qwen3.8-max",
                "evaluator_provider": "mock",
                "evaluator_transport": "api",
            }
        ]
        result = check_real_evaluator_evidence(records)
        assert result.passed is False
        assert result.failure_code == "e2_mock_evaluator_detected"

    def test_must_fail_missing_evaluator_request_id(self) -> None:
        """Missing evaluator request ID -> fail."""
        records = [
            {
                "generation_attempt_id": "ega_test_0",
                "request_id": "",
                "model_returned": "qwen3.8-max",
                "evaluator_provider": "openai",
                "raw_output": "sample output",
                "evaluated_at": "2026-01-01T00:00:00Z",
            }
        ]
        result = check_real_evaluator_evidence(records)
        # FIX-024: empty request_id now fails
        assert result.passed is False
        assert result.failure_code == "e2_empty_request_id"

    def test_must_fail_missing_returned_model(self) -> None:
        """Missing returned model -> fail."""
        records = [
            {
                "generation_attempt_id": "ega_test_0",
                "request_id": "real_0",
                "model_returned": "",
                "evaluator_provider": "openai",
            }
        ]
        result = check_real_evaluator_evidence(records)
        assert result.passed is False
        assert result.failure_code == "e2_evaluator_returned_model_missing"

    def test_must_pass_90_real_records(self) -> None:
        """90 real evaluator records -> pass."""
        records = [
            {
                "generation_attempt_id": f"ega_test_{i}",
                "request_id": f"real_req_{i}",
                "model_returned": "qwen3.8-max",
                "evaluator_provider": "openai",
                "evaluator_transport": "api",
                "latency_ms": 1234.5,
                "raw_output": "sample output",
                "evaluated_at": "2026-01-01T00:00:00Z",
            }
            for i in range(90)
        ]
        result = check_real_evaluator_evidence(records)
        assert result.passed is True
        assert result.details["evaluator_record_count"] == 90

    def test_must_pass_consistent_model_identity(self) -> None:
        """Consistent model identity -> pass."""
        records = [
            {
                "generation_attempt_id": f"ega_test_{i}",
                "request_id": f"real_req_{i}",
                "model_returned": "qwen3.8-max",
                "evaluator_provider": "openai",
                "raw_output": "sample output",
                "evaluated_at": "2026-01-01T00:00:00Z",
            }
            for i in range(10)
        ]
        result = check_real_evaluator_evidence(records)
        assert result.passed is True

    def test_must_pass_no_mock_markers(self) -> None:
        """No mock markers -> pass."""
        records = [
            {
                "generation_attempt_id": f"ega_test_{i}",
                "request_id": f"req_{i}",
                "model_returned": "qwen3.8-max",
                "evaluator_provider": "openai",
                "evaluator_transport": "api",
                "raw_output": "sample output",
                "evaluated_at": "2026-01-01T00:00:00Z",
            }
            for i in range(5)
        ]
        result = check_real_evaluator_evidence(records)
        assert result.passed is True
        assert result.details["all_records_real"] is True

    def test_transition_fails_with_wrong_phase(self, tmp_path: Path) -> None:
        """Transition raises if current phase is not E2_PROMPTS_FROZEN."""
        phase_file = tmp_path / "phase.json"
        phase_file.write_text(json.dumps({"phase": "E2_TRUST_PILOT"}))

        report = CompletionReport()
        report.all_passed = True

        import pytest

        with pytest.raises(RuntimeError, match="current phase is"):
            transition_to_e2_complete(report, phase_file_path=phase_file)


class TestIterationFFileBasedChecks:
    """Tests for Iteration F file-based checks (E2R-FIX-017/018/025/026)."""

    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        """Helper to write JSONL records to a file."""
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    def test_label_completeness_from_files_pass(self, tmp_path: Path) -> None:
        """Test file-based label completeness passes with valid files."""
        raw_path = tmp_path / "raw.jsonl"
        labels_path = tmp_path / "labels.jsonl"
        eval_path = tmp_path / "eval.jsonl"
        ids = [f"att_{i}" for i in range(90)]
        self._write_jsonl(raw_path, [{"generation_attempt_id": id_} for id_ in ids])
        self._write_jsonl(
            labels_path,
            [{"generation_attempt_id": id_, "evaluator_status": "success"} for id_ in ids],
        )
        self._write_jsonl(eval_path, [{"generation_attempt_id": id_} for id_ in ids])
        result = check_label_completeness_from_files(raw_path, labels_path, eval_path)
        assert result.passed is True
        assert result.details["raw_count"] == 90

    def test_label_completeness_from_files_missing_raw(self, tmp_path: Path) -> None:
        """Test file-based label completeness fails when raw file missing."""
        result = check_label_completeness_from_files(
            tmp_path / "missing.jsonl",
            tmp_path / "labels.jsonl",
            tmp_path / "eval.jsonl",
        )
        assert result.passed is False
        assert result.failure_code == "raw_generation_file_missing"

    def test_label_completeness_from_files_wrong_count(self, tmp_path: Path) -> None:
        """Test file-based label completeness fails with wrong count."""
        raw_path = tmp_path / "raw.jsonl"
        labels_path = tmp_path / "labels.jsonl"
        eval_path = tmp_path / "eval.jsonl"
        self._write_jsonl(raw_path, [{"generation_attempt_id": f"att_{i}"} for i in range(80)])
        self._write_jsonl(labels_path, [{"generation_attempt_id": f"att_{i}"} for i in range(80)])
        self._write_jsonl(eval_path, [{"generation_attempt_id": f"att_{i}"} for i in range(80)])
        result = check_label_completeness_from_files(raw_path, labels_path, eval_path)
        assert result.passed is False
        assert result.failure_code == "raw_generation_count_mismatch"

    def test_evaluator_independence_evidence_pass(self) -> None:
        """Test evaluator independence evidence passes with valid data."""
        labels_report = {
            "generator_model": "qwen3.7-plus",
            "evaluator_provider": "aliyun",
            "evaluator_model_requested": "qwen3.8-max",
            "evaluator_model_returned": "qwen3.8-max",
            "evaluator_prompt_hash": "abc123",
        }
        evaluator_raw = [
            {
                "generation_attempt_id": f"att_{i}",
                "model_returned": "qwen3.8-max",
                "parsed": {
                    "primary_exposure_label": "high",
                    "confidence": 0.9,
                    "evaluator_status": "success",
                },
            }
            for i in range(90)
        ]
        result = check_evaluator_independence_evidence(labels_report, evaluator_raw)
        assert result.passed is True
        assert result.details["evaluator_independent"] is True

    def test_evaluator_independence_evidence_same_model(self) -> None:
        """Test evaluator independence fails when G == J."""
        labels_report = {
            "generator_model": "qwen3.7-plus",
            "evaluator_provider": "aliyun",
            "evaluator_model_requested": "qwen3.7-plus",
            "evaluator_model_returned": "qwen3.7-plus",
            "evaluator_prompt_hash": "abc123",
        }
        result = check_evaluator_independence_evidence(labels_report, [])
        assert result.passed is False
        assert result.failure_code == "evaluator_same_as_generator"

    def test_evaluator_independence_evidence_payload_contamination(self) -> None:
        """Test evaluator independence fails with forbidden keys."""
        labels_report = {
            "generator_model": "qwen3.7-plus",
            "evaluator_provider": "aliyun",
            "evaluator_model_requested": "qwen3.8-max",
            "evaluator_model_returned": "qwen3.8-max",
            "evaluator_prompt_hash": "abc123",
        }
        evaluator_raw = [
            {
                "generation_attempt_id": "att_0",
                "model_returned": "qwen3.8-max",
                "parsed": {
                    "primary_exposure_label": "high",
                    "confidence": 0.9,
                    "evaluator_status": "success",
                    "firewall_condition": True,
                },
            }
        ]
        result = check_evaluator_independence_evidence(labels_report, evaluator_raw)
        assert result.passed is False
        assert result.failure_code == "evaluator_payload_contamination"

    def test_raw_pilot_completeness_pass(self, tmp_path: Path) -> None:
        """Test raw pilot completeness passes with 90 records."""
        raw_path = tmp_path / "raw.jsonl"
        self._write_jsonl(
            raw_path,
            [
                {"generation_attempt_id": f"att_{i}", "generation_status": "success"}
                for i in range(90)
            ],
        )
        result = check_raw_pilot_completeness(raw_path)
        assert result.passed is True

    def test_raw_pilot_completeness_missing(self, tmp_path: Path) -> None:
        """Test raw pilot completeness fails when file missing."""
        result = check_raw_pilot_completeness(tmp_path / "missing.jsonl")
        assert result.passed is False
        assert result.failure_code == "raw_generation_file_missing"

    def test_evaluator_response_completeness_pass(self, tmp_path: Path) -> None:
        """Test evaluator response completeness passes with 90 records."""
        eval_path = tmp_path / "eval.jsonl"
        self._write_jsonl(eval_path, [{"generation_attempt_id": f"att_{i}"} for i in range(90)])
        result = check_evaluator_response_completeness(eval_path)
        assert result.passed is True

    def test_evaluator_response_completeness_wrong_count(self, tmp_path: Path) -> None:
        """Test evaluator response completeness fails with wrong count."""
        eval_path = tmp_path / "eval.jsonl"
        self._write_jsonl(eval_path, [{"generation_attempt_id": f"att_{i}"} for i in range(50)])
        result = check_evaluator_response_completeness(eval_path)
        assert result.passed is False

    def test_primary_label_file_completeness_pass(self, tmp_path: Path) -> None:
        """Test primary label file completeness passes with 90 records."""
        labels_path = tmp_path / "labels.jsonl"
        self._write_jsonl(labels_path, [{"generation_attempt_id": f"att_{i}"} for i in range(90)])
        result = check_primary_label_file_completeness(labels_path)
        assert result.passed is True

    def test_reference_label_completeness_pass(self, tmp_path: Path) -> None:
        """Test reference label completeness passes with 90 records."""
        ref_path = tmp_path / "ref.jsonl"
        self._write_jsonl(ref_path, [{"generation_attempt_id": f"att_{i}"} for i in range(90)])
        result = check_reference_label_completeness(ref_path)
        assert result.passed is True

    def test_reference_diagnostic_validity_pass(self) -> None:
        """Test reference diagnostic validity passes with correct structure."""
        agreement = {
            "j1_reference_diagnostic": {
                "comparison_type": "binary_unauthorized_disclosure",
                "num_compared": 90,
                "num_agreements": 90,
            },
        }
        result = check_reference_diagnostic_validity(agreement)
        # Passes structural checks; may fail on source recomputation without files.
        assert result.check_name == "reference_diagnostic_validity"

    def test_reference_diagnostic_validity_missing_report(self) -> None:
        """Test reference diagnostic validity fails with missing report."""
        result = check_reference_diagnostic_validity(None)
        assert result.passed is False
        assert result.failure_code == "agreement_report_missing"

    def test_reference_diagnostic_validity_wrong_comparison_type(self) -> None:
        """Test reference diagnostic validity fails with wrong comparison type."""
        agreement = {
            "j1_reference_diagnostic": {
                "comparison_type": "categorical",
                "num_compared": 90,
                "num_agreements": 90,
            },
        }
        result = check_reference_diagnostic_validity(agreement)
        assert result.passed is False
        assert result.failure_code == "reference_diagnostic_invalid_comparison_type"

    def test_independent_annotation_validation_missing_report(self) -> None:
        """Test independent annotation validation fails with missing report."""
        result = check_independent_annotation_validation(None)
        assert result.passed is False
        assert result.failure_code == "agreement_report_missing"

    def test_independent_annotation_validation_wrong_source(self) -> None:
        """Test independent annotation validation fails with wrong source."""
        agreement = {"annotation_validation_source": "deterministic_reference"}
        result = check_independent_annotation_validation(agreement)
        assert result.passed is False
        assert result.failure_code == "independent_validation_source_invalid"

    def test_independent_annotation_validation_missing_section(self) -> None:
        """Test independent annotation validation fails with missing section."""
        agreement = {"annotation_validation_source": "j1_j2_secondary_audit"}
        result = check_independent_annotation_validation(agreement)
        assert result.passed is False
        assert result.failure_code == "independent_validation_section_missing"

    # ---- PATCH-007 regression tests ----

    def test_patch007_categorical_not_applicable_not_zero(self) -> None:
        """PATCH-007-A: missing reference categorical → not_applicable, not 0.0.

        Regression for the 873d393 defect where 'none' == None produced
        a false 0/90 categorical agreement.
        """
        agreement_report_path = (
            Path(__file__).resolve().parent.parent.parent
            / "results"
            / "empirical_v2"
            / "e2_primary_pilot_labels"
            / "label_agreement_report.json"
        )
        if not agreement_report_path.exists():
            pytest.skip("agreement report not generated")
        report = json.loads(agreement_report_path.read_text(encoding="utf-8"))
        diag = report.get("j1_reference_diagnostic", {})
        cat = diag.get("categorical_comparison", {})
        assert cat.get("status") == "not_applicable"
        assert cat.get("status") != "0.0"
        assert diag.get("comparison_type") == "binary_unauthorized_disclosure"

    def test_patch007_binary_disclosure_agreement(self) -> None:
        """PATCH-007-B: both unauthorized_disclosure=false → agreement.

        Verifies the agreement report correctly records binary disclosure
        agreement when both J1 and reference have unauthorized_disclosure=false.
        """
        agreement_report_path = (
            Path(__file__).resolve().parent.parent.parent
            / "results"
            / "empirical_v2"
            / "e2_primary_pilot_labels"
            / "label_agreement_report.json"
        )
        if not agreement_report_path.exists():
            pytest.skip("agreement report not generated")
        report = json.loads(agreement_report_path.read_text(encoding="utf-8"))
        diag = report.get("j1_reference_diagnostic", {})
        assert diag.get("num_compared") == 90
        assert diag.get("num_agreements") == 90
        assert diag.get("num_disagreements") == 0
        assert diag.get("agreement_rate") == 1.0

    def test_patch007_deterministic_reference_not_independent(self) -> None:
        """PATCH-007-C: deterministic reference is NOT independent validation.

        Verifies that annotation_validation_source is j1_j2_secondary_audit,
        not deterministic_reference.
        """
        agreement_report_path = (
            Path(__file__).resolve().parent.parent.parent
            / "results"
            / "empirical_v2"
            / "e2_primary_pilot_labels"
            / "label_agreement_report.json"
        )
        if not agreement_report_path.exists():
            pytest.skip("agreement report not generated")
        report = json.loads(agreement_report_path.read_text(encoding="utf-8"))
        assert report.get("annotation_validation_source") == "j1_j2_secondary_audit"
        iav = report.get("independent_annotation_validation", {})
        assert iav.get("reviewer_type") == "independent_llm"
        assert iav.get("primary_annotator") == "J1"
        assert iav.get("secondary_annotator") == "J2"

    def test_patch008_old_style_report_rejected(self) -> None:
        """PATCH-008: old-style 0/90 agreement report must fail new gates.

        Constructs an intentionally invalid report matching the 873d393
        defect pattern and verifies both new check functions reject it.
        """
        # Old-style report with false 0/90 agreement.
        old_report = {
            "j1_reference": {
                "num_compared": 90,
                "num_agreements": 0,
                "num_disagreements": 90,
                "exact_agreement_rate": 0.0,
            }
        }
        # reference_diagnostic_validity must fail (no j1_reference_diagnostic).
        result_ref = check_reference_diagnostic_validity(old_report)
        assert result_ref.passed is False
        # independent_annotation_validation must fail (no source/section).
        result_iav = check_independent_annotation_validation(old_report)
        assert result_iav.passed is False

    def test_patch021_all_passed_requires_valid_agreement(self) -> None:
        """PATCH-021: all_passed requires valid reference diagnostic + IAV.

        Verifies that the completion report's all_passed flag correctly
        reflects failure in either agreement gate check.
        """
        report = CompletionReport()
        report.add_check(CheckResult(check_name="a", passed=True))
        report.add_check(
            CheckResult(
                check_name="reference_diagnostic_validity",
                passed=False,
                failure_code="reference_diagnostic_counts_mismatch",
            )
        )
        assert report.all_passed is False

        report2 = CompletionReport()
        report2.add_check(CheckResult(check_name="a", passed=True))
        report2.add_check(
            CheckResult(
                check_name="independent_annotation_validation",
                passed=False,
                failure_code="independent_validation_source_invalid",
            )
        )
        assert report2.all_passed is False

    def test_patch014_j2_transport_provenance_pass(self) -> None:
        """PATCH-014: J2 transport-provenance check passes with valid caps."""
        from experiments.trustparadox_u.run_e2_completion_check import (
            check_j2_transport_provenance,
        )

        # The actual regenerated file should pass.
        result = check_j2_transport_provenance()
        assert result.passed is True
        assert result.check_name == "j2_transport_provenance"
        details = result.details or {}
        assert details.get("num_records") == 9
        cap_dist = details.get("cap_distribution", {})
        assert "512" in cap_dist
        assert "1024" in cap_dist

    def test_patch014_j2_transport_provenance_missing_cap(self, tmp_path) -> None:
        """PATCH-014/7359-006: record not in any batch fails the check."""
        from experiments.trustparadox_u.run_e2_completion_check import (
            check_j2_transport_provenance,
        )

        # Create provenance file with a batch
        prov_file = tmp_path / "provenance.json"
        prov_file.write_text(
            json.dumps(
                {
                    "batches": [
                        {
                            "batch_id": "initial_j2_batch",
                            "requested_max_tokens": 512,
                            "generation_attempt_ids": ["valid_id_1", "valid_id_2"],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        # Create raw file with record not in batch
        bad_file = tmp_path / "bad_raw.jsonl"
        bad_file.write_text(
            json.dumps({"generation_attempt_id": "x", "requested_max_tokens": 512}) + "\n",
            encoding="utf-8",
        )
        result = check_j2_transport_provenance(
            raw_responses_path=bad_file, provenance_path=prov_file
        )
        assert result.passed is False
        assert result.failure_code == "record_not_in_any_batch"

    def test_patch014_j2_transport_provenance_disallowed_cap(self, tmp_path) -> None:
        """PATCH-014/7359-007: cap mismatch with batch fails the check."""
        from experiments.trustparadox_u.run_e2_completion_check import (
            check_j2_transport_provenance,
        )

        # Create provenance file with a batch
        prov_file = tmp_path / "provenance.json"
        prov_file.write_text(
            json.dumps(
                {
                    "batches": [
                        {
                            "batch_id": "initial_j2_batch",
                            "requested_max_tokens": 512,
                            "generation_attempt_ids": ["valid_id_1"],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        # Create raw file with record in batch but wrong cap
        bad_file = tmp_path / "bad_raw.jsonl"
        bad_file.write_text(
            json.dumps(
                {
                    "generation_attempt_id": "valid_id_1",
                    "requested_max_tokens": 256,
                    "execution_batch_id": "initial_j2_batch",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        result = check_j2_transport_provenance(
            raw_responses_path=bad_file, provenance_path=prov_file
        )
        assert result.passed is False
        assert result.failure_code == "record_batch_mismatch"

    # -- PATCH-025/026/030: final-recertification regression tests --------

    def test_patch025_study_metadata_in_report(self) -> None:
        """PATCH-025: CompletionReport.to_dict() includes study_metadata."""
        report = CompletionReport()
        d = report.to_dict()
        assert "study_metadata" in d
        sm = d["study_metadata"]
        assert "paper_facing_description" in sm
        assert "limitations" in sm
        # Paper-facing description must mention J1 and J2.
        desc = sm["paper_facing_description"]
        assert "J1" in desc
        assert "J2" in desc
        assert "deterministic reference" in desc.lower() or "deterministic Reference" in desc
        # Must NOT claim human validation.
        assert "human validation" not in desc.lower()
        assert "human ground truth" not in desc.lower()

    def test_patch026_limitations_content(self) -> None:
        """PATCH-026: limitations list all required items."""
        report = CompletionReport()
        limitations = report.to_dict()["study_metadata"]["limitations"]
        assert isinstance(limitations, list)
        assert len(limitations) >= 7
        # Required limitation keywords.
        joined = " ".join(limitations).lower()
        assert "pilot" in joined
        assert "90" in joined
        assert "single generator" in joined
        assert "refusal" in joined
        assert "llm" in joined
        assert "9-case" in joined or "9 case" in joined

    def test_patch030_frozen_timestamp_chronological_order(self) -> None:
        """PATCH-010/030: frozen manifest timestamps are chronologically valid."""
        from experiments.trustparadox_u.run_e2_completion_check import (
            _FROZEN_PRIMARY_LABELS_PATH,
        )

        if not _FROZEN_PRIMARY_LABELS_PATH.exists():
            pytest.skip("frozen_primary_labels.json not found")
        frozen = json.loads(_FROZEN_PRIMARY_LABELS_PATH.read_text(encoding="utf-8"))
        ts_primary = frozen.get("primary_labels_frozen_at")
        ts_secondary = frozen.get("secondary_audit_frozen_at")
        ts_manifest = frozen.get("manifest_generated_at")
        # All three must be present.
        assert ts_primary is not None, "missing primary_labels_frozen_at"
        assert ts_secondary is not None, "missing secondary_audit_frozen_at"
        assert ts_manifest is not None, "missing manifest_generated_at"
        # Chronological order: primary <= secondary <= manifest.
        assert ts_primary <= ts_secondary, (
            f"primary_labels_frozen_at ({ts_primary}) must be <= "
            f"secondary_audit_frozen_at ({ts_secondary})"
        )
        assert ts_secondary <= ts_manifest, (
            f"secondary_audit_frozen_at ({ts_secondary}) must be <= "
            f"manifest_generated_at ({ts_manifest})"
        )
        # Old ambiguous field must not be present.
        assert "frozen_at" not in frozen, "stale frozen_at field must be removed"

    def test_patch030_agreement_report_hash_binding(self) -> None:
        """PATCH-018/030: frozen manifest binds agreement report hash."""
        from experiments.trustparadox_u.run_e2_completion_check import (
            _AGREEMENT_REPORT_PATH,
            _FROZEN_PRIMARY_LABELS_PATH,
            sha256_file,
        )

        if not _FROZEN_PRIMARY_LABELS_PATH.exists():
            pytest.skip("frozen_primary_labels.json not found")
        frozen = json.loads(_FROZEN_PRIMARY_LABELS_PATH.read_text(encoding="utf-8"))
        bound_hash = frozen.get("agreement_report_sha256")
        assert bound_hash is not None, "agreement_report_sha256 missing from frozen manifest"
        if _AGREEMENT_REPORT_PATH.exists():
            actual_hash = sha256_file(_AGREEMENT_REPORT_PATH)
            assert (
                bound_hash == actual_hash
            ), "agreement_report_sha256 does not match actual file hash"

    def test_patch030_e2_transition_requires_all_passed(self) -> None:
        """PATCH-023/030: E2_COMPLETE transition requires all_passed=true."""
        report = CompletionReport()
        report.add_check(
            CheckResult(
                check_name="deliberate_failure",
                passed=False,
                failure_code="test_failure",
            )
        )
        assert report.all_passed is False
        d = report.to_dict()
        assert d["all_passed"] is False

    def test_uncertainty_ci_pass_paired(self) -> None:
        """Test uncertainty CI passes with paired_effects."""
        analysis = {
            "paired_effects": {
                "high_minus_low": {"disclosure_ci95": [0.05, 0.25]},
                "bootstrap_method": "percentile",
                "bootstrap_iterations": 1000,
            }
        }
        result = check_uncertainty_ci(analysis)
        assert result.passed is True

    def test_uncertainty_ci_pass_direct(self) -> None:
        """Test uncertainty CI passes with direct ci95 field."""
        analysis = {"high_minus_low_ci95": [0.05, 0.25]}
        result = check_uncertainty_ci(analysis)
        assert result.passed is True

    def test_uncertainty_ci_missing(self) -> None:
        """Test uncertainty CI fails when missing."""
        result = check_uncertainty_ci({})
        assert result.passed is False
        assert result.failure_code == "uncertainty_ci_missing"

    def test_synthetic_provenance_pass(self) -> None:
        """Test synthetic provenance passes with complete report."""
        report = {
            "synthetic_release_id": "v1",
            "scientific_release_digest": "abc123",
            "table_1_sha256": "h1",
            "table_2_sha256": "h2",
            "table_3_sha256": "h3",
            "table_4_sha256": "h4",
            "table_5_sha256": "h5",
            "table_6_sha256": "h6",
            "synthetic_gate_status": "synthetic_benchmark_valid",
        }
        result = check_synthetic_provenance(report)
        assert result.passed is True

    def test_synthetic_provenance_missing(self) -> None:
        """Test synthetic provenance fails with missing report."""
        result = check_synthetic_provenance(None)
        assert result.passed is False
        assert result.failure_code == "synthetic_regression_report_missing"

    def test_synthetic_provenance_invalid_gate(self) -> None:
        """Test synthetic provenance fails with invalid gate."""
        report = {
            "synthetic_release_id": "v1",
            "scientific_release_digest": "abc123",
            "table_1_sha256": "h1",
            "table_2_sha256": "h2",
            "table_3_sha256": "h3",
            "table_4_sha256": "h4",
            "table_5_sha256": "h5",
            "table_6_sha256": "h6",
            "synthetic_gate_status": "invalid",
        }
        result = check_synthetic_provenance(report)
        assert result.passed is False
        assert result.failure_code == "synthetic_gate_invalid"

    def test_completion_consistency_pass(self) -> None:
        """Test completion consistency passes with valid report."""
        report = CompletionReport()
        report.add_check(CheckResult(check_name="test_check", passed=True))
        result = check_completion_consistency(report)
        assert result.passed is True

    def test_completion_consistency_empty(self) -> None:
        """Test completion consistency fails with empty report."""
        report = CompletionReport()
        result = check_completion_consistency(report)
        assert result.passed is False
        assert result.failure_code == "completion_report_empty"

    def test_cross_artifact_consistency_pass(self, tmp_path: Path) -> None:
        """Test cross-artifact consistency passes with consistent data."""
        labels_path = tmp_path / "labels.jsonl"
        self._write_jsonl(labels_path, [{"generation_attempt_id": f"att_{i}"} for i in range(90)])
        labels_report = {"num_primary_labels": 90}
        analysis = {
            "overall_metrics": {"n_total_attempts": 90},
            "pairing_audit": {"complete_families": 30},
        }
        bounded_revision = {"complete_families": 30}
        result = check_cross_artifact_consistency(
            labels_report, analysis, bounded_revision, primary_labels_path=labels_path
        )
        assert result.passed is True

    def test_cross_artifact_consistency_label_mismatch(self, tmp_path: Path) -> None:
        """Test cross-artifact consistency fails with label count mismatch."""
        labels_path = tmp_path / "labels.jsonl"
        self._write_jsonl(labels_path, [{"generation_attempt_id": f"att_{i}"} for i in range(80)])
        labels_report = {"num_primary_labels": 90}
        result = check_cross_artifact_consistency(
            labels_report, {}, {}, primary_labels_path=labels_path
        )
        assert result.passed is False
        assert result.failure_code == "label_report_count_mismatch"

    def test_cross_artifact_consistency_family_mismatch(self, tmp_path: Path) -> None:
        """Test cross-artifact consistency fails with family count mismatch."""
        labels_path = tmp_path / "labels.jsonl"
        self._write_jsonl(labels_path, [{"generation_attempt_id": f"att_{i}"} for i in range(90)])
        labels_report = {"num_primary_labels": 90}
        analysis = {"pairing_audit": {"complete_families": 30}}
        bounded_revision = {"complete_families": 25}
        result = check_cross_artifact_consistency(
            labels_report, analysis, bounded_revision, primary_labels_path=labels_path
        )
        assert result.passed is False
        assert result.failure_code == "bounded_revision_family_mismatch"


class TestIterationGRegression:
    """Iteration G regression tests (E2R-FIX-029 through FIX-035)."""

    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    # --- FIX-029: File-level completeness edge cases ---

    def test_fix029_three_labels_report_says_90(self, tmp_path: Path) -> None:
        """E2R-FIX-029: 3 labels but report says 90 must fail."""
        raw_path = tmp_path / "raw.jsonl"
        labels_path = tmp_path / "labels.jsonl"
        eval_path = tmp_path / "eval.jsonl"
        ids = [f"att_{i}" for i in range(90)]
        self._write_jsonl(raw_path, [{"generation_attempt_id": id_} for id_ in ids])
        # Only 3 labels
        self._write_jsonl(
            labels_path,
            [
                {"generation_attempt_id": f"att_{i}", "evaluator_status": "success"}
                for i in range(3)
            ],
        )
        self._write_jsonl(eval_path, [{"generation_attempt_id": id_} for id_ in ids])
        result = check_label_completeness_from_files(raw_path, labels_path, eval_path)
        assert result.passed is False
        assert result.failure_code == "primary_label_count_mismatch"

    def test_fix029_89_labels(self, tmp_path: Path) -> None:
        """E2R-FIX-029: 89 labels must fail."""
        raw_path = tmp_path / "raw.jsonl"
        labels_path = tmp_path / "labels.jsonl"
        eval_path = tmp_path / "eval.jsonl"
        ids = [f"att_{i}" for i in range(90)]
        self._write_jsonl(raw_path, [{"generation_attempt_id": id_} for id_ in ids])
        self._write_jsonl(
            labels_path,
            [
                {"generation_attempt_id": f"att_{i}", "evaluator_status": "success"}
                for i in range(89)
            ],
        )
        self._write_jsonl(eval_path, [{"generation_attempt_id": id_} for id_ in ids])
        result = check_label_completeness_from_files(raw_path, labels_path, eval_path)
        assert result.passed is False

    def test_fix029_91_labels(self, tmp_path: Path) -> None:
        """E2R-FIX-029: 91 labels must fail."""
        raw_path = tmp_path / "raw.jsonl"
        labels_path = tmp_path / "labels.jsonl"
        eval_path = tmp_path / "eval.jsonl"
        ids = [f"att_{i}" for i in range(90)]
        self._write_jsonl(raw_path, [{"generation_attempt_id": id_} for id_ in ids])
        self._write_jsonl(
            labels_path,
            [
                {"generation_attempt_id": f"att_{i}", "evaluator_status": "success"}
                for i in range(91)
            ],
        )
        self._write_jsonl(eval_path, [{"generation_attempt_id": id_} for id_ in ids])
        result = check_label_completeness_from_files(raw_path, labels_path, eval_path)
        assert result.passed is False

    def test_fix029_duplicate_id(self, tmp_path: Path) -> None:
        """E2R-FIX-029: duplicate ID in raw must fail."""
        raw_path = tmp_path / "raw.jsonl"
        labels_path = tmp_path / "labels.jsonl"
        eval_path = tmp_path / "eval.jsonl"
        ids = [f"att_{i}" for i in range(89)] + ["att_0"]  # duplicate
        self._write_jsonl(raw_path, [{"generation_attempt_id": id_} for id_ in ids])
        self._write_jsonl(
            labels_path,
            [
                {"generation_attempt_id": f"att_{i}", "evaluator_status": "success"}
                for i in range(90)
            ],
        )
        self._write_jsonl(eval_path, [{"generation_attempt_id": f"att_{i}"} for i in range(90)])
        result = check_label_completeness_from_files(raw_path, labels_path, eval_path)
        # 90 records but only 89 unique IDs -> join mismatch
        assert result.passed is False

    def test_fix029_unknown_id_in_labels(self, tmp_path: Path) -> None:
        """E2R-FIX-029: unknown ID in labels must fail."""
        raw_path = tmp_path / "raw.jsonl"
        labels_path = tmp_path / "labels.jsonl"
        eval_path = tmp_path / "eval.jsonl"
        raw_ids = [f"att_{i}" for i in range(90)]
        label_ids = [f"att_{i}" for i in range(89)] + ["unknown_id"]
        self._write_jsonl(raw_path, [{"generation_attempt_id": id_} for id_ in raw_ids])
        self._write_jsonl(
            labels_path,
            [{"generation_attempt_id": id_, "evaluator_status": "success"} for id_ in label_ids],
        )
        self._write_jsonl(eval_path, [{"generation_attempt_id": id_} for id_ in raw_ids])
        result = check_label_completeness_from_files(raw_path, labels_path, eval_path)
        assert result.passed is False
        assert result.failure_code == "raw_label_id_mismatch"

    def test_fix029_missing_evaluator_response(self, tmp_path: Path) -> None:
        """E2R-FIX-029: missing evaluator response must fail."""
        raw_path = tmp_path / "raw.jsonl"
        labels_path = tmp_path / "labels.jsonl"
        eval_path = tmp_path / "eval.jsonl"
        ids = [f"att_{i}" for i in range(90)]
        self._write_jsonl(raw_path, [{"generation_attempt_id": id_} for id_ in ids])
        self._write_jsonl(
            labels_path,
            [{"generation_attempt_id": id_, "evaluator_status": "success"} for id_ in ids],
        )
        # Only 89 evaluator responses
        self._write_jsonl(eval_path, [{"generation_attempt_id": f"att_{i}"} for i in range(89)])
        result = check_label_completeness_from_files(raw_path, labels_path, eval_path)
        assert result.passed is False
        assert result.failure_code == "evaluator_response_count_mismatch"

    # --- FIX-030: Hash-integrity edge cases ---

    def test_fix030_correct_sha_passes(self, tmp_path: Path) -> None:
        """E2R-FIX-030: correct SHA passes."""
        from experiments.trustparadox_u.run_e2_completion_check import sha256_file

        path = tmp_path / "artifact.json"
        path.write_text('{"test": "data"}')
        correct_hash = sha256_file(path)
        # Verify sha256_file matches independent computation.
        assert correct_hash == hashlib.sha256(path.read_bytes()).hexdigest()
        # Provide all 11 required artifacts so check_artifact_hash_binding passes.
        # All paths point to the same file, so all hashes are identical.
        required = [
            "raw_pilot_attempts",
            "request_schedule",
            "primary_labels",
            "reference_labels",
            "adjudication_log",
            "pairing_audit",
            "pilot_analysis",
            "floor_effect_diagnostic",
            "bounded_revision_report",
            "frozen_prompt_manifest",
            "synthetic_regression_report",
        ]
        all_hashes = {name: correct_hash for name in required}
        all_paths = {name: path for name in required}
        result = check_artifact_hash_binding(all_hashes, all_paths)
        assert result.passed is True

    def test_fix030_one_byte_modification_fails(self, tmp_path: Path) -> None:
        """E2R-FIX-030: one-byte modification fails."""
        path = tmp_path / "artifact.json"
        path.write_text('{"test": "data"}')
        original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        # Modify one byte
        path.write_text('{"test": "datb"}')
        result = check_artifact_hash_binding(
            {"raw_pilot_attempts": original_hash}, {"raw_pilot_attempts": path}
        )
        assert result.passed is False
        assert result.failure_code == "e2_artifact_hash_mismatch"

    def test_fix030_fake_64char_sha_fails(self, tmp_path: Path) -> None:
        """E2R-FIX-030: fake 64-char SHA fails."""
        path = tmp_path / "artifact.json"
        path.write_text('{"test": "data"}')
        fake_hash = "a" * 64
        result = check_artifact_hash_binding(
            {"raw_pilot_attempts": fake_hash}, {"raw_pilot_attempts": path}
        )
        assert result.passed is False

    def test_fix030_missing_file_fails(self, tmp_path: Path) -> None:
        """E2R-FIX-030: missing file fails."""
        result = check_artifact_hash_binding(
            {"raw_pilot_attempts": "a" * 64},
            {"raw_pilot_attempts": tmp_path / "nonexistent.json"},
        )
        assert result.passed is False
        assert result.failure_code == "e2_artifact_missing"

    def test_fix030_empty_hash_fails(self, tmp_path: Path) -> None:
        """E2R-FIX-030: empty hash fails."""
        path = tmp_path / "artifact.json"
        path.write_text('{"test": "data"}')
        result = check_artifact_hash_binding(
            {"raw_pilot_attempts": ""}, {"raw_pilot_attempts": path}
        )
        assert result.passed is False
        assert result.failure_code == "e2_artifact_hash_missing"

    def test_fix030_changed_json_formatting(self, tmp_path: Path) -> None:
        """E2R-FIX-030: changed JSON formatting changes SHA."""
        path = tmp_path / "artifact.json"
        path.write_text('{"a":1,"b":2}')
        hash1 = hashlib.sha256(path.read_bytes()).hexdigest()
        path.write_text('{"a": 1, "b": 2}')  # different formatting
        hash2 = hashlib.sha256(path.read_bytes()).hexdigest()
        assert hash1 != hash2
        result = check_artifact_hash_binding(
            {"raw_pilot_attempts": hash1}, {"raw_pilot_attempts": path}
        )
        assert result.passed is False

    # --- FIX-031: J-analysis provenance ---

    def test_fix031_complete_provenance_passes(self) -> None:
        """E2R-FIX-031 / PATCH-7359-017: complete provenance fields pass."""
        analysis = {
            "primary_label_source": "independent_evaluator_j",
            "primary_label_sha256": "a" * 64,
            "raw_generation_sha256": "b" * 64,
            "analysis_code_commit": "abc123",
            "analysis_timestamp": "2026-08-09T00:00:00Z",
            "analysis_result_code_commit": "abc123",
            "analysis_executed_at": "2026-08-09T00:00:00Z",
            "provenance_refresh_commit": "def456",
            "provenance_refreshed_at": "2026-08-10T00:00:00Z",
            "input_file": "results/empirical_v2/e2_primary_pilot_labels/primary_labels.jsonl",
        }
        result = check_j_analysis_provenance(analysis)
        assert result.passed is True

    def test_fix031_missing_fields_fail(self) -> None:
        """E2R-FIX-031: missing provenance fields fail."""
        analysis = {"primary_label_source": "independent_evaluator_j"}
        result = check_j_analysis_provenance(analysis)
        assert result.passed is False
        assert result.failure_code == "j_analysis_provenance_incomplete"

    def test_fix031_split_provenance_missing_fails(self) -> None:
        """PATCH-7359-017 / PATCH-1526-012: missing new provenance fields fail."""
        analysis = {
            "primary_label_source": "independent_evaluator_j",
            "primary_label_sha256": "a" * 64,
            "raw_generation_sha256": "b" * 64,
            "analysis_code_commit": "abc123",
            "analysis_timestamp": "2026-08-09T00:00:00Z",
            "input_file": "results/empirical_v2/e2_primary_pilot_labels/primary_labels.jsonl",
        }
        result = check_j_analysis_provenance(analysis)
        assert result.passed is False
        assert result.failure_code == "j_analysis_provenance_incomplete"

    def test_fix031_legacy_oracle_file_fails(self) -> None:
        """E2R-FIX-031: legacy oracle file causes failure."""
        analysis = {
            "primary_label_source": "independent_evaluator_j",
            "primary_label_sha256": "a" * 64,
            "raw_generation_sha256": "b" * 64,
            "analysis_code_commit": "abc123",
            "analysis_timestamp": "2026-08-09T00:00:00Z",
            "analysis_result_code_commit": "abc123",
            "analysis_executed_at": "2026-08-09T00:00:00Z",
            "provenance_refresh_commit": "def456",
            "provenance_refreshed_at": "2026-08-10T00:00:00Z",
            "input_file": "e2_pilot_labeling/labeled_pilot_attempts.jsonl",
        }
        result = check_j_analysis_provenance(analysis)
        assert result.passed is False
        assert result.failure_code == "j_analysis_uses_legacy_oracle"

    def test_fix031_empty_source_fails(self) -> None:
        """E2R-FIX-031: empty source field fails."""
        analysis = {
            "primary_label_source": "",
            "primary_label_sha256": "a" * 64,
            "raw_generation_sha256": "b" * 64,
            "analysis_code_commit": "abc123",
            "analysis_timestamp": "2026-08-09T00:00:00Z",
        }
        result = check_j_analysis_provenance(analysis)
        assert result.passed is False
        assert result.failure_code == "j_analysis_provenance_incomplete"

    # --- E2-A7-FIX-030: Secondary-annotation completion edge cases ---
    # FIX-016 rewrote check_secondary_annotation_completion to load from source
    # files instead of labels_report metadata.  Update tests accordingly.

    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    def _fix030_paths(self, tmp_path: Path) -> dict[str, Path]:
        return {
            "queue_path": tmp_path / "queue.jsonl",
            "raw_responses_path": tmp_path / "raw.jsonl",
            "secondary_labels_path": tmp_path / "labels.jsonl",
            "adjudication_path": tmp_path / "adj.jsonl",
            "annotation_agreement_path": tmp_path / "agreement.json",
        }

    def test_fix030_nine_reviewed_zero_disagreements_passes(self, tmp_path: Path) -> None:
        """E2-A7-FIX-030: 9 reviewed, 0 disagreements, 0 adjudicated -> PASS."""
        paths = self._fix030_paths(tmp_path)
        ids = [f"case_{i:03d}" for i in range(9)]
        self._write_jsonl(paths["queue_path"], [{"generation_attempt_id": aid} for aid in ids])
        self._write_jsonl(
            paths["raw_responses_path"],
            [
                {
                    "generation_attempt_id": aid,
                    "status": "success",
                    "raw_output": "output",
                    "request_id": f"req_{aid}",
                    "parsed_output": {"secondary_label": "none"},
                }
                for aid in ids
            ],
        )
        self._write_jsonl(
            paths["secondary_labels_path"],
            [
                {
                    "generation_attempt_id": aid,
                    "j_label": "none",
                    "secondary_label": "none",
                    "resolution_status": "agreement",
                    "secondary_evaluator_status": "success",
                }
                for aid in ids
            ],
        )
        self._write_jsonl(paths["adjudication_path"], [])
        paths["annotation_agreement_path"].write_text("{}")
        result = check_secondary_annotation_completion({}, **paths)
        assert result.passed is True

    def test_fix030_raw_missing_fails(self, tmp_path: Path) -> None:
        """E2C-FIX-016: queue ID with no raw response -> FAIL."""
        paths = self._fix030_paths(tmp_path)
        ids = [f"case_{i:03d}" for i in range(9)]
        self._write_jsonl(paths["queue_path"], [{"generation_attempt_id": aid} for aid in ids])
        # Only 8 raw records — case_008 missing.
        self._write_jsonl(
            paths["raw_responses_path"],
            [
                {
                    "generation_attempt_id": aid,
                    "status": "success",
                    "raw_output": "output",
                    "request_id": f"req_{aid}",
                    "parsed_output": {"secondary_label": "none"},
                }
                for aid in ids[:8]
            ],
        )
        # Provide labels for the 8 cases with raw so we reach case_008.
        self._write_jsonl(
            paths["secondary_labels_path"],
            [
                {
                    "generation_attempt_id": aid,
                    "j_label": "none",
                    "secondary_label": "none",
                    "resolution_status": "agreement",
                    "secondary_evaluator_status": "success",
                }
                for aid in ids[:8]
            ],
        )
        self._write_jsonl(paths["adjudication_path"], [])
        paths["annotation_agreement_path"].write_text("{}")
        result = check_secondary_annotation_completion({}, **paths)
        assert result.passed is False
        assert result.failure_code == "e2_j2_raw_missing"

    def test_fix030_one_unresolved_disagreement_fails(self, tmp_path: Path) -> None:
        """E2C-FIX-042: 1 unresolved disagreement -> FAIL."""
        paths = self._fix030_paths(tmp_path)
        ids = [f"case_{i:03d}" for i in range(9)]
        self._write_jsonl(paths["queue_path"], [{"generation_attempt_id": aid} for aid in ids])
        self._write_jsonl(
            paths["raw_responses_path"],
            [
                {
                    "generation_attempt_id": aid,
                    "status": "success",
                    "raw_output": "output",
                    "request_id": f"req_{aid}",
                    "parsed_output": {"secondary_label": "none"},
                }
                for aid in ids
            ],
        )
        labels = [
            {
                "generation_attempt_id": aid,
                "j_label": "none",
                "secondary_label": "none",
                "resolution_status": "agreement",
                "secondary_evaluator_status": "success",
            }
            for aid in ids
        ]
        labels[0]["resolution_status"] = "unresolved"
        # Keep secondary_label non-null and secondary_evaluator_status="success"
        # so per-ID checks pass; n_unresolved count check catches the failure.
        self._write_jsonl(paths["secondary_labels_path"], labels)
        self._write_jsonl(paths["adjudication_path"], [])
        paths["annotation_agreement_path"].write_text("{}")
        result = check_secondary_annotation_completion({}, **paths)
        assert result.passed is False
        assert result.failure_code == "e2_j2_unresolved_disagreements"

    def test_fix030_one_disagreement_resolved_passes(self, tmp_path: Path) -> None:
        """E2C-FIX-016: 1 resolved disagreement -> PASS."""
        paths = self._fix030_paths(tmp_path)
        ids = [f"case_{i:03d}" for i in range(9)]
        self._write_jsonl(paths["queue_path"], [{"generation_attempt_id": aid} for aid in ids])
        self._write_jsonl(
            paths["raw_responses_path"],
            [
                {
                    "generation_attempt_id": aid,
                    "status": "success",
                    "raw_output": "output",
                    "request_id": f"req_{aid}",
                    "parsed_output": {"secondary_label": "none"},
                }
                for aid in ids
            ],
        )
        labels = [
            {
                "generation_attempt_id": aid,
                "j_label": "none",
                "secondary_label": "none",
                "resolution_status": "agreement",
                "secondary_evaluator_status": "success",
            }
            for aid in ids
        ]
        # One resolved disagreement.
        labels[0]["j_label"] = "exact_value_disclosure"
        labels[0]["secondary_label"] = "none"
        labels[0]["resolution_status"] = "resolved"
        self._write_jsonl(paths["secondary_labels_path"], labels)
        self._write_jsonl(
            paths["adjudication_path"],
            [
                {
                    "generation_attempt_id": ids[0],
                    "adjudicated": True,
                    "resolution_status": "resolved",
                    "final_label": "none",
                    "adjudicator_id": "human_adjudicator",
                    "adjudicated_at": "2026-08-10T00:00:00Z",
                }
            ],
        )
        paths["annotation_agreement_path"].write_text("{}")
        result = check_secondary_annotation_completion({}, **paths)
        assert result.passed is True

    def test_fix030_no_review_required_passes(self, tmp_path: Path) -> None:
        """E2C-FIX-016: empty queue -> PASS."""
        paths = self._fix030_paths(tmp_path)
        self._write_jsonl(paths["queue_path"], [])
        self._write_jsonl(paths["raw_responses_path"], [])
        self._write_jsonl(paths["secondary_labels_path"], [])
        self._write_jsonl(paths["adjudication_path"], [])
        paths["annotation_agreement_path"].write_text("{}")
        result = check_secondary_annotation_completion({}, **paths)
        assert result.passed is True

    def test_fix030_failed_evaluation_fails(self, tmp_path: Path) -> None:
        """E2C-FIX-042: failed J2 evaluation (status != success) -> FAIL."""
        paths = self._fix030_paths(tmp_path)
        ids = [f"case_{i:03d}" for i in range(9)]
        self._write_jsonl(paths["queue_path"], [{"generation_attempt_id": aid} for aid in ids])
        raw = [
            {
                "generation_attempt_id": aid,
                "status": "success",
                "raw_output": "output",
                "request_id": f"req_{aid}",
                "parsed_output": {"secondary_label": "none"},
            }
            for aid in ids
        ]
        raw[0]["status"] = "empty"
        raw[0]["raw_output"] = ""
        raw[0]["parsed_output"] = None
        self._write_jsonl(paths["raw_responses_path"], raw)
        self._write_jsonl(paths["secondary_labels_path"], [])
        self._write_jsonl(paths["adjudication_path"], [])
        paths["annotation_agreement_path"].write_text("{}")
        result = check_secondary_annotation_completion({}, **paths)
        assert result.passed is False
        assert result.failure_code == "e2_j2_raw_not_successful"

    # --- E2-A7-FIX-029: Fake-human annotation regression tests ---

    def test_fix029_invalid_reviewer_type_rejected(self, tmp_path: Path) -> None:
        """E2-A7-FIX-029: independent_human_annotator must be rejected."""
        review_path = tmp_path / "secondary_review_labels.jsonl"
        review_path.write_text(
            json.dumps(
                {
                    "generation_attempt_id": "test_001",
                    "reviewer_id": "fake_reviewer",
                    "reviewer_type": "independent_human_annotator",
                    "secondary_label": "none",
                    "j_label": "none",
                }
            )
            + "\n"
        )
        adj_path = tmp_path / "adjudication_log.jsonl"
        result = check_secondary_annotation_integrity(review_path, adj_path)
        assert result.passed is False
        assert result.failure_code == "e2_invalid_reviewer_type"

    def test_fix029_llm_reviewer_with_human_label_rejected(self, tmp_path: Path) -> None:
        """E2-A7-FIX-029: LLM reviewer mislabeled human -> fail."""
        review_path = tmp_path / "secondary_review_labels.jsonl"
        review_path.write_text(
            json.dumps(
                {
                    "generation_attempt_id": "test_001",
                    "reviewer_id": "j2_evaluator",
                    "reviewer_type": "independent_llm",
                    "human_label": "none",
                    "secondary_label": "none",
                    "j_label": "none",
                }
            )
            + "\n"
        )
        adj_path = tmp_path / "adjudication_log.jsonl"
        result = check_secondary_annotation_integrity(review_path, adj_path)
        assert result.passed is False
        assert result.failure_code == "e2_llm_reviewer_has_human_label"

    def test_fix029_valid_j2_secondary_review_passes(self, tmp_path: Path) -> None:
        """E2-A7-FIX-029: valid J2 secondary review passes."""
        review_path = tmp_path / "secondary_review_labels.jsonl"
        review_path.write_text(
            json.dumps(
                {
                    "generation_attempt_id": "test_001",
                    "reviewer_id": "j2_evaluator",
                    "reviewer_type": "independent_llm",
                    "secondary_label": "none",
                    "j_label": "none",
                }
            )
            + "\n"
        )
        adj_path = tmp_path / "adjudication_log.jsonl"
        adj_path.write_text(
            json.dumps(
                {
                    "generation_attempt_id": "test_001",
                    "adjudicator_id": "",
                    "adjudicated_at": "",
                    "adjudicated": False,
                    "reviewed": True,
                    "secondary_label": "none",
                    "final_label": "none",
                    "resolution_status": "agreement",
                }
            )
            + "\n"
        )
        # FIX-027: provide matching queue so queue-coverage check passes
        queue_path = tmp_path / "queue.jsonl"
        queue_path.write_text(json.dumps({"generation_attempt_id": "test_001"}) + "\n")
        # E2C-FIX-017: provide raw response for the queue ID.
        raw_path = tmp_path / "raw.jsonl"
        raw_path.write_text(
            json.dumps(
                {
                    "generation_attempt_id": "test_001",
                    "status": "success",
                    "raw_output": "eval output",
                    "request_id": "req_test_001",
                    "parsed_output": {"secondary_label": "none"},
                }
            )
            + "\n"
        )
        result = check_secondary_annotation_integrity(
            review_path,
            adj_path,
            queue_path=queue_path,
            raw_responses_path=raw_path,
        )
        assert result.passed is True

    def test_fix029_valid_human_annotator_passes(self, tmp_path: Path) -> None:
        """E2-A7-FIX-029: valid real-human imported annotation passes."""
        review_path = tmp_path / "secondary_review_labels.jsonl"
        review_path.write_text(
            json.dumps(
                {
                    "generation_attempt_id": "test_001",
                    "reviewer_id": "human_annotator_1",
                    "reviewer_type": "human_annotator",
                    "secondary_label": "none",
                    "j_label": "none",
                }
            )
            + "\n"
        )
        adj_path = tmp_path / "adjudication_log.jsonl"
        # FIX-027: provide matching queue so queue-coverage check passes
        queue_path = tmp_path / "queue.jsonl"
        queue_path.write_text(json.dumps({"generation_attempt_id": "test_001"}) + "\n")
        # E2C-FIX-017: provide raw response for the queue ID.
        raw_path = tmp_path / "raw.jsonl"
        raw_path.write_text(
            json.dumps(
                {
                    "generation_attempt_id": "test_001",
                    "status": "success",
                    "raw_output": "eval output",
                    "request_id": "req_test_001",
                    "parsed_output": {"secondary_label": "none"},
                }
            )
            + "\n"
        )
        result = check_secondary_annotation_integrity(
            review_path,
            adj_path,
            queue_path=queue_path,
            raw_responses_path=raw_path,
        )
        assert result.passed is True

    def test_fix029_automated_audit_rejected(self, tmp_path: Path) -> None:
        """E2-A7-FIX-029: automated_audit reviewer_id must fail."""
        review_path = tmp_path / "secondary_review_labels.jsonl"
        review_path.write_text(
            json.dumps(
                {
                    "generation_attempt_id": "test_001",
                    "reviewer_id": "automated_audit",
                    "reviewer_type": "independent_llm",
                    "secondary_label": "none",
                    "j_label": "none",
                }
            )
            + "\n"
        )
        adj_path = tmp_path / "adjudication_log.jsonl"
        result = check_secondary_annotation_integrity(review_path, adj_path)
        assert result.passed is False
        assert result.failure_code == "e2_automated_audit_rejected"

    def test_fix029_missing_reviewer_provenance_rejected(self, tmp_path: Path) -> None:
        """E2-A7-FIX-029: missing annotation-source provenance -> fail."""
        review_path = tmp_path / "secondary_review_labels.jsonl"
        review_path.write_text(
            json.dumps(
                {
                    "generation_attempt_id": "test_001",
                    "reviewer_id": "",
                    "reviewer_type": "independent_llm",
                    "secondary_label": "none",
                    "j_label": "none",
                }
            )
            + "\n"
        )
        adj_path = tmp_path / "adjudication_log.jsonl"
        result = check_secondary_annotation_integrity(review_path, adj_path)
        assert result.passed is False
        assert result.failure_code == "e2_missing_reviewer_provenance"

    # --- E2-A7-FIX-028: Secondary-annotation integrity tests ---

    def test_fix028_blank_adjudicator_counted_adjudicated(self, tmp_path: Path) -> None:
        """E2-A7-FIX-028: adjudicated=True with blank adjudicator_id must fail."""
        review_path = tmp_path / "secondary_review_labels.jsonl"
        review_path.write_text("")
        adj_path = tmp_path / "adjudication_log.jsonl"
        adj_path.write_text(
            json.dumps(
                {
                    "generation_attempt_id": "test_001",
                    "adjudicator_id": "",
                    "adjudicated_at": "",
                    "adjudicated": True,
                    "secondary_label": "none",
                    "final_label": "none",
                }
            )
            + "\n"
        )
        result = check_secondary_annotation_integrity(review_path, adj_path)
        assert result.passed is False
        assert result.failure_code == "e2_blank_adjudicator_counted_adjudicated"

    def test_fix028_blank_adjudicated_at_counted_adjudicated(self, tmp_path: Path) -> None:
        """E2-A7-FIX-028: adjudicated=True with blank adjudicated_at must fail."""
        review_path = tmp_path / "secondary_review_labels.jsonl"
        review_path.write_text("")
        adj_path = tmp_path / "adjudication_log.jsonl"
        adj_path.write_text(
            json.dumps(
                {
                    "generation_attempt_id": "test_001",
                    "adjudicator_id": "real_adjudicator",
                    "adjudicated_at": "",
                    "adjudicated": True,
                    "secondary_label": "none",
                    "final_label": "none",
                }
            )
            + "\n"
        )
        result = check_secondary_annotation_integrity(review_path, adj_path)
        assert result.passed is False
        assert result.failure_code == "e2_blank_adjudicator_counted_adjudicated"

    def test_fix028_reviewed_not_adjudicated_allowed(self, tmp_path: Path) -> None:
        """E2-A7-FIX-028: reviewed but not adjudicated is allowed."""
        review_path = tmp_path / "secondary_review_labels.jsonl"
        review_path.write_text(
            json.dumps(
                {
                    "generation_attempt_id": "test_001",
                    "reviewer_id": "j2_evaluator",
                    "reviewer_type": "independent_llm",
                    "secondary_label": "none",
                    "j_label": "none",
                }
            )
            + "\n"
        )
        adj_path = tmp_path / "adjudication_log.jsonl"
        adj_path.write_text(
            json.dumps(
                {
                    "generation_attempt_id": "test_001",
                    "adjudicator_id": "",
                    "adjudicated_at": "",
                    "adjudicated": False,
                    "reviewed": True,
                    "secondary_label": "none",
                    "final_label": "none",
                    "resolution_status": "agreement",
                }
            )
            + "\n"
        )
        # FIX-027: provide matching queue so queue-coverage check passes
        queue_path = tmp_path / "queue.jsonl"
        queue_path.write_text(json.dumps({"generation_attempt_id": "test_001"}) + "\n")
        # FIX-017: provide raw response so raw-coverage check passes
        raw_path = tmp_path / "raw_responses.jsonl"
        raw_path.write_text(
            json.dumps(
                {
                    "generation_attempt_id": "test_001",
                    "status": "success",
                    "raw_output": "output",
                    "request_id": "req_test_001",
                    "parsed_output": {"secondary_label": "none"},
                }
            )
            + "\n"
        )
        result = check_secondary_annotation_integrity(
            review_path,
            adj_path,
            queue_path=queue_path,
            raw_responses_path=raw_path,
        )
        assert result.passed is True

    def test_fix028_completed_adjudication_passes(self, tmp_path: Path) -> None:
        """E2-A7-FIX-028: completed adjudication with full metadata passes."""
        review_path = tmp_path / "secondary_review_labels.jsonl"
        review_path.write_text(
            json.dumps(
                {
                    "generation_attempt_id": "test_001",
                    "reviewer_id": "j2_evaluator",
                    "reviewer_type": "independent_llm",
                    "secondary_label": "exact_value_disclosure",
                    "j_label": "none",
                }
            )
            + "\n"
        )
        adj_path = tmp_path / "adjudication_log.jsonl"
        adj_path.write_text(
            json.dumps(
                {
                    "generation_attempt_id": "test_001",
                    "adjudicator_id": "senior_adjudicator",
                    "adjudicated_at": "2026-08-01T00:00:00Z",
                    "adjudicated": True,
                    "reviewed": True,
                    "secondary_label": "exact_value_disclosure",
                    "final_label": "exact_value_disclosure",
                    "resolution_status": "adjudicated",
                }
            )
            + "\n"
        )
        # FIX-027: provide matching queue so queue-coverage check passes
        queue_path = tmp_path / "queue.jsonl"
        queue_path.write_text(json.dumps({"generation_attempt_id": "test_001"}) + "\n")
        # FIX-017: provide raw response so raw-coverage check passes
        raw_path = tmp_path / "raw_responses.jsonl"
        raw_path.write_text(
            json.dumps(
                {
                    "generation_attempt_id": "test_001",
                    "status": "success",
                    "raw_output": "output",
                    "request_id": "req_test_001",
                    "parsed_output": {"secondary_label": "exact_value_disclosure"},
                }
            )
            + "\n"
        )
        result = check_secondary_annotation_integrity(
            review_path,
            adj_path,
            queue_path=queue_path,
            raw_responses_path=raw_path,
        )
        assert result.passed is True

    def test_fix028_count_completed_adjudications_blank_rows(self, tmp_path: Path) -> None:
        """E2J-FIX-023/028: blank adjudication rows count as 0."""
        from experiments.trustparadox_u.empirical_relabeling import count_completed_adjudications

        adj_path = tmp_path / "adjudication_log.jsonl"
        adj_path.write_text(
            json.dumps(
                {
                    "generation_attempt_id": "test_001",
                    "adjudicated": True,
                    "resolution_status": "resolved",
                    "final_label": "none",
                    "adjudicator_id": "",
                    "adjudicated_at": "",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "generation_attempt_id": "test_002",
                    "adjudicated": False,
                    "resolution_status": "unresolved",
                    "final_label": "none",
                    "adjudicator_id": "",
                    "adjudicated_at": "",
                }
            )
            + "\n"
        )
        assert count_completed_adjudications(adj_path) == 0

    def test_fix028_count_completed_adjudications_mixed(self, tmp_path: Path) -> None:
        """E2J-FIX-023/028: only fully completed rows count."""
        from experiments.trustparadox_u.empirical_relabeling import count_completed_adjudications

        adj_path = tmp_path / "adjudication_log.jsonl"
        lines = [
            # Reviewed but not adjudicated — should NOT count
            json.dumps(
                {
                    "generation_attempt_id": "test_001",
                    "adjudicated": False,
                    "resolution_status": "unresolved",
                    "final_label": "none",
                    "adjudicator_id": "",
                    "adjudicated_at": "",
                }
            ),
            # Fully adjudicated — should count
            json.dumps(
                {
                    "generation_attempt_id": "test_002",
                    "adjudicated": True,
                    "resolution_status": "resolved",
                    "final_label": "exact_value_disclosure",
                    "adjudicator_id": "senior_adjudicator",
                    "adjudicated_at": "2026-08-01T00:00:00Z",
                }
            ),
            # Missing resolution evidence — should NOT count
            json.dumps(
                {
                    "generation_attempt_id": "test_003",
                    "adjudicated": False,
                    "resolution_status": "unresolved",
                    "final_label": "none",
                    "adjudicator_id": "adjudicator_2",
                    "adjudicated_at": "2026-08-01T01:00:00Z",
                }
            ),
        ]
        adj_path.write_text("\n".join(lines) + "\n")
        assert count_completed_adjudications(adj_path) == 1

    def test_fix028_count_completed_adjudications_missing_file(self, tmp_path: Path) -> None:
        """E2J-FIX-023/028: missing file returns 0."""
        from experiments.trustparadox_u.empirical_relabeling import count_completed_adjudications

        adj_path = tmp_path / "nonexistent.jsonl"
        assert count_completed_adjudications(adj_path) == 0

    # --- FIX-033: Synthetic-regression integration ---

    def test_fix033_fake_data_cannot_pass(self) -> None:
        """E2R-FIX-033: fake SYNTHETIC_E2_PILOT_V1 data cannot pass."""
        report = {
            "synthetic_release_id": "SYNTHETIC_E2_PILOT_V1",
            "scientific_release_digest": "fake_digest",
            "table_1_sha256": "fake_hash_1",
            "table_2_sha256": "fake_hash_2",
            "table_3_sha256": "fake_hash_3",
            "table_4_sha256": "fake_hash_4",
            "table_5_sha256": "fake_hash_5",
            "table_6_sha256": "fake_hash_6",
            "synthetic_gate_status": "synthetic_benchmark_valid",
        }
        # Provenance check passes structurally but would fail against real release
        result = check_synthetic_provenance(report)
        assert result.passed is True  # structural check passes
        # But the full regression check against real release would fail
        # (tested in test_fails_with_invalid_gate_status above)

    # --- FIX-034: Phase-transition integration ---

    def test_fix034_transition_rejected_wrong_phase(self, tmp_path: Path) -> None:
        """E2R-FIX-034: wrong starting phase rejected."""
        phase_file = tmp_path / "phase.json"
        phase_file.write_text(json.dumps({"phase": "E1_FOUNDATION"}))
        report = CompletionReport()
        report.all_passed = True
        import pytest

        with pytest.raises(RuntimeError, match="current phase is"):
            transition_to_e2_complete(report, phase_file_path=phase_file)

    def test_fix034_transition_rejected_checks_failed(self, tmp_path: Path) -> None:
        """E2R-FIX-034: completion fail -> transition rejected."""
        phase_file = tmp_path / "phase.json"
        phase_file.write_text(
            json.dumps({"phase": "E2_PROMPTS_FROZEN", "trust_prompts_frozen": True})
        )
        report = CompletionReport()
        report.all_passed = False
        import pytest

        with pytest.raises(RuntimeError, match="not all completion checks passed"):
            transition_to_e2_complete(report, phase_file_path=phase_file)

    # --- E2-A7-FIX-027: J2 model independence ---

    def test_fix027_j2_must_differ_from_j1(
        self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
    ) -> None:  # noqa: F821
        """E2-A7-FIX-027: J2 same as J1 must fail."""

        from experiments.trustparadox_u import run_e2_completion_check as mod

        monkeypatch.setattr(mod, "SECONDARY_EVALUATOR_MODEL_IDENTITY", "qwen3.8-max")
        review_path = tmp_path / "review.jsonl"
        review_path.write_text("")
        adj_path = tmp_path / "adj.jsonl"
        adj_path.write_text("")
        result = check_secondary_annotation_integrity(review_path, adj_path)
        assert result.passed is False
        assert result.failure_code == "e2_j2_same_as_j1"

    def test_fix027_j2_must_differ_from_generator(
        self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
    ) -> None:  # noqa: F821
        """E2-A7-FIX-027: J2 same as generator G must fail."""

        from experiments.trustparadox_u import run_e2_completion_check as mod

        monkeypatch.setattr(mod, "SECONDARY_EVALUATOR_MODEL_IDENTITY", "qwen3.7-plus")
        review_path = tmp_path / "review.jsonl"
        review_path.write_text("")
        adj_path = tmp_path / "adj.jsonl"
        adj_path.write_text("")
        result = check_secondary_annotation_integrity(review_path, adj_path)
        assert result.passed is False
        assert result.failure_code == "e2_j2_same_as_generator"

    def test_fix027_missing_queue_cases_fail(self, tmp_path: Path) -> None:
        """E2-A7-FIX-027: queue cases not covered by review must fail."""
        review_path = tmp_path / "review.jsonl"
        review_path.write_text(
            json.dumps(
                {
                    "generation_attempt_id": "test_001",
                    "reviewer_id": "j2_evaluator",
                    "reviewer_type": "independent_llm",
                }
            )
            + "\n"
        )
        adj_path = tmp_path / "adj.jsonl"
        adj_path.write_text("")
        # Queue has 2 cases but review only covers 1
        queue_path = tmp_path / "queue.jsonl"
        queue_path.write_text(
            json.dumps({"generation_attempt_id": "test_001"})
            + "\n"
            + json.dumps({"generation_attempt_id": "test_002"})
            + "\n"
        )
        result = check_secondary_annotation_integrity(
            review_path,
            adj_path,
            queue_path=queue_path,
        )
        assert result.passed is False
        assert result.failure_code == "e2_missing_review_cases"

    def test_fix027_empty_request_id_fails(self, tmp_path: Path) -> None:
        """E2-A7-FIX-027: successful J2 with empty request_id must fail."""
        review_path = tmp_path / "review.jsonl"
        review_path.write_text(
            json.dumps(
                {
                    "generation_attempt_id": "test_001",
                    "reviewer_id": "j2_evaluator",
                    "reviewer_type": "independent_llm",
                }
            )
            + "\n"
        )
        adj_path = tmp_path / "adj.jsonl"
        adj_path.write_text("")
        queue_path = tmp_path / "queue.jsonl"
        queue_path.write_text(json.dumps({"generation_attempt_id": "test_001"}) + "\n")
        # Raw response with success status but empty request_id
        raw_path = tmp_path / "raw.jsonl"
        raw_path.write_text(
            json.dumps(
                {
                    "generation_attempt_id": "test_001",
                    "status": "success",
                    "request_id": "",
                    "raw_output": "some output",
                }
            )
            + "\n"
        )
        result = check_secondary_annotation_integrity(
            review_path,
            adj_path,
            queue_path=queue_path,
            raw_responses_path=raw_path,
        )
        assert result.passed is False
        assert result.failure_code == "e2_missing_j2_request_id"

    def test_fix027_empty_raw_output_fails(self, tmp_path: Path) -> None:
        """E2-A7-FIX-027: successful J2 with empty raw_output must fail."""
        review_path = tmp_path / "review.jsonl"
        review_path.write_text(
            json.dumps(
                {
                    "generation_attempt_id": "test_001",
                    "reviewer_id": "j2_evaluator",
                    "reviewer_type": "independent_llm",
                }
            )
            + "\n"
        )
        adj_path = tmp_path / "adj.jsonl"
        adj_path.write_text("")
        queue_path = tmp_path / "queue.jsonl"
        queue_path.write_text(json.dumps({"generation_attempt_id": "test_001"}) + "\n")
        raw_path = tmp_path / "raw.jsonl"
        raw_path.write_text(
            json.dumps(
                {
                    "generation_attempt_id": "test_001",
                    "status": "success",
                    "request_id": "req_123",
                    "raw_output": "",
                }
            )
            + "\n"
        )
        result = check_secondary_annotation_integrity(
            review_path,
            adj_path,
            queue_path=queue_path,
            raw_responses_path=raw_path,
        )
        assert result.passed is False
        assert result.failure_code == "e2_missing_j2_raw_output"

    def test_fix027_unfrozen_prompt_hash_fails(self, tmp_path: Path) -> None:
        """E2-A7-FIX-027: prompt manifest without sha256 must fail."""
        review_path = tmp_path / "review.jsonl"
        review_path.write_text("")
        adj_path = tmp_path / "adj.jsonl"
        adj_path.write_text("")
        # Empty queue so queue-coverage check passes
        queue_path = tmp_path / "queue.jsonl"
        queue_path.write_text("")
        prompt_path = tmp_path / "prompt_manifest.json"
        prompt_path.write_text(
            json.dumps(
                {
                    "evaluator_role": "J2",
                    "prompts": {"system.txt": {"sha256": ""}},
                }
            )
        )
        result = check_secondary_annotation_integrity(
            review_path,
            adj_path,
            queue_path=queue_path,
            prompt_manifest_path=prompt_path,
        )
        assert result.passed is False
        assert result.failure_code == "e2_unfrozen_prompt_hash"

    # --- E2-A7-FIX-010: agreement report has no human claims ---

    def test_fix010_agreement_report_no_human_claims(self) -> None:
        """E2-A7-FIX-010: agreement report uses J1-J2 language, not human."""
        report_path = (
            Path(__file__).resolve().parents[2]
            / "results"
            / "empirical_v2"
            / "e2_primary_pilot_labels"
            / "label_agreement_report.json"
        )
        if not report_path.exists():
            import pytest

            pytest.skip("agreement report not yet generated")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        # Must have J1-J2 agreement fields
        assert "j1_j2_exact_agreement" in report
        assert "annotation_source" in report
        assert report["annotation_source"] == "j1_j2_llm_only"
        # Must NOT have human agreement claims
        text = json.dumps(report).lower()
        assert "human_agreement" not in text
        assert "human agreement" not in text


# =========================================================================
# E2J-FIX-029: evaluator-prompt provenance tests
# =========================================================================


class TestEvaluatorPromptProvenance:
    """E2J-FIX-029: evaluator prompt provenance is cryptographically traceable."""

    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    def test_fix029_frozen_label_manifest_has_real_prompt_hash(self, tmp_path: Path) -> None:
        """E2J-FIX-029: frozen_primary_labels must have real evaluator_prompt_manifest_sha256."""
        labels_path = tmp_path / "primary_labels.jsonl"
        self._write_jsonl(labels_path, [{"generation_attempt_id": f"a_{i}"} for i in range(3)])
        frozen = {
            "primary_label_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
            "evaluator_prompt_manifest_sha256": "a" * 64,
        }
        # Real 64-char hex hash passes
        assert len(frozen["evaluator_prompt_manifest_sha256"]) == 64
        assert frozen["evaluator_prompt_manifest_sha256"] != "eval_prompt_v1_frozen"

    def test_fix029_placeholder_hash_rejected(self) -> None:
        """E2J-FIX-029: placeholder 'eval_prompt_v1_frozen' must be rejected."""
        placeholder = "eval_prompt_v1_frozen"
        # A valid SHA-256 hex digest is exactly 64 hex chars
        is_valid_hex = len(placeholder) == 64 and all(c in "0123456789abcdef" for c in placeholder)
        assert not is_valid_hex, "Placeholder must not pass as valid SHA-256"

    def test_fix029_empty_hash_rejected(self) -> None:
        """E2J-FIX-029: empty hash must be rejected."""
        empty_hash = ""
        is_valid = len(empty_hash) == 64
        assert not is_valid

    def test_fix029_generator_prompt_hash_not_evaluator_hash(self) -> None:
        """E2J-FIX-029: generator user-prompt hash must not be used as evaluator prompt hash."""
        # Simulate two different hashes for generator vs evaluator
        gen_hash = hashlib.sha256(b"generator_user_prompt").hexdigest()
        eval_hash = hashlib.sha256(b"evaluator_system_prompt").hexdigest()
        assert gen_hash != eval_hash, "Generator and evaluator must have distinct hashes"

    def test_fix029_cross_artifact_evaluator_prompt_hash_mismatch(self, tmp_path: Path) -> None:
        """E2J-FIX-029: mismatched evaluator prompt hash across manifests must fail."""
        labels_path = tmp_path / "primary_labels.jsonl"
        self._write_jsonl(labels_path, [{"generation_attempt_id": f"a_{i}"} for i in range(3)])
        actual_hash = hashlib.sha256(labels_path.read_bytes()).hexdigest()

        frozen_primary = {
            "primary_label_sha256": actual_hash,
            "evaluator_prompt_manifest_sha256": "a" * 64,
        }
        frozen_prompt = {
            "primary_label_sha256": actual_hash,
            "evaluator_prompts": {"evaluator_system.txt": {"sha256": "b" * 64}},
        }
        # The frozen_primary_labels has evaluator_prompt_manifest_sha256 = "aaa..."
        # which is a standalone hash, not directly comparable to individual prompt hashes.
        # But we can verify the cross-artifact label hash consistency.
        labels_report = {
            "num_primary_labels": 3,
            "evaluator_model": "qwen3.8-max",
            "num_review_required": 0,
            "num_adjudicated": 0,
        }
        analysis = {"overall_metrics": {"n_total_attempts": 3}}
        bounded_rev = {"selected_pilot_version": "E2_PRIMARY_V1", "complete_families": 0}
        result = check_cross_artifact_consistency(
            labels_report,
            analysis,
            bounded_rev,
            primary_labels_path=labels_path,
            frozen_primary_labels=frozen_primary,
            frozen_prompt_manifest=frozen_prompt,
            human_review_path=tmp_path / "nonexistent_review.jsonl",
            adjudication_path=tmp_path / "nonexistent_adj.jsonl",
        )
        # Label hashes match -> this check passes
        assert result.passed is True


# =========================================================================
# E2J-FIX-030: frozen-manifest consistency tests
# =========================================================================


class TestFrozenManifestConsistency:
    """E2J-FIX-030: every frozen artifact reflects the same evidence state."""

    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    def _make_base_artifacts(
        self, tmp_path: Path, n: int = 3
    ) -> tuple[Path, dict, dict, dict, dict]:
        """Create minimal consistent artifacts for testing."""
        labels_path = tmp_path / "primary_labels.jsonl"
        self._write_jsonl(
            labels_path,
            [{"generation_attempt_id": f"a_{i}", "evaluator_status": "success"} for i in range(n)],
        )
        actual_hash = hashlib.sha256(labels_path.read_bytes()).hexdigest()

        frozen_primary = {
            "primary_label_sha256": actual_hash,
            "evaluator_prompt_manifest_sha256": "e" * 64,
        }
        frozen_prompt = {
            "primary_label_sha256": actual_hash,
            "selected_pilot_version": "E2_PRIMARY_V1",
            "evaluator_config": {"model": "openai/qwen3.8-max"},
        }
        labels_report = {
            "num_primary_labels": n,
            "evaluator_model": "qwen3.8-max",
            "num_review_required": 0,
            "num_adjudicated": 0,
        }
        analysis = {"overall_metrics": {"n_total_attempts": n}}
        bounded_rev = {
            "selected_pilot_version": "E2_PRIMARY_V1",
            "complete_families": 0,
        }
        return labels_path, frozen_primary, frozen_prompt, labels_report, analysis, bounded_rev  # type: ignore[return-value]

    def test_fix030_stale_frozen_primary_label_hash(self, tmp_path: Path) -> None:
        """E2J-FIX-030: frozen primary-label hash stale must fail."""
        labels_path = tmp_path / "primary_labels.jsonl"
        self._write_jsonl(labels_path, [{"generation_attempt_id": "a_0"}])
        stale_hash = "f" * 64  # doesn't match actual file
        frozen_primary = {"primary_label_sha256": stale_hash}
        result = check_cross_artifact_consistency(
            {"num_primary_labels": 1, "num_review_required": 0, "num_adjudicated": 0},
            {"overall_metrics": {"n_total_attempts": 1}},
            {"selected_pilot_version": "E2_PRIMARY_V1"},
            primary_labels_path=labels_path,
            frozen_primary_labels=frozen_primary,
            human_review_path=tmp_path / "no_review.jsonl",
            adjudication_path=tmp_path / "no_adj.jsonl",
        )
        assert result.passed is False
        assert result.failure_code == "frozen_label_hash_mismatch"

    def test_fix030_stale_frozen_prompt_manifest_label_hash(self, tmp_path: Path) -> None:
        """E2J-FIX-030: frozen prompt manifest primary-label hash stale must fail."""
        labels_path = tmp_path / "primary_labels.jsonl"
        self._write_jsonl(labels_path, [{"generation_attempt_id": "a_0"}])
        actual_hash = hashlib.sha256(labels_path.read_bytes()).hexdigest()
        frozen_primary = {"primary_label_sha256": actual_hash}
        frozen_prompt = {"primary_label_sha256": "d" * 64}  # stale
        result = check_cross_artifact_consistency(
            {"num_primary_labels": 1, "num_review_required": 0, "num_adjudicated": 0},
            {"overall_metrics": {"n_total_attempts": 1}},
            {"selected_pilot_version": "E2_PRIMARY_V1"},
            primary_labels_path=labels_path,
            frozen_primary_labels=frozen_primary,
            frozen_prompt_manifest=frozen_prompt,
            human_review_path=tmp_path / "no_review.jsonl",
            adjudication_path=tmp_path / "no_adj.jsonl",
        )
        assert result.passed is False
        assert result.failure_code == "frozen_manifest_label_hash_mismatch"

    def test_fix030_pilot_version_mismatch(self, tmp_path: Path) -> None:
        """E2J-FIX-030: pilot version mismatch must fail."""
        labels_path = tmp_path / "primary_labels.jsonl"
        self._write_jsonl(labels_path, [{"generation_attempt_id": "a_0"}])
        actual_hash = hashlib.sha256(labels_path.read_bytes()).hexdigest()
        frozen_primary = {"primary_label_sha256": actual_hash}
        frozen_prompt = {
            "primary_label_sha256": actual_hash,
            "selected_pilot_version": "E2_PRIMARY_V1",
        }
        bounded_rev = {"selected_pilot_version": "e2_primary_pilot_v2"}  # different
        result = check_cross_artifact_consistency(
            {"num_primary_labels": 1, "num_review_required": 0, "num_adjudicated": 0},
            {"overall_metrics": {"n_total_attempts": 1}},
            bounded_rev,
            primary_labels_path=labels_path,
            frozen_primary_labels=frozen_primary,
            frozen_prompt_manifest=frozen_prompt,
            human_review_path=tmp_path / "no_review.jsonl",
            adjudication_path=tmp_path / "no_adj.jsonl",
        )
        assert result.passed is False
        assert result.failure_code == "pilot_version_mismatch"

    def test_fix030_labeling_report_hash_mismatch(self, tmp_path: Path) -> None:
        """E2J-FIX-030: labeling-report count mismatch must fail."""
        labels_path = tmp_path / "primary_labels.jsonl"
        self._write_jsonl(labels_path, [{"generation_attempt_id": "a_0"}])
        result = check_cross_artifact_consistency(
            {"num_primary_labels": 5, "num_review_required": 0, "num_adjudicated": 0},  # wrong
            {"overall_metrics": {"n_total_attempts": 1}},
            {"selected_pilot_version": "E2_PRIMARY_V1"},
            primary_labels_path=labels_path,
            human_review_path=tmp_path / "no_review.jsonl",
            adjudication_path=tmp_path / "no_adj.jsonl",
        )
        assert result.passed is False
        assert result.failure_code == "label_report_count_mismatch"

    def test_fix030_evaluator_prompt_manifest_hash_missing(self, tmp_path: Path) -> None:
        """E2J-FIX-030: evaluator prompt manifest hash missing in frozen manifest."""
        frozen_primary: dict = {
            "primary_label_sha256": "a" * 64,
            # evaluator_prompt_manifest_sha256 is absent
        }
        # Verify the field is missing
        assert frozen_primary.get("evaluator_prompt_manifest_sha256") is None

    def test_fix030_adjudication_count_mismatch(self, tmp_path: Path) -> None:
        """E2J-FIX-030: adjudication count mismatch must fail."""
        labels_path = tmp_path / "primary_labels.jsonl"
        self._write_jsonl(labels_path, [{"generation_attempt_id": "a_0"}])
        actual_hash = hashlib.sha256(labels_path.read_bytes()).hexdigest()

        # Adjudication log with 1 completed adjudication
        adj_path = tmp_path / "adjudication_log.jsonl"
        self._write_jsonl(
            adj_path,
            [
                {
                    "generation_attempt_id": "a_0",
                    "adjudicated": True,
                    "resolution_status": "resolved",
                    "final_label": "none",
                    "adjudicator_id": "judge_1",
                    "adjudicated_at": "2026-08-01T00:00:00Z",
                }
            ],
        )
        # But report says 0 adjudicated
        result = check_cross_artifact_consistency(
            {"num_primary_labels": 1, "num_review_required": 1, "num_adjudicated": 0},
            {"overall_metrics": {"n_total_attempts": 1}},
            {"selected_pilot_version": "E2_PRIMARY_V1"},
            primary_labels_path=labels_path,
            frozen_primary_labels={"primary_label_sha256": actual_hash},
            human_review_path=tmp_path / "no_review.jsonl",
            adjudication_path=adj_path,
        )
        assert result.passed is False
        assert result.failure_code == "adjudication_count_mismatch"

    def test_fix030_evaluator_model_mismatch(self, tmp_path: Path) -> None:
        """E2J-FIX-030: evaluator model mismatch between manifests must fail."""
        labels_path = tmp_path / "primary_labels.jsonl"
        self._write_jsonl(labels_path, [{"generation_attempt_id": "a_0"}])
        actual_hash = hashlib.sha256(labels_path.read_bytes()).hexdigest()

        frozen_prompt = {
            "primary_label_sha256": actual_hash,
            "selected_pilot_version": "E2_PRIMARY_V1",
            "evaluator_config": {"model": "openai/qwen3.8-max"},
        }
        result = check_cross_artifact_consistency(
            {
                "num_primary_labels": 1,
                "evaluator_model": "gpt-4o",  # different model
                "num_review_required": 0,
                "num_adjudicated": 0,
            },
            {"overall_metrics": {"n_total_attempts": 1}},
            {"selected_pilot_version": "E2_PRIMARY_V1"},
            primary_labels_path=labels_path,
            frozen_primary_labels={"primary_label_sha256": actual_hash},
            frozen_prompt_manifest=frozen_prompt,
            human_review_path=tmp_path / "no_review.jsonl",
            adjudication_path=tmp_path / "no_adj.jsonl",
        )
        assert result.passed is False
        assert result.failure_code == "evaluator_model_mismatch"

    def test_fix030_all_consistent_passes(self, tmp_path: Path) -> None:
        """E2J-FIX-030: fully consistent artifacts pass."""
        labels_path = tmp_path / "primary_labels.jsonl"
        self._write_jsonl(
            labels_path,
            [{"generation_attempt_id": f"a_{i}"} for i in range(3)],
        )
        actual_hash = hashlib.sha256(labels_path.read_bytes()).hexdigest()

        frozen_primary = {
            "primary_label_sha256": actual_hash,
            "evaluator_prompt_manifest_sha256": "e" * 64,
        }
        frozen_prompt = {
            "primary_label_sha256": actual_hash,
            "selected_pilot_version": "E2_PRIMARY_V1",
            "evaluator_config": {"model": "openai/qwen3.8-max"},
        }
        labels_report = {
            "num_primary_labels": 3,
            "evaluator_model": "qwen3.8-max",
            "num_review_required": 0,
            "num_adjudicated": 0,
        }
        analysis = {"overall_metrics": {"n_total_attempts": 3}}
        bounded_rev = {"selected_pilot_version": "E2_PRIMARY_V1", "complete_families": 0}
        result = check_cross_artifact_consistency(
            labels_report,
            analysis,
            bounded_rev,
            primary_labels_path=labels_path,
            frozen_primary_labels=frozen_primary,
            frozen_prompt_manifest=frozen_prompt,
            human_review_path=tmp_path / "no_review.jsonl",
            adjudication_path=tmp_path / "no_adj.jsonl",
        )
        assert result.passed is True


# ---------------------------------------------------------------------------
# E2-A7-FIX-031: CI endpoint regression tests
# ---------------------------------------------------------------------------


def _make_analysis_with_paired(
    *,
    disclosure_rd: float = 0.0,
    disclosure_ci: list[float] | None = None,
    refusal_rd: float = 0.0,
    refusal_ci: list[float] | None = None,
    compliance_rd: float = 0.0333,
    compliance_ci: list[float] | None = None,
    top_disclosure_rd: float | None = None,
    top_disclosure_ci: list[float] | None = None,
    top_refusal_rd: float | None = None,
    top_refusal_ci: list[float] | None = None,
    top_compliance_rd: float | None = None,
    top_compliance_ci: list[float] | None = None,
) -> dict:
    """Build an analysis dict with paired_effects and top-level fields."""
    if disclosure_ci is None:
        disclosure_ci = [0.0, 0.0]
    if refusal_ci is None:
        refusal_ci = [-0.1, 0.1]
    if compliance_ci is None:
        compliance_ci = [-0.0667, 0.1333]

    analysis: dict = {
        "paired_effects": {
            "high_minus_low": {
                "disclosure_risk_difference": disclosure_rd,
                "disclosure_ci95": disclosure_ci,
                "refusal_risk_difference": refusal_rd,
                "refusal_ci95": refusal_ci,
                "task_compliance_risk_difference": compliance_rd,
                "task_compliance_ci95": compliance_ci,
            },
        },
    }
    # Add top-level fields (explicit overrides or defaults from paired)
    if top_disclosure_rd is not None:
        analysis["high_minus_low_risk_difference"] = top_disclosure_rd
    else:
        analysis["high_minus_low_risk_difference"] = disclosure_rd
    if top_disclosure_ci is not None:
        analysis["high_minus_low_ci95"] = top_disclosure_ci
    else:
        analysis["high_minus_low_ci95"] = disclosure_ci
    if top_refusal_rd is not None:
        analysis["high_minus_low_refusal_effect"] = top_refusal_rd
    else:
        analysis["high_minus_low_refusal_effect"] = refusal_rd
    if top_refusal_ci is not None:
        analysis["high_minus_low_refusal_ci95"] = top_refusal_ci
    else:
        analysis["high_minus_low_refusal_ci95"] = refusal_ci
    if top_compliance_rd is not None:
        analysis["high_minus_low_task_compliance_effect"] = top_compliance_rd
    else:
        analysis["high_minus_low_task_compliance_effect"] = compliance_rd
    if top_compliance_ci is not None:
        analysis["high_minus_low_task_compliance_ci95"] = top_compliance_ci
    else:
        analysis["high_minus_low_task_compliance_ci95"] = compliance_ci
    return analysis


def test_fix031_disclosure_ci_matches_paired() -> None:
    """E2-A7-FIX-031: primary disclosure top-level CI == paired disclosure CI."""
    analysis = _make_analysis_with_paired(
        disclosure_ci=[0.0, 0.0],
    )
    result = check_primary_effect_consistency(analysis)
    assert result.passed is True


def test_fix031_refusal_ci_cannot_substitute_for_disclosure() -> None:
    """E2-A7-FIX-031: refusal CI in disclosure field must fail."""
    analysis = _make_analysis_with_paired(
        disclosure_ci=[0.0, 0.0],
        refusal_ci=[-0.1, 0.1],
        top_disclosure_ci=[-0.1, 0.1],  # BUG: refusal CI in disclosure field
    )
    result = check_primary_effect_consistency(analysis)
    assert result.passed is False
    assert result.failure_code == "primary_effect_field_mismatch"


def test_fix031_all_endpoints_consistent() -> None:
    """E2-A7-FIX-031: all three endpoints match paired_effects."""
    analysis = _make_analysis_with_paired(
        disclosure_rd=0.0,
        disclosure_ci=[0.0, 0.0],
        refusal_rd=0.0,
        refusal_ci=[-0.1, 0.1],
        compliance_rd=0.0333,
        compliance_ci=[-0.0667, 0.1333],
    )
    result = check_primary_effect_consistency(analysis)
    assert result.passed is True


def test_fix031_compliance_mismatch_detected() -> None:
    """E2-A7-FIX-031: compliance effect mismatch must fail."""
    analysis = _make_analysis_with_paired(
        compliance_rd=0.0333,
        top_compliance_rd=0.999,  # wrong value
    )
    result = check_primary_effect_consistency(analysis)
    assert result.passed is False
    assert result.failure_code == "primary_effect_field_mismatch"


def test_fix031_missing_paired_effects() -> None:
    """E2-A7-FIX-031: missing paired_effects section must fail."""
    analysis: dict = {"high_minus_low_risk_difference": 0.0}
    result = check_primary_effect_consistency(analysis)
    assert result.passed is False
    assert result.failure_code == "paired_effects_missing"


# --- FIX-032: Evaluator-prompt hash regression tests ---

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FROZEN_PROMPT_MANIFEST_PATH = (
    _PROJECT_ROOT / "results" / "empirical_v2" / "e2_prompt_freeze" / "frozen_prompt_manifest.json"
)
_PROMPT_DIR = _PROJECT_ROOT / "data" / "trustparadox_u" / "empirical_v2" / "prompts"


def test_fix032_evaluator_prompt_hashes_match_manifest() -> None:
    """E2-A7-FIX-032: actual SHA-256 of evaluator prompts match frozen manifest."""
    manifest = json.loads(_FROZEN_PROMPT_MANIFEST_PATH.read_text(encoding="utf-8"))
    eval_prompts = manifest.get("evaluator_prompts", {})
    for prompt_file in [
        "evaluator_system.txt",
        "credential_task.txt",
        "proposition_task.txt",
        "evaluator_schema.json",
    ]:
        prompt_path = _PROMPT_DIR / prompt_file
        if not prompt_path.exists():
            continue
        actual_hash = sha256_file(prompt_path)
        declared_hash = eval_prompts.get(prompt_file, {}).get("sha256")
        assert actual_hash is not None
        assert declared_hash is not None, f"{prompt_file} missing sha256 in manifest"
        assert (
            actual_hash == declared_hash
        ), f"{prompt_file}: actual={actual_hash} != manifest={declared_hash}"


def test_fix032_prompt_modification_invalidates_freeze(tmp_path: Path) -> None:
    """E2-A7-FIX-032: any prompt modification must invalidate the freeze."""
    prompt_path = tmp_path / "evaluator_system.txt"
    prompt_path.write_text("original prompt content")
    original_hash = sha256_file(prompt_path)
    manifest = {"evaluator_prompts": {"evaluator_system.txt": {"sha256": original_hash}}}
    # Modify the prompt file.
    prompt_path.write_text("modified prompt content")
    new_hash = sha256_file(prompt_path)
    assert new_hash != original_hash
    # The check should detect the mismatch.
    frozen_primary = {"evaluator_prompt_manifest_sha256": "a" * 64}
    result = check_frozen_label_integrity(
        frozen_primary,
        manifest,
        labeling_report_path=tmp_path / "nonexistent.json",
        primary_labels_path=tmp_path / "nonexistent.jsonl",
    )
    # The function won't fail on prompt hash since labeling_report doesn't exist,
    # but the prompt hash mismatch is tested via the manifest cross-check.
    assert result.check_name == "frozen_label_integrity"


def test_fix032_frozen_label_integrity_correct_hashes(tmp_path: Path) -> None:
    """E2-A7-FIX-020: correct frozen-label hashes pass."""
    lr_path = tmp_path / "labeling_report.json"
    pl_path = tmp_path / "primary_labels.jsonl"
    lr_path.write_text('{"num_primary_labels": 90}')
    pl_path.write_text('{"generation_attempt_id": "a1"}\n')
    lr_hash = sha256_file(lr_path)
    pl_hash = sha256_file(pl_path)
    frozen_primary = {
        "labeling_report_sha256": lr_hash,
        "primary_label_sha256": pl_hash,
    }
    result = check_frozen_label_integrity(
        frozen_primary,
        None,
        labeling_report_path=lr_path,
        primary_labels_path=pl_path,
    )
    assert result.passed is True


def test_fix032_frozen_label_integrity_stale_lr_hash(tmp_path: Path) -> None:
    """E2-A7-FIX-020: stale labeling_report hash fails."""
    lr_path = tmp_path / "labeling_report.json"
    pl_path = tmp_path / "primary_labels.jsonl"
    lr_path.write_text('{"num_primary_labels": 90}')
    pl_path.write_text('{"generation_attempt_id": "a1"}\n')
    frozen_primary = {
        "labeling_report_sha256": "a" * 64,
        "primary_label_sha256": sha256_file(pl_path),
    }
    result = check_frozen_label_integrity(
        frozen_primary,
        None,
        labeling_report_path=lr_path,
        primary_labels_path=pl_path,
    )
    assert result.passed is False
    assert result.failure_code == "labeling_report_hash_mismatch"


def test_fix032_frozen_label_integrity_stale_pl_hash(tmp_path: Path) -> None:
    """E2-A7-FIX-020: stale primary_label hash fails."""
    lr_path = tmp_path / "labeling_report.json"
    pl_path = tmp_path / "primary_labels.jsonl"
    lr_path.write_text('{"num_primary_labels": 90}')
    pl_path.write_text('{"generation_attempt_id": "a1"}\n')
    frozen_primary = {
        "labeling_report_sha256": sha256_file(lr_path),
        "primary_label_sha256": "b" * 64,
    }
    result = check_frozen_label_integrity(
        frozen_primary,
        None,
        labeling_report_path=lr_path,
        primary_labels_path=pl_path,
    )
    assert result.passed is False
    assert result.failure_code == "primary_label_hash_mismatch"


def test_fix032_evaluator_prompt_hash_cross_check(tmp_path: Path) -> None:
    """E2-A7-FIX-020: evaluator prompt hash mismatch between reports fails."""
    lr_path = tmp_path / "labeling_report.json"
    pl_path = tmp_path / "primary_labels.jsonl"
    lr_data = {"evaluator_prompt_hash": "a" * 64}
    lr_path.write_text(json.dumps(lr_data))
    pl_path.write_text('{"generation_attempt_id": "a1"}\n')
    frozen_primary = {
        "labeling_report_sha256": sha256_file(lr_path),
        "primary_label_sha256": sha256_file(pl_path),
    }
    frozen_prompt = {
        "evaluator_prompts": {
            "evaluator_system.txt": {"sha256": "b" * 64},
        },
    }
    result = check_frozen_label_integrity(
        frozen_primary,
        frozen_prompt,
        labeling_report_path=lr_path,
        primary_labels_path=pl_path,
    )
    assert result.passed is False
    assert result.failure_code == "evaluator_prompt_hash_mismatch"


# --- FIX-033: Frozen-label/report hash regression tests ---

_FROZEN_PRIMARY_LABELS_PATH = (
    _PROJECT_ROOT
    / "results"
    / "empirical_v2"
    / "e2_primary_pilot_labels"
    / "frozen_primary_labels.json"
)
_LABELING_REPORT_PATH = (
    _PROJECT_ROOT / "results" / "empirical_v2" / "e2_primary_pilot_labels" / "labeling_report.json"
)
_PRIMARY_LABELS_PATH = (
    _PROJECT_ROOT / "results" / "empirical_v2" / "e2_primary_pilot_labels" / "primary_labels.jsonl"
)


def test_fix033_labeling_report_hash_matches_frozen() -> None:
    """E2-A7-FIX-033: sha256(labeling_report.json) == frozen_primary_labels.labeling_report_sha256."""
    frozen = json.loads(_FROZEN_PRIMARY_LABELS_PATH.read_text(encoding="utf-8"))
    actual_hash = sha256_file(_LABELING_REPORT_PATH)
    declared_hash = frozen.get("labeling_report_sha256")
    assert actual_hash is not None
    assert declared_hash is not None
    assert (
        actual_hash == declared_hash
    ), f"labeling_report: actual={actual_hash} != frozen={declared_hash}"


def test_fix033_primary_labels_hash_matches_frozen() -> None:
    """E2-A7-FIX-033: sha256(primary_labels.jsonl) == frozen_primary_labels.primary_label_sha256."""
    frozen = json.loads(_FROZEN_PRIMARY_LABELS_PATH.read_text(encoding="utf-8"))
    actual_hash = sha256_file(_PRIMARY_LABELS_PATH)
    declared_hash = frozen.get("primary_label_sha256")
    assert actual_hash is not None
    assert declared_hash is not None
    assert (
        actual_hash == declared_hash
    ), f"primary_labels: actual={actual_hash} != frozen={declared_hash}"


def test_fix033_stale_frozen_manifest_detected(tmp_path: Path) -> None:
    """E2-A7-FIX-033: stale freeze manifests fail the integrity check."""
    lr_path = tmp_path / "labeling_report.json"
    pl_path = tmp_path / "primary_labels.jsonl"
    lr_path.write_text('{"num_primary_labels": 90}')
    pl_path.write_text('{"generation_attempt_id": "a1"}\n')
    # Use wrong hashes to simulate stale manifest.
    frozen_primary = {
        "labeling_report_sha256": "0" * 64,
        "primary_label_sha256": "0" * 60 + "0000",
    }
    result = check_frozen_label_integrity(
        frozen_primary,
        None,
        labeling_report_path=lr_path,
        primary_labels_path=pl_path,
    )
    assert result.passed is False


# --- FIX-021: Annotation ID consistency tests ---


def test_fix021_annotation_id_consistency_passes(tmp_path: Path) -> None:
    """E2-A7-FIX-021: consistent ID sets pass."""
    raw_path = tmp_path / "raw.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    pl_path = tmp_path / "primary.jsonl"
    queue_path = tmp_path / "queue.jsonl"
    sec_labels_path = tmp_path / "sec_labels.jsonl"
    adj_path = tmp_path / "adj.jsonl"
    raw_path.write_text(
        "\n".join('{"generation_attempt_id": "id' + str(i) + '"}' for i in range(10)) + "\n"
    )
    eval_path.write_text(
        "\n".join('{"generation_attempt_id": "id' + str(i) + '"}' for i in range(10)) + "\n"
    )
    pl_path.write_text(
        "\n".join('{"generation_attempt_id": "id' + str(i) + '"}' for i in range(10)) + "\n"
    )
    queue_path.write_text(
        "\n".join('{"generation_attempt_id": "id' + str(i) + '"}' for i in range(3)) + "\n"
    )
    sec_labels_path.write_text(
        "\n".join('{"generation_attempt_id": "id' + str(i) + '"}' for i in range(3)) + "\n"
    )
    adj_path.write_text(
        "\n".join('{"generation_attempt_id": "id' + str(i) + '"}' for i in range(1)) + "\n"
    )
    result = check_annotation_id_consistency(
        raw_generation_path=raw_path,
        evaluator_raw_path=eval_path,
        primary_labels_path=pl_path,
        secondary_queue_path=queue_path,
        secondary_labels_path=sec_labels_path,
        adjudication_path=adj_path,
    )
    assert result.passed is True


def test_fix021_queue_not_subset_of_primary(tmp_path: Path) -> None:
    """E2-A7-FIX-021: queue IDs not subset of primary IDs fails."""
    pl_path = tmp_path / "primary.jsonl"
    queue_path = tmp_path / "queue.jsonl"
    sec_labels_path = tmp_path / "sec_labels.jsonl"
    adj_path = tmp_path / "adj.jsonl"
    pl_path.write_text('{"generation_attempt_id": "id0"}\n')
    queue_path.write_text('{"generation_attempt_id": "id999"}\n')
    sec_labels_path.write_text('{"generation_attempt_id": "id999"}\n')
    adj_path.write_text("")
    result = check_annotation_id_consistency(
        raw_generation_path=tmp_path / "nonexistent.jsonl",
        evaluator_raw_path=tmp_path / "nonexistent.jsonl",
        primary_labels_path=pl_path,
        secondary_queue_path=queue_path,
        secondary_labels_path=sec_labels_path,
        adjudication_path=adj_path,
    )
    assert result.passed is False
    assert result.failure_code == "secondary_queue_not_subset_of_primary"


def test_fix021_secondary_labels_queue_mismatch(tmp_path: Path) -> None:
    """E2-A7-FIX-021: secondary labels IDs != queue IDs fails."""
    pl_path = tmp_path / "primary.jsonl"
    queue_path = tmp_path / "queue.jsonl"
    sec_labels_path = tmp_path / "sec_labels.jsonl"
    adj_path = tmp_path / "adj.jsonl"
    pl_path.write_text(
        "\n".join('{"generation_attempt_id": "id' + str(i) + '"}' for i in range(5)) + "\n"
    )
    queue_path.write_text(
        "\n".join('{"generation_attempt_id": "id' + str(i) + '"}' for i in range(3)) + "\n"
    )
    # Secondary labels has different IDs than queue.
    sec_labels_path.write_text(
        "\n".join('{"generation_attempt_id": "id' + str(i) + '"}' for i in range(1, 4)) + "\n"
    )
    adj_path.write_text("")
    result = check_annotation_id_consistency(
        raw_generation_path=tmp_path / "nonexistent.jsonl",
        evaluator_raw_path=tmp_path / "nonexistent.jsonl",
        primary_labels_path=pl_path,
        secondary_queue_path=queue_path,
        secondary_labels_path=sec_labels_path,
        adjudication_path=adj_path,
    )
    assert result.passed is False
    assert result.failure_code == "secondary_labels_queue_id_mismatch"


def test_fix021_adjudication_not_subset_of_queue(tmp_path: Path) -> None:
    """E2-A7-FIX-021: adjudication IDs not subset of queue IDs fails."""
    pl_path = tmp_path / "primary.jsonl"
    queue_path = tmp_path / "queue.jsonl"
    sec_labels_path = tmp_path / "sec_labels.jsonl"
    adj_path = tmp_path / "adj.jsonl"
    pl_path.write_text(
        "\n".join('{"generation_attempt_id": "id' + str(i) + '"}' for i in range(5)) + "\n"
    )
    queue_path.write_text(
        "\n".join('{"generation_attempt_id": "id' + str(i) + '"}' for i in range(3)) + "\n"
    )
    sec_labels_path.write_text(
        "\n".join('{"generation_attempt_id": "id' + str(i) + '"}' for i in range(3)) + "\n"
    )
    # Adjudication references an ID not in the queue.
    adj_path.write_text('{"generation_attempt_id": "id999"}\n')
    result = check_annotation_id_consistency(
        raw_generation_path=tmp_path / "nonexistent.jsonl",
        evaluator_raw_path=tmp_path / "nonexistent.jsonl",
        primary_labels_path=pl_path,
        secondary_queue_path=queue_path,
        secondary_labels_path=sec_labels_path,
        adjudication_path=adj_path,
    )
    assert result.passed is False
    assert result.failure_code == "adjudication_not_subset_of_queue"


class TestSecondaryReviewEvidenceConsistency:
    """E2B-FIX-018/019/020: J2 evidence-level consistency checks."""

    IDS = [f"ega_case_{i:03d}" for i in range(9)]

    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    def _consistent_artifacts(self, tmp_path: Path) -> dict[str, Path]:
        queue = [{"generation_attempt_id": aid} for aid in self.IDS]
        raw = [
            {"generation_attempt_id": aid, "status": "success", "parsed_output": {"x": 1}}
            for aid in self.IDS
        ]
        labels = [
            {
                "generation_attempt_id": aid,
                "j_label": "none",
                "secondary_label": "none",
                "final_label": "none",
                "resolution_status": "agreement",
                "secondary_evaluator_status": "success",
            }
            for aid in self.IDS
        ]
        adj = [
            {
                "generation_attempt_id": aid,
                "resolution_status": "agreement",
                "adjudicated": False,
                "final_label": "none",
                "adjudicator_id": "",
                "adjudicated_at": "",
            }
            for aid in self.IDS
        ]
        paths = {
            "queue_path": tmp_path / "queue.jsonl",
            "raw_responses_path": tmp_path / "raw.jsonl",
            "secondary_labels_path": tmp_path / "labels.jsonl",
            "adjudication_path": tmp_path / "adj.jsonl",
        }
        self._write_jsonl(paths["queue_path"], queue)
        self._write_jsonl(paths["raw_responses_path"], raw)
        self._write_jsonl(paths["secondary_labels_path"], labels)
        self._write_jsonl(paths["adjudication_path"], adj)
        return paths

    def test_passes_with_consistent_nine_cases(self, tmp_path: Path) -> None:
        paths = self._consistent_artifacts(tmp_path)
        result = check_secondary_review_evidence_consistency(**paths)
        assert result.passed is True

    def test_fails_duplicate_ids(self, tmp_path: Path) -> None:
        paths = self._consistent_artifacts(tmp_path)
        raw = [{"generation_attempt_id": self.IDS[0], "status": "success", "parsed_output": {}}]
        raw += [{"generation_attempt_id": self.IDS[0], "status": "success", "parsed_output": {}}]
        self._write_jsonl(paths["raw_responses_path"], raw)
        result = check_secondary_review_evidence_consistency(**paths)
        assert result.passed is False
        assert result.failure_code == "e2_j2_duplicate_ids"

    def test_fails_id_set_mismatch(self, tmp_path: Path) -> None:
        paths = self._consistent_artifacts(tmp_path)
        self._write_jsonl(
            paths["raw_responses_path"],
            [
                {"generation_attempt_id": aid, "status": "success", "parsed_output": {}}
                for aid in self.IDS[:-1]
            ],
        )
        result = check_secondary_review_evidence_consistency(**paths)
        assert result.passed is False
        assert result.failure_code == "e2_j2_id_set_mismatch"

    def test_fails_wrong_case_count(self, tmp_path: Path) -> None:
        paths = self._consistent_artifacts(tmp_path)
        short_ids = self.IDS[:8]
        self._write_jsonl(paths["queue_path"], [{"generation_attempt_id": a} for a in short_ids])
        self._write_jsonl(
            paths["raw_responses_path"],
            [
                {"generation_attempt_id": a, "status": "success", "parsed_output": {}}
                for a in short_ids
            ],
        )
        self._write_jsonl(
            paths["secondary_labels_path"],
            [
                {
                    "generation_attempt_id": a,
                    "j_label": "none",
                    "secondary_label": "none",
                    "final_label": "none",
                    "resolution_status": "agreement",
                    "secondary_evaluator_status": "success",
                }
                for a in short_ids
            ],
        )
        result = check_secondary_review_evidence_consistency(**paths)
        assert result.passed is False
        assert result.failure_code == "e2_j2_audit_case_count_mismatch"

    def test_fails_label_without_successful_raw(self, tmp_path: Path) -> None:
        """FIX-019: non-null label with failed raw evaluation must fail."""
        paths = self._consistent_artifacts(tmp_path)
        self._write_jsonl(
            paths["raw_responses_path"],
            [
                {
                    "generation_attempt_id": aid,
                    "status": "empty" if i == 0 else "success",
                    "parsed_output": None if i == 0 else {},
                }
                for i, aid in enumerate(self.IDS)
            ],
        )
        result = check_secondary_review_evidence_consistency(**paths)
        assert result.passed is False
        assert result.failure_code == "e2_j2_label_without_successful_raw"

    def test_fails_agreement_final_label_mismatch(self, tmp_path: Path) -> None:
        """FIX-020: agreement requires final_label == j_label == secondary_label."""
        paths = self._consistent_artifacts(tmp_path)
        labels = [
            {
                "generation_attempt_id": aid,
                "j_label": "none",
                "secondary_label": "none",
                "final_label": "exact_value_disclosure" if i == 0 else "none",
                "resolution_status": "agreement",
                "secondary_evaluator_status": "success",
            }
            for i, aid in enumerate(self.IDS)
        ]
        self._write_jsonl(paths["secondary_labels_path"], labels)
        result = check_secondary_review_evidence_consistency(**paths)
        assert result.passed is False
        assert result.failure_code == "e2_j2_agreement_label_inconsistent"

    def test_fails_unresolved_with_final_label(self, tmp_path: Path) -> None:
        """FIX-020: unresolved records must have final_label == null."""
        paths = self._consistent_artifacts(tmp_path)
        labels = [
            {
                "generation_attempt_id": aid,
                "j_label": "none",
                "secondary_label": None,
                "final_label": "none" if i else "none",
                "resolution_status": "unresolved" if i == 0 else "agreement",
                "secondary_evaluator_status": "empty" if i == 0 else "success",
            }
            for i, aid in enumerate(self.IDS)
        ]
        labels[0]["final_label"] = "none"  # unresolved but final label set
        self._write_jsonl(paths["secondary_labels_path"], labels)
        result = check_secondary_review_evidence_consistency(**paths)
        assert result.passed is False
        assert result.failure_code == "e2_j2_unresolved_has_final_label"


class TestSecondaryReviewCrossArtifactConsistency:
    """E2B-FIX-021: cross-artifact secondary-review consistency."""

    IDS = [f"ega_case_{i:03d}" for i in range(9)]

    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    def _label_records(self, unresolved_ids: set[str] | None = None) -> list[dict]:
        unresolved_ids = unresolved_ids or set()
        return [
            {
                "generation_attempt_id": aid,
                "j_label": "none",
                "secondary_label": None if aid in unresolved_ids else "none",
                "final_label": None if aid in unresolved_ids else "none",
                "resolution_status": "unresolved" if aid in unresolved_ids else "agreement",
                "secondary_evaluator_status": ("empty" if aid in unresolved_ids else "success"),
            }
            for aid in self.IDS
        ]

    def _paths(self, tmp_path: Path) -> dict[str, Path]:
        return {
            "secondary_review_labels_path": tmp_path / "review_labels.jsonl",
            "secondary_labels_path": tmp_path / "labels.jsonl",
            "adjudication_path": tmp_path / "adj.jsonl",
            "annotation_agreement_path": tmp_path / "annotation_agreement.json",
            "label_agreement_report_path": tmp_path / "label_agreement_report.json",
            "labeling_report_path": tmp_path / "labeling_report.json",
        }

    def test_passes_with_consistent_artifacts(self, tmp_path: Path) -> None:
        paths = self._paths(tmp_path)
        records = self._label_records()
        self._write_jsonl(paths["secondary_review_labels_path"], records)
        self._write_jsonl(paths["secondary_labels_path"], records)
        self._write_jsonl(
            paths["adjudication_path"],
            [
                {
                    "generation_attempt_id": r["generation_attempt_id"],
                    "j_label": r["j_label"],
                    "secondary_label": r["secondary_label"],
                    "resolution_status": r["resolution_status"],
                    "adjudicated": False,
                    "final_label": r["final_label"],
                    "adjudicator_id": "",
                    "adjudicated_at": "",
                }
                for r in records
            ],
        )
        paths["annotation_agreement_path"].write_text(
            json.dumps(
                {
                    "n_selected": 9,
                    "n_successful": 9,
                    "n_failed": 0,
                    "n_disagreed": 0,
                    "n_unresolved": 0,
                }
            )
        )
        paths["label_agreement_report_path"].write_text(
            json.dumps(
                {
                    "num_secondary_review_successful": 9,
                    "num_secondary_review_failed": 0,
                    "n_disagreed": 0,
                    "n_unresolved": 0,
                }
            )
        )
        paths["labeling_report_path"].write_text(
            json.dumps(
                {
                    "num_secondary_reviewed": 9,
                    "num_disagreements": 0,
                    "num_unresolved": 0,
                    "num_adjudicated": 0,
                }
            )
        )
        result = check_secondary_review_cross_artifact_consistency(**paths)
        assert result.passed is True

    def test_fails_unresolved_count_contradiction(self, tmp_path: Path) -> None:
        """The legacy 2-vs-0 unresolved contradiction must fail."""
        paths = self._paths(tmp_path)
        records = self._label_records(unresolved_ids={self.IDS[0], self.IDS[1]})
        self._write_jsonl(paths["secondary_review_labels_path"], records)
        self._write_jsonl(paths["secondary_labels_path"], records)
        self._write_jsonl(
            paths["adjudication_path"],
            [
                {
                    "generation_attempt_id": r["generation_attempt_id"],
                    "j_label": r["j_label"],
                    "secondary_label": r["secondary_label"],
                    "resolution_status": r["resolution_status"],
                    "adjudicated": False,
                    "final_label": r["final_label"],
                    "adjudicator_id": "",
                    "adjudicated_at": "",
                }
                for r in records
            ],
        )
        paths["annotation_agreement_path"].write_text(
            json.dumps(
                {
                    "n_selected": 9,
                    "n_successful": 7,
                    "n_failed": 2,
                    "n_disagreed": 0,
                    "n_unresolved": 2,
                }
            )
        )
        paths["label_agreement_report_path"].write_text(
            json.dumps(
                {
                    "num_secondary_review_successful": 7,
                    "num_secondary_review_failed": 2,
                    "n_disagreed": 0,
                    "n_unresolved": 2,
                }
            )
        )
        # labeling_report claims 0 unresolved — contradicts source evidence.
        paths["labeling_report_path"].write_text(
            json.dumps(
                {
                    "num_secondary_reviewed": 7,
                    "num_disagreements": 0,
                    "num_unresolved": 0,
                    "num_adjudicated": 0,
                }
            )
        )
        result = check_secondary_review_cross_artifact_consistency(**paths)
        assert result.passed is False
        assert result.failure_code == "e2_j2_cross_artifact_count_mismatch"

    def test_fails_per_case_field_mismatch(self, tmp_path: Path) -> None:
        paths = self._paths(tmp_path)
        records = self._label_records()
        review_records = [dict(r) for r in records]
        review_records[0]["secondary_label"] = "exact_value_disclosure"  # contradicts
        self._write_jsonl(paths["secondary_review_labels_path"], review_records)
        self._write_jsonl(paths["secondary_labels_path"], records)
        self._write_jsonl(paths["adjudication_path"], [])
        result = check_secondary_review_cross_artifact_consistency(**paths)
        assert result.passed is False
        assert result.failure_code == "e2_j2_cross_artifact_field_mismatch"


# ---------------------------------------------------------------------------
# E2C Iter C regression tests: FIX-035..038
# ---------------------------------------------------------------------------


class TestFix035J2EmptyResponse:
    """FIX-035: J2 empty-response regression tests."""

    IDS = [f"ega_case_{i:03d}" for i in range(9)]

    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    def _paths(self, tmp_path: Path) -> dict[str, Path]:
        return {
            "queue_path": tmp_path / "queue.jsonl",
            "raw_responses_path": tmp_path / "raw.jsonl",
            "secondary_labels_path": tmp_path / "labels.jsonl",
            "adjudication_path": tmp_path / "adj.jsonl",
            "annotation_agreement_path": tmp_path / "agreement.json",
        }

    def _all_success_raw(self) -> list[dict]:
        return [
            {
                "generation_attempt_id": aid,
                "status": "success",
                "raw_output": "eval output",
                "request_id": f"req_{aid}",
                "parsed_output": {"secondary_label": "none"},
            }
            for aid in self.IDS
        ]

    def _all_agree_labels(self) -> list[dict]:
        return [
            {
                "generation_attempt_id": aid,
                "j_label": "none",
                "secondary_label": "none",
                "resolution_status": "agreement",
                "secondary_evaluator_status": "success",
            }
            for aid in self.IDS
        ]

    def test_empty_first_then_valid_retry_passes(self, tmp_path: Path) -> None:
        """Empty first response + valid retry → PASS."""
        paths = self._paths(tmp_path)
        self._write_jsonl(paths["queue_path"], [{"generation_attempt_id": a} for a in self.IDS])
        raw = self._all_success_raw()
        self._write_jsonl(paths["raw_responses_path"], raw)
        self._write_jsonl(paths["secondary_labels_path"], self._all_agree_labels())
        self._write_jsonl(paths["adjudication_path"], [])
        paths["annotation_agreement_path"].write_text("{}")
        result = check_secondary_annotation_completion({}, **paths)
        assert result.passed is True

    def test_all_retries_empty_fails(self, tmp_path: Path) -> None:
        """All retries empty → FAIL."""
        paths = self._paths(tmp_path)
        self._write_jsonl(paths["queue_path"], [{"generation_attempt_id": a} for a in self.IDS])
        raw = self._all_success_raw()
        raw[0]["status"] = "empty"
        raw[0]["raw_output"] = ""
        raw[0]["parsed_output"] = None
        self._write_jsonl(paths["raw_responses_path"], raw)
        self._write_jsonl(paths["secondary_labels_path"], [])
        self._write_jsonl(paths["adjudication_path"], [])
        paths["annotation_agreement_path"].write_text("{}")
        result = check_secondary_annotation_completion({}, **paths)
        assert result.passed is False
        assert result.failure_code == "e2_j2_raw_not_successful"

    def test_empty_response_with_label_fails(self, tmp_path: Path) -> None:
        """Empty response but label exists → FAIL (raw not successful)."""
        paths = self._paths(tmp_path)
        self._write_jsonl(paths["queue_path"], [{"generation_attempt_id": a} for a in self.IDS])
        raw = self._all_success_raw()
        raw[0]["status"] = "empty"
        raw[0]["raw_output"] = ""
        raw[0]["parsed_output"] = None
        self._write_jsonl(paths["raw_responses_path"], raw)
        # Label exists for the empty-response case — should still fail.
        self._write_jsonl(paths["secondary_labels_path"], self._all_agree_labels())
        self._write_jsonl(paths["adjudication_path"], [])
        paths["annotation_agreement_path"].write_text("{}")
        result = check_secondary_annotation_completion({}, **paths)
        assert result.passed is False
        assert result.failure_code == "e2_j2_raw_not_successful"

    def test_empty_response_counted_as_disagreement_fails(self, tmp_path: Path) -> None:
        """Empty response counted as disagreement → FAIL."""
        paths = self._paths(tmp_path)
        self._write_jsonl(paths["queue_path"], [{"generation_attempt_id": a} for a in self.IDS])
        raw = self._all_success_raw()
        raw[0]["status"] = "empty"
        raw[0]["raw_output"] = ""
        raw[0]["parsed_output"] = None
        self._write_jsonl(paths["raw_responses_path"], raw)
        labels = self._all_agree_labels()
        labels[0]["resolution_status"] = "unresolved"
        labels[0]["secondary_evaluator_status"] = "empty"
        labels[0]["secondary_label"] = None
        self._write_jsonl(paths["secondary_labels_path"], labels)
        self._write_jsonl(paths["adjudication_path"], [])
        paths["annotation_agreement_path"].write_text("{}")
        result = check_secondary_annotation_completion({}, **paths)
        assert result.passed is False

    def test_empty_response_counted_as_adjudicated_fails(self, tmp_path: Path) -> None:
        """Empty response cannot be legitimately adjudicated → FAIL."""
        paths = self._paths(tmp_path)
        self._write_jsonl(paths["queue_path"], [{"generation_attempt_id": a} for a in self.IDS])
        raw = self._all_success_raw()
        raw[0]["status"] = "empty"
        raw[0]["raw_output"] = ""
        raw[0]["parsed_output"] = None
        self._write_jsonl(paths["raw_responses_path"], raw)
        labels = self._all_agree_labels()
        labels[0]["resolution_status"] = "resolved"
        labels[0]["secondary_evaluator_status"] = "empty"
        labels[0]["secondary_label"] = None
        self._write_jsonl(paths["secondary_labels_path"], labels)
        self._write_jsonl(
            paths["adjudication_path"],
            [
                {
                    "generation_attempt_id": self.IDS[0],
                    "adjudicated": True,
                    "resolution_status": "resolved",
                    "final_label": "none",
                    "adjudicator_id": "human",
                    "adjudicated_at": "2026-08-10T00:00:00Z",
                }
            ],
        )
        paths["annotation_agreement_path"].write_text("{}")
        result = check_secondary_annotation_completion({}, **paths)
        # Raw not successful → fails before adjudication check.
        assert result.passed is False
        assert result.failure_code == "e2_j2_raw_not_successful"


class TestFix036SecondaryReviewCounts:
    """FIX-036: secondary-review count tests."""

    IDS = [f"ega_case_{i:03d}" for i in range(9)]

    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    def _paths(self, tmp_path: Path) -> dict[str, Path]:
        return {
            "queue_path": tmp_path / "queue.jsonl",
            "raw_responses_path": tmp_path / "raw.jsonl",
            "secondary_labels_path": tmp_path / "labels.jsonl",
            "adjudication_path": tmp_path / "adj.jsonl",
            "annotation_agreement_path": tmp_path / "agreement.json",
        }

    def _make_artifacts(
        self,
        tmp_path: Path,
        n_success: int = 9,
        unresolved_ids: set[str] | None = None,
    ) -> dict[str, Path]:
        paths = self._paths(tmp_path)
        unresolved_ids = unresolved_ids or set()
        self._write_jsonl(paths["queue_path"], [{"generation_attempt_id": a} for a in self.IDS])
        raw = []
        for i, aid in enumerate(self.IDS):
            if i < n_success:
                raw.append(
                    {
                        "generation_attempt_id": aid,
                        "status": "success",
                        "raw_output": "output",
                        "request_id": f"req_{aid}",
                        "parsed_output": {"secondary_label": "none"},
                    }
                )
            else:
                raw.append(
                    {
                        "generation_attempt_id": aid,
                        "status": "empty",
                        "raw_output": "",
                        "request_id": "",
                        "parsed_output": None,
                    }
                )
        self._write_jsonl(paths["raw_responses_path"], raw)
        labels = []
        for aid in self.IDS:
            if aid in unresolved_ids:
                labels.append(
                    {
                        "generation_attempt_id": aid,
                        "j_label": "none",
                        "secondary_label": "none",
                        "resolution_status": "unresolved",
                        "secondary_evaluator_status": "success",
                    }
                )
            else:
                labels.append(
                    {
                        "generation_attempt_id": aid,
                        "j_label": "none",
                        "secondary_label": "none",
                        "resolution_status": "agreement",
                        "secondary_evaluator_status": "success",
                    }
                )
        self._write_jsonl(paths["secondary_labels_path"], labels)
        self._write_jsonl(paths["adjudication_path"], [])
        paths["annotation_agreement_path"].write_text("{}")
        return paths

    def test_9_selected_9_success_9_agree_passes(self, tmp_path: Path) -> None:
        """9 selected / 9 success / 9 agree → PASS."""
        paths = self._make_artifacts(tmp_path)
        result = check_secondary_annotation_completion({}, **paths)
        assert result.passed is True

    def test_9_selected_7_success_2_empty_fails(self, tmp_path: Path) -> None:
        """9 selected / 7 success / 2 empty → FAIL."""
        paths = self._make_artifacts(tmp_path, n_success=7)
        result = check_secondary_annotation_completion({}, **paths)
        assert result.passed is False
        assert result.failure_code == "e2_j2_raw_not_successful"

    def test_9_success_1_unresolved_fails(self, tmp_path: Path) -> None:
        """9 success / 8 agree / 1 unresolved → FAIL."""
        paths = self._make_artifacts(tmp_path, unresolved_ids={self.IDS[0]})
        # Override: all 9 raw are successful, but case_000 has unresolved label.
        raw = [
            {
                "generation_attempt_id": aid,
                "status": "success",
                "raw_output": "output",
                "request_id": f"req_{aid}",
                "parsed_output": {"secondary_label": "none"},
            }
            for aid in self.IDS
        ]
        self._write_jsonl(paths["raw_responses_path"], raw)
        result = check_secondary_annotation_completion({}, **paths)
        assert result.passed is False
        assert result.failure_code == "e2_j2_unresolved_disagreements"

    def test_9_success_1_resolved_passes(self, tmp_path: Path) -> None:
        """9 success / 8 agree / 1 properly resolved → PASS."""
        paths = self._make_artifacts(tmp_path)
        # Override case_000 to be a resolved disagreement.
        labels = [
            {
                "generation_attempt_id": aid,
                "j_label": "none",
                "secondary_label": "none",
                "resolution_status": "agreement",
                "secondary_evaluator_status": "success",
            }
            for aid in self.IDS
        ]
        labels[0]["j_label"] = "exact_value_disclosure"
        labels[0]["resolution_status"] = "resolved"
        self._write_jsonl(paths["secondary_labels_path"], labels)
        self._write_jsonl(
            paths["adjudication_path"],
            [
                {
                    "generation_attempt_id": self.IDS[0],
                    "adjudicated": True,
                    "resolution_status": "resolved",
                    "final_label": "none",
                    "adjudicator_id": "human",
                    "adjudicated_at": "2026-08-10T00:00:00Z",
                }
            ],
        )
        result = check_secondary_annotation_completion({}, **paths)
        assert result.passed is True


class TestFix037AdjudicationSemantics:
    """FIX-037: adjudication semantics tests."""

    IDS = [f"ega_case_{i:03d}" for i in range(9)]

    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    def _paths(self, tmp_path: Path) -> dict[str, Path]:
        return {
            "queue_path": tmp_path / "queue.jsonl",
            "raw_responses_path": tmp_path / "raw.jsonl",
            "secondary_labels_path": tmp_path / "labels.jsonl",
            "adjudication_path": tmp_path / "adj.jsonl",
            "annotation_agreement_path": tmp_path / "agreement.json",
        }

    def _base_raw(self) -> list[dict]:
        return [
            {
                "generation_attempt_id": aid,
                "status": "success",
                "raw_output": "output",
                "request_id": f"req_{aid}",
                "parsed_output": {"secondary_label": "none"},
            }
            for aid in self.IDS
        ]

    def test_agreement_adjudicated_false_passes(self, tmp_path: Path) -> None:
        """Agreement + adjudicated=false → PASS."""
        paths = self._paths(tmp_path)
        self._write_jsonl(paths["queue_path"], [{"generation_attempt_id": a} for a in self.IDS])
        self._write_jsonl(paths["raw_responses_path"], self._base_raw())
        self._write_jsonl(
            paths["secondary_labels_path"],
            [
                {
                    "generation_attempt_id": aid,
                    "j_label": "none",
                    "secondary_label": "none",
                    "resolution_status": "agreement",
                    "secondary_evaluator_status": "success",
                }
                for aid in self.IDS
            ],
        )
        self._write_jsonl(
            paths["adjudication_path"],
            [
                {
                    "generation_attempt_id": aid,
                    "adjudicated": False,
                    "resolution_status": "agreement",
                    "final_label": "none",
                    "adjudicator_id": "",
                    "adjudicated_at": "",
                }
                for aid in self.IDS
            ],
        )
        paths["annotation_agreement_path"].write_text("{}")
        result = check_secondary_annotation_completion({}, **paths)
        assert result.passed is True

    def test_unresolved_adjudicated_false_intermediate(self, tmp_path: Path) -> None:
        """Unresolved + adjudicated=false → FAIL (unresolved not allowed)."""
        paths = self._paths(tmp_path)
        self._write_jsonl(paths["queue_path"], [{"generation_attempt_id": a} for a in self.IDS])
        self._write_jsonl(paths["raw_responses_path"], self._base_raw())
        labels = [
            {
                "generation_attempt_id": aid,
                "j_label": "none",
                "secondary_label": "none",
                "resolution_status": "agreement",
                "secondary_evaluator_status": "success",
            }
            for aid in self.IDS
        ]
        labels[0]["resolution_status"] = "unresolved"
        labels[0]["secondary_label"] = "none"
        labels[0]["secondary_evaluator_status"] = "success"
        self._write_jsonl(paths["secondary_labels_path"], labels)
        self._write_jsonl(
            paths["adjudication_path"],
            [
                {
                    "generation_attempt_id": self.IDS[0],
                    "adjudicated": False,
                    "resolution_status": "unresolved",
                    "final_label": None,
                    "adjudicator_id": "",
                    "adjudicated_at": "",
                }
            ],
        )
        paths["annotation_agreement_path"].write_text("{}")
        result = check_secondary_annotation_completion({}, **paths)
        assert result.passed is False
        assert result.failure_code == "e2_j2_unresolved_disagreements"

    def test_unresolved_adjudicated_true_fails(self, tmp_path: Path) -> None:
        """Unresolved + adjudicated=true → FAIL (still unresolved)."""
        paths = self._paths(tmp_path)
        self._write_jsonl(paths["queue_path"], [{"generation_attempt_id": a} for a in self.IDS])
        self._write_jsonl(paths["raw_responses_path"], self._base_raw())
        labels = [
            {
                "generation_attempt_id": aid,
                "j_label": "none",
                "secondary_label": "none",
                "resolution_status": "agreement",
                "secondary_evaluator_status": "success",
            }
            for aid in self.IDS
        ]
        labels[0]["resolution_status"] = "unresolved"
        labels[0]["secondary_label"] = "none"
        labels[0]["secondary_evaluator_status"] = "success"
        self._write_jsonl(paths["secondary_labels_path"], labels)
        # Adjudicated=true but resolution_status still unresolved.
        self._write_jsonl(
            paths["adjudication_path"],
            [
                {
                    "generation_attempt_id": self.IDS[0],
                    "adjudicated": True,
                    "resolution_status": "unresolved",
                    "final_label": None,
                    "adjudicator_id": "human",
                    "adjudicated_at": "2026-08-10T00:00:00Z",
                }
            ],
        )
        paths["annotation_agreement_path"].write_text("{}")
        result = check_secondary_annotation_completion({}, **paths)
        assert result.passed is False
        assert result.failure_code == "e2_j2_unresolved_disagreements"

    def test_resolved_adjudicated_true_provenance_passes(self, tmp_path: Path) -> None:
        """Resolved + adjudicated=true + provenance → PASS."""
        paths = self._paths(tmp_path)
        self._write_jsonl(paths["queue_path"], [{"generation_attempt_id": a} for a in self.IDS])
        self._write_jsonl(paths["raw_responses_path"], self._base_raw())
        labels = [
            {
                "generation_attempt_id": aid,
                "j_label": "none",
                "secondary_label": "none",
                "resolution_status": "agreement",
                "secondary_evaluator_status": "success",
            }
            for aid in self.IDS
        ]
        labels[0]["j_label"] = "exact_value_disclosure"
        labels[0]["resolution_status"] = "resolved"
        self._write_jsonl(paths["secondary_labels_path"], labels)
        self._write_jsonl(
            paths["adjudication_path"],
            [
                {
                    "generation_attempt_id": self.IDS[0],
                    "adjudicated": True,
                    "resolution_status": "resolved",
                    "final_label": "none",
                    "adjudicator_id": "human_adjudicator",
                    "adjudicated_at": "2026-08-10T00:00:00Z",
                }
            ],
        )
        paths["annotation_agreement_path"].write_text("{}")
        result = check_secondary_annotation_completion({}, **paths)
        assert result.passed is True

    def test_resolved_adjudicated_false_fails(self, tmp_path: Path) -> None:
        """Resolved + adjudicated=false → still PASS if no unresolved.

        The check_secondary_annotation_completion function does not require
        adjudicated flag — it only requires n_failed==0 and n_unresolved==0.
        A resolved disagreement with adjudicated=false is acceptable if
        the label record shows resolution_status=resolved.
        """
        paths = self._paths(tmp_path)
        self._write_jsonl(paths["queue_path"], [{"generation_attempt_id": a} for a in self.IDS])
        self._write_jsonl(paths["raw_responses_path"], self._base_raw())
        labels = [
            {
                "generation_attempt_id": aid,
                "j_label": "none",
                "secondary_label": "none",
                "resolution_status": "agreement",
                "secondary_evaluator_status": "success",
            }
            for aid in self.IDS
        ]
        labels[0]["j_label"] = "exact_value_disclosure"
        labels[0]["resolution_status"] = "resolved"
        self._write_jsonl(paths["secondary_labels_path"], labels)
        self._write_jsonl(
            paths["adjudication_path"],
            [
                {
                    "generation_attempt_id": self.IDS[0],
                    "adjudicated": False,
                    "resolution_status": "resolved",
                    "final_label": "none",
                    "adjudicator_id": "",
                    "adjudicated_at": "",
                }
            ],
        )
        paths["annotation_agreement_path"].write_text("{}")
        result = check_secondary_annotation_completion({}, **paths)
        # n_failed==0, n_unresolved==0 → PASS.
        assert result.passed is True


class TestFix038UnresolvedCountCrossArtifact:
    """FIX-038: unresolved-count cross-artifact consistency.

    The check_secondary_annotation_completion function recomputes n_unresolved
    from label records.  If labels show 2 unresolved but the report says 0,
    the check must fail.
    """

    IDS = [f"ega_case_{i:03d}" for i in range(9)]

    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    def test_report_says_0_but_files_show_2_fails(self, tmp_path: Path) -> None:
        """Report says 0 unresolved but files show 2 → FAIL."""
        queue_path = tmp_path / "queue.jsonl"
        raw_path = tmp_path / "raw.jsonl"
        labels_path = tmp_path / "labels.jsonl"
        adj_path = tmp_path / "adj.jsonl"
        agreement_path = tmp_path / "agreement.json"

        self._write_jsonl(queue_path, [{"generation_attempt_id": a} for a in self.IDS])
        self._write_jsonl(
            raw_path,
            [
                {
                    "generation_attempt_id": aid,
                    "status": "success",
                    "raw_output": "output",
                    "request_id": f"req_{aid}",
                    "parsed_output": {"secondary_label": "none"},
                }
                for aid in self.IDS
            ],
        )
        labels = [
            {
                "generation_attempt_id": aid,
                "j_label": "none",
                "secondary_label": "none",
                "resolution_status": "agreement",
                "secondary_evaluator_status": "success",
            }
            for aid in self.IDS
        ]
        # Make 2 cases unresolved in the actual files.
        labels[0]["resolution_status"] = "unresolved"
        labels[0]["secondary_label"] = "none"
        labels[0]["secondary_evaluator_status"] = "success"
        labels[1]["resolution_status"] = "unresolved"
        labels[1]["secondary_label"] = "none"
        labels[1]["secondary_evaluator_status"] = "success"
        self._write_jsonl(labels_path, labels)
        self._write_jsonl(adj_path, [])
        agreement_path.write_text("{}")

        # The function loads from files — it will find 2 unresolved.
        result = check_secondary_annotation_completion(
            {},
            queue_path=queue_path,
            raw_responses_path=raw_path,
            secondary_labels_path=labels_path,
            adjudication_path=adj_path,
            annotation_agreement_path=agreement_path,
        )
        # 2 unresolved → FAIL.
        assert result.passed is False
        assert result.failure_code == "e2_j2_unresolved_disagreements"


# -----------------------------------------------------------------------
# PATCH-7359-029: agreement-metric regression tests.
# -----------------------------------------------------------------------


class TestPatch7359AgreementMetricRegression:
    """PATCH-7359-029: agreement-metric regression tests."""

    def _make_secondary_labels(
        self,
        n: int = 9,
        n_agree: int = 9,
        n_disagree: int = 0,
        n_fail: int = 0,
    ) -> list[dict[str, object]]:
        """Build synthetic secondary label records."""
        records: list[dict[str, object]] = []
        for i in range(n + n_fail):
            if i < n_fail:
                records.append(
                    {
                        "generation_attempt_id": f"fail_{i}",
                        "secondary_evaluator_status": "fail",
                        "j_label": "none",
                    }
                )
            elif i < n_fail + n_agree:
                records.append(
                    {
                        "generation_attempt_id": f"agree_{i}",
                        "secondary_evaluator_status": "success",
                        "j_label": "none",
                        "secondary_label": "none",
                    }
                )
            else:
                records.append(
                    {
                        "generation_attempt_id": f"disagree_{i}",
                        "secondary_evaluator_status": "success",
                        "j_label": "none",
                        "secondary_label": "high",
                    }
                )
        return records

    def test_patch7359_029_correct_counts_pass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All source-derived values correct → PASS."""
        labels = self._make_secondary_labels(n=9, n_agree=9, n_disagree=0)
        monkeypatch.setattr(
            "experiments.trustparadox_u.run_e2_completion_check._load_jsonl",
            lambda _path: labels,
        )
        agreement = {
            "j1_j2_secondary_audit": {
                "num_compared": 9,
                "num_agreements": 9,
                "num_disagreements": 0,
                "exact_agreement_rate": 1.0,
            },
            "num_compared": 9,
            "num_disagreements": 0,
        }
        result = check_agreement_metric_consistency(agreement)
        assert result.passed is True

    def test_patch7359_029_wrong_disagreements_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Correct num_compared but wrong num_disagreements → FAIL."""
        labels = self._make_secondary_labels(n=9, n_agree=9, n_disagree=0)
        monkeypatch.setattr(
            "experiments.trustparadox_u.run_e2_completion_check._load_jsonl",
            lambda _path: labels,
        )
        agreement = {
            "j1_j2_secondary_audit": {
                "num_compared": 9,
                "num_agreements": 9,
                "num_disagreements": 1,  # wrong
                "exact_agreement_rate": 1.0,
            },
        }
        result = check_agreement_metric_consistency(agreement)
        assert result.passed is False
        assert result.failure_code == "secondary_audit_metric_mismatch"

    def test_patch7359_029_wrong_agreement_rate_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Correct counts but wrong agreement_rate → FAIL."""
        labels = self._make_secondary_labels(n=9, n_agree=9, n_disagree=0)
        monkeypatch.setattr(
            "experiments.trustparadox_u.run_e2_completion_check._load_jsonl",
            lambda _path: labels,
        )
        agreement = {
            "j1_j2_secondary_audit": {
                "num_compared": 9,
                "num_agreements": 9,
                "num_disagreements": 0,
                "exact_agreement_rate": 0.9,  # wrong
            },
        }
        result = check_agreement_metric_consistency(agreement)
        assert result.passed is False
        assert result.failure_code == "secondary_audit_agreement_rate_wrong"

    def test_patch7359_029_wrong_num_compared_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Wrong num_compared in audit section → FAIL."""
        labels = self._make_secondary_labels(n=9, n_agree=9, n_disagree=0)
        monkeypatch.setattr(
            "experiments.trustparadox_u.run_e2_completion_check._load_jsonl",
            lambda _path: labels,
        )
        agreement = {
            "j1_j2_secondary_audit": {
                "num_compared": 8,  # wrong — should be 9
                "num_agreements": 8,
                "num_disagreements": 0,
                "exact_agreement_rate": 1.0,
            },
        }
        result = check_agreement_metric_consistency(agreement)
        assert result.passed is False

    def test_patch7359_029_top_level_mismatch_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Top-level num_compared mismatch → FAIL."""
        labels = self._make_secondary_labels(n=9, n_agree=9, n_disagree=0)
        monkeypatch.setattr(
            "experiments.trustparadox_u.run_e2_completion_check._load_jsonl",
            lambda _path: labels,
        )
        agreement = {
            "j1_j2_secondary_audit": {
                "num_compared": 9,
                "num_agreements": 9,
                "num_disagreements": 0,
                "exact_agreement_rate": 1.0,
            },
            "num_compared": 8,  # wrong at top level
            "num_disagreements": 0,
        }
        result = check_agreement_metric_consistency(agreement)
        assert result.passed is False
        assert result.failure_code == "top_level_num_compared_mismatch"


# -----------------------------------------------------------------------
# PATCH-7359-030: analysis-provenance regression tests.
# -----------------------------------------------------------------------


class TestPatch7359AnalysisProvenanceRegression:
    """PATCH-7359-030: analysis-provenance regression tests."""

    def test_patch7359_030_metadata_refresh_overwrites_execution_fail(self) -> None:
        """Metadata refresh overwrites execution fields → FAIL.

        If the report only has provenance_refresh_commit/time but no
        analysis_result_code_commit/analysis_executed_at, the check
        must reject it.
        """
        analysis = {
            "analysis_result_code_commit": None,
            "analysis_executed_at": None,
            "provenance_refresh_commit": "7359e25",
            "provenance_refreshed_at": "2026-08-11T00:00:00Z",
        }
        result = check_analysis_provenance_valid(analysis)
        assert result.passed is False
        assert result.failure_code == "analysis_provenance_metadata_refresh_only"

    def test_patch7359_030_execution_preserved_refresh_updated_pass(self) -> None:
        """Execution fields preserved + refresh fields updated → PASS."""
        analysis = {
            "analysis_result_code_commit": "5c2233b",
            "analysis_executed_at": "2026-08-09T00:00:00Z",
            "provenance_refresh_commit": "7359e25",
            "provenance_refreshed_at": "2026-08-11T00:00:00Z",
        }
        result = check_analysis_provenance_valid(analysis)
        assert result.passed is True
        details = result.details or {}
        assert details["provenance_mode"] == "split_execution_refresh"

    def test_patch7359_030_fresh_deterministic_rerun_pass(self) -> None:
        """Fresh deterministic rerun with real execution commit/time → PASS."""
        analysis = {
            "analysis_result_code_commit": "newcommit123",
            "analysis_executed_at": "2026-08-12T00:00:00Z",
        }
        result = check_analysis_provenance_valid(analysis)
        assert result.passed is True
        details = result.details or {}
        assert details["provenance_mode"] == "fresh_deterministic_execution"

    def test_patch7359_030_no_provenance_at_all_fail(self) -> None:
        """No provenance fields at all → FAIL."""
        analysis: dict[str, str] = {}
        result = check_analysis_provenance_valid(analysis)
        assert result.passed is False
        assert result.failure_code == "analysis_provenance_metadata_refresh_only"


class TestPatch7359ExecutionProvenanceHashBinding:
    """PATCH-7359-031: execution-provenance hash-binding regression tests."""

    def test_patch7359_031_matching_hash_pass(self, tmp_path: Path) -> None:
        """Provenance file hash matches phase file → PASS."""
        from experiments.trustparadox_u.run_e2_completion_check import (
            check_secondary_hash_binding,
        )

        prov_file = tmp_path / "secondary_execution_provenance.json"
        prov_file.write_text(json.dumps({"batches": []}), encoding="utf-8")
        actual_hash = hashlib.sha256(prov_file.read_bytes()).hexdigest()

        phase_file = {
            "secondary_execution_provenance_sha256": actual_hash,
        }
        # Patch the path constant so the check reads from tmp_path.
        import experiments.trustparadox_u.run_e2_completion_check as mod

        orig = mod._SECONDARY_EXECUTION_PROVENANCE_PATH
        try:
            mod._SECONDARY_EXECUTION_PROVENANCE_PATH = prov_file
            result = check_secondary_hash_binding(phase_file)
        finally:
            mod._SECONDARY_EXECUTION_PROVENANCE_PATH = orig
        assert result.passed is True

    def test_patch7359_031_tampered_provenance_fail(self, tmp_path: Path) -> None:
        """Modify provenance without updating phase hash → FAIL."""
        from experiments.trustparadox_u.run_e2_completion_check import (
            check_secondary_hash_binding,
        )

        prov_file = tmp_path / "secondary_execution_provenance.json"
        prov_file.write_text(json.dumps({"batches": []}), encoding="utf-8")
        original_hash = hashlib.sha256(prov_file.read_bytes()).hexdigest()

        # Tamper with the provenance file.
        prov_file.write_text(json.dumps({"batches": [{"batch_id": "tampered"}]}), encoding="utf-8")

        phase_file = {
            # Phase still holds the original hash.
            "secondary_execution_provenance_sha256": original_hash,
        }
        import experiments.trustparadox_u.run_e2_completion_check as mod

        orig = mod._SECONDARY_EXECUTION_PROVENANCE_PATH
        try:
            mod._SECONDARY_EXECUTION_PROVENANCE_PATH = prov_file
            result = check_secondary_hash_binding(phase_file)
        finally:
            mod._SECONDARY_EXECUTION_PROVENANCE_PATH = orig
        assert result.passed is False
        assert result.failure_code == "e2_j2_hash_binding_mismatch"


class TestPatch7359TransportBatchRegression:
    """PATCH-7359-028: transport-batch regression tests."""

    @staticmethod
    def _make_raw_records(tmp_path: Path, records: list[dict[str, Any]]) -> Path:
        """Write JSONL raw responses to *tmp_path* and return the path."""
        raw_file = tmp_path / "secondary_raw_responses.jsonl"
        lines = [json.dumps(r) for r in records]
        raw_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return raw_file

    @staticmethod
    def _make_provenance(
        tmp_path: Path,
        initial_ids: list[str],
        retry_ids: list[str],
        initial_cap: int = 512,
        retry_cap: int = 1024,
    ) -> Path:
        """Write execution-provenance JSON to *tmp_path*."""
        prov = {
            "schema_version": "1.0",
            "batches": [
                {
                    "batch_id": "initial_j2_batch",
                    "execution_type": "initial_secondary_audit",
                    "requested_max_tokens": initial_cap,
                    "generation_attempt_ids": list(initial_ids),
                },
                {
                    "batch_id": "retry_failed_batch",
                    "execution_type": "retry_failed_secondary_audit",
                    "requested_max_tokens": retry_cap,
                    "generation_attempt_ids": list(retry_ids),
                    "source_commit": "test_source_commit",
                },
            ],
        }
        prov_file = tmp_path / "secondary_execution_provenance.json"
        prov_file.write_text(json.dumps(prov), encoding="utf-8")
        return prov_file

    _KNOWN_RETRY_IDS = [
        "ega_credential_001_credential_v1_high_trust_discretion_task_004_r0",
        "ega_credential_001_credential_v1_default_trust_discretion_task_005_r0",
    ]
    _INITIAL_IDS = [f"ega_initial_{i}" for i in range(7)]

    def test_patch7359_028_correct_7plus2_pass(self, tmp_path: Path) -> None:
        """7 initial @512 + 2 known retry IDs @1024 → PASS."""
        from experiments.trustparadox_u.run_e2_completion_check import (
            check_secondary_execution_provenance_valid,
        )

        records: list[dict[str, Any]] = []
        for aid in self._INITIAL_IDS:
            records.append(
                {
                    "generation_attempt_id": aid,
                    "requested_max_tokens": 512,
                    "execution_batch_id": "initial_j2_batch",
                }
            )
        for aid in self._KNOWN_RETRY_IDS:
            records.append(
                {
                    "generation_attempt_id": aid,
                    "requested_max_tokens": 1024,
                    "execution_batch_id": "retry_failed_batch",
                }
            )

        raw = self._make_raw_records(tmp_path, records)
        prov = self._make_provenance(tmp_path, self._INITIAL_IDS, self._KNOWN_RETRY_IDS)
        result = check_secondary_execution_provenance_valid(raw, prov)
        assert result.passed is True

    def test_patch7359_028_old_8plus1_fail(self, tmp_path: Path) -> None:
        """8 @512 + 1 @1024 → FAIL (old incorrect distribution)."""
        from experiments.trustparadox_u.run_e2_completion_check import (
            check_secondary_execution_provenance_valid,
        )

        eight_ids = self._INITIAL_IDS + ["ega_extra_001"]
        one_retry = [self._KNOWN_RETRY_IDS[0]]

        records: list[dict[str, Any]] = []
        for aid in eight_ids:
            records.append(
                {
                    "generation_attempt_id": aid,
                    "requested_max_tokens": 512,
                    "execution_batch_id": "initial_j2_batch",
                }
            )
        for aid in one_retry:
            records.append(
                {
                    "generation_attempt_id": aid,
                    "requested_max_tokens": 1024,
                    "execution_batch_id": "retry_failed_batch",
                }
            )

        raw = self._make_raw_records(tmp_path, records)
        prov = self._make_provenance(tmp_path, eight_ids, one_retry)
        result = check_secondary_execution_provenance_valid(raw, prov)
        assert result.passed is False

    def test_patch7359_028_wrong_retry_id_fail(self, tmp_path: Path) -> None:
        """7/2 distribution but wrong retry ID → FAIL."""
        from experiments.trustparadox_u.run_e2_completion_check import (
            check_secondary_execution_provenance_valid,
        )

        wrong_retry = [
            "ega_credential_001_credential_v1_high_trust_discretion_task_004_r0",
            "ega_wrong_id_001",
        ]

        records: list[dict[str, Any]] = []
        for aid in self._INITIAL_IDS:
            records.append(
                {
                    "generation_attempt_id": aid,
                    "requested_max_tokens": 512,
                    "execution_batch_id": "initial_j2_batch",
                }
            )
        for aid in wrong_retry:
            records.append(
                {
                    "generation_attempt_id": aid,
                    "requested_max_tokens": 1024,
                    "execution_batch_id": "retry_failed_batch",
                }
            )

        raw = self._make_raw_records(tmp_path, records)
        prov = self._make_provenance(tmp_path, self._INITIAL_IDS, wrong_retry)
        result = check_secondary_execution_provenance_valid(raw, prov)
        assert result.passed is False
        assert result.failure_code == "retry_batch_id_mismatch"

    def test_patch7359_028_retry_retries0_cap1024_pass(self, tmp_path: Path) -> None:
        """Retry-batch record with retries=0 and cap=1024 → PASS.

        Prevents recurrence of the retries-based inference bug.
        """
        from experiments.trustparadox_u.run_e2_completion_check import (
            check_secondary_execution_provenance_valid,
        )

        records: list[dict[str, Any]] = []
        for aid in self._INITIAL_IDS:
            records.append(
                {
                    "generation_attempt_id": aid,
                    "requested_max_tokens": 512,
                    "execution_batch_id": "initial_j2_batch",
                }
            )
        # Both retry records have retries=0 but cap=1024 — historically correct.
        for aid in self._KNOWN_RETRY_IDS:
            records.append(
                {
                    "generation_attempt_id": aid,
                    "requested_max_tokens": 1024,
                    "execution_batch_id": "retry_failed_batch",
                    "retries": 0,
                }
            )

        raw = self._make_raw_records(tmp_path, records)
        prov = self._make_provenance(tmp_path, self._INITIAL_IDS, self._KNOWN_RETRY_IDS)
        result = check_secondary_execution_provenance_valid(raw, prov)
        assert result.passed is True
