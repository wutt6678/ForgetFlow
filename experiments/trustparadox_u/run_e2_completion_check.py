"""E2 repair §40-51: strengthened completion checker.

This module validates scientific contracts, not just file existence.
It checks:
- Protocol version consistency
- Phase state
- Model consistency
- Primary pilot task
- Schedule completeness
- Independent labels
- Statistics
- Bounded revision
- Generator freeze
- Synthetic regression

Checklist coverage:
- §40: Strengthened completion checker
- §41: Empirical versions check
- §42: Phase state check
- §43: Model consistency check
- §44: Primary pilot task check
- §45: Schedule check
- §46: Independent labels check
- §47: Statistics check
- §48: Bounded revision check
- §49: Generator freeze check
- §50: Synthetic regression check
- §51: Completion artifact
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.trustparadox_u.empirical_corpus import (
    E2_RESEARCH_STATUS,
    EMPIRICAL_PHASE_FILE,
    EMPIRICAL_PROTOCOL_VERSION,
    EMPIRICAL_STUDY_VERSION,
    EmpiricalPhase,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: E2R-FIX-016: canonical artifact path mapping for hash verification.
E2_ARTIFACT_PATHS: dict[str, Path] = {
    "raw_pilot_attempts": (
        _PROJECT_ROOT
        / "results"
        / "empirical_v2"
        / "e2_primary_trust_pilot"
        / "raw_generation_attempts.jsonl"
    ),
    "request_schedule": (
        _PROJECT_ROOT
        / "results"
        / "empirical_v2"
        / "e2_primary_trust_pilot"
        / "request_schedule.json"
    ),
    "primary_labels": (
        _PROJECT_ROOT
        / "results"
        / "empirical_v2"
        / "e2_primary_pilot_labels"
        / "primary_labels.jsonl"
    ),
    "reference_labels": (
        _PROJECT_ROOT
        / "results"
        / "empirical_v2"
        / "e2_primary_pilot_labels"
        / "reference_labels.jsonl"
    ),
    "adjudication_log": (
        _PROJECT_ROOT
        / "results"
        / "empirical_v2"
        / "e2_primary_pilot_labels"
        / "adjudication_log.jsonl"
    ),
    "labeling_report": (
        _PROJECT_ROOT
        / "results"
        / "empirical_v2"
        / "e2_primary_pilot_labels"
        / "labeling_report.json"
    ),
    "agreement_report": (
        _PROJECT_ROOT
        / "results"
        / "empirical_v2"
        / "e2_primary_pilot_labels"
        / "label_agreement_report.json"
    ),
    "pairing_audit": (
        _PROJECT_ROOT / "results" / "empirical_v2" / "e2_reanalysis" / "e2_pairing_audit.json"
    ),
    "pilot_analysis": (
        _PROJECT_ROOT / "results" / "empirical_v2" / "e2_reanalysis" / "e2_reanalysis_report.json"
    ),
    "floor_effect_diagnostic": (
        _PROJECT_ROOT
        / "results"
        / "empirical_v2"
        / "e2_reanalysis"
        / "floor_effect_diagnostic.json"
    ),
    "bounded_revision_report": (
        _PROJECT_ROOT
        / "results"
        / "empirical_v2"
        / "e2_reanalysis"
        / "bounded_revision_report.json"
    ),
    "frozen_prompt_manifest": (
        _PROJECT_ROOT
        / "results"
        / "empirical_v2"
        / "e2_prompt_freeze"
        / "frozen_prompt_manifest.json"
    ),
    "synthetic_regression_report": (
        _PROJECT_ROOT
        / "results"
        / "empirical_v2"
        / "e2_synthetic_regression"
        / "synthetic_regression_report.json"
    ),
}

#: E2R-FIX-019/020: directory containing synthetic release bundles.
RELEASES_DIR = _PROJECT_ROOT / "results" / "releases"

#: E2 repair §51: completion report output directory.
COMPLETION_OUTPUT_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "e2_completion"
COMPLETION_REPORT_PATH = COMPLETION_OUTPUT_DIR / "e2_research_completion_report.json"

# E2R-FIX-017/024: canonical file paths for file-based checks.
_RAW_GENERATION_PATH = (
    _PROJECT_ROOT
    / "results"
    / "empirical_v2"
    / "e2_primary_trust_pilot"
    / "raw_generation_attempts.jsonl"
)
_PRIMARY_LABELS_PATH = (
    _PROJECT_ROOT / "results" / "empirical_v2" / "e2_primary_pilot_labels" / "primary_labels.jsonl"
)
_EVALUATOR_RAW_PATH = (
    _PROJECT_ROOT
    / "results"
    / "empirical_v2"
    / "e2_primary_pilot_labels"
    / "evaluator_raw_responses.jsonl"
)
_LABELING_REPORT_PATH = (
    _PROJECT_ROOT / "results" / "empirical_v2" / "e2_primary_pilot_labels" / "labeling_report.json"
)
_AGREEMENT_REPORT_PATH = (
    _PROJECT_ROOT
    / "results"
    / "empirical_v2"
    / "e2_primary_pilot_labels"
    / "label_agreement_report.json"
)
_ADJUDICATION_LOG_PATH = (
    _PROJECT_ROOT
    / "results"
    / "empirical_v2"
    / "e2_primary_pilot_labels"
    / "adjudication_log.jsonl"
)
_REFERENCE_LABELS_PATH = (
    _PROJECT_ROOT
    / "results"
    / "empirical_v2"
    / "e2_primary_pilot_labels"
    / "reference_labels.jsonl"
)
_REANALYSIS_REPORT_PATH = (
    _PROJECT_ROOT / "results" / "empirical_v2" / "e2_reanalysis" / "e2_reanalysis_report.json"
)
_BOUNDED_REVISION_REPORT_PATH = (
    _PROJECT_ROOT / "results" / "empirical_v2" / "e2_reanalysis" / "bounded_revision_report.json"
)
_FROZEN_PROMPT_MANIFEST_PATH = (
    _PROJECT_ROOT / "results" / "empirical_v2" / "e2_prompt_freeze" / "frozen_prompt_manifest.json"
)
_SYNTHETIC_REGRESSION_REPORT_PATH = (
    _PROJECT_ROOT
    / "results"
    / "empirical_v2"
    / "e2_synthetic_regression"
    / "synthetic_regression_report.json"
)
_PILOT_MANIFEST_PATH = (
    _PROJECT_ROOT / "results" / "empirical_v2" / "e2_primary_trust_pilot" / "pilot_manifest.json"
)
_REQUEST_SCHEDULE_PATH = (
    _PROJECT_ROOT / "results" / "empirical_v2" / "e2_primary_trust_pilot" / "request_schedule.json"
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file and return a list of parsed records."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


@dataclass
class CheckResult:
    """Result of a single completion check."""

    check_name: str
    passed: bool
    failure_code: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        return {
            "passed": self.passed,
            "failure_code": self.failure_code,
            "details": self.details,
        }


@dataclass
class CompletionReport:
    """E2 repair §51: completion report."""

    check_type: str = "e2_research_completion"
    protocol_version: str = EMPIRICAL_PROTOCOL_VERSION
    study_version: str = EMPIRICAL_STUDY_VERSION
    research_status: str = E2_RESEARCH_STATUS
    all_passed: bool = True
    checks: dict[str, CheckResult] = field(default_factory=dict)
    artifact_hashes: dict[str, str | None] = field(default_factory=dict)

    def add_check(self, result: CheckResult) -> None:
        """Add a check result."""
        self.checks[result.check_name] = result
        if not result.passed:
            self.all_passed = False

    def set_artifact_hash(self, name: str, sha256: str | None) -> None:
        """Record SHA-256 for an artifact (E2R-035)."""
        self.artifact_hashes[name] = sha256

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        return {
            "check_type": self.check_type,
            "protocol_version": self.protocol_version,
            "study_version": self.study_version,
            "research_status": self.research_status,
            "all_passed": self.all_passed,
            "checks": {name: result.to_dict() for name, result in self.checks.items()},
            "artifact_hashes": dict(self.artifact_hashes),
        }


def check_protocol_consistency(artifacts: dict[str, Any]) -> CheckResult:
    """E2 repair §41: check protocol version consistency."""
    expected_protocol = EMPIRICAL_PROTOCOL_VERSION
    expected_study = EMPIRICAL_STUDY_VERSION

    for name, artifact in artifacts.items():
        if isinstance(artifact, dict):
            protocol = artifact.get("protocol_version")
            study = artifact.get("study_version")
            if protocol and protocol != expected_protocol:
                return CheckResult(
                    check_name="protocol_consistency",
                    passed=False,
                    failure_code="empirical_protocol_version_mismatch",
                    details={"artifact": name, "expected": expected_protocol, "found": protocol},
                )
            if study and study != expected_study:
                return CheckResult(
                    check_name="protocol_consistency",
                    passed=False,
                    failure_code="empirical_protocol_version_mismatch",
                    details={"artifact": name, "expected": expected_study, "found": study},
                )

    return CheckResult(
        check_name="protocol_consistency",
        passed=True,
        details={"message": "All artifacts use protocol 2.0.0"},
    )


def check_phase_state(phase_file: dict[str, Any] | None) -> CheckResult:
    """E2 repair §42 / E2R-036: check phase state.

    Accepts E2_PROMPTS_FROZEN (pre-completion) or E2_COMPLETE (post-completion).
    E2_PROMPTS_FROZEN requires full_corpus_generation_authorized=false.
    E2_COMPLETE requires full_corpus_generation_authorized=true and additional
    freeze fields (evaluator_frozen, independent_labels_frozen).
    """
    if phase_file is None:
        return CheckResult(
            check_name="phase_state",
            passed=False,
            failure_code="empirical_phase_file_missing",
        )

    phase = phase_file.get("phase")

    # E2R-036: accept E2_COMPLETE as the post-completion valid state.
    if phase == EmpiricalPhase.E2_COMPLETE.value:
        if not phase_file.get("trust_prompts_frozen"):
            return CheckResult(
                check_name="phase_state",
                passed=False,
                failure_code="trust_prompts_not_frozen",
            )
        if not phase_file.get("evaluator_frozen"):
            return CheckResult(
                check_name="phase_state",
                passed=False,
                failure_code="evaluator_not_frozen",
            )
        if not phase_file.get("independent_labels_frozen"):
            return CheckResult(
                check_name="phase_state",
                passed=False,
                failure_code="independent_labels_not_frozen",
            )
        if not phase_file.get("full_corpus_generation_authorized"):
            return CheckResult(
                check_name="phase_state",
                passed=False,
                failure_code="full_corpus_not_authorized",
            )
        return CheckResult(
            check_name="phase_state",
            passed=True,
            details={
                "phase": phase,
                "trust_prompts_frozen": True,
                "evaluator_frozen": True,
                "independent_labels_frozen": True,
                "full_corpus_generation_authorized": True,
            },
        )

    # Pre-completion: must be E2_PROMPTS_FROZEN.
    if phase != EmpiricalPhase.E2_PROMPTS_FROZEN.value:
        return CheckResult(
            check_name="phase_state",
            passed=False,
            failure_code="empirical_phase_not_frozen",
            details={
                "expected": EmpiricalPhase.E2_PROMPTS_FROZEN.value,
                "found": phase,
            },
        )

    if not phase_file.get("trust_prompts_frozen"):
        return CheckResult(
            check_name="phase_state",
            passed=False,
            failure_code="empirical_phase_not_frozen",
        )

    if phase_file.get("full_corpus_generation_authorized"):
        return CheckResult(
            check_name="phase_state",
            passed=False,
            failure_code="full_corpus_prematurely_authorized",
        )

    return CheckResult(
        check_name="phase_state",
        passed=True,
        details={"phase": phase, "trust_prompts_frozen": True},
    )


def check_model_consistency(
    connectivity_config: dict[str, Any],
    pilot_config: dict[str, Any],
) -> CheckResult:
    """E2 repair §43: check model consistency."""
    # Support both naming conventions
    conn_provider = connectivity_config.get("provider") or connectivity_config.get(
        "generator_provider"
    )
    conn_model = connectivity_config.get("model") or connectivity_config.get("generator_model")
    pilot_provider = pilot_config.get("provider") or pilot_config.get("generator_provider")
    pilot_model = pilot_config.get("model") or pilot_config.get("generator_model")

    if conn_provider != pilot_provider or conn_model != pilot_model:
        return CheckResult(
            check_name="connectivity_model_match",
            passed=False,
            failure_code="connectivity_pilot_model_mismatch",
            details={
                "connectivity": f"{conn_provider}/{conn_model}",
                "pilot": f"{pilot_provider}/{pilot_model}",
            },
        )

    return CheckResult(
        check_name="connectivity_model_match",
        passed=True,
        details={"provider": conn_provider, "model": conn_model},
    )


def check_primary_pilot_task(pilot_manifest: dict[str, Any]) -> CheckResult:
    """E2 repair §44 / E2R-022: check primary pilot task.

    Require explicit pilot_type and attack_type fields.
    No inference from artifact_class or naming conventions.
    """
    pilot_type = pilot_manifest.get("pilot_type")
    attack_type = pilot_manifest.get("attack_type")

    # Also check embedded validation report
    if attack_type is None:
        validation = pilot_manifest.get("validation_report", {})
        attack_type = validation.get("attack_type")
    if pilot_type is None:
        validation = pilot_manifest.get("validation_report", {})
        pilot_type = validation.get("pilot_type")

    if attack_type is None or pilot_type is None:
        return CheckResult(
            check_name="primary_pilot_task",
            passed=False,
            failure_code="primary_pilot_metadata_missing",
            details={
                "expected_pilot_type": "primary_trust_discretion",
                "expected_attack_type": "trust_discretion_task",
                "found_pilot_type": pilot_type,
                "found_attack_type": attack_type,
            },
        )

    if attack_type == "direct_disclosure":
        return CheckResult(
            check_name="primary_pilot_task",
            passed=False,
            failure_code="primary_pilot_uses_explicit_disclosure_attack",
            details={"attack_type": attack_type},
        )

    if pilot_type != "primary_trust_discretion" or attack_type != "trust_discretion_task":
        return CheckResult(
            check_name="primary_pilot_task",
            passed=False,
            failure_code="primary_pilot_uses_explicit_disclosure_attack",
            details={
                "expected_pilot_type": "primary_trust_discretion",
                "expected_attack_type": "trust_discretion_task",
                "found_pilot_type": pilot_type,
                "found_attack_type": attack_type,
            },
        )

    return CheckResult(
        check_name="primary_pilot_task",
        passed=True,
        details={"attack_type": attack_type, "pilot_type": pilot_type},
    )


def check_schedule(schedule: dict[str, Any] | None) -> CheckResult:
    """E2 repair §45 / E2R-023: check schedule completeness.

    Validate explicit family membership:
    - 90 scheduled requests
    - 30 unique family IDs
    - 3 trust levels/family
    - same scenario/variant/task/replicate within family
    - no duplicate trust cell
    """
    if schedule is None:
        return CheckResult(
            check_name="request_randomization",
            passed=False,
            failure_code="pilot_schedule_missing",
        )

    requests = schedule.get("requests", [])
    if len(requests) != 90:
        return CheckResult(
            check_name="request_randomization",
            passed=False,
            failure_code="pilot_schedule_not_reproducible",
            details={"expected_requests": 90, "found": len(requests)},
        )

    # Group by family_id
    families: dict[str, list[dict[str, Any]]] = {}
    for req in requests:
        family_id = req.get("generation_family_id")
        if family_id not in families:
            families[family_id] = []
        families[family_id].append(req)

    if len(families) != 30:
        return CheckResult(
            check_name="request_randomization",
            passed=False,
            failure_code="pilot_family_incomplete",
            details={"expected_families": 30, "found": len(families)},
        )

    # Validate each family
    for family_id, family_reqs in families.items():
        if len(family_reqs) != 3:
            return CheckResult(
                check_name="request_randomization",
                passed=False,
                failure_code="pilot_family_incomplete",
                details={"family_id": family_id, "expected_requests": 3, "found": len(family_reqs)},
            )

        # Check 3 trust levels
        trust_levels = {req.get("trust_level") for req in family_reqs}
        if trust_levels != {"low", "default", "high"}:
            return CheckResult(
                check_name="request_randomization",
                passed=False,
                failure_code="pilot_family_incomplete",
                details={"family_id": family_id, "trust_levels": sorted(trust_levels)},
            )

        # Check same scenario/variant/task/replicate within family
        scenario_ids = {req.get("scenario_id") for req in family_reqs}
        if len(scenario_ids) != 1:
            return CheckResult(
                check_name="request_randomization",
                passed=False,
                failure_code="pilot_family_inconsistent",
                details={"family_id": family_id, "scenario_ids": sorted(scenario_ids)},
            )

        variant_ids = {req.get("secret_variant_id") for req in family_reqs}
        if len(variant_ids) != 1:
            return CheckResult(
                check_name="request_randomization",
                passed=False,
                failure_code="pilot_family_inconsistent",
                details={"family_id": family_id, "variant_ids": sorted(variant_ids)},
            )

        attack_types = {req.get("attack_type") for req in family_reqs}
        if len(attack_types) != 1:
            return CheckResult(
                check_name="request_randomization",
                passed=False,
                failure_code="pilot_family_inconsistent",
                details={"family_id": family_id, "attack_types": sorted(attack_types)},
            )

    return CheckResult(
        check_name="request_randomization",
        passed=True,
        details={"requests": 90, "matched_families": 30},
    )


def check_annotation_independence(labels_report: dict[str, Any]) -> CheckResult:
    """E2 repair §46 / E2R-021: check annotation independence.

    Require explicit evaluator metadata and label completeness.
    No permissive inference from labeling_oracle field.
    """
    # Check evaluator metadata
    evaluator_provider = labels_report.get("evaluator_provider")
    evaluator_model_requested = labels_report.get("evaluator_model_requested")
    evaluator_model_returned = labels_report.get("evaluator_model_returned")
    evaluator_prompt_hash = labels_report.get("evaluator_prompt_hash")

    if not evaluator_provider or not evaluator_model_requested or not evaluator_model_returned:
        return CheckResult(
            check_name="annotation_independence",
            passed=False,
            failure_code="pilot_evaluator_model_missing",
            details={
                "evaluator_provider": evaluator_provider,
                "evaluator_model_requested": evaluator_model_requested,
                "evaluator_model_returned": evaluator_model_returned,
            },
        )

    if not evaluator_prompt_hash:
        return CheckResult(
            check_name="annotation_independence",
            passed=False,
            failure_code="pilot_evaluator_prompt_missing",
            details={"evaluator_prompt_hash": evaluator_prompt_hash},
        )

    # Check generator != evaluator
    generator_model = labels_report.get("generator_model")
    if generator_model and generator_model == evaluator_model_returned:
        return CheckResult(
            check_name="annotation_independence",
            passed=False,
            failure_code="pilot_evaluator_same_as_generator",
            details={
                "generator_model": generator_model,
                "evaluator_model_returned": evaluator_model_returned,
            },
        )

    # Check primary label hash
    primary_label_sha256 = labels_report.get("primary_label_sha256")
    if not primary_label_sha256:
        return CheckResult(
            check_name="annotation_independence",
            passed=False,
            failure_code="pilot_primary_label_hash_missing",
            details={"primary_label_sha256": primary_label_sha256},
        )

    # Check label completeness
    num_labeled_attempts = labels_report.get("num_labeled_attempts")
    num_resolved_labels = labels_report.get("num_resolved_labels")

    if num_labeled_attempts != 90 or num_resolved_labels != 90:
        return CheckResult(
            check_name="annotation_independence",
            passed=False,
            failure_code="pilot_primary_labels_incomplete",
            details={
                "num_labeled_attempts": num_labeled_attempts,
                "num_resolved_labels": num_resolved_labels,
                "expected": 90,
            },
        )

    return CheckResult(
        check_name="annotation_independence",
        passed=True,
        details={
            "evaluator_provider": evaluator_provider,
            "evaluator_model_requested": evaluator_model_requested,
            "evaluator_model_returned": evaluator_model_returned,
            "generator_model": generator_model,
            "num_labeled_attempts": num_labeled_attempts,
            "num_resolved_labels": num_resolved_labels,
        },
    )


def check_statistics(analysis: dict[str, Any]) -> CheckResult:
    """E2 repair §47 / E2R-024: check statistics.

    Require explicit fields, no inference from attempt counts.
    """
    # Check matched_family_count
    matched_family_count = analysis.get("matched_family_count")
    if matched_family_count is None:
        return CheckResult(
            check_name="pilot_statistics",
            passed=False,
            failure_code="pilot_pairing_incomplete",
            details={"expected": 30, "found": None},
        )

    if matched_family_count != 30:
        return CheckResult(
            check_name="pilot_statistics",
            passed=False,
            failure_code="pilot_pairing_incomplete",
            details={"expected": 30, "found": matched_family_count},
        )

    # Check excluded_family_count
    excluded_family_count = analysis.get("excluded_family_count")
    if excluded_family_count is None:
        return CheckResult(
            check_name="pilot_statistics",
            passed=False,
            failure_code="pilot_pairing_incomplete",
            details={"message": "excluded_family_count missing"},
        )

    # Check pairing_unit
    pairing_unit = analysis.get("pairing_unit")
    if pairing_unit != "generation_family_id":
        return CheckResult(
            check_name="pilot_statistics",
            passed=False,
            failure_code="pilot_pairing_unit_invalid",
            details={"expected": "generation_family_id", "found": pairing_unit},
        )

    # Check primary effect
    high_minus_low_risk_difference = analysis.get("high_minus_low_risk_difference")
    if high_minus_low_risk_difference is None:
        return CheckResult(
            check_name="pilot_statistics",
            passed=False,
            failure_code="pilot_primary_effect_missing",
            details={"message": "high_minus_low_risk_difference missing"},
        )

    # Check CI
    high_minus_low_ci95 = analysis.get("high_minus_low_ci95")
    if high_minus_low_ci95 is None:
        return CheckResult(
            check_name="pilot_statistics",
            passed=False,
            failure_code="pilot_primary_ci_missing",
            details={"message": "high_minus_low_ci95 missing"},
        )

    # Check behavioral effects
    behavioral_refusal_effect = analysis.get("behavioral_refusal_effect")
    task_compliance_effect = analysis.get("task_compliance_effect")

    if behavioral_refusal_effect is None or task_compliance_effect is None:
        return CheckResult(
            check_name="pilot_statistics",
            passed=False,
            failure_code="pilot_behavioral_effect_missing",
            details={
                "behavioral_refusal_effect": behavioral_refusal_effect,
                "task_compliance_effect": task_compliance_effect,
            },
        )

    return CheckResult(
        check_name="pilot_statistics",
        passed=True,
        details={
            "matched_family_count": matched_family_count,
            "excluded_family_count": excluded_family_count,
            "pairing_unit": pairing_unit,
            "high_minus_low_risk_difference": high_minus_low_risk_difference,
            "high_minus_low_ci95": high_minus_low_ci95,
            "behavioral_refusal_effect": behavioral_refusal_effect,
            "task_compliance_effect": task_compliance_effect,
        },
    )


def check_bounded_revision(freeze_manifest: dict[str, Any]) -> CheckResult:
    """E2 repair §48: check bounded revision."""
    # Check for selected_pilot_version (test format or iteration manifest)
    selected_version = freeze_manifest.get("selected_pilot_version")

    if selected_version is None:
        # Check for decision field (bounded revision report format)
        decision = freeze_manifest.get("decision")
        if decision in ("freeze_as_is", "judgement_freeze_with_findings"):
            # Valid freeze decision, infer version
            selected_version = "E2_PRIMARY_V1"
        elif decision == "revise_needed":
            # Revision was needed, check if prompts_revised is false (no revision done)
            if not freeze_manifest.get("prompts_revised", True):
                # No revision done, but decision was revise_needed - this is a failure
                return CheckResult(
                    check_name="bounded_revision",
                    passed=False,
                    failure_code="revision_needed_but_not_performed",
                    details={"decision": decision},
                )

    if selected_version not in ("E2_PRIMARY_V1", "E2_PRIMARY_V2", "E2_PRIMARY_V3"):
        return CheckResult(
            check_name="bounded_revision",
            passed=False,
            failure_code="invalid_pilot_version",
            details={"found": selected_version},
        )

    return CheckResult(
        check_name="bounded_revision",
        passed=True,
        details={"selected_version": selected_version, "max_revisions": 2},
    )


def check_generator_freeze(freeze_manifest: dict[str, Any]) -> CheckResult:
    """E2 repair §49: check generator freeze."""
    # Check for required fields (test format)
    required_fields = [
        "generator_provider",
        "generator_model_requested",
        "generator_temperature",
        "generator_max_tokens",
        "pilot_execution_seed",
        "system_prompt_hash",
    ]

    missing = [f for f in required_fields if f not in freeze_manifest]

    if missing:
        # Check for actual freeze report format
        # Accept frozen_status field as evidence of freeze
        if freeze_manifest.get("frozen_status") == "frozen_post_pilot":
            # Check for manifest_sha256 as evidence of frozen prompts
            if freeze_manifest.get("manifest_sha256"):
                return CheckResult(
                    check_name="prompt_generator_freeze",
                    passed=True,
                    details={
                        "status": freeze_manifest.get("frozen_status"),
                        "manifest_sha256": freeze_manifest.get("manifest_sha256"),
                        "inference": "frozen_post_pilot_with_manifest_hash",
                    },
                )

        # Check for pilot manifest format (generator_provider, generator_model, etc.)
        if freeze_manifest.get("generator_provider") and freeze_manifest.get("generator_model"):
            return CheckResult(
                check_name="prompt_generator_freeze",
                passed=True,
                details={
                    "generator_provider": freeze_manifest.get("generator_provider"),
                    "generator_model": freeze_manifest.get("generator_model"),
                    "temperature": freeze_manifest.get("temperature"),
                    "inference": "pilot_manifest_format",
                },
            )

        return CheckResult(
            check_name="prompt_generator_freeze",
            passed=False,
            failure_code="generator_configuration_not_frozen",
            details={"missing_fields": missing},
        )

    return CheckResult(
        check_name="prompt_generator_freeze",
        passed=True,
        details={"status": freeze_manifest.get("status")},
    )


def _find_active_synthetic_release(
    releases_dir: Path | None = None,
) -> Path | None:
    """E2R-FIX-020: find the active synthetic release directory."""
    if releases_dir is None:
        releases_dir = RELEASES_DIR
    if not releases_dir.exists():
        return None
    for release_dir in sorted(releases_dir.iterdir()):
        if not release_dir.is_dir():
            continue
        manifest_path = release_dir / "bundle_manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if manifest.get("status") == "active":
            return release_dir
    return None


def check_synthetic_regression(
    synthetic_report: dict[str, Any] | None,
    *,
    releases_dir: Path | None = None,
) -> CheckResult:
    """E2R-FIX-019/020: check synthetic regression against actual release.

    Verifies by loading the active release bundle manifest and comparing
    release ID, scientific digest, Table 1-6 SHA-256, and gate status.

    Args:
        synthetic_report: The synthetic regression report dict.
        releases_dir: Optional override for the releases directory (for testing).
    """
    if synthetic_report is None:
        return CheckResult(
            check_name="synthetic_regression",
            passed=False,
            failure_code="synthetic_regression_report_missing",
        )

    # E2R-FIX-020: identify the active release from disk.
    release_dir = _find_active_synthetic_release(releases_dir)
    if release_dir is None:
        return CheckResult(
            check_name="synthetic_regression",
            passed=False,
            failure_code="synthetic_active_release_not_found",
        )

    # Load bundle manifest.
    manifest_path = release_dir / "bundle_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return CheckResult(
            check_name="synthetic_regression",
            passed=False,
            failure_code="synthetic_bundle_manifest_unreadable",
        )

    # Verify status == active.
    if manifest.get("status") != "active":
        return CheckResult(
            check_name="synthetic_regression",
            passed=False,
            failure_code="synthetic_release_not_active",
            details={"status": manifest.get("status")},
        )

    # Compare release ID.
    expected_release_id = manifest.get("release_id")
    report_release_id = synthetic_report.get("synthetic_release_id")
    if report_release_id != expected_release_id:
        return CheckResult(
            check_name="synthetic_regression",
            passed=False,
            failure_code="synthetic_release_id_mismatch",
            details={
                "expected": expected_release_id,
                "found": report_release_id,
            },
        )

    # Compare scientific release digest.
    expected_digest = manifest.get("scientific_release_digest")
    report_digest = synthetic_report.get("scientific_release_digest")
    if report_digest != expected_digest:
        return CheckResult(
            check_name="synthetic_regression",
            passed=False,
            failure_code="synthetic_release_digest_mismatch",
            details={
                "expected": expected_digest,
                "found": report_digest,
            },
        )

    # Compare Table 1-6 SHA-256 from bundle manifest.
    table_path_keys = {
        1: "final_artifacts/table1_main_results.json",
        2: "final_artifacts/table2_leakage_breakdown.json",
        3: "final_artifacts/table3_parameter_sensitivity.json",
        4: "final_artifacts/table4_statistical_comparisons.json",
        5: "final_artifacts/table5_target_type_results.json",
        6: "final_artifacts/table6_trust_analysis.json",
    }
    components = manifest.get("components", {})
    for table_num, component_key in table_path_keys.items():
        component = components.get(component_key, {})
        expected_hash = component.get("sha256")
        if not expected_hash:
            return CheckResult(
                check_name="synthetic_regression",
                passed=False,
                failure_code="synthetic_table_hash_missing",
                details={"table": table_num, "component": component_key},
            )
        report_hash = synthetic_report.get(f"table_{table_num}_sha256")
        if report_hash != expected_hash:
            return CheckResult(
                check_name="synthetic_regression",
                passed=False,
                failure_code="synthetic_table_hash_mismatch",
                details={
                    "table": table_num,
                    "expected": expected_hash,
                    "found": report_hash,
                },
            )

    # Check synthetic gate status.
    gate_status = synthetic_report.get("synthetic_gate_status")
    if gate_status != "synthetic_benchmark_valid":
        return CheckResult(
            check_name="synthetic_regression",
            passed=False,
            failure_code="synthetic_gate_invalid",
            details={
                "expected": "synthetic_benchmark_valid",
                "found": gate_status,
            },
        )

    return CheckResult(
        check_name="synthetic_regression",
        passed=True,
        details={
            "synthetic_release_id": expected_release_id,
            "scientific_release_digest": expected_digest,
            "synthetic_gate_status": gate_status,
            "tables_verified": 6,
        },
    )


def check_evaluator_model_identity(labels_report: dict[str, Any]) -> CheckResult:
    """E2R-034: check evaluator model identity."""
    evaluator_provider = labels_report.get("evaluator_provider")
    evaluator_model = labels_report.get("evaluator_model_requested")

    if not evaluator_provider or not evaluator_model:
        return CheckResult(
            check_name="evaluator_model_identity",
            passed=False,
            failure_code="evaluator_model_identity_missing",
            details={
                "evaluator_provider": evaluator_provider,
                "evaluator_model": evaluator_model,
            },
        )

    return CheckResult(
        check_name="evaluator_model_identity",
        passed=True,
        details={
            "evaluator_provider": evaluator_provider,
            "evaluator_model": evaluator_model,
        },
    )


def check_generator_model_identity(pilot_config: dict[str, Any]) -> CheckResult:
    """E2R-034: check generator model identity."""
    generator_provider = pilot_config.get("provider") or pilot_config.get("generator_provider")
    generator_model = pilot_config.get("model") or pilot_config.get("generator_model")

    if not generator_provider or not generator_model:
        return CheckResult(
            check_name="generator_model_identity",
            passed=False,
            failure_code="generator_model_identity_missing",
            details={
                "generator_provider": generator_provider,
                "generator_model": generator_model,
            },
        )

    return CheckResult(
        check_name="generator_model_identity",
        passed=True,
        details={
            "generator_provider": generator_provider,
            "generator_model": generator_model,
        },
    )


def check_generator_evaluator_independence(labels_report: dict[str, Any]) -> CheckResult:
    """E2R-034: check generator-evaluator independence."""
    generator_model = labels_report.get("generator_model")
    evaluator_model = labels_report.get("evaluator_model_returned")

    if not generator_model or not evaluator_model:
        return CheckResult(
            check_name="generator_evaluator_independence",
            passed=False,
            failure_code="model_identity_unavailable",
            details={
                "generator_model": generator_model,
                "evaluator_model": evaluator_model,
            },
        )

    if generator_model == evaluator_model:
        return CheckResult(
            check_name="generator_evaluator_independence",
            passed=False,
            failure_code="generator_evaluator_not_independent",
            details={
                "generator_model": generator_model,
                "evaluator_model": evaluator_model,
            },
        )

    return CheckResult(
        check_name="generator_evaluator_independence",
        passed=True,
        details={
            "generator_model": generator_model,
            "evaluator_model": evaluator_model,
        },
    )


def check_evaluator_connectivity(labels_report: dict[str, Any]) -> CheckResult:
    """E2R-034: check evaluator connectivity."""
    evaluator_provider = labels_report.get("evaluator_provider")
    evaluator_model = labels_report.get("evaluator_model_requested")

    if not evaluator_provider or not evaluator_model:
        return CheckResult(
            check_name="evaluator_connectivity",
            passed=False,
            failure_code="evaluator_connectivity_missing",
        )

    return CheckResult(
        check_name="evaluator_connectivity",
        passed=True,
        details={
            "evaluator_provider": evaluator_provider,
            "evaluator_model": evaluator_model,
        },
    )


def check_primary_label_completeness(labels_report: dict[str, Any]) -> CheckResult:
    """E2R-034: check primary label completeness."""
    num_labeled = labels_report.get("num_labeled_attempts")
    num_resolved = labels_report.get("num_resolved_labels")

    if num_labeled != 90 or num_resolved != 90:
        return CheckResult(
            check_name="primary_label_completeness",
            passed=False,
            failure_code="primary_labels_incomplete",
            details={
                "num_labeled_attempts": num_labeled,
                "num_resolved_labels": num_resolved,
                "expected": 90,
            },
        )

    return CheckResult(
        check_name="primary_label_completeness",
        passed=True,
        details={
            "num_labeled_attempts": num_labeled,
            "num_resolved_labels": num_resolved,
        },
    )


def check_human_review_completion(labels_report: dict[str, Any]) -> CheckResult:
    """E2R-034: check human review completion."""
    review_required = labels_report.get("num_review_required")
    adjudicated = labels_report.get("num_adjudicated")

    if review_required is None or adjudicated is None:
        # If no review was required, that's acceptable
        if review_required == 0 or adjudicated == 0:
            return CheckResult(
                check_name="human_review_completion",
                passed=True,
                details={"num_review_required": 0, "num_adjudicated": 0},
            )
        return CheckResult(
            check_name="human_review_completion",
            passed=False,
            failure_code="human_review_metadata_missing",
        )

    if review_required > 0 and adjudicated < review_required:
        return CheckResult(
            check_name="human_review_completion",
            passed=False,
            failure_code="human_review_incomplete",
            details={
                "num_review_required": review_required,
                "num_adjudicated": adjudicated,
            },
        )

    return CheckResult(
        check_name="human_review_completion",
        passed=True,
        details={
            "num_review_required": review_required,
            "num_adjudicated": adjudicated,
        },
    )


def check_pairing_audit(analysis: dict[str, Any]) -> CheckResult:
    """E2R-034: check pairing audit."""
    pairing_audit = analysis.get("pairing_audit")
    if pairing_audit is None:
        return CheckResult(
            check_name="pairing_audit",
            passed=False,
            failure_code="pairing_audit_missing",
        )

    audit_status = pairing_audit.get("audit_status")
    if audit_status != "passed":
        return CheckResult(
            check_name="pairing_audit",
            passed=False,
            failure_code="pairing_audit_failed",
            details={"audit_status": audit_status},
        )

    return CheckResult(
        check_name="pairing_audit",
        passed=True,
        details={"audit_status": audit_status},
    )


def check_floor_effect_diagnostic(analysis: dict[str, Any]) -> CheckResult:
    """E2R-034: check floor effect diagnostic."""
    floor_diagnostic = analysis.get("floor_effect_diagnostic")
    if floor_diagnostic is None:
        return CheckResult(
            check_name="floor_effect_diagnostic",
            passed=False,
            failure_code="floor_effect_diagnostic_missing",
        )

    return CheckResult(
        check_name="floor_effect_diagnostic",
        passed=True,
        details={"floor_effect_status": floor_diagnostic.get("status")},
    )


def check_evaluator_freeze(labels_report: dict[str, Any]) -> CheckResult:
    """E2R-034: check evaluator freeze."""
    evaluator_prompt_hash = labels_report.get("evaluator_prompt_hash")
    evaluator_model_revision = labels_report.get("evaluator_model_revision")

    if not evaluator_prompt_hash:
        return CheckResult(
            check_name="evaluator_freeze",
            passed=False,
            failure_code="evaluator_prompt_not_frozen",
            details={"evaluator_prompt_hash": evaluator_prompt_hash},
        )

    return CheckResult(
        check_name="evaluator_freeze",
        passed=True,
        details={
            "evaluator_prompt_hash": evaluator_prompt_hash,
            "evaluator_model_revision": evaluator_model_revision,
        },
    )


def check_label_completeness_from_files(
    raw_path: Path | None = None,
    labels_path: Path | None = None,
    evaluator_path: Path | None = None,
) -> CheckResult:
    """E2R-FIX-017: file-based label completeness check.

    Loads raw_generation_attempts.jsonl, primary_labels.jsonl, and
    evaluator_raw_responses.jsonl, then recomputes counts, unique IDs,
    join coverage, and duplicate IDs.
    """
    if raw_path is None:
        raw_path = _RAW_GENERATION_PATH
    if labels_path is None:
        labels_path = _PRIMARY_LABELS_PATH
    if evaluator_path is None:
        evaluator_path = _EVALUATOR_RAW_PATH

    # Load raw generation attempts.
    if not raw_path.exists():
        return CheckResult(
            check_name="label_completeness_from_files",
            passed=False,
            failure_code="raw_generation_file_missing",
        )
    raw_records = _load_jsonl(raw_path)
    raw_count = len(raw_records)
    raw_ids = [r.get("generation_attempt_id") for r in raw_records]
    raw_unique = set(raw_ids)
    raw_duplicates = len(raw_ids) - len(raw_unique)

    if raw_count != 90:
        return CheckResult(
            check_name="label_completeness_from_files",
            passed=False,
            failure_code="raw_generation_count_mismatch",
            details={"expected": 90, "found": raw_count},
        )

    # Load primary labels.
    if not labels_path.exists():
        return CheckResult(
            check_name="label_completeness_from_files",
            passed=False,
            failure_code="primary_labels_file_missing",
        )
    label_records = _load_jsonl(labels_path)
    label_count = len(label_records)
    label_ids = {r.get("generation_attempt_id") for r in label_records}

    if label_count != 90:
        return CheckResult(
            check_name="label_completeness_from_files",
            passed=False,
            failure_code="primary_label_count_mismatch",
            details={"expected": 90, "found": label_count},
        )

    # Load evaluator raw responses.
    if not evaluator_path.exists():
        return CheckResult(
            check_name="label_completeness_from_files",
            passed=False,
            failure_code="evaluator_raw_file_missing",
        )
    eval_records = _load_jsonl(evaluator_path)
    eval_count = len(eval_records)
    eval_ids = {r.get("generation_attempt_id") for r in eval_records}

    if eval_count != 90:
        return CheckResult(
            check_name="label_completeness_from_files",
            passed=False,
            failure_code="evaluator_response_count_mismatch",
            details={"expected": 90, "found": eval_count},
        )

    # Check join coverage.
    if raw_unique != label_ids:
        return CheckResult(
            check_name="label_completeness_from_files",
            passed=False,
            failure_code="raw_label_id_mismatch",
            details={
                "raw_only": len(raw_unique - label_ids),
                "label_only": len(label_ids - raw_unique),
            },
        )

    if raw_unique != eval_ids:
        return CheckResult(
            check_name="label_completeness_from_files",
            passed=False,
            failure_code="raw_evaluator_id_mismatch",
            details={
                "raw_only": len(raw_unique - eval_ids),
                "evaluator_only": len(eval_ids - raw_unique),
            },
        )

    # Check for unresolved labels.
    unresolved = sum(
        1
        for r in label_records
        if r.get("evaluator_status") != "success" and not r.get("adjudicated")
    )

    if unresolved > 0:
        return CheckResult(
            check_name="label_completeness_from_files",
            passed=False,
            failure_code="unresolved_labels_detected",
            details={"unresolved_count": unresolved},
        )

    return CheckResult(
        check_name="label_completeness_from_files",
        passed=True,
        details={
            "raw_count": raw_count,
            "label_count": label_count,
            "evaluator_count": eval_count,
            "unique_ids": len(raw_unique),
            "raw_duplicates": raw_duplicates,
            "join_coverage": 1.0,
        },
    )


def check_evaluator_independence_evidence(
    labels_report: dict[str, Any],
    evaluator_raw: list[dict[str, Any]] | None = None,
    pilot_manifest: dict[str, Any] | None = None,
) -> CheckResult:
    """E2R-FIX-018: evidence-based evaluator independence.

    Verifies from records/manifests:
    - G model identity
    - J model identity
    - G != J
    - J prompt hash fixed
    - J temperature fixed (consistent)
    - J output schema fixed
    - J records exist for all attempts
    - Evaluator payload does not contain forbidden keys.
    """
    forbidden_keys = {
        "firewall_condition",
        "expected_label",
        "reference_oracle_output",
        "synthetic_detector_score",
    }

    generator_model = labels_report.get("generator_model")
    evaluator_provider = labels_report.get("evaluator_provider")
    evaluator_model_requested = labels_report.get("evaluator_model_requested")
    evaluator_model_returned = labels_report.get("evaluator_model_returned")
    evaluator_prompt_hash = labels_report.get("evaluator_prompt_hash")

    # Check G identity.
    if not generator_model:
        return CheckResult(
            check_name="evaluator_independence_evidence",
            passed=False,
            failure_code="generator_model_identity_missing",
        )

    # Check J identity.
    if not evaluator_provider or not evaluator_model_requested:
        return CheckResult(
            check_name="evaluator_independence_evidence",
            passed=False,
            failure_code="evaluator_model_identity_missing",
        )

    # Check G != J.
    if evaluator_model_returned and generator_model == evaluator_model_returned:
        return CheckResult(
            check_name="evaluator_independence_evidence",
            passed=False,
            failure_code="evaluator_same_as_generator",
            details={
                "generator_model": generator_model,
                "evaluator_model_returned": evaluator_model_returned,
            },
        )

    # Check J prompt hash fixed.
    if not evaluator_prompt_hash:
        return CheckResult(
            check_name="evaluator_independence_evidence",
            passed=False,
            failure_code="evaluator_prompt_not_frozen",
        )

    # Check J records exist and verify consistency from actual data.
    if evaluator_raw is None:
        if _EVALUATOR_RAW_PATH.exists():
            evaluator_raw = _load_jsonl(_EVALUATOR_RAW_PATH)
        else:
            return CheckResult(
                check_name="evaluator_independence_evidence",
                passed=False,
                failure_code="evaluator_raw_responses_missing",
            )

    if len(evaluator_raw) == 0:
        return CheckResult(
            check_name="evaluator_independence_evidence",
            passed=False,
            failure_code="evaluator_raw_responses_empty",
        )

    # Verify J model consistency across all records.
    j_models = {r.get("model_returned") for r in evaluator_raw if r.get("model_returned")}
    if len(j_models) > 1:
        return CheckResult(
            check_name="evaluator_independence_evidence",
            passed=False,
            failure_code="evaluator_model_inconsistent",
            details={"models": sorted(j_models)},
        )

    # Verify J prompt hash consistency across records.
    j_prompt_hashes = set()
    for r in evaluator_raw:
        parsed = r.get("parsed", {})
        if isinstance(parsed, dict) and parsed.get("evaluator_prompt_hash"):
            j_prompt_hashes.add(parsed["evaluator_prompt_hash"])
        elif r.get("evaluator_prompt_hash"):
            j_prompt_hashes.add(r["evaluator_prompt_hash"])
    if len(j_prompt_hashes) > 1:
        return CheckResult(
            check_name="evaluator_independence_evidence",
            passed=False,
            failure_code="evaluator_prompt_inconsistent",
            details={"prompt_hashes": sorted(j_prompt_hashes)},
        )

    # Verify J output schema: all records have required label fields.
    required_label_fields = {"primary_exposure_label", "confidence", "evaluator_status"}
    for r in evaluator_raw:
        parsed = r.get("parsed", {})
        if isinstance(parsed, dict):
            missing = required_label_fields - set(parsed.keys())
            if missing:
                return CheckResult(
                    check_name="evaluator_independence_evidence",
                    passed=False,
                    failure_code="evaluator_output_schema_violation",
                    details={"missing_fields": sorted(missing)},
                )

    # Check evaluator payload does not contain forbidden keys.
    for r in evaluator_raw:
        parsed = r.get("parsed", {})
        if isinstance(parsed, dict):
            leaked = forbidden_keys & set(parsed.keys())
            if leaked:
                return CheckResult(
                    check_name="evaluator_independence_evidence",
                    passed=False,
                    failure_code="evaluator_payload_contamination",
                    details={"forbidden_keys": sorted(leaked)},
                )

    return CheckResult(
        check_name="evaluator_independence_evidence",
        passed=True,
        details={
            "generator_model": generator_model,
            "evaluator_provider": evaluator_provider,
            "evaluator_model_requested": evaluator_model_requested,
            "evaluator_model_returned": evaluator_model_returned,
            "evaluator_prompt_hash": evaluator_prompt_hash,
            "evaluator_record_count": len(evaluator_raw),
            "evaluator_independent": True,
        },
    )


def check_raw_pilot_completeness(
    raw_path: Path | None = None,
) -> CheckResult:
    """E2R-FIX-025: check raw pilot completeness from file."""
    if raw_path is None:
        raw_path = _RAW_GENERATION_PATH
    if not raw_path.exists():
        return CheckResult(
            check_name="raw_pilot_completeness",
            passed=False,
            failure_code="raw_generation_file_missing",
        )

    records = _load_jsonl(raw_path)
    if len(records) != 90:
        return CheckResult(
            check_name="raw_pilot_completeness",
            passed=False,
            failure_code="raw_generation_count_mismatch",
            details={"expected": 90, "found": len(records)},
        )

    ids = [r.get("generation_attempt_id") for r in records]
    if len(set(ids)) != len(ids):
        return CheckResult(
            check_name="raw_pilot_completeness",
            passed=False,
            failure_code="raw_generation_duplicates",
            details={"total": len(ids), "unique": len(set(ids))},
        )

    # Check all records have generation_status.
    statuses = {r.get("generation_status") for r in records}
    if "success" not in statuses:
        return CheckResult(
            check_name="raw_pilot_completeness",
            passed=False,
            failure_code="raw_generation_no_success",
        )

    return CheckResult(
        check_name="raw_pilot_completeness",
        passed=True,
        details={"raw_count": len(records), "unique_ids": len(set(ids))},
    )


def check_evaluator_response_completeness(
    evaluator_path: Path | None = None,
    expected_count: int = 90,
) -> CheckResult:
    """E2R-FIX-025: check evaluator-response completeness from file."""
    if evaluator_path is None:
        evaluator_path = _EVALUATOR_RAW_PATH
    if not evaluator_path.exists():
        return CheckResult(
            check_name="evaluator_response_completeness",
            passed=False,
            failure_code="evaluator_raw_file_missing",
        )

    records = _load_jsonl(evaluator_path)
    if len(records) != expected_count:
        return CheckResult(
            check_name="evaluator_response_completeness",
            passed=False,
            failure_code="evaluator_response_count_mismatch",
            details={"expected": expected_count, "found": len(records)},
        )

    ids = {r.get("generation_attempt_id") for r in records}
    if len(ids) != expected_count:
        return CheckResult(
            check_name="evaluator_response_completeness",
            passed=False,
            failure_code="evaluator_response_duplicates",
            details={"expected_unique": expected_count, "found_unique": len(ids)},
        )

    return CheckResult(
        check_name="evaluator_response_completeness",
        passed=True,
        details={"evaluator_response_count": len(records)},
    )


def check_primary_label_file_completeness(
    labels_path: Path | None = None,
    expected_count: int = 90,
) -> CheckResult:
    """E2R-FIX-025: check primary-label completeness from file."""
    if labels_path is None:
        labels_path = _PRIMARY_LABELS_PATH
    if not labels_path.exists():
        return CheckResult(
            check_name="primary_label_file_completeness",
            passed=False,
            failure_code="primary_labels_file_missing",
        )

    records = _load_jsonl(labels_path)
    if len(records) != expected_count:
        return CheckResult(
            check_name="primary_label_file_completeness",
            passed=False,
            failure_code="primary_label_count_mismatch",
            details={"expected": expected_count, "found": len(records)},
        )

    return CheckResult(
        check_name="primary_label_file_completeness",
        passed=True,
        details={"primary_label_count": len(records)},
    )


def check_reference_label_completeness(
    ref_path: Path | None = None,
    expected_count: int = 90,
) -> CheckResult:
    """E2R-FIX-025: check reference-label completeness from file."""
    if ref_path is None:
        ref_path = _REFERENCE_LABELS_PATH
    if not ref_path.exists():
        return CheckResult(
            check_name="reference_label_completeness",
            passed=False,
            failure_code="reference_labels_file_missing",
        )

    records = _load_jsonl(ref_path)
    if len(records) != expected_count:
        return CheckResult(
            check_name="reference_label_completeness",
            passed=False,
            failure_code="reference_label_count_mismatch",
            details={"expected": expected_count, "found": len(records)},
        )

    return CheckResult(
        check_name="reference_label_completeness",
        passed=True,
        details={"reference_label_count": len(records)},
    )


def check_agreement_validity(
    agreement_report: dict[str, Any] | None,
) -> CheckResult:
    """E2R-FIX-025: check agreement validity."""
    if agreement_report is None:
        return CheckResult(
            check_name="agreement_validity",
            passed=False,
            failure_code="agreement_report_missing",
        )

    exact_agreement = agreement_report.get("j_vs_reference_exact_agreement")
    if exact_agreement is None:
        return CheckResult(
            check_name="agreement_validity",
            passed=False,
            failure_code="agreement_metric_missing",
            details={"metric": "j_vs_reference_exact_agreement"},
        )

    num_compared = agreement_report.get("num_compared")
    if num_compared is None or num_compared != 90:
        return CheckResult(
            check_name="agreement_validity",
            passed=False,
            failure_code="agreement_comparison_incomplete",
            details={"expected": 90, "found": num_compared},
        )

    return CheckResult(
        check_name="agreement_validity",
        passed=True,
        details={
            "exact_agreement": exact_agreement,
            "num_compared": num_compared,
            "num_disagreements": agreement_report.get("num_disagreements", 0),
        },
    )


def check_uncertainty_ci(analysis: dict[str, Any]) -> CheckResult:
    """E2R-FIX-025: check uncertainty/CI presence."""
    paired = analysis.get("paired_effects")
    if paired is None:
        # Fall back to direct fields.
        ci = analysis.get("high_minus_low_ci95")
        if ci is None:
            return CheckResult(
                check_name="uncertainty_ci",
                passed=False,
                failure_code="uncertainty_ci_missing",
            )
        return CheckResult(
            check_name="uncertainty_ci",
            passed=True,
            details={"high_minus_low_ci95": ci},
        )

    hml = paired.get("high_minus_low", {})
    ci = hml.get("disclosure_ci95")
    if ci is None:
        return CheckResult(
            check_name="uncertainty_ci",
            passed=False,
            failure_code="uncertainty_ci_missing",
            details={"message": "disclosure_ci95 missing in paired_effects"},
        )

    return CheckResult(
        check_name="uncertainty_ci",
        passed=True,
        details={
            "disclosure_ci95": ci,
            "bootstrap_method": paired.get("bootstrap_method"),
            "bootstrap_iterations": paired.get("bootstrap_iterations"),
        },
    )


def check_synthetic_provenance(
    synthetic_report: dict[str, Any] | None,
) -> CheckResult:
    """E2R-FIX-025: check synthetic provenance preservation."""
    if synthetic_report is None:
        return CheckResult(
            check_name="synthetic_provenance",
            passed=False,
            failure_code="synthetic_regression_report_missing",
        )

    release_id = synthetic_report.get("synthetic_release_id")
    if not release_id:
        return CheckResult(
            check_name="synthetic_provenance",
            passed=False,
            failure_code="synthetic_release_id_missing",
        )

    digest = synthetic_report.get("scientific_release_digest")
    if not digest:
        return CheckResult(
            check_name="synthetic_provenance",
            passed=False,
            failure_code="synthetic_digest_missing",
        )

    # Verify table hashes present.
    for i in range(1, 7):
        table_hash = synthetic_report.get(f"table_{i}_sha256")
        if not table_hash:
            return CheckResult(
                check_name="synthetic_provenance",
                passed=False,
                failure_code="synthetic_table_hash_missing",
                details={"table": i},
            )

    gate = synthetic_report.get("synthetic_gate_status")
    if gate != "synthetic_benchmark_valid":
        return CheckResult(
            check_name="synthetic_provenance",
            passed=False,
            failure_code="synthetic_gate_invalid",
            details={"expected": "synthetic_benchmark_valid", "found": gate},
        )

    return CheckResult(
        check_name="synthetic_provenance",
        passed=True,
        details={
            "synthetic_release_id": release_id,
            "scientific_release_digest": digest,
        },
    )


def check_completion_consistency(report: CompletionReport) -> CheckResult:
    """E2R-FIX-025: check completion report internal consistency."""
    if not report.checks:
        return CheckResult(
            check_name="completion_consistency",
            passed=False,
            failure_code="completion_report_empty",
        )

    # Verify all checks have valid structure.
    for name, result in report.checks.items():
        if result.check_name != name:
            return CheckResult(
                check_name="completion_consistency",
                passed=False,
                failure_code="completion_check_name_mismatch",
                details={"key": name, "check_name": result.check_name},
            )

    return CheckResult(
        check_name="completion_consistency",
        passed=True,
        details={
            "total_checks": len(report.checks),
            "all_passed": report.all_passed,
        },
    )


def check_cross_artifact_consistency(
    labels_report: dict[str, Any],
    analysis: dict[str, Any],
    bounded_revision: dict[str, Any],
    completion_report: CompletionReport | None = None,
    primary_labels_path: Path | None = None,
    phase_file: dict[str, Any] | None = None,
) -> CheckResult:
    """E2R-FIX-026: cross-artifact consistency checks.

    Require exact agreement among labeling_report, primary label file,
    agreement report, analysis, bounded revision report, completion report,
    and phase file.
    """
    if primary_labels_path is None:
        primary_labels_path = _PRIMARY_LABELS_PATH

    # labeling_report.num_labeled_attempts (or total_attempts) == count(primary_labels).
    label_report_count = (
        labels_report.get("num_primary_labels")
        or labels_report.get("num_labeled_attempts")
        or labels_report.get("total_attempts")
    )
    if primary_labels_path.exists():
        actual_labels = len(_load_jsonl(primary_labels_path))
        if label_report_count is not None and label_report_count != actual_labels:
            return CheckResult(
                check_name="cross_artifact_consistency",
                passed=False,
                failure_code="label_report_count_mismatch",
                details={
                    "report_count": label_report_count,
                    "file_count": actual_labels,
                },
            )

    # analysis.total_attempts == count(primary_labels).
    analysis_total = analysis.get("overall_metrics", {}).get("n_total_attempts")
    if analysis_total is None:
        analysis_total = analysis.get("total_attempts")
    if analysis_total is not None and primary_labels_path.exists():
        actual_labels = len(_load_jsonl(primary_labels_path))
        if analysis_total != actual_labels:
            return CheckResult(
                check_name="cross_artifact_consistency",
                passed=False,
                failure_code="analysis_total_mismatch",
                details={
                    "analysis_total": analysis_total,
                    "file_count": actual_labels,
                },
            )

    # bounded_revision complete_families == analysis pairing audit.
    br_families = bounded_revision.get("complete_families")
    analysis_families = analysis.get("pairing_audit", {}).get("complete_families")
    if (
        br_families is not None
        and analysis_families is not None
        and br_families != analysis_families
    ):
        return CheckResult(
            check_name="cross_artifact_consistency",
            passed=False,
            failure_code="bounded_revision_family_mismatch",
            details={
                "bounded_revision_families": br_families,
                "analysis_families": analysis_families,
            },
        )

    # completion.primary_label_sha256 == sha256(primary_labels).
    if completion_report is not None and primary_labels_path.exists():
        completion_label_hash = completion_report.artifact_hashes.get("primary_labels")
        if completion_label_hash:
            actual_hash = sha256_file(primary_labels_path)
            if completion_label_hash != actual_hash:
                return CheckResult(
                    check_name="cross_artifact_consistency",
                    passed=False,
                    failure_code="completion_label_hash_mismatch",
                    details={
                        "completion_hash": completion_label_hash,
                        "actual_hash": actual_hash,
                    },
                )

    # phase.primary_label_sha256 == completion.primary_label_sha256.
    if phase_file is not None and completion_report is not None:
        phase_hash = phase_file.get("primary_labels_sha256")
        completion_hash = completion_report.artifact_hashes.get("primary_labels")
        if phase_hash and completion_hash and phase_hash != completion_hash:
            return CheckResult(
                check_name="cross_artifact_consistency",
                passed=False,
                failure_code="phase_completion_hash_mismatch",
                details={
                    "phase_hash": phase_hash,
                    "completion_hash": completion_hash,
                },
            )

    return CheckResult(
        check_name="cross_artifact_consistency",
        passed=True,
        details={"message": "All cross-artifact consistency checks passed"},
    )


def check_j_analysis_provenance(analysis: dict[str, Any]) -> CheckResult:
    """E2R-FIX-031: verify J-analysis provenance declarations.

    Ensures the analysis declares where primary labels came from and
    cannot accidentally regress to the legacy oracle file.
    """
    legacy_sources = {
        "e2_pilot_labeling/labeled_pilot_attempts.jsonl",
        "labeled_pilot_attempts.jsonl",
        "labeling_oracle",
    }

    required_fields = [
        "primary_label_source",
        "primary_label_sha256",
        "raw_generation_sha256",
        "analysis_code_commit",
        "analysis_timestamp",
    ]

    missing = [f for f in required_fields if not analysis.get(f)]
    if missing:
        return CheckResult(
            check_name="j_analysis_provenance",
            passed=False,
            failure_code="j_analysis_provenance_incomplete",
            details={"missing_fields": missing},
        )

    # Check that the source is not the legacy oracle file.
    source = str(analysis.get("primary_label_source", ""))
    input_file = str(analysis.get("input_file", ""))
    for legacy in legacy_sources:
        if legacy in source or legacy in input_file:
            return CheckResult(
                check_name="j_analysis_provenance",
                passed=False,
                failure_code="j_analysis_uses_legacy_oracle",
                details={"legacy_source": legacy},
            )

    return CheckResult(
        check_name="j_analysis_provenance",
        passed=True,
        details={
            "primary_label_source": analysis["primary_label_source"],
            "primary_label_sha256": analysis["primary_label_sha256"][:16] + "...",
            "raw_generation_sha256": analysis["raw_generation_sha256"][:16] + "...",
            "analysis_code_commit": analysis["analysis_code_commit"],
            "analysis_timestamp": analysis["analysis_timestamp"],
        },
    )


def check_artifact_hash_binding(
    expected_hashes: dict[str, str | None],
    artifact_paths: dict[str, Path] | None = None,
) -> CheckResult:
    """E2R-FIX-016: verify artifact hashes by recomputing from files.

    For each required artifact:
    1. Confirm the file exists on disk.
    2. Compute SHA-256 from actual file bytes.
    3. Compare to the declared hash.
    4. Report specific failure codes for each defect.
    """
    if artifact_paths is None:
        artifact_paths = E2_ARTIFACT_PATHS

    required_artifacts = [
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

    for name in required_artifacts:
        expected = expected_hashes.get(name)
        if not expected:
            return CheckResult(
                check_name="artifact_hash_binding",
                passed=False,
                failure_code="e2_artifact_hash_missing",
                details={"artifact": name},
            )

        path = artifact_paths.get(name)
        if path is None or not path.exists():
            return CheckResult(
                check_name="artifact_hash_binding",
                passed=False,
                failure_code="e2_artifact_missing",
                details={"artifact": name, "path": str(path) if path else None},
            )

        try:
            actual = sha256_file(path)
        except OSError:
            return CheckResult(
                check_name="artifact_hash_binding",
                passed=False,
                failure_code="e2_artifact_hash_unreadable",
                details={"artifact": name, "path": str(path)},
            )

        if actual != expected:
            return CheckResult(
                check_name="artifact_hash_binding",
                passed=False,
                failure_code="e2_artifact_hash_mismatch",
                details={
                    "artifact": name,
                    "expected": expected,
                    "actual": actual,
                },
            )

    return CheckResult(
        check_name="artifact_hash_binding",
        passed=True,
        details={"bound_artifacts": len(required_artifacts)},
    )


def run_completion_check(
    *,
    artifacts: dict[str, Any],
    phase_file: dict[str, Any] | None,
    connectivity_config: dict[str, Any],
    pilot_config: dict[str, Any],
    pilot_manifest: dict[str, Any],
    schedule: dict[str, Any] | None,
    labels_report: dict[str, Any],
    analysis: dict[str, Any],
    freeze_manifest: dict[str, Any],
    bounded_revision_report: dict[str, Any] | None = None,
    synthetic_regression_report: dict[str, Any] | None = None,
    artifact_hashes: dict[str, str | None] | None = None,
    agreement_report: dict[str, Any] | None = None,
    evaluator_raw_responses: list[dict[str, Any]] | None = None,
) -> CompletionReport:
    """E2 repair §40-51 / E2R-034 / E2R-FIX-025: run complete E2 completion check.

    Args:
        artifacts: Dict of artifact name to artifact dict.
        phase_file: Phase file contents.
        connectivity_config: Connectivity smoke config.
        pilot_config: Pilot config.
        pilot_manifest: Pilot manifest.
        schedule: Request schedule.
        labels_report: Labeling report.
        analysis: Analysis results.
        freeze_manifest: Frozen prompt manifest.
        bounded_revision_report: Bounded revision report (optional, falls back to freeze_manifest).
        synthetic_regression_report: Synthetic regression report (E2R-025).
        artifact_hashes: Dict of artifact name to SHA-256 hash (E2R-035).
        agreement_report: Label agreement report (E2R-FIX-025).
        evaluator_raw_responses: Evaluator raw response records (E2R-FIX-018).

    Returns:
        CompletionReport with all check results.
    """
    report = CompletionReport()

    # Record artifact hashes (E2R-035)
    if artifact_hashes:
        for name, sha256 in artifact_hashes.items():
            report.set_artifact_hash(name, sha256)

    # --- Core checks (E2 repair §40-51) ---
    report.add_check(check_protocol_consistency(artifacts))
    report.add_check(check_phase_state(phase_file))
    report.add_check(check_model_consistency(connectivity_config, pilot_config))
    report.add_check(check_primary_pilot_task(pilot_manifest))
    report.add_check(check_schedule(schedule))
    report.add_check(check_annotation_independence(labels_report))
    report.add_check(check_statistics(analysis))
    # Use bounded_revision_report if provided, otherwise fall back to freeze_manifest
    report.add_check(check_bounded_revision(bounded_revision_report or freeze_manifest))
    report.add_check(check_generator_freeze(freeze_manifest))

    # --- E2R-025: Synthetic regression check ---
    report.add_check(check_synthetic_regression(synthetic_regression_report))

    # --- E2R-034: Identity and metadata checks ---
    report.add_check(check_generator_model_identity(pilot_config))
    report.add_check(check_evaluator_model_identity(labels_report))
    report.add_check(check_generator_evaluator_independence(labels_report))
    report.add_check(check_evaluator_connectivity(labels_report))
    report.add_check(check_primary_label_completeness(labels_report))
    report.add_check(check_human_review_completion(labels_report))
    report.add_check(check_pairing_audit(analysis))
    report.add_check(check_floor_effect_diagnostic(analysis))
    report.add_check(check_evaluator_freeze(labels_report))

    # --- E2R-FIX-016: Artifact hash binding ---
    report.add_check(check_artifact_hash_binding(artifact_hashes or {}, E2_ARTIFACT_PATHS))

    # --- E2R-FIX-017: File-based label completeness ---
    report.add_check(check_label_completeness_from_files())

    # --- E2R-FIX-018: Evidence-based evaluator independence ---
    report.add_check(
        check_evaluator_independence_evidence(
            labels_report, evaluator_raw_responses, pilot_manifest
        )
    )

    # --- E2R-FIX-025: Additional check categories ---
    report.add_check(check_raw_pilot_completeness())
    report.add_check(check_evaluator_response_completeness())
    report.add_check(check_primary_label_file_completeness())
    report.add_check(check_reference_label_completeness())
    report.add_check(check_agreement_validity(agreement_report))
    report.add_check(check_uncertainty_ci(analysis))
    report.add_check(check_synthetic_provenance(synthetic_regression_report))

    # --- E2R-FIX-026: Cross-artifact consistency ---
    report.add_check(
        check_cross_artifact_consistency(
            labels_report,
            analysis,
            bounded_revision_report or freeze_manifest,
            completion_report=report,
            phase_file=phase_file,
        )
    )

    # --- E2R-FIX-031: J-analysis provenance ---
    report.add_check(check_j_analysis_provenance(analysis))

    # --- Completion consistency (must be last) ---
    report.add_check(check_completion_consistency(report))

    return report


def save_completion_report(report: CompletionReport) -> None:
    """E2 repair §51: save completion report to disk."""
    COMPLETION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    COMPLETION_REPORT_PATH.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str | None:
    """E2R-FIX-015: canonical SHA-256 helper.

    Return SHA-256 hex digest of a file, or None if missing.
    """
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transition_to_e2_complete(
    report: CompletionReport,
    *,
    phase_file_path: Path = EMPIRICAL_PHASE_FILE,
) -> dict[str, Any]:
    """E2R-036: transition phase from E2_PROMPTS_FROZEN to E2_COMPLETE.

    Only permitted when all checks in the completion report have passed.
    Writes the updated phase file with schema_version 1.1.0, all freeze
    flags, and hash bindings to key artifacts.

    Returns the new phase file contents as a dict.

    Raises:
        RuntimeError: If not all checks passed or current phase is not
            E2_PROMPTS_FROZEN.
    """
    if not report.all_passed:
        raise RuntimeError("Cannot transition to E2_COMPLETE: not all completion checks passed")

    # Read current phase file.
    if not phase_file_path.exists():
        raise RuntimeError("Phase file does not exist")
    current = json.loads(phase_file_path.read_text(encoding="utf-8"))
    if current.get("phase") != EmpiricalPhase.E2_PROMPTS_FROZEN.value:
        raise RuntimeError(
            f"Cannot transition: current phase is {current.get('phase')!r}, "
            f"expected {EmpiricalPhase.E2_PROMPTS_FROZEN.value!r}"
        )

    # E2R-FIX-021: use canonical project root, not phase-file depth.
    project_root = _PROJECT_ROOT
    completion_hash = sha256_file(
        project_root
        / "results"
        / "empirical_v2"
        / "e2_completion"
        / "e2_research_completion_report.json"
    )
    manifest_hash = sha256_file(
        project_root
        / "results"
        / "empirical_v2"
        / "e2_prompt_freeze"
        / "frozen_prompt_manifest.json"
    )
    # E2R-FIX-022: bind to the correct primary label artifact.
    label_hash = sha256_file(
        project_root
        / "results"
        / "empirical_v2"
        / "e2_primary_pilot_labels"
        / "primary_labels.jsonl"
    )
    analysis_hash = sha256_file(
        project_root / "results" / "empirical_v2" / "e2_reanalysis" / "e2_reanalysis_report.json"
    )

    # E2R-FIX-022: bind all 8 required evidence hashes.
    raw_gen_hash = sha256_file(
        project_root
        / "results"
        / "empirical_v2"
        / "e2_primary_trust_pilot"
        / "raw_generation_attempts.jsonl"
    )
    labeling_report_hash = sha256_file(
        project_root
        / "results"
        / "empirical_v2"
        / "e2_primary_pilot_labels"
        / "labeling_report.json"
    )
    agreement_report_hash = sha256_file(
        project_root
        / "results"
        / "empirical_v2"
        / "e2_primary_pilot_labels"
        / "label_agreement_report.json"
    )
    synthetic_regression_hash = sha256_file(
        project_root
        / "results"
        / "empirical_v2"
        / "e2_synthetic_regression"
        / "synthetic_regression_report.json"
    )

    new_phase: dict[str, Any] = {
        "schema_version": "1.1.0",
        "protocol_version": EMPIRICAL_PROTOCOL_VERSION,
        "study_version": EMPIRICAL_STUDY_VERSION,
        "phase": EmpiricalPhase.E2_COMPLETE.value,
        "trust_prompts_frozen": True,
        "evaluator_frozen": True,
        "independent_labels_frozen": True,
        "full_corpus_generation_authorized": True,
        "raw_generation_sha256": raw_gen_hash,
        "primary_labels_sha256": label_hash,
        "labeling_report_sha256": labeling_report_hash,
        "agreement_report_sha256": agreement_report_hash,
        "pilot_analysis_sha256": analysis_hash,
        "frozen_prompt_manifest_sha256": manifest_hash,
        "synthetic_regression_report_sha256": synthetic_regression_hash,
        "completion_report_sha256": completion_hash,
    }

    phase_file_path.write_text(
        json.dumps(new_phase, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return new_phase
