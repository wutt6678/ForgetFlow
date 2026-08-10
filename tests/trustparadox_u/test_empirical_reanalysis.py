"""Tests for empirical_reanalysis.py (E2R-013/014/015/016/032)."""

from __future__ import annotations

from experiments.trustparadox_u.empirical_corpus import (
    EmpiricalGenerationAttempt,
    TrustLevel,
)
from experiments.trustparadox_u.empirical_reanalysis import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DECISION_FREEZE_AS_IS,
    DECISION_FREEZE_WITH_LIMITATION,
    DECISION_REVISE_AND_RERUN,
    FLOOR_MANIPULATION_INFORMATIVE,
    FLOOR_MANIPULATION_PARTIALLY_INFORMATIVE,
    FLOOR_MANIPULATION_UNINFORMATIVE,
    MatchedFamilyIndependent,
    PairingAudit,
    audit_pairing,
    build_matched_families,
    compute_overall_metrics,
    compute_paired_effects,
    compute_scenario_trust_metrics,
    compute_trust_level_metrics,
    diagnose_floor_effect,
    generate_bounded_revision_report,
    run_reanalysis,
)
from experiments.trustparadox_u.empirical_relabeling import IndependentPrimaryLabel


def _make_attempt(
    attempt_id: str,
    family_id: str,
    scenario_id: str,
    trust_level: str,
    candidate_text: str = "test response",
    generation_status: str = "success",
) -> EmpiricalGenerationAttempt:
    """Helper to create a test attempt."""
    return EmpiricalGenerationAttempt(
        generation_attempt_id=attempt_id,
        candidate_family_id=family_id,
        scenario_id=scenario_id,
        secret_variant_id="sv_001",
        split="development",
        trust_level=trust_level,
        attack_type="trust_discretion_task",
        sample_index=0,
        generation_replicate=0,
        sender_id="sender_001",
        recipient_id="recipient_001",
        sequence_family_id=None,
        sequence_id=None,
        sequence_step_index=None,
        sequence_step_count=None,
        candidate_text=candidate_text,
        generation_status=generation_status,
        refusal=False,
        malformed=False,
        off_topic=False,
        generator_provider="openai",
        generator_model="test-model",
        generator_revision=None,
        temperature=0.3,
        seed=None,
        system_prompt_hash="abc123",
        user_prompt_hash="def456",
        request_id=f"req_{attempt_id}",
        retry_index=0,
        generated_at="2026-01-01T00:00:00Z",
    )


def _make_label(
    attempt_id: str,
    primary_exposure_label: str = "none",
    behavioral_refusal: bool = False,
    task_compliance: bool = True,
    task_relevant: bool = True,
    evaluator_status: str = "success",
) -> IndependentPrimaryLabel:
    """Helper to create a test label."""
    return IndependentPrimaryLabel(
        generation_attempt_id=attempt_id,
        eligible=True,
        unauthorized_disclosure=primary_exposure_label != "none",
        exact_value_disclosure=primary_exposure_label == "exact_value_disclosure",
        semantic_disclosure=primary_exposure_label == "semantic_disclosure",
        alias_or_coreference_disclosure=False,
        behavioral_refusal=behavioral_refusal,
        task_compliance=task_compliance,
        task_relevant=task_relevant,
        question=False,
        negation=False,
        historical=False,
        conditional_or_modal=False,
        primary_exposure_label=primary_exposure_label,
        evaluator_provider="mock",
        evaluator_model_requested="qwen3.8-max",
        evaluator_model_returned="qwen3.8-max",
        evaluator_model_revision=None,
        evaluator_prompt_hash="user_hash",
        evaluator_request_id=f"req_{attempt_id}",
        confidence=0.95,
        review_required=False,
        adjudicated=False,
        adjudication_label=None,
        evaluator_status=evaluator_status,
    )


def _make_complete_family(
    family_id: str,
    scenario_id: str,
    low_disclosure: bool = False,
    default_disclosure: bool = False,
    high_disclosure: bool = False,
) -> tuple[list[EmpiricalGenerationAttempt], list[IndependentPrimaryLabel]]:
    """Helper to create a complete family with 3 trust levels."""
    attempts = [
        _make_attempt(f"{family_id}_low", family_id, scenario_id, "low"),
        _make_attempt(f"{family_id}_default", family_id, scenario_id, "default"),
        _make_attempt(f"{family_id}_high", family_id, scenario_id, "high"),
    ]
    labels = [
        _make_label(
            f"{family_id}_low",
            primary_exposure_label="exact_value_disclosure" if low_disclosure else "none",
        ),
        _make_label(
            f"{family_id}_default",
            primary_exposure_label="exact_value_disclosure" if default_disclosure else "none",
        ),
        _make_label(
            f"{family_id}_high",
            primary_exposure_label="exact_value_disclosure" if high_disclosure else "none",
        ),
    ]
    return attempts, labels


class TestAuditPairing:
    """Test E2R-014: pairing audit."""

    def test_complete_families(self) -> None:
        """Test audit with 30 complete families."""
        attempts: list[EmpiricalGenerationAttempt] = []
        labels: list[IndependentPrimaryLabel] = []
        for i in range(30):
            fam_attempts, fam_labels = _make_complete_family(f"family_{i:03d}", "credential_001")
            attempts.extend(fam_attempts)
            labels.extend(fam_labels)

        audit = audit_pairing(attempts, labels)
        assert audit.total_families == 30
        assert audit.complete_families == 30
        assert audit.excluded_families == 0
        assert audit.duplicate_families == 0
        assert len(audit.complete_family_ids) == 30

    def test_missing_trust_level(self) -> None:
        """Test audit with missing trust level."""
        attempts = [
            _make_attempt("fam1_low", "fam1", "credential_001", "low"),
            _make_attempt("fam1_high", "fam1", "credential_001", "high"),
        ]
        labels = [
            _make_label("fam1_low"),
            _make_label("fam1_high"),
        ]
        audit = audit_pairing(attempts, labels)
        assert audit.total_families == 1
        assert audit.complete_families == 0
        assert audit.excluded_families == 1
        assert audit.missing_default == 1

    def test_duplicate_trust_level(self) -> None:
        """Test audit with duplicate trust level."""
        attempts = [
            _make_attempt("fam1_low_1", "fam1", "credential_001", "low"),
            _make_attempt("fam1_low_2", "fam1", "credential_001", "low"),
            _make_attempt("fam1_default", "fam1", "credential_001", "default"),
            _make_attempt("fam1_high", "fam1", "credential_001", "high"),
        ]
        labels = [
            _make_label("fam1_low_1"),
            _make_label("fam1_low_2"),
            _make_label("fam1_default"),
            _make_label("fam1_high"),
        ]
        audit = audit_pairing(attempts, labels)
        assert audit.duplicate_families == 1
        assert audit.excluded_families == 1

    def test_scenario_mismatch(self) -> None:
        """Test audit with scenario mismatch within family."""
        attempts = [
            _make_attempt("fam1_low", "fam1", "credential_001", "low"),
            _make_attempt("fam1_default", "fam1", "private_attribute_001", "default"),
            _make_attempt("fam1_high", "fam1", "credential_001", "high"),
        ]
        labels = [
            _make_label("fam1_low"),
            _make_label("fam1_default"),
            _make_label("fam1_high"),
        ]
        audit = audit_pairing(attempts, labels)
        assert audit.content_mismatches == 1
        assert audit.excluded_families == 1

    def test_missing_label(self) -> None:
        """Test audit with missing label."""
        attempts = [
            _make_attempt("fam1_low", "fam1", "credential_001", "low"),
            _make_attempt("fam1_default", "fam1", "credential_001", "default"),
            _make_attempt("fam1_high", "fam1", "credential_001", "high"),
        ]
        labels = [
            _make_label("fam1_low"),
            _make_label("fam1_default"),
        ]
        audit = audit_pairing(attempts, labels)
        assert audit.complete_families == 0
        assert audit.excluded_families == 1


class TestBuildMatchedFamilies:
    """Test matched family construction."""

    def test_build_from_complete_audit(self) -> None:
        """Test building matched families from complete audit."""
        attempts: list[EmpiricalGenerationAttempt] = []
        labels: list[IndependentPrimaryLabel] = []
        for i in range(5):
            fam_attempts, fam_labels = _make_complete_family(f"family_{i:03d}", "credential_001")
            attempts.extend(fam_attempts)
            labels.extend(fam_labels)

        audit = audit_pairing(attempts, labels)
        families = build_matched_families(attempts, labels, audit)
        assert len(families) == 5
        assert all(isinstance(f, MatchedFamilyIndependent) for f in families)
        assert families[0].scenario_id == "credential_001"

    def test_exclude_incomplete_families(self) -> None:
        """Test that incomplete families are excluded."""
        attempts: list[EmpiricalGenerationAttempt] = []
        labels: list[IndependentPrimaryLabel] = []

        # Complete family
        fam_attempts, fam_labels = _make_complete_family("fam1", "credential_001")
        attempts.extend(fam_attempts)
        labels.extend(fam_labels)

        # Incomplete family (missing high)
        attempts.append(_make_attempt("fam2_low", "fam2", "credential_001", "low"))
        labels.append(_make_label("fam2_low"))

        audit = audit_pairing(attempts, labels)
        families = build_matched_families(attempts, labels, audit)
        assert len(families) == 1
        assert families[0].family_id == "fam1"


class TestComputeOverallMetrics:
    """Test E2R-013: overall metrics."""

    def test_basic_metrics(self) -> None:
        """Test basic overall metrics computation."""
        attempts = [
            _make_attempt("a1", "f1", "credential_001", "low"),
            _make_attempt("a2", "f2", "credential_001", "default"),
            _make_attempt("a3", "f3", "credential_001", "high"),
        ]
        labels = [
            _make_label("a1", primary_exposure_label="none"),
            _make_label("a2", primary_exposure_label="exact_value_disclosure"),
            _make_label("a3", primary_exposure_label="none", behavioral_refusal=True),
        ]
        metrics = compute_overall_metrics(attempts, labels)
        assert metrics["n_total_attempts"] == 3
        assert metrics["n_positive_disclosures"] == 1
        assert metrics["n_behavioral_refusals"] == 1
        assert metrics["n_task_compliant"] == 3
        assert metrics["n_task_relevant"] == 3

    def test_evaluator_failures(self) -> None:
        """Test metrics with evaluator failures."""
        attempts = [
            _make_attempt("a1", "f1", "credential_001", "low"),
            _make_attempt("a2", "f2", "credential_001", "default"),
        ]
        labels = [
            _make_label("a1", evaluator_status="success"),
            _make_label("a2", evaluator_status="provider_error"),
        ]
        metrics = compute_overall_metrics(attempts, labels)
        assert metrics["n_evaluator_failures"] == 1
        assert metrics["n_positive_disclosures"] == 0


class TestComputeTrustLevelMetrics:
    """Test E2R-013: trust-level metrics."""

    def test_per_trust_metrics(self) -> None:
        """Test per-trust-level metrics."""
        attempts = [
            _make_attempt("a1", "f1", "credential_001", "low"),
            _make_attempt("a2", "f2", "credential_001", "default"),
            _make_attempt("a3", "f3", "credential_001", "high"),
        ]
        labels = [
            _make_label("a1", primary_exposure_label="none"),
            _make_label("a2", primary_exposure_label="none"),
            _make_label("a3", primary_exposure_label="exact_value_disclosure"),
        ]
        metrics = compute_trust_level_metrics(attempts, labels)
        assert metrics["low"]["n"] == 1
        assert metrics["low"]["disclosure_rate"] == 0.0
        assert metrics["default"]["n"] == 1
        assert metrics["default"]["disclosure_rate"] == 0.0
        assert metrics["high"]["n"] == 1
        assert metrics["high"]["disclosure_rate"] == 1.0


class TestComputeScenarioTrustMetrics:
    """Test E2R-013: scenario × trust breakdown."""

    def test_scenario_trust_breakdown(self) -> None:
        """Test scenario × trust metrics."""
        attempts = [
            _make_attempt("a1", "f1", "credential_001", "low"),
            _make_attempt("a2", "f2", "credential_001", "high"),
            _make_attempt("a3", "f3", "private_attribute_001", "low"),
            _make_attempt("a4", "f4", "private_attribute_001", "high"),
        ]
        labels = [
            _make_label("a1", primary_exposure_label="none"),
            _make_label("a2", primary_exposure_label="exact_value_disclosure"),
            _make_label("a3", primary_exposure_label="none"),
            _make_label("a4", primary_exposure_label="none"),
        ]
        metrics = compute_scenario_trust_metrics(attempts, labels)
        assert "credential_001" in metrics
        assert "private_attribute_001" in metrics
        assert metrics["credential_001"]["low"]["disclosure_rate"] == 0.0
        assert metrics["credential_001"]["high"]["disclosure_rate"] == 1.0
        assert metrics["private_attribute_001"]["high"]["disclosure_rate"] == 0.0


class TestComputePairedEffects:
    """Test E2R-015: paired trust-effect statistics."""

    def test_paired_effects_with_disclosure(self) -> None:
        """Test paired effects with disclosure difference."""
        families = []
        for i in range(10):
            _, labels = _make_complete_family(
                f"fam_{i:03d}",
                "credential_001",
                low_disclosure=False,
                high_disclosure=True,
            )
            families.append(
                MatchedFamilyIndependent(
                    family_id=f"fam_{i:03d}",
                    scenario_id="credential_001",
                    low_label=labels[0],
                    default_label=labels[1],
                    high_label=labels[2],
                )
            )
        effects = compute_paired_effects(families)
        assert effects["n_families"] == 10
        assert effects["high_minus_low"]["disclosure_risk_difference"] == 1.0
        assert len(effects["high_minus_low"]["disclosure_ci95"]) == 2

    def test_paired_effects_no_disclosure(self) -> None:
        """Test paired effects with no disclosure."""
        families = []
        for i in range(10):
            _, labels = _make_complete_family(
                f"fam_{i:03d}",
                "credential_001",
                low_disclosure=False,
                high_disclosure=False,
            )
            families.append(
                MatchedFamilyIndependent(
                    family_id=f"fam_{i:03d}",
                    scenario_id="credential_001",
                    low_label=labels[0],
                    default_label=labels[1],
                    high_label=labels[2],
                )
            )
        effects = compute_paired_effects(families)
        assert effects["high_minus_low"]["disclosure_risk_difference"] == 0.0

    def test_bootstrap_reproducibility(self) -> None:
        """Test that bootstrap CIs are reproducible with frozen seed."""
        families = []
        for i in range(10):
            _, labels = _make_complete_family(
                f"fam_{i:03d}",
                "credential_001",
                low_disclosure=False,
                high_disclosure=True,
            )
            families.append(
                MatchedFamilyIndependent(
                    family_id=f"fam_{i:03d}",
                    scenario_id="credential_001",
                    low_label=labels[0],
                    default_label=labels[1],
                    high_label=labels[2],
                )
            )
        effects1 = compute_paired_effects(families)
        effects2 = compute_paired_effects(families)
        assert (
            effects1["high_minus_low"]["disclosure_ci95"]
            == effects2["high_minus_low"]["disclosure_ci95"]
        )

    def test_empty_families(self) -> None:
        """Test paired effects with no families."""
        effects = compute_paired_effects([])
        assert "error" in effects


class TestDiagnoseFloorEffect:
    """Test E2R-016: floor effect diagnosis."""

    def test_manipulation_informative(self) -> None:
        """Test informative manipulation (disclosure observed)."""
        labels = [
            _make_label("a1", primary_exposure_label="none"),
            _make_label("a2", primary_exposure_label="exact_value_disclosure"),
            _make_label("a3", primary_exposure_label="none"),
        ]
        trust_metrics = {
            "low": {"n": 1, "disclosure_rate": 0.0, "mean_response_length": 100.0},
            "default": {"n": 1, "disclosure_rate": 0.0, "mean_response_length": 100.0},
            "high": {"n": 1, "disclosure_rate": 1.0, "mean_response_length": 100.0},
        }
        scenario_trust_metrics = {
            "credential_001": {
                "low": {"disclosure_rate": 0.0},
                "default": {"disclosure_rate": 0.0},
                "high": {"disclosure_rate": 1.0},
            }
        }
        diagnostic = diagnose_floor_effect(labels, trust_metrics, scenario_trust_metrics)
        assert diagnostic["decision"] == FLOOR_MANIPULATION_INFORMATIVE
        assert diagnostic["overall_disclosure_rate"] > 0.0

    def test_manipulation_partially_informative(self) -> None:
        """Test partially informative manipulation (floor but refusal ceiling)."""
        labels = [
            _make_label("a1", primary_exposure_label="none", behavioral_refusal=True),
            _make_label("a2", primary_exposure_label="none", behavioral_refusal=True),
            _make_label("a3", primary_exposure_label="none", behavioral_refusal=True),
        ]
        trust_metrics = {
            "low": {"n": 1, "disclosure_rate": 0.0, "mean_response_length": 100.0},
            "default": {"n": 1, "disclosure_rate": 0.0, "mean_response_length": 100.0},
            "high": {"n": 1, "disclosure_rate": 0.0, "mean_response_length": 100.0},
        }
        scenario_trust_metrics = {
            "credential_001": {
                "low": {"disclosure_rate": 0.0},
                "default": {"disclosure_rate": 0.0},
                "high": {"disclosure_rate": 0.0},
            }
        }
        diagnostic = diagnose_floor_effect(labels, trust_metrics, scenario_trust_metrics)
        assert diagnostic["decision"] == FLOOR_MANIPULATION_PARTIALLY_INFORMATIVE
        assert diagnostic["refusal_ceiling"] is True

    def test_manipulation_uninformative(self) -> None:
        """Test uninformative manipulation (all floors)."""
        labels = [
            _make_label("a1", primary_exposure_label="none", task_compliance=False),
            _make_label("a2", primary_exposure_label="none", task_compliance=False),
            _make_label("a3", primary_exposure_label="none", task_compliance=False),
        ]
        trust_metrics = {
            "low": {"n": 1, "disclosure_rate": 0.0, "mean_response_length": 100.0},
            "default": {"n": 1, "disclosure_rate": 0.0, "mean_response_length": 100.0},
            "high": {"n": 1, "disclosure_rate": 0.0, "mean_response_length": 100.0},
        }
        scenario_trust_metrics = {
            "credential_001": {
                "low": {"disclosure_rate": 0.0},
                "default": {"disclosure_rate": 0.0},
                "high": {"disclosure_rate": 0.0},
            }
        }
        diagnostic = diagnose_floor_effect(labels, trust_metrics, scenario_trust_metrics)
        assert diagnostic["decision"] == FLOOR_MANIPULATION_UNINFORMATIVE
        assert diagnostic["task_compliance_floor"] is True


class TestGenerateBoundedRevisionReport:
    """Test E2R-032: bounded-revision report."""

    def test_freeze_as_is(self) -> None:
        """Test freeze_as_is decision (non-zero disclosure effect)."""
        paired_effects = {
            "high_minus_low": {
                "disclosure_risk_difference": 0.5,
                "refusal_risk_difference": 0.0,
                "task_compliance_risk_difference": 0.0,
            }
        }
        floor_diagnostic = {"decision": FLOOR_MANIPULATION_INFORMATIVE}
        audit = PairingAudit(
            total_families=30,
            complete_families=30,
            excluded_families=0,
            incomplete_families=0,
            duplicate_families=0,
            missing_low=0,
            missing_default=0,
            missing_high=0,
            content_mismatches=0,
            complete_family_ids=tuple([f"fam_{i:03d}" for i in range(30)]),
            excluded_family_ids=(),
            exclusion_reasons=(),
            trust_level_coverage=(),
        )
        report = generate_bounded_revision_report(
            paired_effects=paired_effects,
            floor_diagnostic=floor_diagnostic,
            pairing_audit=audit,
        )
        assert report["decision"] == DECISION_FREEZE_AS_IS
        assert report["decision_rule"] == "non_zero_disclosure_effect"

    def test_freeze_with_limitation(self) -> None:
        """Test freeze_with_manipulation_limitation decision."""
        paired_effects = {
            "high_minus_low": {
                "disclosure_risk_difference": 0.0,
                "refusal_risk_difference": 0.0,
                "task_compliance_risk_difference": 0.0,
            }
        }
        floor_diagnostic = {"decision": FLOOR_MANIPULATION_UNINFORMATIVE}
        audit = PairingAudit(
            total_families=30,
            complete_families=30,
            excluded_families=0,
            incomplete_families=0,
            duplicate_families=0,
            missing_low=0,
            missing_default=0,
            missing_high=0,
            content_mismatches=0,
            complete_family_ids=tuple([f"fam_{i:03d}" for i in range(30)]),
            excluded_family_ids=(),
            exclusion_reasons=(),
            trust_level_coverage=(),
        )
        report = generate_bounded_revision_report(
            paired_effects=paired_effects,
            floor_diagnostic=floor_diagnostic,
            pairing_audit=audit,
        )
        assert report["decision"] == DECISION_FREEZE_WITH_LIMITATION
        assert "zero_disclosure_effect" in report["decision_rule"]

    def test_revise_and_rerun(self) -> None:
        """Test revise_and_rerun decision (ambiguous with budget)."""
        paired_effects = {
            "high_minus_low": {
                "disclosure_risk_difference": 0.0,
                "refusal_risk_difference": 0.0,
                "task_compliance_risk_difference": 0.0,
            }
        }
        floor_diagnostic = {"decision": FLOOR_MANIPULATION_INFORMATIVE}
        audit = PairingAudit(
            total_families=30,
            complete_families=30,
            excluded_families=0,
            incomplete_families=0,
            duplicate_families=0,
            missing_low=0,
            missing_default=0,
            missing_high=0,
            content_mismatches=0,
            complete_family_ids=tuple([f"fam_{i:03d}" for i in range(30)]),
            excluded_family_ids=(),
            exclusion_reasons=(),
            trust_level_coverage=(),
        )
        report = generate_bounded_revision_report(
            paired_effects=paired_effects,
            floor_diagnostic=floor_diagnostic,
            pairing_audit=audit,
            revision_count=0,
            max_revision_budget=2,
        )
        assert report["decision"] == DECISION_REVISE_AND_RERUN
        assert report["remaining_revision_budget"] == 2


class TestRunReanalysis:
    """Test full reanalysis pipeline."""

    def test_full_pipeline(self, tmp_path) -> None:
        """Test full reanalysis pipeline with mock data."""
        attempts: list[EmpiricalGenerationAttempt] = []
        labels: list[IndependentPrimaryLabel] = []
        for i in range(30):
            scenario = ["credential_001", "private_attribute_001", "authorization_001"][i % 3]
            fam_attempts, fam_labels = _make_complete_family(f"family_{i:03d}", scenario)
            attempts.extend(fam_attempts)
            labels.extend(fam_labels)

        output_dir = tmp_path / "reanalysis"
        output_dir.mkdir()
        result = run_reanalysis(attempts, labels, output_dir=output_dir)

        assert "pairing_audit" in result
        assert "overall_metrics" in result
        assert "floor_effect_diagnostic" in result
        assert "bounded_revision_decision" in result
        assert result["pairing_audit"]["complete_families"] == 30
