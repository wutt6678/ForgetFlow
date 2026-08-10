"""Tests for empirical_revision_decision.py (E2R-017/018/019/020)."""

from __future__ import annotations

from experiments.trustparadox_u.empirical_revision_decision import (
    DECISION_FREEZE_AS_LIMITED,
    DECISION_FREEZE_V1,
    DECISION_REVISE_TO_V2,
    FREEZE_CRITERION_BEHAVIORAL_VARIATION,
    FREEZE_CRITERION_COMPLETE_FAMILIES,
    FREEZE_CRITERION_NO_PROTOCOL_VIOLATIONS,
    compute_manipulation_freeze_rule,
    compute_revision_decision,
    run_revision_decision,
)


class TestComputeRevisionDecision:
    """Test E2R-017: revision decision."""

    def test_freeze_v1_with_disclosure(self) -> None:
        """Test freeze V1 when disclosure effect is non-zero."""
        floor_diagnostic = {"decision": "manipulation_informative"}
        paired_effects = {
            "high_minus_low": {
                "disclosure_risk_difference": 0.5,
                "refusal_risk_difference": 0.0,
                "task_compliance_risk_difference": 0.0,
            }
        }
        pairing_audit = {"complete_families": 30}

        decision = compute_revision_decision(
            floor_diagnostic=floor_diagnostic,
            paired_effects=paired_effects,
            pairing_audit=pairing_audit,
        )

        assert decision.decision == DECISION_FREEZE_V1
        assert decision.decision_rule == "non_zero_disclosure_effect"
        assert decision.v1_immutable is True
        assert decision.v2_required is False
        assert decision.remaining_budget == 2

    def test_freeze_v1_with_behavioral_variation(self) -> None:
        """Test freeze V1 when behavioral variation observed."""
        floor_diagnostic = {"decision": "manipulation_partially_informative"}
        paired_effects = {
            "high_minus_low": {
                "disclosure_risk_difference": 0.0,
                "refusal_risk_difference": 0.3,
                "task_compliance_risk_difference": 0.0,
            }
        }
        pairing_audit = {"complete_families": 30}

        decision = compute_revision_decision(
            floor_diagnostic=floor_diagnostic,
            paired_effects=paired_effects,
            pairing_audit=pairing_audit,
        )

        assert decision.decision == DECISION_FREEZE_V1
        assert decision.decision_rule == "behavioral_variation_on_secondary_endpoints"
        assert decision.v2_required is False

    def test_revise_to_v2_with_budget(self) -> None:
        """Test revise to V2 when floor effect and budget remains."""
        floor_diagnostic = {"decision": "manipulation_uninformative_floor"}
        paired_effects = {
            "high_minus_low": {
                "disclosure_risk_difference": 0.0,
                "refusal_risk_difference": 0.0,
                "task_compliance_risk_difference": 0.0,
            }
        }
        pairing_audit = {"complete_families": 30}

        decision = compute_revision_decision(
            floor_diagnostic=floor_diagnostic,
            paired_effects=paired_effects,
            pairing_audit=pairing_audit,
            revision_count=0,
            max_revision_budget=2,
        )

        assert decision.decision == DECISION_REVISE_TO_V2
        assert decision.decision_rule == "floor_effect_with_remaining_budget"
        assert decision.v2_required is True
        assert decision.v2_directory == "results/empirical_v2/e2_primary_trust_pilot_v2"
        assert decision.remaining_budget == 2

    def test_freeze_as_limited_budget_exhausted(self) -> None:
        """Test freeze as limited when budget exhausted."""
        floor_diagnostic = {"decision": "manipulation_uninformative_floor"}
        paired_effects = {
            "high_minus_low": {
                "disclosure_risk_difference": 0.0,
                "refusal_risk_difference": 0.0,
                "task_compliance_risk_difference": 0.0,
            }
        }
        pairing_audit = {"complete_families": 30}

        decision = compute_revision_decision(
            floor_diagnostic=floor_diagnostic,
            paired_effects=paired_effects,
            pairing_audit=pairing_audit,
            revision_count=2,
            max_revision_budget=2,
        )

        assert decision.decision == DECISION_FREEZE_AS_LIMITED
        assert decision.decision_rule == "budget_exhausted_with_no_variation"
        assert decision.v2_required is False
        assert decision.remaining_budget == 0

    def test_allowed_and_forbidden_revisions(self) -> None:
        """Test allowed and forbidden revision lists."""
        floor_diagnostic = {"decision": "manipulation_informative"}
        paired_effects = {
            "high_minus_low": {
                "disclosure_risk_difference": 0.0,
                "refusal_risk_difference": 0.0,
                "task_compliance_risk_difference": 0.0,
            }
        }
        pairing_audit = {"complete_families": 30}

        decision = compute_revision_decision(
            floor_diagnostic=floor_diagnostic,
            paired_effects=paired_effects,
            pairing_audit=pairing_audit,
        )

        assert "trust_framing_text" in decision.allowed_revisions
        assert "canonical_target" in decision.forbidden_revisions
        assert "privacy_prohibition" in decision.forbidden_revisions


class TestComputeManipulationFreezeRule:
    """Test E2R-020: manipulation-freeze rule."""

    def test_freeze_with_full_manipulation(self) -> None:
        """Test freeze with full manipulation when all criteria met."""
        floor_diagnostic = {"decision": "manipulation_informative"}
        paired_effects = {
            "high_minus_low": {
                "disclosure_risk_difference": 0.5,
                "refusal_risk_difference": 0.0,
                "task_compliance_risk_difference": 0.0,
            }
        }
        pairing_audit = {"complete_families": 30}

        rule = compute_manipulation_freeze_rule(
            floor_diagnostic=floor_diagnostic,
            paired_effects=paired_effects,
            pairing_audit=pairing_audit,
            evaluator_independence_verified=True,
            protocol_violations=[],
        )

        assert rule.decision == "freeze_with_full_manipulation"
        assert rule.manipulation_sensitivity == "informative"
        assert rule.e3_authorization == "authorized"
        assert rule.freeze_criteria_met[FREEZE_CRITERION_COMPLETE_FAMILIES] is True
        assert rule.freeze_criteria_met[FREEZE_CRITERION_BEHAVIORAL_VARIATION] is True

    def test_freeze_with_partial_manipulation(self) -> None:
        """Test freeze with partial manipulation (behavioral only, no disclosure)."""
        floor_diagnostic = {"decision": "manipulation_partially_informative"}
        paired_effects = {
            "high_minus_low": {
                "disclosure_risk_difference": 0.0,
                "refusal_risk_difference": 0.3,
                "task_compliance_risk_difference": 0.0,
            }
        }
        pairing_audit = {"complete_families": 30}

        rule = compute_manipulation_freeze_rule(
            floor_diagnostic=floor_diagnostic,
            paired_effects=paired_effects,
            pairing_audit=pairing_audit,
            evaluator_independence_verified=True,
            protocol_violations=[],
        )

        # With behavioral variation but no disclosure, it's partial manipulation
        assert rule.decision == "freeze_with_partial_manipulation"
        assert rule.manipulation_sensitivity == "partial"
        assert rule.e3_authorization == "authorized_with_limitation"

    def test_freeze_as_manipulation_limited(self) -> None:
        """Test freeze as manipulation limited (no variation)."""
        floor_diagnostic = {"decision": "manipulation_uninformative_floor"}
        paired_effects = {
            "high_minus_low": {
                "disclosure_risk_difference": 0.0,
                "refusal_risk_difference": 0.0,
                "task_compliance_risk_difference": 0.0,
            }
        }
        pairing_audit = {"complete_families": 30}

        rule = compute_manipulation_freeze_rule(
            floor_diagnostic=floor_diagnostic,
            paired_effects=paired_effects,
            pairing_audit=pairing_audit,
            evaluator_independence_verified=True,
            protocol_violations=[],
        )

        assert rule.decision == "freeze_as_manipulation_limited"
        assert rule.manipulation_sensitivity == "limited"
        assert rule.e3_authorization == "authorized_with_strong_limitation"
        assert "RQ7 manipulation sensitivity is limited" in rule.rq7_implication

    def test_incomplete_families_blocks_freeze(self) -> None:
        """Test that incomplete families block full freeze."""
        floor_diagnostic = {"decision": "manipulation_informative"}
        paired_effects = {
            "high_minus_low": {
                "disclosure_risk_difference": 0.5,
                "refusal_risk_difference": 0.0,
                "task_compliance_risk_difference": 0.0,
            }
        }
        pairing_audit = {"complete_families": 25}

        rule = compute_manipulation_freeze_rule(
            floor_diagnostic=floor_diagnostic,
            paired_effects=paired_effects,
            pairing_audit=pairing_audit,
            evaluator_independence_verified=True,
            protocol_violations=[],
        )

        assert rule.freeze_criteria_met[FREEZE_CRITERION_COMPLETE_FAMILIES] is False

    def test_protocol_violations_block_freeze(self) -> None:
        """Test that protocol violations block full freeze."""
        floor_diagnostic = {"decision": "manipulation_informative"}
        paired_effects = {
            "high_minus_low": {
                "disclosure_risk_difference": 0.5,
                "refusal_risk_difference": 0.0,
                "task_compliance_risk_difference": 0.0,
            }
        }
        pairing_audit = {"complete_families": 30}

        rule = compute_manipulation_freeze_rule(
            floor_diagnostic=floor_diagnostic,
            paired_effects=paired_effects,
            pairing_audit=pairing_audit,
            evaluator_independence_verified=True,
            protocol_violations=["violation_1"],
        )

        assert rule.freeze_criteria_met[FREEZE_CRITERION_NO_PROTOCOL_VIOLATIONS] is False


class TestRunRevisionDecision:
    """Test full revision decision pipeline."""

    def test_full_pipeline(self, tmp_path) -> None:
        """Test full revision decision pipeline."""
        floor_diagnostic = {"decision": "manipulation_informative"}
        paired_effects = {
            "high_minus_low": {
                "disclosure_risk_difference": 0.5,
                "refusal_risk_difference": 0.0,
                "task_compliance_risk_difference": 0.0,
            }
        }
        pairing_audit = {"complete_families": 30}

        output_dir = tmp_path / "revision"
        output_dir.mkdir()

        result = run_revision_decision(
            floor_diagnostic=floor_diagnostic,
            paired_effects=paired_effects,
            pairing_audit=pairing_audit,
            output_dir=output_dir,
        )

        assert "revision_decision" in result
        assert "manipulation_freeze_rule" in result
        assert result["analysis_type"] == "e2_revision_decision"

        # Check artifacts were written
        assert (output_dir / "e2_revision_decision.json").exists()
        assert (output_dir / "e2_manipulation_freeze_rule.json").exists()
