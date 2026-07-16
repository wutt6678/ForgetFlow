"""Tests for experiment result auditor."""

from experiments.trustparadox_u.audit_results import (
    audit_episode_result,
    audit_metric_value,
    validate_for_aggregation,
)
from experiments.trustparadox_u.runner import EpisodeResult, TurnResult


def _valid_result(**overrides) -> EpisodeResult:
    """Create a minimal valid EpisodeResult for testing."""
    result = EpisodeResult(
        run_id="r1",
        episode_id="ep1",
        scenario_id="s1",
        trust_level="default",
        seed=42,
    )
    result.metadata = {
        "forbidden_strings": ["secret"],
        "secret_variant_id": "sv1",
        "attack_type": "direct",
        "config_hash": "a" * 64,
        "seed": 42,
    }
    for k, v in overrides.items():
        setattr(result, k, v)
    return result


class TestTurnRules:
    """Turn-level audit rules."""

    def test_block_implies_none_released(self) -> None:
        """block action implies released_text is None."""
        from marble.firewall.types import DetectorResult, FirewallDecision

        dr = DetectorResult(
            exact_score=1.0, entity_score=0.0, semantic_score=0.0,
            reconstruction_score=0.0, matched_forget_ids=[], evidence=[],
        )
        decision = FirewallDecision(
            action="block", released_text=None, detector_result=dr,
            reason_codes=("exact",), policy_version="v1", latency_ms=0.1,
        )
        turn = TurnResult(
            turn_id=0, phase="POST_FORGET_ATTACK",
            sender_id="SK", recipient_id="CK",
            candidate_text="secret", released_text=None, decision=decision,
        )
        findings = audit_episode_result(_valid_result(turns=[turn]))
        block_findings = [f for f in findings if f.code == "BLOCK_WITH_RELEASED_TEXT"]
        assert len(block_findings) == 0

    def test_block_with_released_text_flagged(self) -> None:
        """block action with released_text is flagged."""
        from marble.firewall.types import DetectorResult, FirewallDecision

        dr = DetectorResult(
            exact_score=1.0, entity_score=0.0, semantic_score=0.0,
            reconstruction_score=0.0, matched_forget_ids=[], evidence=[],
        )
        # Create a valid block decision, then override turn.released_text
        decision = FirewallDecision(
            action="block", released_text=None, detector_result=dr,
            reason_codes=("exact",), policy_version="v1", latency_ms=0.1,
        )
        turn = TurnResult(
            turn_id=0, phase="POST_FORGET_ATTACK",
            sender_id="SK", recipient_id="CK",
            candidate_text="secret", released_text="leaked", decision=decision,
        )
        findings = audit_episode_result(_valid_result(turns=[turn]))
        block_findings = [f for f in findings if f.code == "BLOCK_WITH_RELEASED_TEXT"]
        assert len(block_findings) == 1

    def test_exposure_requires_released_text(self) -> None:
        """target_exposed=True without released_text is flagged."""
        turn = TurnResult(
            turn_id=0, phase="POST_FORGET_ATTACK",
            sender_id="SK", recipient_id="CK",
            candidate_text="secret", released_text=None,
            target_exposed=True,
        )
        findings = audit_episode_result(_valid_result(turns=[turn]))
        exp_findings = [f for f in findings if f.code == "EXPOSED_WITHOUT_RELEASED_TEXT"]
        assert len(exp_findings) == 1

    def test_reconstruction_requires_attempt(self) -> None:
        """target_reconstructed without is_reconstruction_attempt is flagged."""
        turn = TurnResult(
            turn_id=0, phase="POST_FORGET_ATTACK",
            sender_id="SK", recipient_id="CK",
            candidate_text="parts", released_text="parts",
            target_reconstructed=True, is_reconstruction_attempt=False,
        )
        findings = audit_episode_result(_valid_result(turns=[turn]))
        recon_findings = [f for f in findings if f.code == "RECONSTRUCTED_WITHOUT_ATTEMPT"]
        assert len(recon_findings) == 1

    def test_reintroduction_requires_recontamination(self) -> None:
        """target_reintroduced without is_recontamination_attempt is flagged."""
        turn = TurnResult(
            turn_id=0, phase="POST_FORGET_ATTACK",
            sender_id="SK", recipient_id="CK",
            candidate_text="msg", released_text="msg",
            target_reintroduced=True, is_recontamination_attempt=False,
        )
        findings = audit_episode_result(_valid_result(turns=[turn]))
        reint_findings = [f for f in findings if f.code == "REINTRODUCED_WITHOUT_ATTEMPT"]
        assert len(reint_findings) == 1

    def test_task_contribution_requires_relevance(self) -> None:
        """task_contribution_successful without task_relevant is flagged."""
        turn = TurnResult(
            turn_id=0, phase="POST_FORGET_ATTACK",
            sender_id="SK", recipient_id="CK",
            candidate_text="msg", released_text="msg",
            task_contribution_successful=True, task_relevant=False,
        )
        findings = audit_episode_result(_valid_result(turns=[turn]))
        task_findings = [f for f in findings if f.code == "TASK_CONTRIBUTION_WITHOUT_RELEVANCE"]
        assert len(task_findings) == 1


class TestEpisodeRules:
    """Episode-level audit rules."""

    def test_config_hash_valid_length(self) -> None:
        """Config hash with invalid length is flagged."""
        result = _valid_result()
        result.metadata["config_hash"] = "short"
        findings = audit_episode_result(result)
        hash_findings = [f for f in findings if f.code == "INVALID_CONFIG_HASH_LENGTH"]
        assert len(hash_findings) == 1

    def test_secret_variant_id_exists(self) -> None:
        """Missing secret_variant_id is flagged."""
        result = _valid_result()
        result.metadata["secret_variant_id"] = ""
        findings = audit_episode_result(result)
        sv_findings = [f for f in findings if f.code == "MISSING_SECRET_VARIANT_ID"]
        assert len(sv_findings) == 1

    def test_attack_type_exists(self) -> None:
        """Missing attack_type is flagged."""
        result = _valid_result()
        result.metadata["attack_type"] = ""
        findings = audit_episode_result(result)
        atk_findings = [f for f in findings if f.code == "MISSING_ATTACK_TYPE"]
        assert len(atk_findings) == 1

    def test_cleaned_agents_not_negative(self) -> None:
        """Negative cleaned_agents_exposed is flagged."""
        result = _valid_result(cleaned_agents_exposed=-1)
        findings = audit_episode_result(result)
        neg_findings = [f for f in findings if f.code == "NEGATIVE_CLEANED_AGENTS_DENOMINATOR"]
        assert len(neg_findings) == 1

    def test_numerator_cannot_exceed_denominator(self) -> None:
        """recontaminated_agents > cleaned_agents_exposed is flagged."""
        result = _valid_result(
            cleaned_agents_exposed=1, recontaminated_agents=2,
        )
        findings = audit_episode_result(result)
        num_findings = [f for f in findings if f.code == "NUMERATOR_EXCEEDS_DENOMINATOR"]
        assert len(num_findings) == 1

    def test_valid_result_passes(self) -> None:
        """A valid result has no errors."""
        result = _valid_result()
        findings = audit_episode_result(result)
        errors = [f for f in findings if f.level == "error"]
        assert len(errors) == 0


class TestMetricRules:
    """Metric-level audit rules."""

    def test_numerator_exceeds_denominator(self) -> None:
        """numerator > denominator is flagged."""
        findings = audit_metric_value(5, 3, 0.5, "PU-RER")
        assert any(f.code == "METRIC_NUMERATOR_EXCEEDS_DENOMINATOR" for f in findings)

    def test_zero_denominator_with_value(self) -> None:
        """Zero denominator with non-None value is flagged."""
        findings = audit_metric_value(0, 0, 0.5, "RR")
        assert any(f.code == "METRIC_ZERO_DENOMINATOR_WITH_VALUE" for f in findings)

    def test_zero_denominator_none_value_ok(self) -> None:
        """Zero denominator with None value is valid."""
        findings = audit_metric_value(0, 0, None, "RR")
        assert len(findings) == 0

    def test_value_out_of_range(self) -> None:
        """Value outside [0, 1] is flagged."""
        findings = audit_metric_value(1, 2, 1.5, "CRR")
        assert any(f.code == "METRIC_VALUE_OUT_OF_RANGE" for f in findings)

    def test_valid_metric_passes(self) -> None:
        """Valid metric has no findings."""
        findings = audit_metric_value(1, 2, 0.5, "PU-RER")
        assert len(findings) == 0


class TestAggregationGate:
    """Aggregation refuses unaudited or invalid results."""

    def test_valid_results_pass_aggregation(self) -> None:
        """Valid results pass aggregation gate."""
        results = [_valid_result()]
        is_valid, report = validate_for_aggregation(results)
        assert is_valid is True

    def test_invalid_results_fail_aggregation(self) -> None:
        """Results with errors fail aggregation gate."""
        result = _valid_result(cleaned_agents_exposed=-1)
        is_valid, report = validate_for_aggregation([result])
        assert is_valid is False
        assert report.has_errors
