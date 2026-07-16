"""Tests for experiment result auditor."""

from experiments.trustparadox_u.audit_results import (
    InvalidExperimentResults,
    PolicyAblationPair,
    audit_duplicate_keys,
    audit_embedding_metadata,
    audit_episode_result,
    audit_fragmentation_result,
    audit_metric_value,
    audit_monitoring_metadata,
    audit_policy_ablation_pair,
    audit_utility_value,
    validate_for_aggregation,
)
from experiments.trustparadox_u.runner import EpisodeResult, TurnResult


def _valid_result(**overrides) -> EpisodeResult:
    """Create a minimal valid EpisodeResult for testing."""
    result = EpisodeResult(
        run_id="run_0001",
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
            exact_score=1.0,
            entity_score=0.0,
            semantic_score=0.0,
            reconstruction_score=0.0,
            matched_forget_ids=[],
            evidence=[],
        )
        decision = FirewallDecision(
            action="block",
            released_text=None,
            detector_result=dr,
            reason_codes=("exact",),
            policy_version="v1",
            latency_ms=0.1,
        )
        turn = TurnResult(
            turn_id=0,
            phase="POST_FORGET_ATTACK",
            sender_id="SK",
            recipient_id="CK",
            candidate_text="secret",
            released_text=None,
            decision=decision,
        )
        findings = audit_episode_result(_valid_result(turns=[turn]))
        block_findings = [f for f in findings if f.code == "BLOCK_WITH_RELEASED_TEXT"]
        assert len(block_findings) == 0

    def test_block_with_released_text_flagged(self) -> None:
        """block action with released_text is flagged."""
        from marble.firewall.types import DetectorResult, FirewallDecision

        dr = DetectorResult(
            exact_score=1.0,
            entity_score=0.0,
            semantic_score=0.0,
            reconstruction_score=0.0,
            matched_forget_ids=[],
            evidence=[],
        )
        # Create a valid block decision, then override turn.released_text
        decision = FirewallDecision(
            action="block",
            released_text=None,
            detector_result=dr,
            reason_codes=("exact",),
            policy_version="v1",
            latency_ms=0.1,
        )
        turn = TurnResult(
            turn_id=0,
            phase="POST_FORGET_ATTACK",
            sender_id="SK",
            recipient_id="CK",
            candidate_text="secret",
            released_text="leaked",
            decision=decision,
        )
        findings = audit_episode_result(_valid_result(turns=[turn]))
        block_findings = [f for f in findings if f.code == "BLOCK_WITH_RELEASED_TEXT"]
        assert len(block_findings) == 1

    def test_exposure_requires_released_text(self) -> None:
        """target_exposed=True without released_text is flagged."""
        turn = TurnResult(
            turn_id=0,
            phase="POST_FORGET_ATTACK",
            sender_id="SK",
            recipient_id="CK",
            candidate_text="secret",
            released_text=None,
            target_exposed=True,
        )
        findings = audit_episode_result(_valid_result(turns=[turn]))
        exp_findings = [f for f in findings if f.code == "EXPOSED_WITHOUT_RELEASED_TEXT"]
        assert len(exp_findings) == 1

    def test_reconstruction_requires_attempt(self) -> None:
        """target_reconstructed without is_reconstruction_attempt is flagged."""
        turn = TurnResult(
            turn_id=0,
            phase="POST_FORGET_ATTACK",
            sender_id="SK",
            recipient_id="CK",
            candidate_text="parts",
            released_text="parts",
            target_reconstructed=True,
            is_reconstruction_attempt=False,
        )
        findings = audit_episode_result(_valid_result(turns=[turn]))
        recon_findings = [f for f in findings if f.code == "RECONSTRUCTED_WITHOUT_ATTEMPT"]
        assert len(recon_findings) == 1

    def test_reintroduction_requires_recontamination(self) -> None:
        """target_reintroduced without is_recontamination_attempt is flagged."""
        turn = TurnResult(
            turn_id=0,
            phase="POST_FORGET_ATTACK",
            sender_id="SK",
            recipient_id="CK",
            candidate_text="msg",
            released_text="msg",
            target_reintroduced=True,
            is_recontamination_attempt=False,
        )
        findings = audit_episode_result(_valid_result(turns=[turn]))
        reint_findings = [f for f in findings if f.code == "REINTRODUCED_WITHOUT_ATTEMPT"]
        assert len(reint_findings) == 1

    def test_task_contribution_requires_relevance(self) -> None:
        """task_contribution_successful without task_relevant is flagged."""
        turn = TurnResult(
            turn_id=0,
            phase="POST_FORGET_ATTACK",
            sender_id="SK",
            recipient_id="CK",
            candidate_text="msg",
            released_text="msg",
            task_contribution_successful=True,
            task_relevant=False,
        )
        findings = audit_episode_result(_valid_result(turns=[turn]))
        task_findings = [f for f in findings if f.code == "TASK_CONTRIBUTION_WITHOUT_RELEVANCE"]
        assert len(task_findings) == 1

    def test_blocked_message_at_risk(self) -> None:
        """Blocked message with target_exposed=True is flagged."""
        turn = TurnResult(
            turn_id=0,
            phase="POST_FORGET_ATTACK",
            sender_id="SK",
            recipient_id="CK",
            candidate_text="secret",
            released_text=None,
            target_exposed=True,
        )
        findings = audit_episode_result(_valid_result(turns=[turn]))
        at_risk = [f for f in findings if f.code == "BLOCKED_MESSAGE_AT_RISK"]
        assert len(at_risk) == 1

    def test_reconstruction_without_released_history(self) -> None:
        """Reconstructed target without released_text is flagged."""
        turn = TurnResult(
            turn_id=0,
            phase="POST_FORGET_ATTACK",
            sender_id="SK",
            recipient_id="CK",
            candidate_text="parts",
            released_text=None,
            target_reconstructed=True,
            is_reconstruction_attempt=True,
        )
        findings = audit_episode_result(_valid_result(turns=[turn]))
        recon = [f for f in findings if f.code == "RECONSTRUCTION_WITHOUT_RELEASED_HISTORY"]
        assert len(recon) == 1

    def test_legitimate_task_not_task_relevant(self) -> None:
        """Legitimate message without task_relevant is flagged."""
        turn = TurnResult(
            turn_id=0,
            phase="POST_FORGET_ATTACK",
            sender_id="SK",
            recipient_id="CK",
            candidate_text="help with task",
            released_text="help with task",
            is_legitimate_message=True,
            task_relevant=False,
        )
        findings = audit_episode_result(_valid_result(turns=[turn]))
        legit = [f for f in findings if f.code == "LEGITIMATE_TASK_NOT_TASK_RELEVANT"]
        assert len(legit) == 1


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
            cleaned_agents_exposed=1,
            recontaminated_agents=2,
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

    def test_invalid_run_id_too_short(self) -> None:
        """Run ID shorter than 8 chars is flagged."""
        result = _valid_result(run_id="abc")
        findings = audit_episode_result(result)
        rid = [f for f in findings if f.code == "INVALID_RUN_ID"]
        assert len(rid) == 1

    def test_invalid_run_id_too_long(self) -> None:
        """Run ID longer than 64 chars is flagged."""
        result = _valid_result(run_id="x" * 65)
        findings = audit_episode_result(result)
        rid = [f for f in findings if f.code == "INVALID_RUN_ID"]
        assert len(rid) == 1


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
        """Results with errors raise InvalidExperimentResults."""
        import pytest

        result = _valid_result(cleaned_agents_exposed=-1)
        with pytest.raises(InvalidExperimentResults) as exc_info:
            validate_for_aggregation([result])
        assert exc_info.value.report.has_errors

    def test_invalid_results_allow_errors(self) -> None:
        """Results with errors pass when allow_errors=True."""
        result = _valid_result(cleaned_agents_exposed=-1)
        is_valid, report = validate_for_aggregation([result], allow_errors=True)
        assert is_valid is True
        assert report.has_errors


class TestEmbeddingAudit:
    """Embedding metadata audit rules."""

    def test_experiment_missing_provider(self) -> None:
        findings = audit_embedding_metadata({}, run_mode="experiment", semantic_enabled=True)
        assert any(f.code == "MISSING_EMBEDDING_PROVIDER" for f in findings)

    def test_experiment_fixed_provider_rejected(self) -> None:
        findings = audit_embedding_metadata(
            {"embedding_provider": "fixed"},
            run_mode="experiment",
            semantic_enabled=True,
        )
        assert any(f.code == "EXPERIMENT_USES_FIXED_PROVIDER" for f in findings)

    def test_experiment_missing_model(self) -> None:
        findings = audit_embedding_metadata(
            {"embedding_provider": "litellm"},
            run_mode="experiment",
            semantic_enabled=True,
        )
        assert any(f.code == "MISSING_EMBEDDING_MODEL" for f in findings)

    def test_experiment_default_model_rejected(self) -> None:
        findings = audit_embedding_metadata(
            {"embedding_provider": "litellm", "embedding_model": "default"},
            run_mode="experiment",
            semantic_enabled=True,
        )
        assert any(f.code == "EMBEDDING_MODEL_IS_DEFAULT" for f in findings)

    def test_experiment_invalid_dimension(self) -> None:
        findings = audit_embedding_metadata(
            {
                "embedding_provider": "litellm",
                "embedding_model": "text-embedding-3-small",
                "embedding_dimension": -1,
            },
            run_mode="experiment",
            semantic_enabled=True,
        )
        assert any(f.code == "INVALID_EMBEDDING_DIMENSION" for f in findings)

    def test_experiment_valid_passes(self) -> None:
        findings = audit_embedding_metadata(
            {
                "embedding_provider": "litellm",
                "embedding_model": "text-embedding-3-small",
                "embedding_dimension": 1536,
            },
            run_mode="experiment",
            semantic_enabled=True,
        )
        errors = [f for f in findings if f.level == "error"]
        assert len(errors) == 0

    def test_test_mode_non_fixed_rejected(self) -> None:
        findings = audit_embedding_metadata(
            {"embedding_provider": "litellm"},
            run_mode="test",
            semantic_enabled=True,
        )
        assert any(f.code == "TEST_MODE_NON_FIXED_PROVIDER" for f in findings)

    def test_test_mode_fixed_passes(self) -> None:
        findings = audit_embedding_metadata(
            {"embedding_provider": "fixed"},
            run_mode="test",
            semantic_enabled=True,
        )
        errors = [f for f in findings if f.level == "error"]
        assert len(errors) == 0

    def test_semantic_disabled_no_findings(self) -> None:
        findings = audit_embedding_metadata({}, run_mode="experiment", semantic_enabled=False)
        assert len(findings) == 0


class TestMonitoringAudit:
    """Monitoring metadata audit rules."""

    def test_negative_duration_flagged(self) -> None:
        findings = audit_monitoring_metadata({"monitoring_duration_rounds": -1})
        assert any(f.code == "NEGATIVE_MONITORING_DURATION" for f in findings)

    def test_negative_round_count_flagged(self) -> None:
        findings = audit_monitoring_metadata({"post_forget_round_count": -3})
        assert any(f.code == "NEGATIVE_ROUND_COUNT" for f in findings)

    def test_valid_monitoring_passes(self) -> None:
        findings = audit_monitoring_metadata(
            {
                "monitoring_duration_rounds": 5,
                "post_forget_round_count": 3,
                "monitoring_continuous": True,
            }
        )
        errors = [f for f in findings if f.level == "error"]
        assert len(errors) == 0


class TestUtilityAudit:
    """Utility value audit rules."""

    def test_utility_out_of_range(self) -> None:
        findings = audit_utility_value(1.5)
        assert any(f.code == "UTILITY_OUT_OF_RANGE" for f in findings)

    def test_utility_negative(self) -> None:
        findings = audit_utility_value(-0.1)
        assert any(f.code == "UTILITY_OUT_OF_RANGE" for f in findings)

    def test_utility_none_valid(self) -> None:
        findings = audit_utility_value(None)
        assert len(findings) == 0

    def test_utility_valid(self) -> None:
        findings = audit_utility_value(0.75)
        assert len(findings) == 0


class TestPolicyAblationAudit:
    """Policy ablation pair audit rules."""

    def test_pairing_key_mismatch(self) -> None:
        b = _valid_result()
        b.metadata["pairing_key"] = "key1"
        r = _valid_result()
        r.metadata["pairing_key"] = "key2"
        pair = PolicyAblationPair(binary=b, rich=r, pairing_key="key1")
        findings = audit_policy_ablation_pair(pair)
        assert any(f.code == "POLICY_PAIR_KEY_MISMATCH" for f in findings)

    def test_candidate_mismatch(self) -> None:
        b = _valid_result()
        b.metadata["pairing_key"] = "k1"
        r = _valid_result()
        r.metadata["pairing_key"] = "k1"
        b.turns = [
            TurnResult(
                turn_id=0,
                phase="POST_FORGET_ATTACK",
                sender_id="A",
                recipient_id="B",
                candidate_text="msg1",
            )
        ]
        r.turns = [
            TurnResult(
                turn_id=0,
                phase="POST_FORGET_ATTACK",
                sender_id="A",
                recipient_id="B",
                candidate_text="msg2",
            )
        ]
        pair = PolicyAblationPair(binary=b, rich=r, pairing_key="k1")
        findings = audit_policy_ablation_pair(pair)
        assert any(f.code == "POLICY_PAIR_CANDIDATE_MISMATCH" for f in findings)

    def test_valid_pair_passes(self) -> None:
        b = _valid_result()
        b.metadata["pairing_key"] = "k1"
        r = _valid_result()
        r.metadata["pairing_key"] = "k1"
        b.turns = [
            TurnResult(
                turn_id=0,
                phase="POST_FORGET_ATTACK",
                sender_id="A",
                recipient_id="B",
                candidate_text="same_msg",
            )
        ]
        r.turns = [
            TurnResult(
                turn_id=0,
                phase="POST_FORGET_ATTACK",
                sender_id="A",
                recipient_id="B",
                candidate_text="same_msg",
            )
        ]
        pair = PolicyAblationPair(binary=b, rich=r, pairing_key="k1")
        findings = audit_policy_ablation_pair(pair)
        errors = [f for f in findings if f.level == "error"]
        assert len(errors) == 0


class TestFragmentationAudit:
    """Fragmentation result audit rules."""

    def test_too_few_steps(self) -> None:
        result = _valid_result()
        result.turns = [
            TurnResult(
                turn_id=0,
                phase="POST_FORGET_ATTACK",
                sender_id="A",
                recipient_id="B",
                candidate_text="frag",
                attack_type="temporal_fragmentation",
            )
        ]
        findings = audit_fragmentation_result(result)
        assert any(f.code == "FRAGMENTATION_TOO_FEW_STEPS" for f in findings)

    def test_too_few_fragments(self) -> None:
        result = _valid_result()
        result.turns = [
            TurnResult(
                turn_id=0,
                phase="POST_FORGET_ATTACK",
                sender_id="A",
                recipient_id="B",
                candidate_text="f1",
                attack_type="temporal_fragmentation",
            ),
            TurnResult(
                turn_id=1,
                phase="POST_FORGET_ATTACK",
                sender_id="A",
                recipient_id="B",
                candidate_text="f2",
                attack_type="temporal_fragmentation",
            ),
        ]
        result.metadata["fragment_count"] = 1
        findings = audit_fragmentation_result(result)
        assert any(f.code == "FRAGMENTATION_TOO_FEW_FRAGMENTS" for f in findings)

    def test_valid_fragmentation_passes(self) -> None:
        result = _valid_result()
        result.turns = [
            TurnResult(
                turn_id=0,
                phase="POST_FORGET_ATTACK",
                sender_id="A",
                recipient_id="B",
                candidate_text="f1",
                attack_type="temporal_fragmentation",
            ),
            TurnResult(
                turn_id=1,
                phase="POST_FORGET_ATTACK",
                sender_id="A",
                recipient_id="B",
                candidate_text="f2",
                attack_type="temporal_fragmentation",
            ),
        ]
        result.metadata["fragment_count"] = 2
        findings = audit_fragmentation_result(result)
        errors = [f for f in findings if f.level == "error"]
        assert len(errors) == 0


class TestDuplicateKeys:
    """Duplicate pairing key detection."""

    def test_duplicate_keys_flagged(self) -> None:
        r1 = _valid_result()
        r1.metadata["pairing_key"] = "dup"
        r2 = _valid_result()
        r2.metadata["pairing_key"] = "dup"
        findings = audit_duplicate_keys([r1, r2])
        assert any(f.code == "DUPLICATE_PAIRING_KEY" for f in findings)

    def test_unique_keys_pass(self) -> None:
        r1 = _valid_result()
        r1.metadata["pairing_key"] = "k1"
        r2 = _valid_result()
        r2.metadata["pairing_key"] = "k2"
        findings = audit_duplicate_keys([r1, r2])
        assert len(findings) == 0
