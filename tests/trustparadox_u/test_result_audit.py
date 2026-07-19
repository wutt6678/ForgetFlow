"""Tests for experiment result auditor."""

import pytest

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
        "pairing_key": {
            "scenario_id": "s1",
            "secret_variant_id": "sv1",
            "trust_level": "default",
            "attack_type": "direct",
            "seed": 42,
        },
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

    def test_experiment_missing_dimension(self) -> None:
        """Semantic experiment mode requires embedding_dimension."""
        findings = audit_embedding_metadata(
            {
                "embedding_provider": "litellm",
                "embedding_model": "text-embedding-3-small",
            },
            run_mode="experiment",
            semantic_enabled=True,
        )
        assert any(f.code == "MISSING_EMBEDDING_DIMENSION" for f in findings)

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
        b.metadata["detector_hash"] = "abc123"
        b.metadata["history_hash"] = "def456"
        b.metadata["monitoring_hash"] = "ghi789"
        b.metadata["models_hash"] = "jkl012"
        b.metadata["policy_base_hash"] = "mno345"
        b.metadata["rich_actions_enabled"] = False
        r = _valid_result()
        r.metadata["pairing_key"] = "k1"
        r.metadata["detector_hash"] = "abc123"
        r.metadata["history_hash"] = "def456"
        r.metadata["monitoring_hash"] = "ghi789"
        r.metadata["models_hash"] = "jkl012"
        r.metadata["policy_base_hash"] = "mno345"
        r.metadata["rich_actions_enabled"] = True
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

    def test_duplicate_run_identities_flagged(self) -> None:
        r1 = _valid_result()
        r2 = _valid_result(episode_id="ep2")
        # Same pairing key + same config_hash => duplicate run identity
        findings = audit_duplicate_keys([r1, r2])
        assert any(f.code == "RUN_IDENTITY_DUPLICATE" for f in findings)

    def test_same_pairing_key_different_config_hash_passes(self) -> None:
        """Same pairing key but different config_hash => valid separate variants."""
        r1 = _valid_result()
        r2 = _valid_result(episode_id="ep2")
        r2.metadata["config_hash"] = "b" * 64  # different config hash
        findings = audit_duplicate_keys([r1, r2])
        dup_findings = [f for f in findings if f.code == "RUN_IDENTITY_DUPLICATE"]
        assert len(dup_findings) == 0

    def test_different_seed_distinct_run_identity(self) -> None:
        r1 = _valid_result()
        r2 = _valid_result(episode_id="ep2", seed=99)
        r2.metadata["seed"] = 99
        findings = audit_duplicate_keys([r1, r2])
        dup_findings = [f for f in findings if f.code == "RUN_IDENTITY_DUPLICATE"]
        assert len(dup_findings) == 0

    def test_missing_config_hash_raises_error(self) -> None:
        r1 = _valid_result()
        del r1.metadata["config_hash"]
        findings = audit_duplicate_keys([r1])
        assert any(f.code == "RUN_IDENTITY_INVALID" for f in findings)

    def test_tuple_pairing_key_duplicate(self) -> None:
        r1 = _valid_result()
        r1.metadata["pairing_key"] = ("s1", "sv1", "default", "direct", 42)
        r2 = _valid_result(episode_id="ep2")
        r2.metadata["pairing_key"] = ("s1", "sv1", "default", "direct", 42)
        findings = audit_duplicate_keys([r1, r2])
        assert any(f.code == "RUN_IDENTITY_DUPLICATE" for f in findings)

    def test_evaluator_auditor_consistency(self) -> None:
        from experiments.trustparadox_u.identity import (
            normalize_pairing_key,
            pairing_key_from_result,
        )

        r1 = _valid_result()
        r1.metadata["pairing_key"] = {
            "scenario_id": "s1",
            "secret_variant_id": "sv1",
            "trust_level": "default",
            "attack_type": "direct",
            "seed": 42,
        }
        from_dict = normalize_pairing_key(r1.metadata["pairing_key"])
        from_result = pairing_key_from_result(r1)
        assert from_dict == from_result

    def test_type_error_in_identity_becomes_finding(self) -> None:
        """TypeError during identity conversion becomes an audit finding."""
        r = _valid_result()
        # Set seed to None which will cause TypeError when converting to int
        r.seed = None  # type: ignore[assignment]

        from experiments.trustparadox_u.audit_results import audit_duplicate_keys

        findings = audit_duplicate_keys([r])
        assert any(f.code == "RUN_IDENTITY_INVALID" for f in findings)


class TestRunnerAuditIntegration:
    """Integration tests using actual runner output."""

    def test_runner_result_passes_aggregation_audit(self) -> None:
        """A result from the real runner passes the full audit pipeline."""
        from pathlib import Path

        from experiments.trustparadox_u.audit_results import audit_results
        from experiments.trustparadox_u.config import (
            DetectorConfig,
            ExperimentConfig,
            HistoryConfig,
            MonitoringConfig,
            PolicyConfig,
        )
        from experiments.trustparadox_u.dataset import load_episode
        from experiments.trustparadox_u.runner import run_episode

        scenarios_dir = Path(__file__).parents[2] / "data" / "trustparadox_u" / "scenarios"
        ep = load_episode(scenarios_dir / "pilot_credential.yaml")
        config = ExperimentConfig(
            seed=42,
            repetitions=1,
            detector=DetectorConfig(semantic_enabled=False),
            history=HistoryConfig(),
            policy=PolicyConfig(),
            monitoring=MonitoringConfig(),
        )
        result = run_episode(ep, config)
        report = audit_results([result])
        assert not report.has_errors

    def test_runner_duplicate_results_fail_audit(self) -> None:
        """Two runs of the same episode with same identity are flagged."""
        from pathlib import Path

        from experiments.trustparadox_u.audit_results import audit_results
        from experiments.trustparadox_u.config import (
            DetectorConfig,
            ExperimentConfig,
            HistoryConfig,
            MonitoringConfig,
            PolicyConfig,
        )
        from experiments.trustparadox_u.dataset import load_episode
        from experiments.trustparadox_u.runner import run_episode

        scenarios_dir = Path(__file__).parents[2] / "data" / "trustparadox_u" / "scenarios"
        ep = load_episode(scenarios_dir / "pilot_credential.yaml")
        config = ExperimentConfig(
            seed=42,
            repetitions=1,
            detector=DetectorConfig(semantic_enabled=False),
            history=HistoryConfig(),
            policy=PolicyConfig(),
            monitoring=MonitoringConfig(),
        )
        result_a = run_episode(ep, config)
        result_b = run_episode(ep, config)
        report = audit_results([result_a, result_b])
        assert report.has_errors
        assert any(f.code == "RUN_IDENTITY_DUPLICATE" for f in report.findings)

    def test_runner_distinct_seeds_pass(self) -> None:
        """Different seeds produce distinct pairing keys."""
        from pathlib import Path

        from experiments.trustparadox_u.audit_results import audit_results
        from experiments.trustparadox_u.config import (
            DetectorConfig,
            ExperimentConfig,
            HistoryConfig,
            MonitoringConfig,
            PolicyConfig,
        )
        from experiments.trustparadox_u.dataset import load_episode
        from experiments.trustparadox_u.runner import run_episode

        scenarios_dir = Path(__file__).parents[2] / "data" / "trustparadox_u" / "scenarios"
        ep = load_episode(scenarios_dir / "pilot_credential.yaml")
        config_a = ExperimentConfig(
            seed=1,
            repetitions=1,
            detector=DetectorConfig(semantic_enabled=False),
            history=HistoryConfig(),
            policy=PolicyConfig(),
            monitoring=MonitoringConfig(),
        )
        config_b = ExperimentConfig(
            seed=2,
            repetitions=1,
            detector=DetectorConfig(semantic_enabled=False),
            history=HistoryConfig(),
            policy=PolicyConfig(),
            monitoring=MonitoringConfig(),
        )
        result_a = run_episode(ep, config_a)
        result_b = run_episode(ep, config_b)
        report = audit_results([result_a, result_b])
        dup_findings = [f for f in report.findings if f.code == "RUN_IDENTITY_DUPLICATE"]
        assert len(dup_findings) == 0

    def test_missing_config_hash_raises_not_typeerror(self) -> None:
        """A missing config_hash raises an audit error, not a raw KeyError."""
        import pytest

        result = _valid_result()
        del result.metadata["config_hash"]
        with pytest.raises(InvalidExperimentResults):
            validate_for_aggregation([result])


class TestAttackStepIndexAudit:
    """Phase 8: attack-step index audit checks."""

    def _attack_turn(
        self,
        turn_id: int = 0,
        attack_type: str = "direct",
        step_index: int | None = 0,
    ) -> "TurnResult":
        from experiments.trustparadox_u.runner import TurnResult

        return TurnResult(
            turn_id=turn_id,
            phase="POST_FORGET_ATTACK",
            sender_id="A",
            recipient_id="B",
            candidate_text="test",
            released_text="test",
            attack_type=attack_type,
            attack_step_index=step_index,
            is_attack_attempt=True,
        )

    def test_valid_attack_step_passes(self) -> None:
        """Valid attack step index passes audit."""
        from experiments.trustparadox_u.audit_results import audit_attack_step_indices
        from experiments.trustparadox_u.runner import EpisodeResult

        result = EpisodeResult(
            run_id="r1",
            episode_id="ep1",
            scenario_id="s1",
            trust_level="high",
            seed=42,
        )
        result.turns.append(self._attack_turn(turn_id=0, step_index=0))
        result.turns.append(self._attack_turn(turn_id=1, step_index=1))
        findings = audit_attack_step_indices(result)
        assert len(findings) == 0

    def test_missing_attack_step_index_fails(self) -> None:
        """Missing attack step index is flagged."""
        from experiments.trustparadox_u.audit_results import audit_attack_step_indices
        from experiments.trustparadox_u.runner import EpisodeResult

        result = EpisodeResult(
            run_id="r1",
            episode_id="ep1",
            scenario_id="s1",
            trust_level="high",
            seed=42,
        )
        result.turns.append(self._attack_turn(step_index=None))
        findings = audit_attack_step_indices(result)
        assert any(f.code == "ATTACK_STEP_INDEX_MISSING" for f in findings)

    def test_negative_step_index_fails(self) -> None:
        """Negative step index is flagged."""
        from experiments.trustparadox_u.audit_results import audit_attack_step_indices
        from experiments.trustparadox_u.runner import EpisodeResult

        result = EpisodeResult(
            run_id="r1",
            episode_id="ep1",
            scenario_id="s1",
            trust_level="high",
            seed=42,
        )
        result.turns.append(self._attack_turn(step_index=-1))
        findings = audit_attack_step_indices(result)
        assert any(f.code == "ATTACK_STEP_INDEX_NEGATIVE" for f in findings)

    def test_duplicate_step_index_fails(self) -> None:
        """Duplicate step indices within one attack type are flagged."""
        from experiments.trustparadox_u.audit_results import audit_attack_step_indices
        from experiments.trustparadox_u.runner import EpisodeResult, TurnResult

        result = EpisodeResult(
            run_id="r1",
            episode_id="ep1",
            scenario_id="s1",
            trust_level="high",
            seed=42,
        )
        # Two response turns with same step index = duplicate
        result.turns.append(
            TurnResult(
                turn_id=0,
                phase="POST_FORGET_ATTACK",
                sender_id="CK",
                recipient_id="SK",
                candidate_text="test",
                released_text="test",
                attack_type="direct",
                attack_step_index=0,
                is_attack_response=True,
            )
        )
        result.turns.append(
            TurnResult(
                turn_id=1,
                phase="POST_FORGET_ATTACK",
                sender_id="CK",
                recipient_id="SK",
                candidate_text="test",
                released_text="test",
                attack_type="direct",
                attack_step_index=0,
                is_attack_response=True,
            )
        )
        findings = audit_attack_step_indices(result)
        assert any(f.code == "ATTACK_STEP_INDEX_DUPLICATE" for f in findings)

    def test_non_monotonic_step_index_fails(self) -> None:
        """Non-monotonic step indices are flagged."""
        from experiments.trustparadox_u.audit_results import audit_attack_step_indices
        from experiments.trustparadox_u.runner import EpisodeResult

        result = EpisodeResult(
            run_id="r1",
            episode_id="ep1",
            scenario_id="s1",
            trust_level="high",
            seed=42,
        )
        result.turns.append(self._attack_turn(turn_id=0, step_index=2))
        result.turns.append(self._attack_turn(turn_id=1, step_index=1))
        findings = audit_attack_step_indices(result)
        assert any(f.code == "ATTACK_STEP_INDEX_NOT_MONOTONIC" for f in findings)

    def test_real_runner_result_passes_step_audit(self) -> None:
        """Real runner results pass the attack-step audit."""
        from pathlib import Path

        from experiments.trustparadox_u.audit_results import audit_attack_step_indices
        from experiments.trustparadox_u.config import (
            DetectorConfig,
            ExperimentConfig,
            HistoryConfig,
            MonitoringConfig,
            PolicyConfig,
        )
        from experiments.trustparadox_u.dataset import load_episode
        from experiments.trustparadox_u.runner import run_episode

        scenarios = Path(__file__).parents[2] / "data" / "trustparadox_u" / "scenarios"
        ep = load_episode(scenarios / "pilot_credential.yaml")
        cfg = ExperimentConfig(
            seed=42,
            repetitions=1,
            detector=DetectorConfig(semantic_enabled=False),
            history=HistoryConfig(),
            policy=PolicyConfig(),
            monitoring=MonitoringConfig(),
        )
        result = run_episode(ep, cfg)
        findings = audit_attack_step_indices(result)
        assert len(findings) == 0


class TestPolicyComponentMatrix:
    """G13: Complete policy-ablation equivalence matrix."""

    @staticmethod
    def _make_pair(
        *,
        binary_hashes: dict | None = None,
        rich_hashes: dict | None = None,
        binary_rich_actions: bool = False,
        rich_rich_actions: bool = True,
        pairing_key: str = "k1",
        candidate: str = "same_msg",
    ) -> PolicyAblationPair:
        """Create a policy pair with configurable component hashes."""
        default_hashes = {
            "detector_hash": "abc123",
            "history_hash": "def456",
            "monitoring_hash": "ghi789",
            "models_hash": "jkl012",
            "policy_base_hash": "mno345",
        }
        b = _valid_result()
        b.metadata["pairing_key"] = pairing_key
        b.metadata["rich_actions_enabled"] = binary_rich_actions
        for k, v in (binary_hashes or default_hashes).items():
            b.metadata[k] = v
        b.turns = [
            TurnResult(
                turn_id=0,
                phase="POST_FORGET_ATTACK",
                sender_id="A",
                recipient_id="B",
                candidate_text=candidate,
            )
        ]

        r = _valid_result()
        r.metadata["pairing_key"] = pairing_key
        r.metadata["rich_actions_enabled"] = rich_rich_actions
        for k, v in (rich_hashes or default_hashes).items():
            r.metadata[k] = v
        r.turns = [
            TurnResult(
                turn_id=0,
                phase="POST_FORGET_ATTACK",
                sender_id="A",
                recipient_id="B",
                candidate_text=candidate,
            )
        ]
        return PolicyAblationPair(binary=b, rich=r, pairing_key=pairing_key)

    @pytest.mark.parametrize(
        "component",
        ["detector_hash", "history_hash", "monitoring_hash", "models_hash", "policy_base_hash"],
    )
    def test_binary_missing_component(self, component: str) -> None:
        """Binary missing a component hash produces an error."""
        b_hashes = {
            "detector_hash": "abc123",
            "history_hash": "def456",
            "monitoring_hash": "ghi789",
            "models_hash": "jkl012",
            "policy_base_hash": "mno345",
        }
        b_hashes[component] = ""
        pair = self._make_pair(binary_hashes=b_hashes)
        findings = audit_policy_ablation_pair(pair)
        assert any(component.upper() in f.code and "MISSING_BINARY" in f.code for f in findings)

    @pytest.mark.parametrize(
        "component",
        ["detector_hash", "history_hash", "monitoring_hash", "models_hash", "policy_base_hash"],
    )
    def test_rich_missing_component(self, component: str) -> None:
        """Rich missing a component hash produces an error."""
        r_hashes = {
            "detector_hash": "abc123",
            "history_hash": "def456",
            "monitoring_hash": "ghi789",
            "models_hash": "jkl012",
            "policy_base_hash": "mno345",
        }
        r_hashes[component] = ""
        pair = self._make_pair(rich_hashes=r_hashes)
        findings = audit_policy_ablation_pair(pair)
        assert any(component.upper() in f.code and "MISSING_RICH" in f.code for f in findings)

    @pytest.mark.parametrize(
        "component",
        ["detector_hash", "history_hash", "monitoring_hash", "models_hash", "policy_base_hash"],
    )
    def test_component_mismatch(self, component: str) -> None:
        """Binary and rich with different component hash produces an error."""
        r_hashes = {
            "detector_hash": "abc123",
            "history_hash": "def456",
            "monitoring_hash": "ghi789",
            "models_hash": "jkl012",
            "policy_base_hash": "mno345",
        }
        r_hashes[component] = "different_value"
        pair = self._make_pair(rich_hashes=r_hashes)
        findings = audit_policy_ablation_pair(pair)
        assert any(component.upper() in f.code and "MISMATCH" in f.code for f in findings)

    def test_valid_pair_no_errors(self) -> None:
        """Valid pair with all matching components passes."""
        pair = self._make_pair()
        findings = audit_policy_ablation_pair(pair)
        errors = [f for f in findings if f.level == "error"]
        assert len(errors) == 0


class TestPolicyOrientation:
    """G13: Policy orientation — binary=false, rich=true."""

    def test_binary_not_false_fails(self) -> None:
        """Binary with rich_actions_enabled=True is rejected."""
        pair = TestPolicyComponentMatrix._make_pair(
            binary_rich_actions=True, rich_rich_actions=True
        )
        findings = audit_policy_ablation_pair(pair)
        assert any(f.code == "POLICY_PAIR_BINARY_NOT_FALSE" for f in findings)

    def test_rich_not_true_fails(self) -> None:
        """Rich with rich_actions_enabled=False is rejected."""
        pair = TestPolicyComponentMatrix._make_pair(
            binary_rich_actions=False, rich_rich_actions=False
        )
        findings = audit_policy_ablation_pair(pair)
        assert any(f.code == "POLICY_PAIR_RICH_NOT_TRUE" for f in findings)

    def test_reversed_orientation_fails(self) -> None:
        """Reversed orientation (binary=True, rich=False) produces two errors."""
        pair = TestPolicyComponentMatrix._make_pair(
            binary_rich_actions=True, rich_rich_actions=False
        )
        findings = audit_policy_ablation_pair(pair)
        codes = [f.code for f in findings]
        assert "POLICY_PAIR_BINARY_NOT_FALSE" in codes
        assert "POLICY_PAIR_RICH_NOT_TRUE" in codes

    def test_same_rich_action_value_both_true(self) -> None:
        """Both binary and rich with rich_actions_enabled=True fails."""
        pair = TestPolicyComponentMatrix._make_pair(
            binary_rich_actions=True, rich_rich_actions=True
        )
        findings = audit_policy_ablation_pair(pair)
        assert any(f.code == "POLICY_PAIR_BINARY_NOT_FALSE" for f in findings)

    def test_same_rich_action_value_both_false(self) -> None:
        """Both binary and rich with rich_actions_enabled=False fails."""
        pair = TestPolicyComponentMatrix._make_pair(
            binary_rich_actions=False, rich_rich_actions=False
        )
        findings = audit_policy_ablation_pair(pair)
        assert any(f.code == "POLICY_PAIR_RICH_NOT_TRUE" for f in findings)

    def test_invalid_hash_type(self) -> None:
        """Non-string hash value is caught by strict validation."""
        b_hashes = {
            "detector_hash": 12345,  # type: ignore[dict-item]
            "history_hash": "def456",
            "monitoring_hash": "ghi789",
            "models_hash": "jkl012",
            "policy_base_hash": "mno345",
        }
        pair = TestPolicyComponentMatrix._make_pair(binary_hashes=b_hashes)
        findings = audit_policy_ablation_pair(pair)
        # Non-string hash should cause mismatch or invalid finding
        assert len(findings) > 0

    def test_candidate_mismatch(self) -> None:
        """Different candidate messages produce POLICY_PAIR_CANDIDATE_MISMATCH."""
        pair = TestPolicyComponentMatrix._make_pair(candidate="same_msg")
        # Override rich candidate
        pair.rich.turns[0] = TurnResult(
            turn_id=0,
            phase="POST_FORGET_ATTACK",
            sender_id="A",
            recipient_id="B",
            candidate_text="different_msg",
        )
        findings = audit_policy_ablation_pair(pair)
        assert any(f.code == "POLICY_PAIR_CANDIDATE_MISMATCH" for f in findings)

    def test_pairing_key_mismatch(self) -> None:
        """Different pairing keys produce POLICY_PAIR_KEY_MISMATCH."""
        pair = TestPolicyComponentMatrix._make_pair(pairing_key="k1")
        pair.rich.metadata["pairing_key"] = "k2"
        findings = audit_policy_ablation_pair(pair)
        assert any(f.code == "POLICY_PAIR_KEY_MISMATCH" for f in findings)


class TestEndpointEquivalence:
    """G12: Endpoint provenance detects different sanitized endpoints."""

    def test_different_endpoints_detected(self) -> None:
        """Different api_base_sanitized values are detected by manifest validation."""
        from experiments.trustparadox_u.manifest import (
            build_manifest,
            validate_manifest_against_results,
        )

        r1 = _valid_result(
            metadata={
                "forbidden_strings": ["secret"],
                "config_hash": "a" * 64,
                "run_mode": "test",
                "semantic_enabled": True,
                "embedding_provider": "litellm",
                "embedding_model": "openai/text-embedding-v3",
                "embedding_dimension": 1024,
                "semantic_threshold": 0.8,
                "api_base_sanitized": "https://endpoint-a.example.com",
            }
        )
        r2 = _valid_result(
            metadata={
                "forbidden_strings": ["secret"],
                "config_hash": "a" * 64,
                "run_mode": "test",
                "semantic_enabled": True,
                "embedding_provider": "litellm",
                "embedding_model": "openai/text-embedding-v3",
                "embedding_dimension": 1024,
                "semantic_threshold": 0.8,
                "api_base_sanitized": "https://endpoint-b.example.com",
            }
        )
        m = build_manifest(results=[r1])
        findings = validate_manifest_against_results(m, [r2])
        assert any("ENDPOINT" in f["code"] for f in findings)


class TestUnexpectedRecontaminationAudit:
    """ST-RR-005: Unexpected recontamination pairs are audited."""

    def test_zero_unexpected_passes(self) -> None:
        """ST-RR-005-zero: Zero unexpected pairs passes."""
        result = _valid_result(
            metadata={
                "forbidden_strings": ["secret"],
                "config_hash": "a" * 64,
                "unexpected_recontaminated_pair_count": 0,
            }
        )
        findings = audit_episode_result(result)
        unexpected = [f for f in findings if f.code == "UNEXPECTED_RECONTAMINATION_PAIRS"]
        assert len(unexpected) == 0

    def test_nonzero_unexpected_reports_info(self) -> None:
        """ST-RR-005-one: Non-zero unexpected pairs produces audit info finding."""
        result = _valid_result(
            metadata={
                "forbidden_strings": ["secret"],
                "config_hash": "a" * 64,
                "unexpected_recontaminated_pair_count": 1,
            }
        )
        findings = audit_episode_result(result)
        unexpected = [f for f in findings if f.code == "UNEXPECTED_RECONTAMINATION_PAIRS"]
        assert len(unexpected) == 1
        # s2: Unexpected recontamination is now info, not error
        assert unexpected[0].level == "info"

    def test_negative_count_fails(self) -> None:
        """Section 6: Negative unexpected count is invalid."""
        result = _valid_result(
            metadata={
                "forbidden_strings": ["secret"],
                "config_hash": "a" * 64,
                "unexpected_recontaminated_pair_count": -1,
            }
        )
        findings = audit_episode_result(result)
        invalid = [f for f in findings if f.code == "UNEXPECTED_RECONTAMINATION_COUNT_INVALID"]
        assert len(invalid) == 1
        assert invalid[0].level == "error"

    def test_non_integer_count_fails(self) -> None:
        """Section 6: Non-integer unexpected count is invalid."""
        result = _valid_result(
            metadata={
                "forbidden_strings": ["secret"],
                "config_hash": "a" * 64,
                "unexpected_recontaminated_pair_count": "1",
            }
        )
        findings = audit_episode_result(result)
        invalid = [f for f in findings if f.code == "UNEXPECTED_RECONTAMINATION_COUNT_INVALID"]
        assert len(invalid) == 1
        assert invalid[0].level == "error"

    def test_unexpected_pairs_not_in_rr_numerator(self) -> None:
        """ST-RR-005-numerator: Unexpected pairs don't enter RR numerator.

        recontaminated_agent_record_pairs counts only attributable
        recontaminations (labeled attempt caused AT_RISK -> RECONTAMINATED).
        Unexpected pairs (recontamination without attributable attempt)
        are tracked separately and don't inflate the RR numerator.
        """
        from experiments.trustparadox_u.evaluator import compute_rr

        # Result with attempted pairs but NO attributable recontaminations
        # (all recontamination was "unexpected" - not from labeled attempts)
        result = _valid_result(
            attempted_agent_record_pairs=2,
            recontaminated_agent_record_pairs=0,
        )
        result.metadata["unexpected_recontaminated_pair_count"] = 2
        # RR numerator must be 0 even though unexpected pairs exist
        rr = compute_rr([result])
        assert rr.numerator == 0
        assert rr.denominator == 2
        assert rr.value == 0.0
        # Audit should report the unexpected pairs as info
        findings = audit_episode_result(result)
        unexpected = [f for f in findings if f.code == "UNEXPECTED_RECONTAMINATION_PAIRS"]
        assert len(unexpected) == 1
        assert unexpected[0].level == "info"


class TestReintroducedIdsConsistency:
    """Section 2: target_reintroduced must agree with reintroduced_forget_ids."""

    def test_consistent_reintroduced_ids(self) -> None:
        """target_reintroduced=True with non-empty reintroduced_forget_ids passes."""
        result = _valid_result()
        result.turns = [
            TurnResult(
                turn_id=0,
                phase="POST_FORGET_ATTACK",
                sender_id="A",
                recipient_id="B",
                candidate_text="test",
                released_text="test",
                is_recontamination_attempt=True,
                target_forget_ids=("F001",),
                exposed_forget_ids=("F001",),
                reintroduced_forget_ids=("F001",),
                target_reintroduced=True,
            )
        ]
        findings = audit_episode_result(result)
        consistency = [f for f in findings if f.code == "REINTRODUCED_IDS_CONSISTENCY"]
        assert len(consistency) == 0

    def test_inconsistent_reintroduced_ids(self) -> None:
        """target_reintroduced=True but empty reintroduced_forget_ids fails."""
        result = _valid_result()
        result.turns = [
            TurnResult(
                turn_id=0,
                phase="POST_FORGET_ATTACK",
                sender_id="A",
                recipient_id="B",
                candidate_text="test",
                released_text="test",
                is_recontamination_attempt=True,
                target_forget_ids=("F001",),
                exposed_forget_ids=("F001",),
                reintroduced_forget_ids=(),
                target_reintroduced=True,
            )
        ]
        findings = audit_episode_result(result)
        consistency = [f for f in findings if f.code == "REINTRODUCED_IDS_CONSISTENCY"]
        assert len(consistency) == 1
        assert consistency[0].level == "error"

    def test_reintroduced_not_subset_of_exposed(self) -> None:
        """reintroduced_forget_ids not subset of exposed_forget_ids fails."""
        result = _valid_result()
        result.turns = [
            TurnResult(
                turn_id=0,
                phase="POST_FORGET_ATTACK",
                sender_id="A",
                recipient_id="B",
                candidate_text="test",
                released_text="test",
                is_recontamination_attempt=True,
                target_forget_ids=("F001",),
                exposed_forget_ids=("F002",),
                reintroduced_forget_ids=("F001",),
                target_reintroduced=True,
            )
        ]
        findings = audit_episode_result(result)
        subset = [f for f in findings if f.code == "REINTRODUCED_NOT_SUBSET_OF_EXPOSED"]
        assert len(subset) == 1

    def test_reintroduced_not_subset_of_targeted(self) -> None:
        """reintroduced_forget_ids not subset of target_forget_ids fails."""
        result = _valid_result()
        result.turns = [
            TurnResult(
                turn_id=0,
                phase="POST_FORGET_ATTACK",
                sender_id="A",
                recipient_id="B",
                candidate_text="test",
                released_text="test",
                is_recontamination_attempt=True,
                target_forget_ids=("F001",),
                exposed_forget_ids=("F001", "F002"),
                reintroduced_forget_ids=("F002",),
                target_reintroduced=True,
            )
        ]
        findings = audit_episode_result(result)
        subset = [f for f in findings if f.code == "REINTRODUCED_NOT_SUBSET_OF_TARGETED"]
        assert len(subset) == 1


class TestReconstructedIdsConsistency:
    """Section 4: target_reconstructed must agree with reconstructed_forget_ids."""

    def test_consistent_reconstructed_ids(self) -> None:
        """target_reconstructed=True with non-empty reconstructed_forget_ids passes."""
        result = _valid_result()
        result.turns = [
            TurnResult(
                turn_id=0,
                phase="POST_FORGET_ATTACK",
                sender_id="A",
                recipient_id="B",
                candidate_text="test",
                released_text="test",
                is_reconstruction_attempt=True,
                reconstructed_forget_ids=("F001",),
                target_reconstructed=True,
            )
        ]
        findings = audit_episode_result(result)
        consistency = [f for f in findings if f.code == "RECONSTRUCTED_IDS_CONSISTENCY"]
        assert len(consistency) == 0

    def test_inconsistent_reconstructed_ids(self) -> None:
        """target_reconstructed=True but empty reconstructed_forget_ids fails."""
        result = _valid_result()
        result.turns = [
            TurnResult(
                turn_id=0,
                phase="POST_FORGET_ATTACK",
                sender_id="A",
                recipient_id="B",
                candidate_text="test",
                released_text="test",
                is_reconstruction_attempt=True,
                reconstructed_forget_ids=(),
                target_reconstructed=True,
            )
        ]
        findings = audit_episode_result(result)
        consistency = [f for f in findings if f.code == "RECONSTRUCTED_IDS_CONSISTENCY"]
        assert len(consistency) == 1
        assert consistency[0].level == "error"


class TestSchemaAwareAudit:
    """Section 7: Schema-aware historical audit."""

    def test_legacy_schema_skips_record_id_checks(self) -> None:
        """Legacy schema (< 1.0) should not enforce record ID consistency."""
        result = _valid_result()
        result.schema_version = "0.9"
        result.turns = [
            TurnResult(
                turn_id=0,
                phase="POST_FORGET_ATTACK",
                sender_id="A",
                recipient_id="B",
                candidate_text="test",
                released_text="test",
                is_recontamination_attempt=True,
                reintroduced_forget_ids=(),
                target_reintroduced=True,
            )
        ]
        findings = audit_episode_result(result)
        # Should NOT have REINTRODUCED_IDS_CONSISTENCY error for legacy schema
        consistency = [f for f in findings if f.code == "REINTRODUCED_IDS_CONSISTENCY"]
        assert len(consistency) == 0

    def test_legacy_schema_warns_on_reconstructed_without_ids(self) -> None:
        """Legacy schema with target_reconstructed but no IDs gets warning."""
        result = _valid_result()
        result.schema_version = "0.9"
        result.turns = [
            TurnResult(
                turn_id=0,
                phase="POST_FORGET_ATTACK",
                sender_id="A",
                recipient_id="B",
                candidate_text="test",
                released_text="test",
                is_reconstruction_attempt=True,
                reconstructed_forget_ids=(),
                target_reconstructed=True,
            )
        ]
        findings = audit_episode_result(result)
        legacy = [f for f in findings if f.code == "LEGACY_SCHEMA_MISSING_RECORD_IDS"]
        assert len(legacy) == 1
        assert legacy[0].level == "warning"

    def test_current_schema_enforces_record_id_checks(self) -> None:
        """Schema >= 1.1 should enforce record ID consistency."""
        result = _valid_result()
        result.schema_version = "1.1"
        result.turns = [
            TurnResult(
                turn_id=0,
                phase="POST_FORGET_ATTACK",
                sender_id="A",
                recipient_id="B",
                candidate_text="test",
                released_text="test",
                is_recontamination_attempt=True,
                reintroduced_forget_ids=(),
                target_reintroduced=True,
            )
        ]
        findings = audit_episode_result(result)
        consistency = [f for f in findings if f.code == "REINTRODUCED_IDS_CONSISTENCY"]
        assert len(consistency) == 1
        assert consistency[0].level == "error"
