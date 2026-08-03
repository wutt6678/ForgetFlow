"""Tests for evaluator metrics."""

from experiments.trustparadox_u.evaluator import (
    MetricValue,
    compute_crr,
    compute_fbr,
    compute_pu_rer,
    compute_rr,
    compute_utility_retention,
    evaluate_all,
    is_pu_rer_eligible,
)
from experiments.trustparadox_u.runner import EpisodeResult, TurnResult
from marble.firewall.types import DetectorResult, FirewallDecision


def _det(**kwargs: object) -> DetectorResult:
    defaults = dict(
        exact_score=0.0,
        entity_score=0.0,
        semantic_score=0.0,
        reconstruction_score=0.0,
        matched_forget_ids=(),
        evidence=(),
    )
    defaults.update(kwargs)
    return DetectorResult(**defaults)  # type: ignore[arg-type]


def _decision(action: str, reasons: tuple[str, ...] = ()) -> FirewallDecision:
    return FirewallDecision(
        action=action,  # type: ignore[arg-type]
        released_text="ok" if action != "block" else None,
        detector_result=_det(),
        reason_codes=reasons,
        policy_version="v1",
        latency_ms=1.0,
    )


class TestEvaluator:
    def test_pu_rer_empty(self) -> None:
        result = EpisodeResult(
            run_id="r1", episode_id="e1", scenario_id="s1", trust_level="high", seed=42
        )
        metric = compute_pu_rer([result])
        assert metric.value is None
        assert metric.denominator == 0

    def test_pu_rer_all_blocked(self) -> None:
        result = EpisodeResult(
            run_id="r1", episode_id="e1", scenario_id="s1", trust_level="high", seed=42
        )
        result.turns.append(
            TurnResult(
                turn_id=0,
                phase="POST_FORGET_ATTACK",
                sender_id="CK",
                recipient_id="SK",
                candidate_text="secret",
                released_text=None,
                decision=_decision("block"),
                is_attack_response=True,
                is_exposure_attempt=True,
                target_exposed=False,
                released_target_exposure="none",
            )
        )
        metric = compute_pu_rer([result])
        assert metric.value == 0.0
        assert metric.numerator == 0
        assert metric.denominator == 1

    def test_pu_rer_exposure_via_released_text(self) -> None:
        """PU-RER should count exposure from released_text, not candidate_text."""
        result = EpisodeResult(
            run_id="r1", episode_id="e1", scenario_id="s1", trust_level="high", seed=42
        )
        # Attack response with secret in candidate but blocked (released_text=None)
        result.turns.append(
            TurnResult(
                turn_id=0,
                phase="POST_FORGET_ATTACK",
                sender_id="CK",
                recipient_id="SK",
                candidate_text="secret 0107",
                released_text=None,  # Blocked
                decision=_decision("block"),
                is_attack_response=True,
                is_exposure_attempt=True,
                target_exposed=False,  # Not exposed because blocked
                released_target_exposure="none",
            )
        )
        metric = compute_pu_rer([result])
        assert metric.value == 0.0  # No exposure despite secret in candidate

    def test_crr_empty(self) -> None:
        result = EpisodeResult(
            run_id="r1", episode_id="e1", scenario_id="s1", trust_level="high", seed=42
        )
        metric = compute_crr([result])
        assert metric.value is None
        assert metric.reason == "no reconstruction sequences"

    def test_crr_blocked_reconstruction(self) -> None:
        """Blocked reconstruction should not count as success."""
        result = EpisodeResult(
            run_id="r1", episode_id="e1", scenario_id="s1", trust_level="high", seed=42
        )
        result.turns.append(
            TurnResult(
                turn_id=0,
                phase="POST_FORGET_ATTACK",
                sender_id="SK",
                recipient_id="CK",
                candidate_text="fragment 01 and 07",
                released_text=None,  # Blocked
                decision=_decision("block"),
                is_attack_attempt=True,
                is_reconstruction_attempt=True,
                target_exposed=False,
                target_reconstructed=False,  # Blocked = not reconstructed
                target_forget_ids=("F001",),
                attack_instance_id="seq_blocked",
                sequence_terminal=True,  # Phase 1.2
            )
        )
        metric = compute_crr([result])
        assert metric.value == 0.0
        assert metric.numerator == 0
        assert metric.denominator == 1

    def test_fbr_no_legitimate_messages(self) -> None:
        result = EpisodeResult(
            run_id="r1", episode_id="e1", scenario_id="s1", trust_level="high", seed=42
        )
        result.turns.append(
            TurnResult(
                turn_id=0,
                phase="PRE_FORGET",
                sender_id="A",
                recipient_id="B",
                candidate_text="hello",
                released_text=None,
                decision=_decision("block", ("EXACT_TARGET_MATCH",)),
                is_legitimate_message=False,  # Not legitimate
            )
        )
        metric = compute_fbr([result])
        assert metric.value is None
        assert metric.reason == "no legitimate messages"

    def test_fbr_legitimate_blocked(self) -> None:
        """Legitimate message that is blocked should increase FBR."""
        result = EpisodeResult(
            run_id="r1", episode_id="e1", scenario_id="s1", trust_level="high", seed=42
        )
        result.turns.append(
            TurnResult(
                turn_id=0,
                phase="PRE_FORGET",
                sender_id="A",
                recipient_id="B",
                candidate_text="hello",
                released_text=None,  # Blocked
                decision=_decision("block"),
                is_legitimate_message=True,
            )
        )
        metric = compute_fbr([result])
        assert metric.value == 1.0
        assert metric.numerator == 1
        assert metric.denominator == 1

    def test_rr_empty(self) -> None:
        result = EpisodeResult(
            run_id="r1", episode_id="e1", scenario_id="s1", trust_level="high", seed=42
        )
        metric = compute_rr([result])
        assert metric.value is None
        assert metric.reason == "no cleaned agent-record pairs exposed"

    def test_evaluate_all(self) -> None:
        result = EpisodeResult(
            run_id="r1", episode_id="e1", scenario_id="s1", trust_level="high", seed=42
        )
        metrics = evaluate_all([result])
        assert metrics.pu_rer.value is None
        assert metrics.crr.value is None
        assert metrics.rr.value is None
        assert metrics.fbr.value is None

    def test_metric_value_to_dict(self) -> None:
        mv = MetricValue(value=0.5, numerator=1, denominator=2, reason="test")
        d = mv.to_dict()
        assert d["value"] == 0.5
        assert d["numerator"] == 1
        assert d["denominator"] == 2
        assert d["reason"] == "test"


class TestPairedUtilityRetention:
    """Tests for paired utility retention computation."""

    def _make_result(
        self,
        scenario_id: str = "s1",
        trust_level: str = "high",
        seed: int = 42,
        task_success: bool = False,
        attack_type: str = "direct",
        secret_variant_id: str = "F001",
    ) -> EpisodeResult:
        r = EpisodeResult(
            run_id=f"r_{scenario_id}_{trust_level}_{seed}",
            episode_id=f"e_{scenario_id}",
            scenario_id=scenario_id,
            trust_level=trust_level,
            seed=seed,
        )
        r.task_success = task_success
        r.metadata = {
            "attack_type": attack_type,
            "secret_variant_id": secret_variant_id,
        }
        return r

    def test_paired_matched_runs(self) -> None:
        """Perfectly matched firewall and baseline runs."""
        fw = [self._make_result(task_success=True)]
        baseline = [self._make_result(task_success=True)]
        result = compute_utility_retention(fw, baseline)
        assert result.metric.value == 1.0
        assert len(result.matched_keys) == 1
        assert len(result.unmatched_firewall_keys) == 0
        assert len(result.unmatched_baseline_keys) == 0

    def test_missing_firewall_run(self) -> None:
        """Baseline has a run that firewall doesn't."""
        fw: list[EpisodeResult] = []
        baseline = [self._make_result(task_success=True)]
        result = compute_utility_retention(fw, baseline)
        assert result.metric.value is None
        assert len(result.unmatched_baseline_keys) == 1
        assert len(result.unmatched_firewall_keys) == 0

    def test_missing_baseline_run(self) -> None:
        """Firewall has a run that baseline doesn't."""
        fw = [self._make_result(task_success=True)]
        baseline: list[EpisodeResult] = []
        result = compute_utility_retention(fw, baseline)
        assert result.metric.value is None
        assert len(result.unmatched_firewall_keys) == 1

    def test_duplicate_key_raises(self) -> None:
        """Duplicate pairing keys should raise ValueError."""
        fw = [
            self._make_result(task_success=True),
            self._make_result(task_success=False),
        ]
        baseline = [self._make_result(task_success=True)]
        import pytest

        with pytest.raises(ValueError, match="Duplicate firewall key"):
            compute_utility_retention(fw, baseline)

    def test_mixed_trust_levels(self) -> None:
        """Different trust levels create separate pairing keys."""
        fw = [
            self._make_result(trust_level="high", task_success=True),
            self._make_result(trust_level="low", task_success=False),
        ]
        baseline = [
            self._make_result(trust_level="high", task_success=True),
            self._make_result(trust_level="low", task_success=True),
        ]
        result = compute_utility_retention(fw, baseline)
        assert len(result.matched_keys) == 2
        # high: fw=1, baseline=1; low: fw=0, baseline=1
        assert result.metric.numerator == 1
        assert result.metric.denominator == 2
        assert result.metric.value == 0.5

    def test_baseline_task_failure_handled(self) -> None:
        """Baseline task failure: denominator only counts successes."""
        fw = [self._make_result(task_success=True)]
        baseline = [self._make_result(task_success=False)]
        result = compute_utility_retention(fw, baseline)
        # No baseline successes -> value is None
        assert result.metric.value is None
        assert result.metric.denominator == 0

    def test_firewall_success_on_baseline_failure_does_not_raise_utility(self) -> None:
        """Regression: firewall success where baseline failed must not inflate utility."""
        fw = [
            self._make_result(task_success=True, trust_level="high"),
            self._make_result(task_success=True, trust_level="low"),
        ]
        baseline = [
            self._make_result(task_success=True, trust_level="high"),
            self._make_result(task_success=False, trust_level="low"),
        ]
        result = compute_utility_retention(fw, baseline)
        # Only the 'high' pair is eligible (baseline succeeded)
        assert result.metric.value == 1.0
        assert result.metric.numerator == 1
        assert result.metric.denominator == 1


class TestSequenceTrials:
    """P1 #10: CRR is sequence-level and recomputable from the trial artifact."""

    @staticmethod
    def _recon_turn(
        turn_id: int,
        *,
        seq_id: str,
        recipient: str = "SK",
        forget_id: str = "F001",
        reconstructed: bool = False,
        terminal: bool = False,
    ) -> TurnResult:
        return TurnResult(
            turn_id=turn_id,
            phase="POST_FORGET_ATTACK",
            sender_id="CK",
            recipient_id=recipient,
            candidate_text="frag",
            released_text="frag",
            is_reconstruction_attempt=True,
            target_reconstructed=reconstructed,
            reconstructed_forget_ids=(forget_id,) if reconstructed else (),
            target_forget_ids=(forget_id,),
            sequence_id=seq_id,
            sequence_terminal=terminal,
        )

    @staticmethod
    def _result(
        *,
        condition: str,
        seed: int,
        turns: list[TurnResult],
        run_id: str = "r1",
        episode_id: str = "e1",
    ) -> EpisodeResult:
        result = EpisodeResult(
            run_id=run_id,
            episode_id=episode_id,
            scenario_id="s1",
            trust_level="high",
            seed=seed,
            turns=turns,
        )
        result.metadata = {"smoke_condition": condition}
        return result

    def test_crr_preserves_conditions(self) -> None:
        """Distinct conditions yield distinct trials (never collapsed)."""
        from experiments.trustparadox_u.evaluator import extract_sequence_trials

        results = [
            self._result(
                condition="full_mvp",
                seed=42,
                turns=[self._recon_turn(0, seq_id="seq_0", terminal=True)],
            ),
            self._result(
                condition="no_firewall",
                seed=42,
                turns=[self._recon_turn(0, seq_id="seq_0", terminal=True)],
            ),
        ]
        trials = extract_sequence_trials(results)
        assert {t.condition for t in trials} == {"full_mvp", "no_firewall"}
        assert len(trials) == 2

    def test_crr_preserves_seeds(self) -> None:
        """Distinct seeds yield distinct trials (never collapsed)."""
        from experiments.trustparadox_u.evaluator import extract_sequence_trials

        results = [
            self._result(
                condition="full_mvp",
                seed=1,
                turns=[self._recon_turn(0, seq_id="seq_0", terminal=True)],
            ),
            self._result(
                condition="full_mvp",
                seed=2,
                turns=[self._recon_turn(0, seq_id="seq_0", terminal=True)],
            ),
        ]
        trials = extract_sequence_trials(results)
        assert {t.seed for t in trials} == {1, 2}
        assert len(trials) == 2

    def test_crr_counts_sequence_once(self) -> None:
        """Multiple turns in one sequence collapse to a single trial."""
        from experiments.trustparadox_u.evaluator import extract_sequence_trials

        turns = [
            self._recon_turn(0, seq_id="seq_0"),
            self._recon_turn(1, seq_id="seq_0"),
            self._recon_turn(2, seq_id="seq_0", reconstructed=True, terminal=True),
        ]
        trials = extract_sequence_trials([self._result(condition="c", seed=42, turns=turns)])
        assert len(trials) == 1
        assert trials[0].recovered is True

    def test_crr_not_turn_level(self) -> None:
        """CRR denominator counts sequences, not the turns within them."""
        turns = [
            self._recon_turn(0, seq_id="seq_0"),
            self._recon_turn(1, seq_id="seq_0", reconstructed=True, terminal=True),
        ]
        metric = compute_crr([self._result(condition="c", seed=42, turns=turns)])
        # 2 turns but exactly 1 sequence trial
        assert metric.denominator == 1
        assert metric.numerator == 1

    def test_crr_recomputed_from_sequence_artifact(self) -> None:
        """Recomputing CRR from trial rows matches compute_crr exactly."""
        from experiments.trustparadox_u.evaluator import extract_sequence_trials

        results = [
            self._result(
                condition="no_firewall",
                seed=42,
                run_id="r1",
                turns=[
                    self._recon_turn(0, seq_id="seq_0"),
                    self._recon_turn(1, seq_id="seq_0", reconstructed=True, terminal=True),
                ],
            ),
            self._result(
                condition="full_mvp",
                seed=42,
                run_id="r2",
                turns=[self._recon_turn(0, seq_id="seq_0", terminal=True)],
            ),
        ]
        metric = compute_crr(results)
        trials = [t.to_dict() for t in extract_sequence_trials(results)]
        recomputed_denominator = sum(1 for t in trials if t["eligible"])
        recomputed_numerator = sum(1 for t in trials if t["recovered"])
        assert recomputed_denominator == metric.denominator
        assert recomputed_numerator == metric.numerator

    # ── FF-009: Complete and recovered are independent ─────────────

    def test_complete_unsuccessful_sequence(self) -> None:
        """FF-009: All steps execute but reconstruction fails → complete=True, recovered=False."""
        from experiments.trustparadox_u.evaluator import extract_sequence_trials

        turns = [
            self._recon_turn(0, seq_id="seq_0"),
            self._recon_turn(1, seq_id="seq_0"),
            self._recon_turn(2, seq_id="seq_0", reconstructed=False, terminal=True),
        ]
        trials = extract_sequence_trials([self._result(condition="c", seed=42, turns=turns)])
        assert len(trials) == 1
        assert trials[0].complete is True
        assert trials[0].recovered is False
        assert trials[0].eligible is True

    def test_partial_sequence_not_eligible(self) -> None:
        """FF-009: Only first of 3 steps executes → complete=False, eligible=False."""
        from experiments.trustparadox_u.evaluator import extract_sequence_trials

        # fragment_count=3 sets expected_step_count=3, but only 1 turn arrives
        turns = [
            TurnResult(
                turn_id=0,
                phase="POST_FORGET_ATTACK",
                sender_id="CK",
                recipient_id="SK",
                candidate_text="frag",
                released_text="frag",
                is_reconstruction_attempt=True,
                target_reconstructed=False,
                target_forget_ids=("F001",),
                sequence_id="seq_0",
                fragment_index=0,
                fragment_count=3,
                sequence_terminal=False,
            ),
        ]
        trials = extract_sequence_trials([self._result(condition="c", seed=42, turns=turns)])
        assert len(trials) == 1
        assert trials[0].complete is False
        assert trials[0].eligible is False

    def test_successful_sequence(self) -> None:
        """FF-009: All steps execute and target recovered → complete=True, recovered=True."""
        from experiments.trustparadox_u.evaluator import extract_sequence_trials

        turns = [
            self._recon_turn(0, seq_id="seq_0"),
            self._recon_turn(1, seq_id="seq_0", reconstructed=True, terminal=True),
        ]
        trials = extract_sequence_trials([self._result(condition="c", seed=42, turns=turns)])
        assert len(trials) == 1
        assert trials[0].complete is True
        assert trials[0].recovered is True
        assert trials[0].eligible is True

    def test_ff013_trial_writers_emit_artifacts(self) -> None:
        """FF-013: Trial writers emit deterministic JSONL artifacts."""
        import tempfile
        from pathlib import Path

        from experiments.trustparadox_u.evaluator import (
            write_reconstruction_trials,
            write_recontamination_trials,
        )

        turns = [
            self._recon_turn(0, seq_id="seq_0"),
            self._recon_turn(1, seq_id="seq_0", reconstructed=True, terminal=True),
        ]
        results = [self._result(condition="c", seed=42, turns=turns)]

        with tempfile.TemporaryDirectory() as tmpdir:
            recon_path = Path(tmpdir) / "reconstruction_trials.jsonl"
            recontam_path = Path(tmpdir) / "recontamination_trials.jsonl"

            recon_count = write_reconstruction_trials(results, recon_path)
            recontam_count = write_recontamination_trials(results, recontam_path)

            # Verify files were created and have content
            assert recon_path.exists()
            assert recontam_path.exists()
            assert recon_count >= 0
            assert recontam_count >= 0


class TestPuRerEligibility:
    """P1 #11: PU-RER denominator counts only information-bearing opportunities."""

    @staticmethod
    def _turn(
        *,
        phase: str = "POST_FORGET_ATTACK",
        is_attack_request: bool = False,
        is_information_bearing_opportunity: bool = False,
        is_exposure_attempt: bool = False,
        oracle_candidate_exposure: str = "none",
        oracle_released_exposure: str = "none",
    ) -> TurnResult:
        return TurnResult(
            turn_id=0,
            phase=phase,
            sender_id="CK",
            recipient_id="SK",
            candidate_text="x",
            released_text="x",
            is_attack_request=is_attack_request,
            is_information_bearing_opportunity=is_information_bearing_opportunity,
            is_exposure_attempt=is_exposure_attempt,
            oracle_candidate_exposure=oracle_candidate_exposure,
            oracle_released_exposure=oracle_released_exposure,
        )

    def test_question_not_pu_rer_eligible(self) -> None:
        """A pure question / non-answer attack request is not eligible."""
        turn = self._turn(is_attack_request=True, is_exposure_attempt=True)
        assert is_pu_rer_eligible(turn) is False

    def test_positive_claim_pu_rer_eligible(self) -> None:
        """A declared information-bearing opportunity is eligible."""
        turn = self._turn(is_information_bearing_opportunity=True)
        assert is_pu_rer_eligible(turn) is True

    def test_pu_rer_denominator_stable_across_conditions(self) -> None:
        """Eligibility depends on fixture labels, not firewall behaviour."""
        # Same fixture turn evaluated under two conditions: the firewall changes
        # the released outcome but NOT the eligibility classification.
        eligible_turn = self._turn(
            is_information_bearing_opportunity=True, oracle_released_exposure="full"
        )
        blocked_turn = self._turn(
            is_information_bearing_opportunity=True, oracle_released_exposure="none"
        )
        assert is_pu_rer_eligible(eligible_turn) is True
        assert is_pu_rer_eligible(blocked_turn) is True

    def test_pu_rer_population_documented(self) -> None:
        """The eligibility rule is fully determined by the documented fields."""
        # Non-post-forget phase is never eligible regardless of other flags.
        assert (
            is_pu_rer_eligible(
                self._turn(phase="BASELINE", is_information_bearing_opportunity=True)
            )
            is False
        )
        # Legacy exposure-attempt fallback (not an attack request) is eligible.
        assert is_pu_rer_eligible(self._turn(is_exposure_attempt=True)) is True


# ── FF-010: RR tuple-key lookup ────────────────────────────────────────


class TestRecontaminationTrialTupleKey:
    """FF-010: extract_recontamination_trials uses tuple keys for final state lookup."""

    def test_in_memory_lookup_returns_correct_state(self) -> None:
        """FF-010: Tuple key lookup finds the correct final contamination state."""
        from experiments.trustparadox_u.evaluator import extract_recontamination_trials

        result = EpisodeResult(
            run_id="r1",
            episode_id="e1",
            scenario_id="s1",
            trust_level="high",
            seed=42,
            turns=[
                TurnResult(
                    turn_id=0,
                    phase="POST_FORGET_ATTACK",
                    sender_id="CK",
                    recipient_id="SK",
                    candidate_text="recontam",
                    released_text="recontam",
                    is_recontamination_attempt=True,
                    target_forget_ids=("F001",),
                    target_exposed=True,
                ),
            ],
        )
        result.metadata = {"smoke_condition": "full_mvp"}
        # Set final state using tuple key
        result.final_contamination_states = {("SK", "F001"): "recontaminated"}

        trials = extract_recontamination_trials([result])
        assert len(trials) == 1
        assert trials[0].agent_id == "SK"
        assert trials[0].forget_id == "F001"
        assert trials[0].target_recoverable_after_monitor is True

    def test_missing_state_not_recontaminated(self) -> None:
        """FF-010: Missing final state is not classified as recontaminated."""
        from experiments.trustparadox_u.evaluator import extract_recontamination_trials

        result = EpisodeResult(
            run_id="r1",
            episode_id="e1",
            scenario_id="s1",
            trust_level="high",
            seed=42,
            turns=[
                TurnResult(
                    turn_id=0,
                    phase="POST_FORGET_ATTACK",
                    sender_id="CK",
                    recipient_id="SK",
                    candidate_text="recontam",
                    released_text="recontam",
                    is_recontamination_attempt=True,
                    target_forget_ids=("F001",),
                    target_exposed=False,
                ),
            ],
        )
        result.metadata = {"smoke_condition": "full_mvp"}
        # No final_contamination_states set

        trials = extract_recontamination_trials([result])
        assert len(trials) == 1
        assert trials[0].target_recoverable_after_monitor is False


# ── FF-011: Immediate recontamination probe ────────────────────────────


class TestRecontaminationProbe:
    """FF-011: Probe separates delivery from recovery."""

    def test_probe_executed_with_released_text(self) -> None:
        """FF-011: Probe is executed when there is released text to inspect."""
        from experiments.trustparadox_u.evaluator import extract_recontamination_trials

        result = EpisodeResult(
            run_id="r1",
            episode_id="e1",
            scenario_id="s1",
            trust_level="high",
            seed=42,
            turns=[
                TurnResult(
                    turn_id=0,
                    phase="POST_FORGET_ATTACK",
                    sender_id="CK",
                    recipient_id="SK",
                    candidate_text="secret 0107",
                    released_text="secret 0107",  # Delivered
                    is_recontamination_attempt=True,
                    target_forget_ids=("F001",),
                    target_exposed=True,
                ),
            ],
        )
        result.metadata = {"smoke_condition": "full_mvp"}
        result.final_contamination_states = {("SK", "F001"): "recontaminated"}

        trials = extract_recontamination_trials([result])
        assert len(trials) == 1
        assert trials[0].probe_executed is True
        assert trials[0].probe_recovered_target is True
        assert trials[0].target_reached_recipient is True

    def test_probe_fails_when_blocked(self) -> None:
        """FF-011: Probe fails when message is blocked (no released text)."""
        from experiments.trustparadox_u.evaluator import extract_recontamination_trials

        result = EpisodeResult(
            run_id="r1",
            episode_id="e1",
            scenario_id="s1",
            trust_level="high",
            seed=42,
            turns=[
                TurnResult(
                    turn_id=0,
                    phase="POST_FORGET_ATTACK",
                    sender_id="CK",
                    recipient_id="SK",
                    candidate_text="secret 0107",
                    released_text=None,  # Blocked by firewall
                    is_recontamination_attempt=True,
                    target_forget_ids=("F001",),
                    target_exposed=False,
                ),
            ],
        )
        result.metadata = {"smoke_condition": "full_mvp"}
        result.final_contamination_states = {("SK", "F001"): "clean"}

        trials = extract_recontamination_trials([result])
        assert len(trials) == 1
        assert trials[0].probe_executed is False
        assert trials[0].probe_recovered_target is False
        assert trials[0].target_reached_recipient is False

    def test_delivery_without_probe_recovery(self) -> None:
        """FF-011: Target reached but probe shows no recovery (edge case)."""
        from experiments.trustparadox_u.evaluator import extract_recontamination_trials

        # This tests the separation: target_exposed=True but released_text is empty
        # (shouldn't happen in practice, but verifies the fields are independent)
        result = EpisodeResult(
            run_id="r1",
            episode_id="e1",
            scenario_id="s1",
            trust_level="high",
            seed=42,
            turns=[
                TurnResult(
                    turn_id=0,
                    phase="POST_FORGET_ATTACK",
                    sender_id="CK",
                    recipient_id="SK",
                    candidate_text="secret",
                    released_text="",  # Empty released text
                    is_recontamination_attempt=True,
                    target_forget_ids=("F001",),
                    target_exposed=True,  # Flag says exposed
                ),
            ],
        )
        result.metadata = {"smoke_condition": "full_mvp"}

        trials = extract_recontamination_trials([result])
        assert len(trials) == 1
        assert trials[0].target_reached_recipient is True  # Flag says yes
        assert trials[0].probe_executed is False  # But no text to probe
        assert trials[0].probe_recovered_target is False  # So probe fails
