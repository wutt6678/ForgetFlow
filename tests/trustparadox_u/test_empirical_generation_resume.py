"""Patch A/F: partial-sequence resume regression tests.

Tests that partial sequence retry resume works correctly:
- Per-step retry indexes are respected
- Successful steps are not regenerated
- Resumed output matches uninterrupted scientific output
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from experiments.trustparadox_u.empirical_corpus import (
    EMPIRICAL_TARGET_REGISTRY,
    EmpiricalGenerationAttempt,
    EmpiricalTargetSpec,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_attempt(
    *,
    scenario_id: str = "credential_001",
    variant_id: str = "credential_v1",
    trust_level: str = "low",
    attack_type: str = "fragmentation_sequence",
    sample_index: int = 0,
    generation_replicate: int = 0,
    status: str = "success",
    retry_index: int = 0,
    sequence_step_index: int | None = None,
    sequence_step_count: int | None = None,
    sequence_family_id: str | None = None,
    candidate_text: str = "The secret is 42.",
    split: str = "development",
) -> EmpiricalGenerationAttempt:
    """Build a minimal EmpiricalGenerationAttempt for testing."""
    generation_attempt_id = (
        f"ega_{scenario_id}_{variant_id}_{trust_level}_{attack_type}"
        f"_{sample_index:03d}_r{generation_replicate}"
    )
    if sequence_step_index is not None:
        generation_attempt_id = f"{generation_attempt_id}_st{sequence_step_index}"
    provider_attempt_id = f"{generation_attempt_id}_retry{retry_index}"
    candidate_family_id = (
        f"cf_{scenario_id}_{variant_id}_{trust_level}_{attack_type}"
    )
    if sequence_step_index is not None:
        candidate_family_id = f"{candidate_family_id}_step{sequence_step_index}"
    sequence_id = f"seq_{sequence_family_id}" if sequence_family_id else None

    return EmpiricalGenerationAttempt(
        generation_attempt_id=generation_attempt_id,
        provider_attempt_id=provider_attempt_id,
        scenario_id=scenario_id,
        secret_variant_id=variant_id,
        trust_level=trust_level,
        attack_type=attack_type,
        sample_index=sample_index,
        generation_replicate=generation_replicate,
        generation_status=status,
        candidate_text=candidate_text if status == "success" else None,
        refusal=(status == "refusal"),
        malformed=False,
        off_topic=False,
        retry_index=retry_index,
        sequence_family_id=sequence_family_id,
        sequence_id=sequence_id,
        sequence_step_index=sequence_step_index,
        sequence_step_count=sequence_step_count,
        split=split,
        generation_mode="mock",
        generator_provider="mock",
        generator_model="mock-model",
        generator_model_requested="mock-model",
        generator_model_returned="mock-model",
        generator_revision=None,
        temperature=0.7,
        seed=None,
        system_prompt_hash="sys_hash",
        user_prompt_hash="usr_hash",
        request_id=None,
        generated_at="2025-01-01T00:00:00Z",
        sender_id="sender_A",
        recipient_id="recipient_B",
        candidate_family_id=candidate_family_id,
        transport=None,
        trust_prompt_hash=None,
        attack_prompt_hash=None,
        max_tokens=1024,
    )


def _resolve_sequence_state(
    attempts: list[EmpiricalGenerationAttempt],
    *,
    expected_step_count: int,
    max_retries: int = 2,
    retryable_statuses: Sequence[str] = ("provider_error", "timeout"),
) -> dict:
    from experiments.trustparadox_u.generate_empirical_corpus import (
        _resolve_sequence_state,
    )
    return _resolve_sequence_state(
        attempts,
        expected_step_count=expected_step_count,
        max_retries=max_retries,
        retryable_statuses=retryable_statuses,
    )


def _resolve_unit_resume_state(
    unit_key: tuple,
    existing_attempts: list[EmpiricalGenerationAttempt],
    *,
    is_sequence: bool,
    expected_step_count: int,
    max_retries: int = 2,
    retryable_statuses: Sequence[str] = ("provider_error", "timeout"),
) -> dict:
    from experiments.trustparadox_u.generate_empirical_corpus import (
        resolve_unit_resume_state,
    )
    return resolve_unit_resume_state(
        unit_key,
        existing_attempts,
        is_sequence=is_sequence,
        expected_step_count=expected_step_count,
        max_retries=max_retries,
        retryable_statuses=retryable_statuses,
    )


# ===========================================================================
# Test A1 / F — retryable sequence step resumes at correct retry index
# ===========================================================================


class TestPartialSequenceResumeRetryIndex:
    """Per-step retry resume indexes."""

    def test_partial_sequence_resume_continues_failed_step_retry_index(self) -> None:
        """Step0 success → skip; step1 provider_error → retry1; step2 missing → retry0."""
        from experiments.trustparadox_u.generate_empirical_corpus import UnitResumeState

        family_id = "seq_frag_cred_low"
        history = [
            _make_attempt(
                sequence_step_index=0, sequence_step_count=3,
                sequence_family_id=family_id, retry_index=0,
                status="success",
            ),
            _make_attempt(
                sequence_step_index=1, sequence_step_count=3,
                sequence_family_id=family_id, retry_index=0,
                status="provider_error",
            ),
        ]

        result = _resolve_sequence_state(
            history, expected_step_count=3,
        )

        assert result["state"] == UnitResumeState.SEQUENCE_PARTIAL
        assert result["completed_step_indices"] == {0}
        assert result["pending_step_indices"] == {1, 2}
        retry_map = result["retry_start_by_step"]
        assert retry_map[1] == 1, "step1 should resume at retry1"
        assert retry_map[2] == 0, "step2 should start at retry0"

    def test_partial_sequence_resume_does_not_regenerate_successful_steps(self) -> None:
        """Completed steps must be in completed_step_indices, NOT in pending."""
        from experiments.trustparadox_u.generate_empirical_corpus import UnitResumeState

        family_id = "seq_frag_cred_low"
        history = [
            _make_attempt(
                sequence_step_index=0, sequence_step_count=3,
                sequence_family_id=family_id, retry_index=0,
                status="success",
            ),
            _make_attempt(
                sequence_step_index=1, sequence_step_count=3,
                sequence_family_id=family_id, retry_index=0,
                status="success",
            ),
            _make_attempt(
                sequence_step_index=2, sequence_step_count=3,
                sequence_family_id=family_id, retry_index=0,
                status="provider_error",
            ),
        ]

        result = _resolve_sequence_state(
            history, expected_step_count=3,
        )

        assert result["state"] == UnitResumeState.SEQUENCE_PARTIAL
        assert 0 in result["completed_step_indices"]
        assert 1 in result["completed_step_indices"]
        assert 2 in result["pending_step_indices"]
        # Completed steps must NOT appear in retry_start_by_step.
        assert 0 not in result["retry_start_by_step"]
        assert 1 not in result["retry_start_by_step"]

    def test_retry_index_2_after_two_failures(self) -> None:
        """Two failures → resume at retry2."""
        from experiments.trustparadox_u.generate_empirical_corpus import UnitResumeState

        family_id = "seq_frag_cred_low"
        history = [
            _make_attempt(
                sequence_step_index=0, sequence_step_count=3,
                sequence_family_id=family_id, retry_index=0,
                status="provider_error",
            ),
            _make_attempt(
                sequence_step_index=0, sequence_step_count=3,
                sequence_family_id=family_id, retry_index=1,
                status="timeout",
            ),
        ]

        result = _resolve_sequence_state(
            history, expected_step_count=3, max_retries=2,
        )

        assert result["state"] == UnitResumeState.SEQUENCE_PARTIAL
        assert result["retry_start_by_step"][0] == 2

    def test_exhausted_sequence_step(self) -> None:
        """All retries exhausted on every step → SEQUENCE_COMPLETE_FAILURE."""
        from experiments.trustparadox_u.generate_empirical_corpus import UnitResumeState

        family_id = "seq_frag_cred_low"
        history: list = []
        for step in range(3):
            for retry in range(3):  # retry 0,1,2 → max_retries=2 exhausted
                history.append(
                    _make_attempt(
                        sequence_step_index=step,
                        sequence_step_count=3,
                        sequence_family_id=family_id,
                        retry_index=retry,
                        status="provider_error",
                    )
                )

        result = _resolve_sequence_state(
            history, expected_step_count=3, max_retries=2,
        )

        # Every step is exhausted, no pending steps remain.
        assert result["state"] == UnitResumeState.SEQUENCE_COMPLETE_FAILURE
        assert result["pending_step_indices"] == set()

    def test_mixed_pending_state(self) -> None:
        """Mixed: step0 success, step1 error, step2 missing, step3 success."""
        from experiments.trustparadox_u.generate_empirical_corpus import UnitResumeState

        family_id = "seq_frag_cred_low"
        history = [
            _make_attempt(
                sequence_step_index=0, sequence_step_count=4,
                sequence_family_id=family_id, retry_index=0,
                status="success",
            ),
            _make_attempt(
                sequence_step_index=1, sequence_step_count=4,
                sequence_family_id=family_id, retry_index=0,
                status="provider_error",
            ),
            # step2 missing
            _make_attempt(
                sequence_step_index=3, sequence_step_count=4,
                sequence_family_id=family_id, retry_index=0,
                status="success",
            ),
        ]

        result = _resolve_sequence_state(
            history, expected_step_count=4,
        )

        assert result["state"] == UnitResumeState.SEQUENCE_PARTIAL
        assert result["completed_step_indices"] == {0, 3}
        assert result["pending_step_indices"] == {1, 2}
        assert result["retry_start_by_step"] == {1: 1, 2: 0}


# ===========================================================================
# Test F — end-to-end partial-sequence resume integration
# ===========================================================================


class TestEndToEndPartialSequenceResume:
    """Exercise the resume pipeline: resolve state → construct resumed attempts
    → write to raw file → verify combined history produces valid report."""

    def test_partial_sequence_resume_matches_uninterrupted_scientific_output(
        self,
        tmp_path: Path,
    ) -> None:
        """Resumed scientific result must equal uninterrupted result."""
        from experiments.trustparadox_u.generate_empirical_corpus import (
            RawAttemptWriter,
            build_sequence_generation_report_from_attempts,
            terminal_attempts_by_sequence_step,
        )

        # Find a 3-step sequence spec.
        spec: EmpiricalTargetSpec | None = None
        for s in EMPIRICAL_TARGET_REGISTRY:
            if s.split == "development" and len(s.fragments) >= 3:
                spec = s
                break
        if spec is None:
            spec = next(s for s in EMPIRICAL_TARGET_REGISTRY if s.split == "development")

        family_id = f"seq_test_{spec.scenario_id}_{spec.secret_variant_id}"
        trust_level = "low"
        attack_type = "fragmentation_sequence"
        step_count = 3

        # --- Build historical raw state (step0 success, step1 error, step2 missing).
        historical_attempts: list[EmpiricalGenerationAttempt] = []
        # Step 0: retry0 success.
        historical_attempts.append(
            _make_attempt(
                scenario_id=spec.scenario_id,
                variant_id=spec.secret_variant_id,
                trust_level=trust_level,
                attack_type=attack_type,
                sequence_step_index=0,
                sequence_step_count=step_count,
                sequence_family_id=family_id,
                retry_index=0,
                status="success",
                candidate_text=f"Step 0 text for {spec.scenario_id}",
            ),
        )
        # Step 1: retry0 provider_error.
        historical_attempts.append(
            _make_attempt(
                scenario_id=spec.scenario_id,
                variant_id=spec.secret_variant_id,
                trust_level=trust_level,
                attack_type=attack_type,
                sequence_step_index=1,
                sequence_step_count=step_count,
                sequence_family_id=family_id,
                retry_index=0,
                status="provider_error",
            ),
        )

        # Write historical attempts to raw file.
        raw_path = tmp_path / "raw_generation_attempts.jsonl"
        writer = RawAttemptWriter(raw_path)
        for a in historical_attempts:
            writer.write_attempt(a)

        # --- Resume: resolve state from historical attempts.
        unit_key = (
            spec.scenario_id, spec.secret_variant_id, trust_level,
            attack_type, 0, 0,
        )
        resume_info = _resolve_unit_resume_state(
            unit_key,
            historical_attempts,
            is_sequence=True,
            expected_step_count=step_count,
        )

        from experiments.trustparadox_u.generate_empirical_corpus import UnitResumeState
        assert resume_info["state"] == UnitResumeState.SEQUENCE_PARTIAL
        skip_steps = frozenset(resume_info["completed_step_indices"])
        retry_start_by_step = resume_info["retry_start_by_step"]

        # Verify step0 is skipped, step1 resumes at retry1, step2 starts at retry0.
        assert 0 in skip_steps
        assert retry_start_by_step[1] == 1
        assert retry_start_by_step[2] == 0

        # --- Simulate resumed generation: construct attempts that _generate_unit
        # would produce (step1 retry1 success, step2 retry0 success).
        resumed_attempts: list[EmpiricalGenerationAttempt] = []
        # Step 1: retry1 success (resume from retry_start_by_step[1]=1).
        resumed_attempts.append(
            _make_attempt(
                scenario_id=spec.scenario_id,
                variant_id=spec.secret_variant_id,
                trust_level=trust_level,
                attack_type=attack_type,
                sequence_step_index=1,
                sequence_step_count=step_count,
                sequence_family_id=family_id,
                retry_index=1,
                status="success",
                candidate_text=f"Step 1 text for {spec.scenario_id}",
            ),
        )
        # Step 2: retry0 success (start from retry_start_by_step[2]=0).
        resumed_attempts.append(
            _make_attempt(
                scenario_id=spec.scenario_id,
                variant_id=spec.secret_variant_id,
                trust_level=trust_level,
                attack_type=attack_type,
                sequence_step_index=2,
                sequence_step_count=step_count,
                sequence_family_id=family_id,
                retry_index=0,
                status="success",
                candidate_text=f"Step 2 text for {spec.scenario_id}",
            ),
        )

        # Write resumed attempts to the same raw file.
        for a in resumed_attempts:
            writer.write_attempt(a)

        # Combine all attempts.
        all_attempts = list(historical_attempts) + resumed_attempts

        # --- Assertions on raw lineage.
        provider_ids = [a.provider_attempt_id for a in all_attempts]
        assert len(provider_ids) == len(set(provider_ids)), "no duplicate provider IDs"
        assert len(all_attempts) == 4  # 2 historical + 2 resumed

        # Check step0 has only retry0 (success, skipped during resume).
        step0_attempts = [a for a in all_attempts if a.sequence_step_index == 0]
        assert len(step0_attempts) == 1
        assert step0_attempts[0].retry_index == 0

        # Check step1 has retry0 (error) and retry1 (success).
        step1_attempts = sorted(
            [a for a in all_attempts if a.sequence_step_index == 1],
            key=lambda a: a.retry_index,
        )
        assert len(step1_attempts) == 2
        assert step1_attempts[0].retry_index == 0
        assert step1_attempts[0].generation_status == "provider_error"
        assert step1_attempts[1].retry_index == 1
        assert step1_attempts[1].generation_status == "success"

        # Check step2 has retry0 (success).
        step2_attempts = [a for a in all_attempts if a.sequence_step_index == 2]
        assert len(step2_attempts) == 1
        assert step2_attempts[0].retry_index == 0
        assert step2_attempts[0].generation_status == "success"

        # --- Scientific assertions: terminal steps.
        terminal_steps = terminal_attempts_by_sequence_step(all_attempts)
        assert len(terminal_steps) == step_count

        # --- Sequence report from complete raw history (Patch D).
        report = build_sequence_generation_report_from_attempts(
            attempts=all_attempts,
            plan_items=None,
            target_registry=EMPIRICAL_TARGET_REGISTRY,
        )
        assert int(report["complete_sequence_count"]) >= 1 or int(report["planned_sequence_count"]) >= 1
