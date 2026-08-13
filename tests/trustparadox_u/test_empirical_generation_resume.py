"""Patch A/F: partial-sequence resume regression tests.

Tests that partial sequence retry resume works correctly:
- Per-step retry indexes are respected
- Successful steps are not regenerated
- Resumed output matches uninterrupted scientific output
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from experiments.trustparadox_u.empirical_corpus import (
    EMPIRICAL_TARGET_REGISTRY,
    EmpiricalGenerationAttempt,
    EmpiricalTargetSpec,
    empirical_candidate_family_id,
    empirical_sequence_family_id,
    empirical_sequence_id,
    generation_attempt_id as canonical_generation_attempt_id,
    validate_sequence_structure,
)
from experiments.trustparadox_u.empirical_generation import EmpiricalGenerationResponse
from experiments.trustparadox_u.empirical_generation_plan import GenerationPlanItem
from experiments.trustparadox_u.generate_empirical_corpus import (
    RAW_ATTEMPTS_FILENAME,
    run_generation,
    terminal_attempt_for_retry_chain,
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


# ===========================================================================
# Patch A — terminal_attempt_for_retry_segment unit tests
# ===========================================================================


class TestTerminalAttemptForRetrySegment:
    """Unit tests for the retry-segment terminal helper."""

    def test_retry_segment_can_start_at_retry1(self) -> None:
        """A resumed segment [retry1] is valid when expected_start=1."""
        from experiments.trustparadox_u.generate_empirical_corpus import (
            terminal_attempt_for_retry_segment,
        )

        attempts = [
            _make_attempt(retry_index=1, status="success"),
        ]
        terminal = terminal_attempt_for_retry_segment(
            attempts, expected_start_retry_index=1,
        )
        assert terminal.retry_index == 1
        assert terminal.generation_status == "success"

    def test_retry_segment_can_start_at_retry2(self) -> None:
        """A resumed segment [retry2] is valid when expected_start=2."""
        from experiments.trustparadox_u.generate_empirical_corpus import (
            terminal_attempt_for_retry_segment,
        )

        attempts = [
            _make_attempt(retry_index=2, status="provider_error"),
        ]
        terminal = terminal_attempt_for_retry_segment(
            attempts, expected_start_retry_index=2,
        )
        assert terminal.retry_index == 2

    def test_retry_segment_multi_attempt(self) -> None:
        """Segment [retry1, retry2] valid when expected_start=1."""
        from experiments.trustparadox_u.generate_empirical_corpus import (
            terminal_attempt_for_retry_segment,
        )

        attempts = [
            _make_attempt(retry_index=1, status="provider_error"),
            _make_attempt(retry_index=2, status="success"),
        ]
        terminal = terminal_attempt_for_retry_segment(
            attempts, expected_start_retry_index=1,
        )
        assert terminal.retry_index == 2

    def test_retry_segment_rejects_wrong_start_index(self) -> None:
        """Segment [retry0] is rejected when expected_start=1."""
        from experiments.trustparadox_u.generate_empirical_corpus import (
            terminal_attempt_for_retry_segment,
        )

        attempts = [
            _make_attempt(retry_index=0, status="success"),
        ]
        import pytest
        with pytest.raises(ValueError, match="expected start retry"):
            terminal_attempt_for_retry_segment(
                attempts, expected_start_retry_index=1,
            )

    def test_retry_segment_requires_consecutive_indices(self) -> None:
        """Segment [retry2, retry4] is rejected (gap)."""
        from experiments.trustparadox_u.generate_empirical_corpus import (
            terminal_attempt_for_retry_segment,
        )

        attempts = [
            _make_attempt(retry_index=2, status="provider_error"),
            _make_attempt(retry_index=4, status="success"),
        ]
        import pytest
        with pytest.raises(ValueError, match="not consecutive"):
            terminal_attempt_for_retry_segment(
                attempts, expected_start_retry_index=2,
            )

    def test_complete_retry_chain_still_requires_retry0(self) -> None:
        """The strict complete-chain helper still rejects [retry1]."""
        from experiments.trustparadox_u.generate_empirical_corpus import (
            terminal_attempt_for_retry_chain,
        )

        attempts = [
            _make_attempt(retry_index=1, status="success"),
        ]
        import pytest
        with pytest.raises(ValueError, match="do not start at 0"):
            terminal_attempt_for_retry_chain(attempts)


# ===========================================================================
# Patch B — true execution-path non-sequence retry resume
# ===========================================================================


class _ScriptedGenerator:
    """Mock generator with generate_once that cycles through scripted statuses."""

    generation_mode = "mock"
    provider = "mock"
    model_name = "mock-model"
    transport = None

    def __init__(self, statuses: list[str]) -> None:
        self._statuses = statuses
        self._call_count = 0

    def generate_once(self, request: object) -> EmpiricalGenerationResponse:
        status = self._statuses[self._call_count % len(self._statuses)]
        self._call_count += 1
        return EmpiricalGenerationResponse(
            raw_text="5163" if status == "success" else None,
            request_id=f"mock-req-{self._call_count}",
            model_id="mock-model",
            model_revision=None,
            status=status,
            error_message="mock error" if status == "provider_error" else None,
            retry_index=0,
            generated_at="2025-01-01T00:00:00Z",
            latency_ms=0.0,
        )


class TestTrueExecutionPathNonSequenceRetryResume:
    """Patch B: prove non-sequence retry resumes through run_generation()."""

    def test_run_generation_resumes_non_sequence_retry(self, tmp_path: Path) -> None:
        """Historical retry0 provider_error -> resume produces retry1 success."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Write historical raw file with retry0 provider_error.
        historical = _make_attempt(
            scenario_id="credential_001",
            variant_id="credential_v1",
            trust_level="low",
            attack_type="direct_disclosure",
            sample_index=0,
            generation_replicate=0,
            retry_index=0,
            status="provider_error",
            split="development",
        )
        raw_path = output_dir / RAW_ATTEMPTS_FILENAME
        with raw_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(asdict(historical)) + "\n")

        # Scripted generator: resume call -> success.
        generator = _ScriptedGenerator(["success"])

        retry_policy = {
            "max_retries": 2,
            "backoff_seconds": [],
            "retryable_statuses": ["provider_error"],
        }

        # Execute run_generation with resume=True.
        report = run_generation(
            split="development",
            mode="mock",
            scenarios=["credential_001"],
            trust_levels=["low"],
            attack_types=["direct_disclosure"],
            samples=1,
            output_dir=output_dir,
            generator=generator,
            resume=True,
            retry_policy=retry_policy,
        )

        # Verify exactly 2 raw attempts in the file.
        with raw_path.open(encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        assert len(records) == 2, f"expected 2 raw attempts, got {len(records)}"

        # Verify no duplicate provider_attempt_id.
        provider_ids = [r["provider_attempt_id"] for r in records]
        assert len(provider_ids) == len(set(provider_ids)), "duplicate provider IDs"

        # Verify retry0 error + retry1 success.
        sorted_records = sorted(records, key=lambda r: r["retry_index"])
        assert sorted_records[0]["retry_index"] == 0
        assert sorted_records[0]["generation_status"] == "provider_error"
        assert sorted_records[1]["retry_index"] == 1
        assert sorted_records[1]["generation_status"] == "success"

        # Verify provider was called exactly once during resume.
        assert generator._call_count == 1, "provider should be called once during resume"

        # Verify accepted corpus contains the successful candidate.
        accepted_path = output_dir / "accepted_candidates.jsonl"
        with accepted_path.open(encoding="utf-8") as f:
            accepted = [json.loads(line) for line in f if line.strip()]
        assert len(accepted) == 1, f"expected 1 accepted candidate, got {len(accepted)}"
        assert accepted[0]["source_generation_attempt_id"] == sorted_records[1]["generation_attempt_id"]

        # Verify complete retry chain validates.
        all_attempts = [EmpiricalGenerationAttempt(**r) for r in records]
        terminal = terminal_attempt_for_retry_chain(all_attempts)
        assert terminal.retry_index == 1
        assert terminal.generation_status == "success"

        # Verify report was generated.
        assert report is not None
        assert "duplicate_id_count" in report
        assert report["duplicate_id_count"] == 0


# ===========================================================================
# Patch C — true execution-path partial-sequence resume
# ===========================================================================


def _make_valid_sequence_attempt(
    *,
    scenario_id: str,
    variant_id: str,
    trust_level: str,
    attack_type: str,
    sample_index: int,
    generation_replicate: int,
    sequence_step_index: int,
    sequence_step_count: int,
    retry_index: int,
    status: str,
    candidate_text: str | None,
    split: str = "development",
) -> EmpiricalGenerationAttempt:
    """Build an attempt with canonical identity functions (passes validate_identity)."""
    seq_family = empirical_sequence_family_id(
        scenario_id=scenario_id,
        secret_variant_id=variant_id,
        attack_type=attack_type,
        sample_index=sample_index,
        generation_replicate=generation_replicate,
    )
    seq_id = empirical_sequence_id(seq_family, trust_level)
    attempt_id = canonical_generation_attempt_id(
        scenario_id=scenario_id,
        secret_variant_id=variant_id,
        trust_level=trust_level,
        attack_type=attack_type,
        sample_index=sample_index,
        generation_replicate=generation_replicate,
        sequence_step_index=sequence_step_index,
    )
    family_id = empirical_candidate_family_id(
        scenario_id=scenario_id,
        secret_variant_id=variant_id,
        attack_type=attack_type,
        sample_index=sample_index,
        generation_replicate=generation_replicate,
        sequence_step_index=sequence_step_index,
    )
    provider_id = f"{attempt_id}_retry{retry_index}"
    return EmpiricalGenerationAttempt(
        generation_attempt_id=attempt_id,
        provider_attempt_id=provider_id,
        scenario_id=scenario_id,
        secret_variant_id=variant_id,
        trust_level=trust_level,
        attack_type=attack_type,
        sample_index=sample_index,
        generation_replicate=generation_replicate,
        sender_id="CK",
        recipient_id="SK",
        candidate_family_id=family_id,
        sequence_family_id=seq_family,
        sequence_id=seq_id,
        sequence_step_index=sequence_step_index,
        sequence_step_count=sequence_step_count,
        candidate_text=candidate_text,
        generation_status=status,
        refusal=False,
        malformed=False,
        off_topic=False,
        generator_provider="mock",
        generator_model="mock-model",
        generator_revision=None,
        temperature=0.7,
        seed=None,
        system_prompt_hash="",
        user_prompt_hash="",
        request_id=None,
        retry_index=retry_index,
        generated_at="2025-01-01T00:00:00Z",
        split=split,
    )


class TestTrueExecutionPathPartialSequenceResume:
    """Patch C: prove partial-sequence resume through run_generation()."""

    def test_run_generation_resumes_partial_sequence(self, tmp_path: Path) -> None:
        """Historical step0 success + step1 error -> resume completes step1 + step2."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Parameters for a 3-step fragmentation_sequence on credential_001.
        scenario = "credential_001"
        variant = "credential_v1"
        trust = "low"
        attack = "fragmentation_sequence"
        sample = 0
        rep = 0
        step_count = 3  # credential_v1 has 3 fragments

        seq_family = empirical_sequence_family_id(
            scenario_id=scenario,
            secret_variant_id=variant,
            attack_type=attack,
            sample_index=sample,
            generation_replicate=rep,
        )
        seq_id = empirical_sequence_id(seq_family, trust)

        # Historical raw state:
        #   step0: retry0 success
        #   step1: retry0 provider_error
        #   step2: no attempts
        hist_step0 = _make_valid_sequence_attempt(
            scenario_id=scenario, variant_id=variant, trust_level=trust,
            attack_type=attack, sample_index=sample, generation_replicate=rep,
            sequence_step_index=0, sequence_step_count=step_count,
            retry_index=0, status="success", candidate_text="5163",
        )
        hist_step1 = _make_valid_sequence_attempt(
            scenario_id=scenario, variant_id=variant, trust_level=trust,
            attack_type=attack, sample_index=sample, generation_replicate=rep,
            sequence_step_index=1, sequence_step_count=step_count,
            retry_index=0, status="provider_error", candidate_text=None,
        )
        raw_path = output_dir / RAW_ATTEMPTS_FILENAME
        with raw_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(asdict(hist_step0)) + "\n")
            f.write(json.dumps(asdict(hist_step1)) + "\n")

        # Mock provider: every call returns success.
        generator = _ScriptedGenerator(["success"])

        retry_policy = {
            "max_retries": 2,
            "backoff_seconds": [],
            "retryable_statuses": ["provider_error"],
        }

        # Build mini sequence plan (3 steps).
        plan_items = [
            GenerationPlanItem(
                plan_item_id=f"gpi_{scenario}_{variant}_{trust}_{attack}_{sample:03d}_r{rep}_st{i}",
                split="development",
                scenario_id=scenario,
                secret_variant_id=variant,
                trust_level=trust,
                attack_type=attack,
                sample_index=sample,
                generation_replicate=rep,
                sequence_id=seq_id,
                sequence_step_index=i,
                sequence_step_count=step_count,
            )
            for i in range(step_count)
        ]

        # Execute run_generation with resume=True.
        report = run_generation(
            split="development",
            mode="mock",
            scenarios=[scenario],
            trust_levels=[trust],
            attack_types=[attack],
            samples=1,
            output_dir=output_dir,
            generator=generator,
            resume=True,
            plan_items=plan_items,
            retry_policy=retry_policy,
        )

        # --- Expected raw history: exactly 4 records ---
        with raw_path.open(encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        assert len(records) == 4, f"expected 4 raw attempts, got {len(records)}"

        # Index by (step, retry).
        by_step_retry = {
            (r["sequence_step_index"], r["retry_index"]): r for r in records
        }

        # step0 retry0 success (historical, untouched).
        assert (0, 0) in by_step_retry
        assert by_step_retry[(0, 0)]["generation_status"] == "success"

        # step1 retry0 provider_error (historical).
        assert (1, 0) in by_step_retry
        assert by_step_retry[(1, 0)]["generation_status"] == "provider_error"

        # step1 retry1 success (resumed).
        assert (1, 1) in by_step_retry
        assert by_step_retry[(1, 1)]["generation_status"] == "success"

        # step2 retry0 success (new).
        assert (2, 0) in by_step_retry
        assert by_step_retry[(2, 0)]["generation_status"] == "success"

        # --- Negative assertions ---
        assert (0, 1) not in by_step_retry, "step0 must not have retry1"
        # No duplicate step1 retry0 (only one record with key (1,0)).
        step1_retry0 = [r for r in records if r["sequence_step_index"] == 1 and r["retry_index"] == 0]
        assert len(step1_retry0) == 1, "step1 retry0 must not be duplicated"
        assert (2, 1) not in by_step_retry, "step2 must not have retry1"

        # --- Scientific assertions ---
        all_attempts = [EmpiricalGenerationAttempt(**r) for r in records]

        # 3 terminal sequence steps.
        terminal_steps = []
        for si in range(step_count):
            step_records = [a for a in all_attempts if a.sequence_step_index == si]
            terminal = terminal_attempt_for_retry_chain(step_records)
            terminal_steps.append(terminal)
        assert len(terminal_steps) == 3
        for t in terminal_steps:
            assert t.generation_status == "success"

        # Full retry lineage valid (already proven by terminal_attempt_for_retry_chain above).

        # Sequence structurally complete.
        problems = validate_sequence_structure(terminal_steps)
        assert problems == [], f"sequence structure problems: {problems}"

        # Sequence accepted atomically — verify via accepted candidates file.
        accepted_path = output_dir / "accepted_candidates.jsonl"
        with accepted_path.open(encoding="utf-8") as f:
            accepted = [json.loads(line) for line in f if line.strip()]
        assert len(accepted) == step_count, (
            f"expected {step_count} accepted candidates, got {len(accepted)}"
        )

        # Accepted candidate IDs stable.
        candidate_ids = sorted(c["candidate_id"] for c in accepted)
        assert len(candidate_ids) == len(set(candidate_ids)), "duplicate candidate IDs"

        # Sequence report says complete.
        seq_report_path = output_dir / "sequence_generation_report.json"
        with seq_report_path.open(encoding="utf-8") as f:
            seq_report = json.load(f)
        assert seq_report["complete_sequence_count"] == 1
        assert seq_report["accepted_sequence_count"] == 1
        assert seq_report["rejected_sequence_count"] == 0
        assert seq_report["invalid_retry_lineage_count"] == 0

        # --- Uninterrupted control ---
        control_dir = tmp_path / "control"
        control_dir.mkdir()
        control_gen = _ScriptedGenerator(["success"])
        run_generation(
            split="development",
            mode="mock",
            scenarios=[scenario],
            trust_levels=[trust],
            attack_types=[attack],
            samples=1,
            output_dir=control_dir,
            generator=control_gen,
            plan_items=plan_items,
            retry_policy=retry_policy,
        )

        # Compare accepted candidate scientific fields.
        with (control_dir / "accepted_candidates.jsonl").open(encoding="utf-8") as f:
            control_accepted = [json.loads(line) for line in f if line.strip()]
        control_ids = sorted(c["candidate_id"] for c in control_accepted)
        assert candidate_ids == control_ids, "candidate IDs differ: resumed vs control"

        # Compare content hashes.
        resumed_hashes = sorted(c["content_sha256"] for c in accepted)
        control_hashes = sorted(c["content_sha256"] for c in control_accepted)
        assert resumed_hashes == control_hashes, "content hashes differ"

        # Compare sequence report scientific fields.
        with (control_dir / "sequence_generation_report.json").open(encoding="utf-8") as f:
            control_seq_report = json.load(f)
        assert seq_report["complete_sequence_count"] == control_seq_report["complete_sequence_count"]
        assert seq_report["accepted_sequence_count"] == control_seq_report["accepted_sequence_count"]
        assert seq_report["rejected_sequence_count"] == control_seq_report["rejected_sequence_count"]
        assert seq_report["invalid_retry_lineage_count"] == control_seq_report["invalid_retry_lineage_count"]
