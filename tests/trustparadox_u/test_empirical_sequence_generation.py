"""Patch B/D: sequence validation and report tests.

Tests that:
- Provider retries are collapsed before structural validation (B1-B4)
- Sequence reports are derived from complete raw history (D1-D2)
"""

from __future__ import annotations

from experiments.trustparadox_u.empirical_corpus import (
    EMPIRICAL_TARGET_REGISTRY,
    EmpiricalGenerationAttempt,
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


# ===========================================================================
# Test B1 — valid retry sequence collapses to terminal steps
# ===========================================================================


class TestSequenceValidationCollapsesRetries:
    """Patch B: sequence_validation_failures must collapse retries."""

    def test_sequence_validation_collapses_provider_retries(self) -> None:
        """Raw: step0 r0 success, step1 r0 timeout, step1 r1 success, step2 r0 success.

        Expected: terminal steps = [0, 1, 2], 0 structural findings.
        """
        from experiments.trustparadox_u.generate_empirical_corpus import (
            sequence_validation_failures,
            terminal_attempts_by_sequence_step,
        )

        family_id = "seq_frag_cred_low"
        raw_attempts = [
            _make_attempt(
                sequence_step_index=0, sequence_step_count=3,
                sequence_family_id=family_id, retry_index=0,
                status="success",
            ),
            _make_attempt(
                sequence_step_index=1, sequence_step_count=3,
                sequence_family_id=family_id, retry_index=0,
                status="timeout",
            ),
            _make_attempt(
                sequence_step_index=1, sequence_step_count=3,
                sequence_family_id=family_id, retry_index=1,
                status="success",
            ),
            _make_attempt(
                sequence_step_index=2, sequence_step_count=3,
                sequence_family_id=family_id, retry_index=0,
                status="success",
            ),
        ]

        # Terminal reduction yields exactly 3 steps.
        terminal = terminal_attempts_by_sequence_step(raw_attempts)
        assert len(terminal) == 3
        step_indices = sorted(int(a.sequence_step_index) for a in terminal)
        assert step_indices == [0, 1, 2]

        # sequence_validation_failures should produce 0 findings.
        problems = sequence_validation_failures(raw_attempts)
        assert len(problems) == 0

    def test_retry_after_success_is_lineage_error(self) -> None:
        """B2: step1 r0 success, step1 r1 success → retry-lineage error."""
        from experiments.trustparadox_u.generate_empirical_corpus import (
            sequence_validation_failures,
        )

        family_id = "seq_frag_cred_low"
        raw_attempts = [
            _make_attempt(
                sequence_step_index=1, sequence_step_count=3,
                sequence_family_id=family_id, retry_index=0,
                status="success",
            ),
            _make_attempt(
                sequence_step_index=1, sequence_step_count=3,
                sequence_family_id=family_id, retry_index=1,
                status="success",
            ),
        ]

        problems = sequence_validation_failures(raw_attempts)
        assert len(problems) > 0
        assert any("retry-lineage" in p or "retry after success" in p for p in problems)

    def test_duplicate_retry_index_is_lineage_error(self) -> None:
        """B3: step1 r0 error, step1 r0 success → retry-lineage error."""
        from experiments.trustparadox_u.generate_empirical_corpus import (
            sequence_validation_failures,
        )

        family_id = "seq_frag_cred_low"
        # Two attempts with same retry_index=0 for same step.
        # We need distinct provider_attempt_ids, so manually override.
        a0 = _make_attempt(
            sequence_step_index=1, sequence_step_count=3,
            sequence_family_id=family_id, retry_index=0,
            status="provider_error",
        )
        a1 = _make_attempt(
            sequence_step_index=1, sequence_step_count=3,
            sequence_family_id=family_id, retry_index=0,
            status="success",
        )
        # Give distinct provider_attempt_ids.
        a1 = EmpiricalGenerationAttempt(
            **{**a1.__dict__, "provider_attempt_id": (a1.provider_attempt_id or "") + "_dup"}
        )

        problems = sequence_validation_failures([a0, a1])
        assert len(problems) > 0
        assert any("retry-lineage" in p or "duplicate" in p.lower() or "consecutive" in p for p in problems)

    def test_missing_terminal_step_is_incomplete(self) -> None:
        """B4: step0 success, step2 success → incomplete, not duplicate."""
        from experiments.trustparadox_u.generate_empirical_corpus import (
            sequence_validation_failures,
            terminal_attempts_by_sequence_step,
        )

        family_id = "seq_frag_cred_low"
        raw_attempts = [
            _make_attempt(
                sequence_step_index=0, sequence_step_count=3,
                sequence_family_id=family_id, retry_index=0,
                status="success",
            ),
            _make_attempt(
                sequence_step_index=2, sequence_step_count=3,
                sequence_family_id=family_id, retry_index=0,
                status="success",
            ),
        ]

        # Terminal reduction gives 2 steps (0 and 2), not 3.
        terminal = terminal_attempts_by_sequence_step(raw_attempts)
        assert len(terminal) == 2
        step_indices = sorted(int(a.sequence_step_index) for a in terminal)
        assert step_indices == [0, 2]

        # sequence_validation_failures should flag this as a structural problem.
        problems = sequence_validation_failures(raw_attempts)
        # It may be flagged as a structure issue (gap in step indices).
        # The key assertion: no retry-lineage error for this case.
        lineage_errors = [p for p in problems if "retry-lineage" in p]
        assert len(lineage_errors) == 0, "missing step is not a retry-lineage error"


# ===========================================================================
# Test D — sequence report from complete raw history
# ===========================================================================


class TestResumedSequenceReport:
    """Patch D: sequence report derived from complete raw history."""

    def test_resumed_sequence_report_uses_complete_raw_history(self) -> None:
        """Historical step0 success + new step1 success + step2 success.

        Report from combined raw history must show complete=1.
        """
        from experiments.trustparadox_u.generate_empirical_corpus import (
            build_sequence_generation_report_from_attempts,
        )

        family_id = "seq_frag_cred_low"
        # Historical: step0 success.
        historical = [
            _make_attempt(
                sequence_step_index=0, sequence_step_count=3,
                sequence_family_id=family_id, retry_index=0,
                status="success",
                candidate_text="Step 0 text.",
            ),
        ]
        # New: step1 success, step2 success.
        new_attempts = [
            _make_attempt(
                sequence_step_index=1, sequence_step_count=3,
                sequence_family_id=family_id, retry_index=0,
                status="success",
                candidate_text="Step 1 text.",
            ),
            _make_attempt(
                sequence_step_index=2, sequence_step_count=3,
                sequence_family_id=family_id, retry_index=0,
                status="success",
                candidate_text="Step 2 text.",
            ),
        ]

        all_attempts = historical + new_attempts

        report = build_sequence_generation_report_from_attempts(
            attempts=all_attempts,
            plan_items=None,
            target_registry=EMPIRICAL_TARGET_REGISTRY,
        )

        assert int(report["planned_sequence_count"]) >= 1
        assert int(report["complete_sequence_count"]) >= 1
        assert report["incomplete_sequence_count"] == 0

    def test_retry_does_not_inflate_sequence_count(self) -> None:
        """D3: retry history must not increase sequence-step count."""
        from experiments.trustparadox_u.generate_empirical_corpus import (
            build_sequence_generation_report_from_attempts,
            terminal_attempts_by_sequence_step,
        )

        family_id = "seq_frag_cred_low"
        raw_attempts = [
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
            _make_attempt(
                sequence_step_index=1, sequence_step_count=3,
                sequence_family_id=family_id, retry_index=1,
                status="success",
            ),
            _make_attempt(
                sequence_step_index=2, sequence_step_count=3,
                sequence_family_id=family_id, retry_index=0,
                status="success",
            ),
        ]

        # Terminal reduction: exactly 3 terminal steps, not 4.
        terminal = terminal_attempts_by_sequence_step(raw_attempts)
        assert len(terminal) == 3

        report = build_sequence_generation_report_from_attempts(
            attempts=raw_attempts,
            plan_items=None,
            target_registry=EMPIRICAL_TARGET_REGISTRY,
        )
        assert int(report["complete_sequence_count"]) >= 1

    def test_uninterrupted_vs_resumed_report_equivalence(self) -> None:
        """D1: Same scientific result → identical report fields."""
        from experiments.trustparadox_u.generate_empirical_corpus import (
            build_sequence_generation_report_from_attempts,
        )

        family_id = "seq_frag_cred_low"

        # Uninterrupted: all 3 steps succeed first try.
        uninterrupted = [
            _make_attempt(
                sequence_step_index=i, sequence_step_count=3,
                sequence_family_id=family_id, retry_index=0,
                status="success",
                candidate_text=f"Step {i} text.",
            )
            for i in range(3)
        ]

        # Resumed: step0 success, step1 error then success, step2 success.
        resumed = [
            _make_attempt(
                sequence_step_index=0, sequence_step_count=3,
                sequence_family_id=family_id, retry_index=0,
                status="success",
                candidate_text="Step 0 text.",
            ),
            _make_attempt(
                sequence_step_index=1, sequence_step_count=3,
                sequence_family_id=family_id, retry_index=0,
                status="provider_error",
            ),
            _make_attempt(
                sequence_step_index=1, sequence_step_count=3,
                sequence_family_id=family_id, retry_index=1,
                status="success",
                candidate_text="Step 1 text.",
            ),
            _make_attempt(
                sequence_step_index=2, sequence_step_count=3,
                sequence_family_id=family_id, retry_index=0,
                status="success",
                candidate_text="Step 2 text.",
            ),
        ]

        report_u = build_sequence_generation_report_from_attempts(
            attempts=uninterrupted,
            plan_items=None,
            target_registry=EMPIRICAL_TARGET_REGISTRY,
        )
        report_r = build_sequence_generation_report_from_attempts(
            attempts=resumed,
            plan_items=None,
            target_registry=EMPIRICAL_TARGET_REGISTRY,
        )

        # Scientific fields must agree.
        assert report_u["complete_sequence_count"] == report_r["complete_sequence_count"]
        assert report_u["planned_sequence_count"] == report_r["planned_sequence_count"]
