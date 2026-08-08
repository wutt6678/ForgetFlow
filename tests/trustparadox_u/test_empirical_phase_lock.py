"""E1-032 / E2-044: development-split lock tests.

Only ``E3_CORPUS_GENERATION`` may unlock non-development generation; in
E1_FOUNDATION, E2_TRUST_PILOT, and E2_PROMPTS_FROZEN validation/test
generation must raise ``EmpiricalPhaseLockedError`` and the CLI must exit
non-zero. Unknown phase strings are rejected and no override flag may
silently bypass the lock.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.trustparadox_u.empirical_corpus import (
    EMPIRICAL_PHASE,
    EmpiricalPhase,
    EmpiricalPhaseLockedError,
    EmpiricalSplit,
    assert_generation_split_unlocked,
)
from experiments.trustparadox_u.generate_empirical_corpus import (
    _build_parser,
    main,
)


class TestPhaseLock:
    def test_development_allowed(self) -> None:
        assert_generation_split_unlocked(EmpiricalSplit.DEVELOPMENT.value)

    def test_validation_rejected(self) -> None:
        with pytest.raises(EmpiricalPhaseLockedError):
            assert_generation_split_unlocked(EmpiricalSplit.VALIDATION.value)

    def test_test_rejected(self) -> None:
        with pytest.raises(EmpiricalPhaseLockedError):
            assert_generation_split_unlocked(EmpiricalSplit.TEST.value)

    def test_phase_is_e1(self) -> None:
        assert EMPIRICAL_PHASE is EmpiricalPhase.E1_FOUNDATION

    @pytest.mark.parametrize(
        "phase",
        [EmpiricalPhase.E1_FOUNDATION, EmpiricalPhase.E2_TRUST_PILOT],
    )
    @pytest.mark.parametrize("split", [EmpiricalSplit.VALIDATION, EmpiricalSplit.TEST])
    def test_e2_phases_do_not_unlock_generation(self, phase: EmpiricalPhase, split: EmpiricalSplit) -> None:
        # E2-044: advancing to the trust pilot must NOT unlock
        # validation/test generation.
        with pytest.raises(EmpiricalPhaseLockedError):
            assert_generation_split_unlocked(split.value, phase=phase)

    def test_prompts_frozen_phase_stays_locked(self) -> None:
        with pytest.raises(EmpiricalPhaseLockedError):
            assert_generation_split_unlocked(
                EmpiricalSplit.VALIDATION.value, phase=EmpiricalPhase.E2_PROMPTS_FROZEN
            )

    def test_e2_phases_allow_development(self) -> None:
        assert_generation_split_unlocked(
            EmpiricalSplit.DEVELOPMENT.value, phase=EmpiricalPhase.E2_TRUST_PILOT
        )
        assert_generation_split_unlocked(
            EmpiricalSplit.DEVELOPMENT.value, phase=EmpiricalPhase.E2_PROMPTS_FROZEN
        )

    @pytest.mark.parametrize("split", [EmpiricalSplit.DEVELOPMENT, EmpiricalSplit.VALIDATION])
    def test_e3_corpus_generation_unlocks(self, split: EmpiricalSplit) -> None:
        # Only the explicit E3 transition permits non-development splits
        # (the test split additionally carries assert_test_split_locked).
        assert_generation_split_unlocked(split.value, phase=EmpiricalPhase.E3_CORPUS_GENERATION)

    def test_unknown_phase_string_is_rejected(self) -> None:
        # No silent unlock via an unrecognized phase value.
        with pytest.raises(ValueError):
            assert_generation_split_unlocked(EmpiricalSplit.VALIDATION.value, phase="E2")


class TestRunnerLock:
    @pytest.mark.parametrize("split", ["validation", "test"])
    def test_cli_rejects_locked_splits(self, split: str, tmp_path: Path) -> None:
        exit_code = main(
            [
                "--split",
                split,
                "--mode",
                "mock",
                "--samples",
                "1",
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
        assert exit_code == 2
        assert not (tmp_path / "out").exists()

    def test_cli_allows_development(self, tmp_path: Path) -> None:
        exit_code = main(
            [
                "--split",
                "development",
                "--mode",
                "mock",
                "--scenario",
                "credential_001",
                "--trust",
                "default",
                "--attack",
                "direct_disclosure",
                "--samples",
                "1",
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
        assert exit_code == 0
        assert (tmp_path / "out" / "raw_generation_attempts.jsonl").exists()

    def test_no_override_flag_bypasses_lock(self) -> None:
        parser = _build_parser()
        option_strings = {option for action in parser._actions for option in action.option_strings}
        bypass_flags = {"--force", "--unlock", "--override", "--allow-locked", "--no-lock"}
        assert not option_strings & bypass_flags

    def test_real_mode_requires_generator_model(self, tmp_path: Path) -> None:
        exit_code = main(
            [
                "--split",
                "development",
                "--mode",
                "real",
                "--samples",
                "1",
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
        assert exit_code == 2
