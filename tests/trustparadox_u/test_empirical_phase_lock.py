"""E1-032: development-split lock tests.

While the empirical phase is E1, only development generation is permitted;
validation/test generation must raise ``EmpiricalPhaseLockedError`` and the
CLI must exit non-zero. No override flag may silently bypass the lock.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.trustparadox_u.empirical_corpus import (
    EMPIRICAL_PHASE,
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
        assert EMPIRICAL_PHASE == "E1"

    def test_future_phase_unlocks_generation(self) -> None:
        # Once the protocol phase advances, validation generation becomes
        # permitted — the lock is phase-driven, not hard-coded forever.
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
