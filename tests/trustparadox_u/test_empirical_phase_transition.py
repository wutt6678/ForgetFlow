"""E3-001: tests for the controlled E2_COMPLETE → E3_CORPUS_GENERATION transition.

Covers every prerequisite documented in the E3 plan:

- correct phase prerequisite (E2_COMPLETE only);
- ``full_corpus_generation_authorized`` must be true;
- all E2 hash fields must be present and non-empty;
- clean working tree;
- frozen prompts / evaluator / labels gates;
- the transition is not silently repeatable once already in E3;
- the updated record preserves all E2 provenance hashes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.trustparadox_u.empirical_corpus import (
    EmpiricalPhase,
    assert_generation_split_unlocked,
)
from experiments.trustparadox_u.transition_empirical_phase import (
    _REQUIRED_BOOLEAN_GATES,
    _REQUIRED_E2_HASH_FIELDS,
    TransitionError,
    main,
    transition_to_e3_corpus_generation,
)


def _valid_e2_record() -> dict:
    """Return a minimal phase record that satisfies every E2 prerequisite."""
    record: dict = {
        "schema_version": "1.1.0",
        "protocol_version": "2.0.0",
        "study_version": "2.0.0",
        "phase": EmpiricalPhase.E2_COMPLETE.value,
    }
    for gate in _REQUIRED_BOOLEAN_GATES:
        record[gate] = True
    for field in _REQUIRED_E2_HASH_FIELDS:
        record[field] = "a" * 64
    return record


class TestTransitionToE3:
    def test_successful_transition(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        phase_file = tmp_path / "empirical_phase.json"
        phase_file.write_text(json.dumps(_valid_e2_record()))

        monkeypatch.setattr(
            "experiments.trustparadox_u.transition_empirical_phase.working_tree_is_fully_clean",
            lambda: True,
        )
        record = transition_to_e3_corpus_generation(phase_file=phase_file)

        assert record["phase"] == EmpiricalPhase.E3_CORPUS_GENERATION.value
        assert record["corpus_frozen"] is False
        assert "e3_started_from_commit" in record
        assert "e3_started_at" in record
        # E2 hashes preserved.
        for field in _REQUIRED_E2_HASH_FIELDS:
            assert record[field] == "a" * 64

    def test_wrong_phase_rejected(self, tmp_path: Path) -> None:
        record = _valid_e2_record()
        record["phase"] = EmpiricalPhase.E2_PROMPTS_FROZEN.value
        phase_file = tmp_path / "empirical_phase.json"
        phase_file.write_text(json.dumps(record))

        with pytest.raises(TransitionError, match="E2_PROMPTS_FROZEN"):
            transition_to_e3_corpus_generation(phase_file=phase_file)

    def test_e1_foundation_rejected(self, tmp_path: Path) -> None:
        record = _valid_e2_record()
        record["phase"] = EmpiricalPhase.E1_FOUNDATION.value
        phase_file = tmp_path / "empirical_phase.json"
        phase_file.write_text(json.dumps(record))

        with pytest.raises(TransitionError, match="E1_FOUNDATION"):
            transition_to_e3_corpus_generation(phase_file=phase_file)

    def test_corpus_generation_not_repeatable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        record = _valid_e2_record()
        record["phase"] = EmpiricalPhase.E3_CORPUS_GENERATION.value
        phase_file = tmp_path / "empirical_phase.json"
        phase_file.write_text(json.dumps(record))

        monkeypatch.setattr(
            "experiments.trustparadox_u.transition_empirical_phase.working_tree_is_fully_clean",
            lambda: True,
        )
        with pytest.raises(TransitionError, match="E3_CORPUS_GENERATION"):
            transition_to_e3_corpus_generation(phase_file=phase_file)

    def test_full_corpus_generation_authorized_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        record = _valid_e2_record()
        record["full_corpus_generation_authorized"] = False
        phase_file = tmp_path / "empirical_phase.json"
        phase_file.write_text(json.dumps(record))

        monkeypatch.setattr(
            "experiments.trustparadox_u.transition_empirical_phase.working_tree_is_fully_clean",
            lambda: True,
        )
        with pytest.raises(TransitionError, match="full_corpus_generation_authorized"):
            transition_to_e3_corpus_generation(phase_file=phase_file)

    def test_missing_hash_field(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        record = _valid_e2_record()
        del record["pilot_analysis_sha256"]
        phase_file = tmp_path / "empirical_phase.json"
        phase_file.write_text(json.dumps(record))

        monkeypatch.setattr(
            "experiments.trustparadox_u.transition_empirical_phase.working_tree_is_fully_clean",
            lambda: True,
        )
        with pytest.raises(TransitionError, match="missing E2 hash"):
            transition_to_e3_corpus_generation(phase_file=phase_file)

    def test_empty_hash_field(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        record = _valid_e2_record()
        record["pilot_analysis_sha256"] = ""
        phase_file = tmp_path / "empirical_phase.json"
        phase_file.write_text(json.dumps(record))

        monkeypatch.setattr(
            "experiments.trustparadox_u.transition_empirical_phase.working_tree_is_fully_clean",
            lambda: True,
        )
        with pytest.raises(TransitionError, match="empty E2 hash"):
            transition_to_e3_corpus_generation(phase_file=phase_file)

    def test_dirty_tree_rejected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        phase_file = tmp_path / "empirical_phase.json"
        phase_file.write_text(json.dumps(_valid_e2_record()))

        monkeypatch.setattr(
            "experiments.trustparadox_u.transition_empirical_phase.working_tree_is_fully_clean",
            lambda: False,
        )
        with pytest.raises(TransitionError, match="not clean"):
            transition_to_e3_corpus_generation(phase_file=phase_file)

    def test_frozen_prompts_gate_required(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        record = _valid_e2_record()
        record["trust_prompts_frozen"] = False
        phase_file = tmp_path / "empirical_phase.json"
        phase_file.write_text(json.dumps(record))

        monkeypatch.setattr(
            "experiments.trustparadox_u.transition_empirical_phase.working_tree_is_fully_clean",
            lambda: True,
        )
        with pytest.raises(TransitionError, match="trust_prompts_frozen"):
            transition_to_e3_corpus_generation(phase_file=phase_file)

    def test_corpus_not_frozen_after_transition(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        phase_file = tmp_path / "empirical_phase.json"
        phase_file.write_text(json.dumps(_valid_e2_record()))

        monkeypatch.setattr(
            "experiments.trustparadox_u.transition_empirical_phase.working_tree_is_fully_clean",
            lambda: True,
        )
        record = transition_to_e3_corpus_generation(phase_file=phase_file)
        assert record["corpus_frozen"] is False

    def test_e2_hashes_preserved_in_updated_record(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        record = _valid_e2_record()
        record["completion_report_sha256"] = "b" * 64
        phase_file = tmp_path / "empirical_phase.json"
        phase_file.write_text(json.dumps(record))

        monkeypatch.setattr(
            "experiments.trustparadox_u.transition_empirical_phase.working_tree_is_fully_clean",
            lambda: True,
        )
        updated = transition_to_e3_corpus_generation(phase_file=phase_file)
        assert updated["completion_report_sha256"] == "b" * 64
        assert updated["phase"] == EmpiricalPhase.E3_CORPUS_GENERATION.value


class TestTransitionExitCriterion:
    """After a successful transition, all splits are unlocked."""

    def test_all_splits_unlocked_after_transition(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        phase_file = tmp_path / "empirical_phase.json"
        phase_file.write_text(json.dumps(_valid_e2_record()))

        monkeypatch.setattr(
            "experiments.trustparadox_u.transition_empirical_phase.working_tree_is_fully_clean",
            lambda: True,
        )
        transition_to_e3_corpus_generation(phase_file=phase_file)

        # All three splits must be unlocked in E3_CORPUS_GENERATION.
        assert_generation_split_unlocked("development", phase=EmpiricalPhase.E3_CORPUS_GENERATION)
        assert_generation_split_unlocked("validation", phase=EmpiricalPhase.E3_CORPUS_GENERATION)
        assert_generation_split_unlocked("test", phase=EmpiricalPhase.E3_CORPUS_GENERATION)


class TestTransitionCLI:
    def test_cli_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        phase_file = tmp_path / "empirical_phase.json"
        phase_file.write_text(json.dumps(_valid_e2_record()))

        monkeypatch.setattr(
            "experiments.trustparadox_u.transition_empirical_phase.working_tree_is_fully_clean",
            lambda: True,
        )
        rc = main(["--to", "E3_CORPUS_GENERATION", "--phase-file", str(phase_file)])
        assert rc == 0

        updated = json.loads(phase_file.read_text())
        assert updated["phase"] == "E3_CORPUS_GENERATION"

    def test_cli_wrong_phase(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        record = _valid_e2_record()
        record["phase"] = EmpiricalPhase.E2_PROMPTS_FROZEN.value
        phase_file = tmp_path / "empirical_phase.json"
        phase_file.write_text(json.dumps(record))

        monkeypatch.setattr(
            "experiments.trustparadox_u.transition_empirical_phase.working_tree_is_fully_clean",
            lambda: True,
        )
        rc = main(["--to", "E3_CORPUS_GENERATION", "--phase-file", str(phase_file)])
        assert rc == 1

    def test_cli_dirty_tree(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        phase_file = tmp_path / "empirical_phase.json"
        phase_file.write_text(json.dumps(_valid_e2_record()))

        monkeypatch.setattr(
            "experiments.trustparadox_u.transition_empirical_phase.working_tree_is_fully_clean",
            lambda: False,
        )
        rc = main(["--to", "E3_CORPUS_GENERATION", "--phase-file", str(phase_file)])
        assert rc == 1
