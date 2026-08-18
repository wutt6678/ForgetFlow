"""E5-007: Held-out test evaluation tests.

Tests condition-specific detection, row evaluation, sequence evaluation,
completeness validation, and manifest generation using synthetic data.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.e5_conditions import (  # noqa: E402
    CONDITIONS,
    RowResult,
)
from experiments.trustparadox_u.e5_sequence_evaluation import (  # noqa: E402
    SequenceResult,
    StepDecision,
)
from experiments.trustparadox_u.e5_test_evaluation import (  # noqa: E402
    EXPECTED_ROW_COUNT,
    EXPECTED_SEQUENCE_COUNT,
    HeldOutEvaluationResult,
    apply_condition_detection,
    build_test_run_manifest,
    determine_policy_action,
    evaluate_row,
    run_condition,
    validate_completeness,
    write_test_results,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeRowLabel:
    candidate_id: str
    resolution_status: str = "resolved"

    @property
    def is_unresolved(self) -> bool:
        return self.resolution_status != "resolved"


@dataclass(frozen=True)
class _FakeCorpus:
    candidate_id: str
    scenario_id: str = "s1"
    trust_level: str = "default"
    content_sha256: str = "abc123"


@dataclass(frozen=True)
class _FakeSeqLabel:
    sequence_annotation_id: str
    ordered_candidate_ids: tuple[str, ...]
    final_sequence_reconstructs_target: bool
    final_earliest_reconstruction_step: int | None = None
    final_reconstruction_strength: str = "unknown"
    trust_level: str = "default"
    resolution_status: str = "resolved"

    @property
    def is_unresolved(self) -> bool:
        return self.resolution_status != "resolved"


@dataclass(frozen=True)
class _FakeSplitData:
    split: str
    row_labels: tuple[_FakeRowLabel, ...]
    sequence_labels: tuple[_FakeSeqLabel, ...]
    corpus: tuple[_FakeCorpus, ...]


def _feat(
    *,
    exact: bool = False,
    alias: bool = False,
    sim: float = 0.0,
) -> dict:
    return {
        "exact_match": exact,
        "alias_match": alias,
        "semantic_similarity": sim,
        "embedding_model": "test-model",
    }


def _make_result(
    *,
    n_rows: int = 10,
    n_seqs: int = 5,
    conditions: tuple[str, ...] = ("C0", "C1", "C2", "C3", "C4"),
) -> HeldOutEvaluationResult:
    """Create a minimal HeldOutEvaluationResult for testing."""
    row_results = {}
    seq_results = {}
    for cid in conditions:
        row_results[cid] = tuple(
            _make_row_result(cid=cid, candidate_id=f"c{i}")
            for i in range(n_rows)
        )
        seq_results[cid] = tuple(
            _make_seq_result(cid=cid, seq_id=f"s{i}")
            for i in range(n_seqs)
        )
    return HeldOutEvaluationResult(
        split="test",
        condition_ids=conditions,
        row_results=row_results,
        sequence_results=seq_results,
        tau_sem=0.75,
        n_rows_per_condition=n_rows,
        n_sequences_per_condition=n_seqs,
    )


def _make_row_result(
    *,
    cid: str = "C4",
    candidate_id: str = "c0",
) -> RowResult:
    return RowResult(
        candidate_id=candidate_id,
        split="test",
        condition_id=cid,
        scenario_id="s1",
        trust_level="default",
        exact_match=False,
        alias_match=False,
        semantic_similarity=0.0,
        policy_action="allow",
        blocked=False,
        allowed=True,
        input_content_sha="abc",
        output_content_sha="abc",
        detector_config_sha="det",
        condition_manifest_sha="cond",
        embedding_model="test",
    )


def _make_seq_result(
    *,
    cid: str = "C4",
    seq_id: str = "s0",
) -> SequenceResult:
    return SequenceResult(
        sequence_annotation_id=seq_id,
        trust_level="default",
        condition_id=cid,
        ordered_candidate_ids=("c1", "c2"),
        step_decisions=(
            StepDecision(
                step_index=0,
                candidate_id="c1",
                exact_match=False,
                alias_match=False,
                semantic_similarity=0.5,
                detected=False,
                policy_action="allow",
            ),
        ),
        predicted_sequence_reconstruction=True,
        predicted_earliest_reconstruction_step=0,
        predicted_reconstruction_strength=1.0,
    )


# ===========================================================================
# apply_condition_detection
# ===========================================================================


class TestApplyConditionDetection:
    """Tests for condition-specific detection logic."""

    def test_c0_no_detection(self):
        """C0: no firewall → never detected."""
        features = _feat(exact=True, alias=True, sim=0.99)
        _, _, _, detected = apply_condition_detection(features, CONDITIONS["C0"], 0.75)
        assert detected is False

    def test_c1_exact_only(self):
        """C1: exact match only."""
        features = _feat(exact=True, alias=True, sim=0.99)
        _, _, _, detected = apply_condition_detection(features, CONDITIONS["C1"], 0.75)
        assert detected is True

    def test_c1_no_exact_no_detection(self):
        """C1: no exact match → not detected even with high sim."""
        features = _feat(exact=False, alias=True, sim=0.99)
        _, _, _, detected = apply_condition_detection(features, CONDITIONS["C1"], 0.75)
        assert detected is False

    def test_c2_alias(self):
        """C2: exact + alias."""
        features = _feat(exact=False, alias=True, sim=0.99)
        _, _, _, detected = apply_condition_detection(features, CONDITIONS["C2"], 0.75)
        assert detected is True

    def test_c3_semantic(self):
        """C3: exact + alias + semantic."""
        features = _feat(exact=False, alias=False, sim=0.85)
        _, _, _, detected = apply_condition_detection(features, CONDITIONS["C3"], 0.75)
        assert detected is True

    def test_c3_semantic_below_threshold(self):
        """C3: semantic below threshold → not detected."""
        features = _feat(exact=False, alias=False, sim=0.60)
        _, _, _, detected = apply_condition_detection(features, CONDITIONS["C3"], 0.75)
        assert detected is False

    def test_c4_full(self):
        """C4: full detection."""
        features = _feat(exact=True, alias=True, sim=0.95)
        _, _, _, detected = apply_condition_detection(features, CONDITIONS["C4"], 0.75)
        assert detected is True

    def test_return_values(self):
        """Returns (exact_used, alias_used, semantic_used, detected)."""
        features = _feat(exact=True, alias=True, sim=0.85)
        exact, alias, semantic, detected = apply_condition_detection(
            features, CONDITIONS["C3"], 0.75
        )
        assert exact is True
        assert alias is True
        assert semantic == 0.85
        assert detected is True


# ===========================================================================
# determine_policy_action
# ===========================================================================


class TestDeterminePolicyAction:
    """Tests for policy action determination."""

    def test_c0_always_allow(self):
        """C0: no firewall → always allow."""
        action, blocked, allowed = determine_policy_action(True, CONDITIONS["C0"])
        assert action == "allow"
        assert blocked is False
        assert allowed is True

    def test_c4_detected_blocks(self):
        """C4: detected → block."""
        action, blocked, allowed = determine_policy_action(True, CONDITIONS["C4"])
        assert action == "block"
        assert blocked is True
        assert allowed is False

    def test_c4_not_detected_allows(self):
        """C4: not detected → allow."""
        action, blocked, allowed = determine_policy_action(False, CONDITIONS["C4"])
        assert action == "allow"
        assert blocked is False
        assert allowed is True


# ===========================================================================
# evaluate_row
# ===========================================================================


class TestEvaluateRow:
    """Tests for single row evaluation."""

    def test_basic_row(self):
        """Evaluate one row under C4."""
        row = _FakeRowLabel(candidate_id="c1")
        corpus = _FakeCorpus(candidate_id="c1", scenario_id="s1", trust_level="high")
        features = _feat(exact=True, sim=0.9)

        result = evaluate_row(
            row_label=row,
            corpus=corpus,
            features=features,
            condition=CONDITIONS["C4"],
            tau_sem=0.75,
            split="test",
        )
        assert result.candidate_id == "c1"
        assert result.split == "test"
        assert result.condition_id == "C4"
        assert result.scenario_id == "s1"
        assert result.trust_level == "high"
        assert result.blocked is True
        assert result.policy_action == "block"

    def test_c0_passthrough(self):
        """C0: always allow regardless of features."""
        row = _FakeRowLabel(candidate_id="c2")
        corpus = _FakeCorpus(candidate_id="c2")
        features = _feat(exact=True, alias=True, sim=0.99)

        result = evaluate_row(
            row_label=row,
            corpus=corpus,
            features=features,
            condition=CONDITIONS["C0"],
            tau_sem=0.75,
            split="test",
        )
        assert result.allowed is True
        assert result.blocked is False
        assert result.policy_action == "allow"

    def test_missing_features_fail_closed(self):
        """Missing features → ValueError (§36-§37 fail closed)."""
        row = _FakeRowLabel(candidate_id="c3")
        corpus = _FakeCorpus(candidate_id="c3")
        features = {}

        with pytest.raises(ValueError, match="Missing features"):
            evaluate_row(
                row_label=row,
                corpus=corpus,
                features=features,
                condition=CONDITIONS["C4"],
                tau_sem=0.75,
                split="test",
            )


# ===========================================================================
# run_condition
# ===========================================================================


class TestRunCondition:
    """Tests for full condition evaluation."""

    def test_basic_condition_run(self):
        """Run one condition on synthetic split data."""
        split = _FakeSplitData(
            split="test",
            row_labels=(
                _FakeRowLabel(candidate_id="c1"),
                _FakeRowLabel(candidate_id="c2"),
            ),
            sequence_labels=(
                _FakeSeqLabel(
                    sequence_annotation_id="s1",
                    ordered_candidate_ids=("c1", "c2"),
                    final_sequence_reconstructs_target=True,
                ),
            ),
            corpus=(
                _FakeCorpus(candidate_id="c1"),
                _FakeCorpus(candidate_id="c2"),
            ),
        )
        features = {
            "c1": _feat(exact=True, sim=0.9),
            "c2": _feat(sim=0.3),
        }

        rows, seqs = run_condition(
            split_data=split,
            condition=CONDITIONS["C4"],
            features_by_id=features,
            tau_sem=0.75,
        )
        assert len(rows) == 2
        assert len(seqs) == 1
        assert rows[0].condition_id == "C4"

    def test_missing_corpus_skipped(self):
        """Rows without corpus entries are skipped."""
        split = _FakeSplitData(
            split="test",
            row_labels=(
                _FakeRowLabel(candidate_id="missing"),
                _FakeRowLabel(candidate_id="c1"),
            ),
            sequence_labels=(),
            corpus=(_FakeCorpus(candidate_id="c1"),),
        )
        features = {"c1": _feat(sim=0.3)}

        rows, _ = run_condition(
            split_data=split,
            condition=CONDITIONS["C4"],
            features_by_id=features,
            tau_sem=0.75,
        )
        assert len(rows) == 1
        assert rows[0].candidate_id == "c1"


# ===========================================================================
# validate_completeness
# ===========================================================================


class TestValidateCompleteness:
    """Tests for completeness validation."""

    def test_complete(self):
        """Full counts → no errors."""
        result = _make_result(
            n_rows=EXPECTED_ROW_COUNT,
            n_seqs=EXPECTED_SEQUENCE_COUNT,
        )
        errors = validate_completeness(result)
        assert errors == []

    def test_missing_rows(self):
        """Wrong row count → errors."""
        result = _make_result(n_rows=100, n_seqs=EXPECTED_SEQUENCE_COUNT)
        errors = validate_completeness(result)
        assert len(errors) > 0
        assert any("rows" in e for e in errors)

    def test_missing_sequences(self):
        """Wrong sequence count → errors."""
        result = _make_result(n_rows=EXPECTED_ROW_COUNT, n_seqs=10)
        errors = validate_completeness(result)
        assert len(errors) > 0
        assert any("sequences" in e for e in errors)

    def test_missing_condition(self):
        """Missing condition → errors."""
        result = _make_result(
            n_rows=EXPECTED_ROW_COUNT,
            n_seqs=EXPECTED_SEQUENCE_COUNT,
            conditions=("C0", "C1"),
        )
        errors = validate_completeness(result)
        assert len(errors) > 0


# ===========================================================================
# build_test_run_manifest
# ===========================================================================


class TestBuildTestRunManifest:
    """Tests for test run manifest generation."""

    def test_manifest_structure(self):
        """Manifest includes all required fields."""
        result = _make_result(n_rows=10, n_seqs=5)
        manifest = build_test_run_manifest(
            result=result,
            code_commit="abc123",
            test_lock_sha="lock_sha",
        )
        assert manifest["schema_version"] == "1.0"
        assert manifest["code_commit"] == "abc123"
        assert manifest["test_lock_sha"] == "lock_sha"
        assert manifest["split"] == "test"
        assert manifest["condition_count"] == 5
        assert manifest["row_count_per_condition"] == 10
        assert manifest["sequence_count_per_condition"] == 5

    def test_completeness_gate(self):
        """Completeness gate reflects actual counts."""
        result = _make_result(
            n_rows=EXPECTED_ROW_COUNT,
            n_seqs=EXPECTED_SEQUENCE_COUNT,
        )
        manifest = build_test_run_manifest(result=result)
        assert manifest["completeness"]["rows_complete"] is True
        assert manifest["completeness"]["sequences_complete"] is True

    def test_incomplete_gate(self):
        """Incomplete counts → gate is False."""
        result = _make_result(n_rows=10, n_seqs=5)
        manifest = build_test_run_manifest(result=result)
        assert manifest["completeness"]["rows_complete"] is False
        assert manifest["completeness"]["sequences_complete"] is False


# ===========================================================================
# write_test_results
# ===========================================================================


class TestWriteTestResults:
    """Tests for writing results to disk."""

    def test_write_creates_files(self, tmp_path):
        """Write creates per-condition JSONL files."""
        result = _make_result(n_rows=3, n_seqs=2)
        run_dir = write_test_results(result, run_dir=tmp_path / "run_test")

        assert run_dir.exists()
        # Check C0 files exist
        assert (run_dir / "C0_row_results.jsonl").exists()
        assert (run_dir / "C0_sequence_results.jsonl").exists()
        # Check C4 files exist
        assert (run_dir / "C4_row_results.jsonl").exists()
        assert (run_dir / "C4_sequence_results.jsonl").exists()

    def test_write_row_content(self, tmp_path):
        """Row results are valid JSON."""
        result = _make_result(n_rows=2, n_seqs=1)
        run_dir = write_test_results(result, run_dir=tmp_path / "run_test2")

        row_path = run_dir / "C4_row_results.jsonl"
        with open(row_path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        for line in lines:
            record = json.loads(line)
            assert "candidate_id" in record
            assert "condition_id" in record

    def test_write_sequence_content(self, tmp_path):
        """Sequence results are valid JSON."""
        result = _make_result(n_rows=1, n_seqs=2)
        run_dir = write_test_results(result, run_dir=tmp_path / "run_test3")

        seq_path = run_dir / "C4_sequence_results.jsonl"
        with open(seq_path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        for line in lines:
            record = json.loads(line)
            assert "sequence_annotation_id" in record
