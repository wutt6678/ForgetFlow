"""E5-006: Sequence / history evaluation tests.

Tests sequence replay, reconstruction prediction, earliest-step analysis,
confusion matrix, and batch evaluation using synthetic data.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.e5_sequence_evaluation import (  # noqa: E402
    SequenceResult,
    StepDecision,
    compute_earliest_step_metrics,
    compute_sequence_confusion_matrix,
    evaluate_sequences,
    predict_sequence_reconstruction,
    replay_sequence,
    sequence_result_to_dict,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _step(
    idx: int,
    cid: str,
    *,
    detected: bool = False,
    sim: float = 0.0,
) -> StepDecision:
    return StepDecision(
        step_index=idx,
        candidate_id=cid,
        exact_match=detected,
        alias_match=False,
        semantic_similarity=sim,
        detected=detected,
        policy_action="block" if detected else "allow",
    )


def _feat(
    cid: str,
    *,
    exact: bool = False,
    alias: bool = False,
    sim: float = 0.0,
) -> dict:
    return {
        "candidate_id": cid,
        "exact_match": exact,
        "alias_match": alias,
        "semantic_similarity": sim,
    }


# ===========================================================================
# predict_sequence_reconstruction
# ===========================================================================


class TestPredictSequenceReconstruction:
    """Tests for reconstruction prediction."""

    def test_all_detected_no_reconstruction(self):
        """All steps detected → no reconstruction."""
        steps = [_step(0, "a", detected=True), _step(1, "b", detected=True)]
        recon, earliest, strength = predict_sequence_reconstruction(steps)
        assert recon is False
        assert earliest is None
        assert strength == 0.0

    def test_one_undetected_reconstructs(self):
        """One undetected step → reconstruction."""
        steps = [_step(0, "a", detected=True), _step(1, "b", detected=False)]
        recon, earliest, strength = predict_sequence_reconstruction(steps)
        assert recon is True
        assert earliest == 1
        assert strength == 0.5

    def test_first_step_undetected(self):
        """First step undetected → earliest = 0."""
        steps = [_step(0, "a", detected=False), _step(1, "b", detected=True)]
        recon, earliest, strength = predict_sequence_reconstruction(steps)
        assert recon is True
        assert earliest == 0
        assert strength == 0.5

    def test_all_undetected(self):
        """All undetected → full reconstruction."""
        steps = [_step(0, "a", detected=False), _step(1, "b", detected=False)]
        recon, earliest, strength = predict_sequence_reconstruction(steps)
        assert recon is True
        assert earliest == 0
        assert strength == 1.0

    def test_empty_steps(self):
        """Empty steps → no reconstruction."""
        recon, earliest, strength = predict_sequence_reconstruction([])
        assert recon is False
        assert earliest is None
        assert strength == 0.0


# ===========================================================================
# replay_sequence
# ===========================================================================


class TestReplaySequence:
    """Tests for step-by-step sequence replay."""

    def test_basic_replay(self):
        """Replay with mixed detection."""
        features = {
            "a": _feat("a", exact=True, sim=0.9),
            "b": _feat("b", sim=0.3),
        }
        result = replay_sequence(
            sequence_annotation_id="seq-001",
            ordered_candidate_ids=("a", "b"),
            trust_level="default",
            condition_id="C4",
            features_by_id=features,
            tau_sem=0.75,
        )
        assert result.sequence_annotation_id == "seq-001"
        assert len(result.step_decisions) == 2
        assert result.step_decisions[0].detected is True
        assert result.step_decisions[1].detected is False
        assert result.predicted_sequence_reconstruction is True
        assert result.predicted_earliest_reconstruction_step == 1

    def test_missing_features_conservative(self):
        """Missing features → treated as undetected."""
        result = replay_sequence(
            sequence_annotation_id="seq-002",
            ordered_candidate_ids=("missing",),
            trust_level="low",
            condition_id="C0",
            features_by_id={},
            tau_sem=0.75,
        )
        assert len(result.step_decisions) == 1
        assert result.step_decisions[0].detected is False
        assert result.step_decisions[0].policy_action == "allow"

    def test_all_detected_no_reconstruction(self):
        """All candidates detected → no reconstruction."""
        features = {
            "a": _feat("a", exact=True, sim=0.95),
            "b": _feat("b", alias=True, sim=0.85),
        }
        result = replay_sequence(
            sequence_annotation_id="seq-003",
            ordered_candidate_ids=("a", "b"),
            trust_level="high",
            condition_id="C4",
            features_by_id=features,
            tau_sem=0.75,
        )
        assert result.predicted_sequence_reconstruction is False
        assert result.predicted_earliest_reconstruction_step is None
        assert result.predicted_reconstruction_strength == 0.0

    def test_step_decisions_frozen(self):
        """Step decisions are stored as a tuple."""
        result = replay_sequence(
            sequence_annotation_id="seq-004",
            ordered_candidate_ids=("a",),
            trust_level="default",
            condition_id="C1",
            features_by_id={"a": _feat("a", sim=0.5)},
            tau_sem=0.75,
        )
        assert isinstance(result.step_decisions, tuple)


# ===========================================================================
# SequenceResult serialisation
# ===========================================================================


class TestSequenceResultSerialisation:
    """Tests for sequence result serialisation."""

    def test_to_dict(self):
        """Serialisation includes all key fields."""
        result = replay_sequence(
            sequence_annotation_id="seq-010",
            ordered_candidate_ids=("x", "y"),
            trust_level="default",
            condition_id="C4",
            features_by_id={
                "x": _feat("x", exact=True, sim=0.9),
                "y": _feat("y", sim=0.3),
            },
            tau_sem=0.75,
        )
        d = sequence_result_to_dict(result)
        assert d["sequence_annotation_id"] == "seq-010"
        assert isinstance(d["ordered_candidate_ids"], list)
        assert len(d["step_decisions"]) == 2
        assert d["predicted_sequence_reconstruction"] is True


# ===========================================================================
# SequenceConfusionMatrix
# ===========================================================================


class TestSequenceConfusionMatrix:
    """Tests for sequence reconstruction confusion matrix."""

    def test_perfect_prediction(self):
        """All predictions match annotations."""
        results = [
            _make_result(pred_recon=True, final_recon=True),
            _make_result(pred_recon=False, final_recon=False),
        ]
        cm = compute_sequence_confusion_matrix(results)
        assert cm.tp == 1
        assert cm.tn == 1
        assert cm.fp == 0
        assert cm.fn == 0
        assert cm.precision == 1.0
        assert cm.recall == 1.0
        assert cm.f1 == 1.0

    def test_false_positive(self):
        """Predicted reconstructs but annotated says no."""
        results = [_make_result(pred_recon=True, final_recon=False)]
        cm = compute_sequence_confusion_matrix(results)
        assert cm.fp == 1
        assert cm.tp == 0
        assert cm.precision == 0.0

    def test_false_negative(self):
        """Predicted no reconstruction but annotated says yes."""
        results = [_make_result(pred_recon=False, final_recon=True)]
        cm = compute_sequence_confusion_matrix(results)
        assert cm.fn == 1
        assert cm.recall == 0.0

    def test_skips_none_annotations(self):
        """Results with None annotations are skipped."""
        results = [
            _make_result(pred_recon=True, final_recon=True),
            _make_result(pred_recon=True, final_recon=None),
        ]
        cm = compute_sequence_confusion_matrix(results)
        assert cm.n_eligible == 1

    def test_n_eligible(self):
        """n_eligible counts all annotated results."""
        results = [
            _make_result(pred_recon=True, final_recon=True),
            _make_result(pred_recon=False, final_recon=True),
            _make_result(pred_recon=True, final_recon=False),
            _make_result(pred_recon=False, final_recon=False),
        ]
        cm = compute_sequence_confusion_matrix(results)
        assert cm.n_eligible == 4

    def test_empty_results(self):
        """Empty results → zero confusion matrix."""
        cm = compute_sequence_confusion_matrix([])
        assert cm.tp == 0
        assert cm.n_eligible == 0
        assert cm.precision == 0.0
        assert cm.recall == 0.0


# ===========================================================================
# EarliestStepMetrics
# ===========================================================================


class TestEarliestStepMetrics:
    """Tests for earliest reconstruction step analysis."""

    def test_exact_match(self):
        """Predicted == annotated → exact accuracy 1.0."""
        results = [
            _make_result(
                pred_recon=True, final_recon=True,
                pred_earliest=2, final_earliest=2,
            ),
        ]
        em = compute_earliest_step_metrics(results)
        assert em.n_compared == 1
        assert em.exact_step_accuracy == 1.0
        assert em.mean_absolute_step_error == 0.0
        assert em.n_exact_match == 1

    def test_step_error(self):
        """Predicted step differs from annotated."""
        results = [
            _make_result(
                pred_recon=True, final_recon=True,
                pred_earliest=1, final_earliest=3,
            ),
        ]
        em = compute_earliest_step_metrics(results)
        assert em.n_compared == 1
        assert em.exact_step_accuracy == 0.0
        assert em.mean_absolute_step_error == 2.0
        assert em.n_predicted_earlier == 1

    def test_predicted_later(self):
        """Predicted step > annotated step."""
        results = [
            _make_result(
                pred_recon=True, final_recon=True,
                pred_earliest=4, final_earliest=1,
            ),
        ]
        em = compute_earliest_step_metrics(results)
        assert em.n_predicted_later == 1

    def test_skips_non_reconstructing(self):
        """Non-reconstructing sequences are skipped."""
        results = [
            _make_result(
                pred_recon=True, final_recon=False,
                pred_earliest=1, final_earliest=None,
            ),
        ]
        em = compute_earliest_step_metrics(results)
        assert em.n_compared == 0

    def test_skips_none_annotations(self):
        """Missing annotations are skipped."""
        results = [
            _make_result(
                pred_recon=True, final_recon=None,
                pred_earliest=1, final_earliest=None,
            ),
        ]
        em = compute_earliest_step_metrics(results)
        assert em.n_compared == 0

    def test_empty(self):
        """Empty results → zero metrics."""
        em = compute_earliest_step_metrics([])
        assert em.n_compared == 0
        assert em.exact_step_accuracy == 0.0
        assert em.mean_absolute_step_error == 0.0


# ===========================================================================
# evaluate_sequences (batch)
# ===========================================================================


class TestEvaluateSequences:
    """Tests for batch sequence evaluation."""

    def test_basic_batch(self):
        """Evaluate multiple sequences."""
        labels = [
            _FakeSeqLabel(
                sequence_annotation_id="s1",
                ordered_candidate_ids=("a", "b"),
                final_sequence_reconstructs_target=True,
                final_earliest_reconstruction_step=1,
                trust_level="default",
            ),
            _FakeSeqLabel(
                sequence_annotation_id="s2",
                ordered_candidate_ids=("c", "d"),
                final_sequence_reconstructs_target=False,
                trust_level="low",
            ),
        ]
        features = {
            "a": _feat("a", exact=True, sim=0.9),
            "b": _feat("b", sim=0.3),
            "c": _feat("c", exact=True, sim=0.95),
            "d": _feat("d", alias=True, sim=0.88),
        }
        results = evaluate_sequences(labels, features, tau_sem=0.75)
        assert len(results) == 2
        # s1: one undetected → reconstructs
        assert results[0].predicted_sequence_reconstruction is True
        assert results[0].final_sequence_reconstructs_target is True
        # s2: all detected → no reconstruction
        assert results[1].predicted_sequence_reconstruction is False
        assert results[1].final_sequence_reconstructs_target is False

    def test_unresolved_excluded(self):
        """Unresolved sequences are excluded."""
        labels = [
            _FakeSeqLabel(
                sequence_annotation_id="s1",
                ordered_candidate_ids=("a",),
                final_sequence_reconstructs_target=True,
                resolution_status="unresolved",
            ),
        ]
        results = evaluate_sequences(labels, {}, tau_sem=0.75)
        assert len(results) == 0

    def test_annotations_joined(self):
        """Annotation labels are joined after execution."""
        labels = [
            _FakeSeqLabel(
                sequence_annotation_id="s1",
                ordered_candidate_ids=("a",),
                final_sequence_reconstructs_target=True,
                final_earliest_reconstruction_step=0,
                final_reconstruction_strength="strong",
            ),
        ]
        results = evaluate_sequences(labels, {}, tau_sem=0.75)
        assert results[0].final_sequence_reconstructs_target is True
        assert results[0].final_earliest_reconstruction_step == 0
        assert results[0].final_reconstruction_strength == "strong"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    *,
    pred_recon: bool,
    final_recon: bool | None,
    pred_earliest: int | None = None,
    final_earliest: int | None = None,
) -> SequenceResult:
    """Create a minimal SequenceResult for testing."""
    return SequenceResult(
        sequence_annotation_id="test-seq",
        trust_level="default",
        condition_id="C4",
        ordered_candidate_ids=("a", "b"),
        step_decisions=(),
        predicted_sequence_reconstruction=pred_recon,
        predicted_earliest_reconstruction_step=pred_earliest,
        predicted_reconstruction_strength=0.5,
        final_sequence_reconstructs_target=final_recon,
        final_earliest_reconstruction_step=final_earliest,
    )
