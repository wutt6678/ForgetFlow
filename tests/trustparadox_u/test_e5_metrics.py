"""E5-003: Core metric function tests.

Tests the pure metric functions in e5_metrics.py using synthetic
feature and label data.  No real embedding or annotation files are
required.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.e5_metrics import (  # noqa: E402
    TAU_SEM_GRID,
    ConfusionCounts,
    RowMetrics,
    compute_row_metrics,
    compute_sequence_metrics,
    is_detected,
    select_threshold,
)

# ---------------------------------------------------------------------------
# Helpers — lightweight stand-ins for e5_loaders dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeRowLabel:
    """Minimal RowLabel-shaped object for testing."""

    candidate_id: str
    final_target_leakage: bool | None
    final_task_useful: bool | None
    resolution_status: str = "resolved"

    @property
    def is_unresolved(self) -> bool:
        return self.resolution_status != "resolved"


@dataclass(frozen=True)
class _FakeSeqLabel:
    """Minimal SequenceLabel-shaped object for testing."""

    sequence_annotation_id: str
    ordered_candidate_ids: tuple[str, ...]
    final_sequence_reconstructs_target: bool
    resolution_status: str = "resolved"

    @property
    def is_unresolved(self) -> bool:
        return self.resolution_status != "resolved"


def _feat(
    cid: str,
    *,
    exact: bool = False,
    alias: bool = False,
    sim: float = 0.0,
) -> dict:
    """Build a minimal feature dict."""
    return {
        "candidate_id": cid,
        "exact_match": exact,
        "alias_match": alias,
        "semantic_similarity": sim,
    }


# ===========================================================================
# is_detected
# ===========================================================================


class TestIsDetected:
    """Tests for the is_detected() decision rule."""

    def test_exact_match_alone(self):
        assert is_detected(True, False, 0.0, 0.75) is True

    def test_alias_match_alone(self):
        assert is_detected(False, True, 0.0, 0.75) is True

    def test_semantic_above_threshold(self):
        assert is_detected(False, False, 0.80, 0.75) is True

    def test_semantic_at_threshold(self):
        """semantic_similarity >= τ_sem should be detected."""
        assert is_detected(False, False, 0.75, 0.75) is True

    def test_semantic_below_threshold(self):
        assert is_detected(False, False, 0.50, 0.75) is False

    def test_no_signal(self):
        assert is_detected(False, False, 0.0, 0.75) is False

    def test_all_signals(self):
        assert is_detected(True, True, 0.99, 0.75) is True


# ===========================================================================
# ConfusionCounts
# ===========================================================================


class TestConfusionCounts:
    """Tests for ConfusionCounts derived properties."""

    def test_properties(self):
        c = ConfusionCounts(tp=10, fp=5, fn=3, tn=82)
        assert c.n_predicted_positive == 15
        assert c.n_actual_positive == 13
        assert c.n_predicted_negative == 85
        assert c.n_actual_negative == 87
        assert c.total == 100

    def test_zero_counts(self):
        c = ConfusionCounts()
        assert c.total == 0
        assert c.n_predicted_positive == 0


# ===========================================================================
# compute_row_metrics
# ===========================================================================


class TestComputeRowMetrics:
    """Tests for compute_row_metrics()."""

    def test_perfect_detection(self):
        """All leaking detected, no false positives."""
        features = [
            _feat("a", exact=True, sim=0.9),
            _feat("b", sim=0.3),
            _feat("c", sim=0.2),
        ]
        labels = {
            "a": _FakeRowLabel("a", final_target_leakage=True, final_task_useful=False),
            "b": _FakeRowLabel("b", final_target_leakage=False, final_task_useful=True),
            "c": _FakeRowLabel("c", final_target_leakage=False, final_task_useful=True),
        }
        m = compute_row_metrics(features, labels, tau_sem=0.75)
        assert m.counts.tp == 1
        assert m.counts.tn == 2
        assert m.counts.fp == 0
        assert m.counts.fn == 0
        assert m.leakage_precision == 1.0
        assert m.leakage_recall == 1.0
        assert m.leakage_f1 == 1.0
        assert m.false_blocking_rate == 0.0
        assert m.utility_retention == 1.0  # both useful rows not blocked
        assert m.n_eligible == 3

    def test_missed_leakage(self):
        """One leaking row missed by detector."""
        features = [
            _feat("a", sim=0.5),  # leaking but not detected
            _feat("b", sim=0.2),
        ]
        labels = {
            "a": _FakeRowLabel("a", final_target_leakage=True, final_task_useful=False),
            "b": _FakeRowLabel("b", final_target_leakage=False, final_task_useful=True),
        }
        m = compute_row_metrics(features, labels, tau_sem=0.75)
        assert m.counts.fn == 1
        assert m.counts.tn == 1
        assert m.leakage_recall == 0.0
        assert m.leakage_precision == 0.0  # no positives predicted
        assert m.utility_retention == 1.0

    def test_false_positive(self):
        """Non-leaking row incorrectly flagged."""
        features = [
            _feat("a", sim=0.9),  # not leaking but detected
            _feat("b", sim=0.2),
        ]
        labels = {
            "a": _FakeRowLabel("a", final_target_leakage=False, final_task_useful=True),
            "b": _FakeRowLabel("b", final_target_leakage=False, final_task_useful=True),
        }
        m = compute_row_metrics(features, labels, tau_sem=0.75)
        assert m.counts.fp == 1
        assert m.counts.tn == 1
        assert m.false_blocking_rate == 0.5
        assert m.utility_retention == 0.5  # 1 of 2 useful not blocked

    def test_unresolved_excluded(self):
        """Unresolved rows are excluded from metrics."""
        features = [
            _feat("a", exact=True, sim=0.9),
            _feat("b", sim=0.2),
            _feat("c", sim=0.8),  # unresolved
        ]
        labels = {
            "a": _FakeRowLabel("a", final_target_leakage=True, final_task_useful=False),
            "b": _FakeRowLabel("b", final_target_leakage=False, final_task_useful=True),
            "c": _FakeRowLabel(
                "c",
                final_target_leakage=None,
                final_task_useful=None,
                resolution_status="unresolved",
            ),
        }
        m = compute_row_metrics(features, labels, tau_sem=0.75)
        assert m.n_unresolved_excluded == 1
        assert m.n_eligible == 2  # only resolved rows
        assert m.counts.tp == 1
        assert m.counts.tn == 1

    def test_missing_label_raises(self):
        """Feature with no matching label should raise ValueError."""
        features = [_feat("x", sim=0.5)]
        labels: dict = {}
        with pytest.raises(ValueError, match="no matching row label"):
            compute_row_metrics(features, labels, tau_sem=0.75)

    def test_empty_features(self):
        """Empty feature list yields zero metrics."""
        m = compute_row_metrics([], {}, tau_sem=0.75)
        assert m.n_eligible == 0
        assert m.leakage_precision == 0.0
        assert m.leakage_recall == 0.0

    def test_threshold_boundary(self):
        """Candidates at exactly τ_sem are detected."""
        features = [_feat("a", sim=0.75)]
        labels = {
            "a": _FakeRowLabel("a", final_target_leakage=True, final_task_useful=False),
        }
        m = compute_row_metrics(features, labels, tau_sem=0.75)
        assert m.counts.tp == 1  # detected

        m2 = compute_row_metrics(features, labels, tau_sem=0.76)
        assert m2.counts.fn == 1  # not detected at higher threshold

    def test_leakage_none_excluded_from_metric(self):
        """Resolved row with final_target_leakage=None → excluded (§39)."""
        features = [_feat("a", exact=True, sim=0.9)]
        labels = {
            "a": _FakeRowLabel("a", final_target_leakage=None, final_task_useful=True),
        }
        m = compute_row_metrics(features, labels, tau_sem=0.75)
        # None → excluded from leakage metric, not treated as FP
        assert m.n_eligible == 0
        assert m.n_unresolved_excluded == 1  # excluded due to field-None
        assert m.counts.fp == 0


# ===========================================================================
# compute_sequence_metrics
# ===========================================================================


class TestComputeSequenceMetrics:
    """Tests for compute_sequence_metrics()."""

    def test_reconstructing_sequence_caught(self):
        """A reconstructing sequence where one candidate is detected."""
        seqs = [
            _FakeSeqLabel(
                "s1",
                ordered_candidate_ids=("c1", "c2"),
                final_sequence_reconstructs_target=True,
            ),
        ]
        feats = {
            "c1": _feat("c1", sim=0.3),
            "c2": _feat("c2", exact=True, sim=0.9),
        }
        m = compute_sequence_metrics(seqs, feats, tau_sem=0.75)
        assert m.n_reconstructing_sequences == 1
        assert m.n_reconstructing_caught == 1
        assert m.sequence_reconstruction_recall == 1.0

    def test_reconstructing_sequence_missed(self):
        """A reconstructing sequence where no candidate is detected."""
        seqs = [
            _FakeSeqLabel(
                "s1",
                ordered_candidate_ids=("c1", "c2"),
                final_sequence_reconstructs_target=True,
            ),
        ]
        feats = {
            "c1": _feat("c1", sim=0.3),
            "c2": _feat("c2", sim=0.4),
        }
        m = compute_sequence_metrics(seqs, feats, tau_sem=0.75)
        assert m.n_reconstructing_caught == 0
        assert m.sequence_reconstruction_recall == 0.0

    def test_non_reconstructing_ignored(self):
        """Non-reconstructing sequences don't count."""
        seqs = [
            _FakeSeqLabel(
                "s1",
                ordered_candidate_ids=("c1",),
                final_sequence_reconstructs_target=False,
            ),
        ]
        feats = {"c1": _feat("c1", exact=True, sim=0.9)}
        m = compute_sequence_metrics(seqs, feats, tau_sem=0.75)
        assert m.n_reconstructing_sequences == 0
        assert m.n_eligible_sequences == 1

    def test_unresolved_sequence_excluded(self):
        """Unresolved sequences are excluded."""
        seqs = [
            _FakeSeqLabel(
                "s1",
                ordered_candidate_ids=("c1",),
                final_sequence_reconstructs_target=True,
                resolution_status="unresolved",
            ),
        ]
        feats = {"c1": _feat("c1", exact=True, sim=0.9)}
        m = compute_sequence_metrics(seqs, feats, tau_sem=0.75)
        assert m.n_unresolved_excluded == 1
        assert m.n_eligible_sequences == 0

    def test_sequence_leakage_rate(self):
        """Sequence leakage rate = detected candidates / total in recon."""
        seqs = [
            _FakeSeqLabel(
                "s1",
                ordered_candidate_ids=("c1", "c2", "c3"),
                final_sequence_reconstructs_target=True,
            ),
        ]
        feats = {
            "c1": _feat("c1", sim=0.9),  # detected
            "c2": _feat("c2", sim=0.3),  # not detected
            "c3": _feat("c3", exact=True, sim=0.95),  # detected
        }
        m = compute_sequence_metrics(seqs, feats, tau_sem=0.75)
        assert m.sequence_leakage_rate == pytest.approx(2.0 / 3.0)

    def test_missing_feature_skipped(self):
        """Candidates with no feature record are skipped."""
        seqs = [
            _FakeSeqLabel(
                "s1",
                ordered_candidate_ids=("c1", "c_missing"),
                final_sequence_reconstructs_target=True,
            ),
        ]
        feats = {"c1": _feat("c1", exact=True, sim=0.9)}
        m = compute_sequence_metrics(seqs, feats, tau_sem=0.75)
        # Only c1 counted (c_missing has no feature)
        assert m.n_reconstructing_caught == 1


# ===========================================================================
# select_threshold
# ===========================================================================


class TestSelectThreshold:
    """Tests for the threshold selection rule."""

    def _make_row_metrics(
        self, tau: float, *, recall: float, fbr: float, util: float
    ) -> RowMetrics:
        return RowMetrics(
            tau_sem=tau,
            counts=ConfusionCounts(),
            leakage_precision=0.0,
            leakage_recall=recall,
            leakage_f1=0.0,
            false_blocking_rate=fbr,
            utility_retention=util,
            n_eligible=0,
            n_unresolved_excluded=0,
            n_useful_eligible=0,
            n_useful_not_blocked=0,
            n_non_leaking_eligible=0,
        )

    def test_select_lowest_fbr_among_qualifying(self):
        """Among recall >= 0.9, pick lowest FBR."""
        results = [
            self._make_row_metrics(0.60, recall=0.95, fbr=0.30, util=0.80),
            self._make_row_metrics(0.75, recall=0.92, fbr=0.10, util=0.90),
            self._make_row_metrics(0.90, recall=0.91, fbr=0.20, util=0.85),
        ]
        sel = select_threshold(results)
        assert sel.selected_tau == 0.75
        assert sel.leakage_recall == 0.92
        assert sel.false_blocking_rate == 0.10

    def test_utility_tiebreaker(self):
        """When FBR ties, higher utility retention wins."""
        results = [
            self._make_row_metrics(0.70, recall=0.95, fbr=0.10, util=0.80),
            self._make_row_metrics(0.80, recall=0.95, fbr=0.10, util=0.90),
        ]
        sel = select_threshold(results)
        assert sel.selected_tau == 0.80

    def test_fallback_when_no_qualifying(self):
        """If no threshold meets min_recall, pick highest recall."""
        results = [
            self._make_row_metrics(0.60, recall=0.70, fbr=0.30, util=0.80),
            self._make_row_metrics(0.75, recall=0.80, fbr=0.10, util=0.90),
            self._make_row_metrics(0.90, recall=0.60, fbr=0.05, util=0.95),
        ]
        sel = select_threshold(results)
        assert sel.selected_tau == 0.75  # highest recall = 0.80

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            select_threshold([])

    def test_selection_rule_documented(self):
        results = [
            self._make_row_metrics(0.75, recall=0.95, fbr=0.10, util=0.90),
        ]
        sel = select_threshold(results)
        assert "recall" in sel.selection_rule
        assert "FBR" in sel.selection_rule


# ===========================================================================
# Constants
# ===========================================================================


class TestConstants:
    """Tests for module-level constants."""

    def test_tau_grid_sorted(self):
        assert TAU_SEM_GRID == sorted(TAU_SEM_GRID)

    def test_tau_grid_values(self):
        assert TAU_SEM_GRID == [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    def test_all_tau_in_range(self):
        for tau in TAU_SEM_GRID:
            assert 0.0 <= tau <= 1.0
