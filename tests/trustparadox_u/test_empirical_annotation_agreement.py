"""E4-001 Sec 59-61: Agreement, review queue, and sequence agreement tests.

Covers:
- Sec 59: Review queue (binary disagreement, leakage-strength disagreement,
  uncertain primary/secondary, low-confidence primary/secondary, full agreement)
- Sec 60: Agreement statistics (perfect agreement, balanced disagreement,
  all-positive prevalence, all-negative prevalence, rare positives,
  not_estimable)
- Sec 61: Sequence agreement (both reconstruct same step, both reconstruct
  different steps, one reconstructs one doesn't, neither reconstructs)
"""

from __future__ import annotations

import pytest

from experiments.trustparadox_u.empirical_annotation import (
    build_review_queue,
    compute_binary_agreement,
    compute_categorical_agreement,
    compute_sequence_agreement,
    should_queue_for_review,
)


# ---------------------------------------------------------------------------
# Sec 59: Review queue
# ---------------------------------------------------------------------------


class TestReviewQueue:
    """Sec 59: Review queue entry conditions."""

    def _make_label(self, **overrides):
        base = {
            "candidate_id": "cand_001",
            "target_relevant": True,
            "target_leakage": True,
            "positive_entailment": False,
            "task_useful": True,
            "leakage_strength": "partial",
            "confidence": 0.9,
            "uncertain": False,
        }
        base.update(overrides)
        return base

    def test_binary_disagreement_queues(self):
        primary = self._make_label(target_relevant=True)
        secondary = self._make_label(target_relevant=False)
        should_queue, reasons = should_queue_for_review(primary, secondary)
        assert should_queue
        assert any("target_relevant" in r for r in reasons)

    def test_leakage_strength_disagreement_queues(self):
        primary = self._make_label(leakage_strength="partial")
        secondary = self._make_label(leakage_strength="none")
        should_queue, reasons = should_queue_for_review(primary, secondary)
        assert should_queue
        assert any("leakage_strength" in r for r in reasons)

    def test_uncertain_primary_queues(self):
        primary = self._make_label(uncertain=True)
        secondary = self._make_label()
        should_queue, reasons = should_queue_for_review(primary, secondary)
        assert should_queue
        assert any("primary uncertain" in r for r in reasons)

    def test_uncertain_secondary_queues(self):
        primary = self._make_label()
        secondary = self._make_label(uncertain=True)
        should_queue, reasons = should_queue_for_review(primary, secondary)
        assert should_queue
        assert any("secondary uncertain" in r for r in reasons)

    def test_low_confidence_primary_queues(self):
        primary = self._make_label(confidence=0.5)
        secondary = self._make_label()
        should_queue, reasons = should_queue_for_review(primary, secondary)
        assert should_queue
        assert any("primary confidence" in r for r in reasons)

    def test_low_confidence_secondary_queues(self):
        primary = self._make_label()
        secondary = self._make_label(confidence=0.6)
        should_queue, reasons = should_queue_for_review(primary, secondary)
        assert should_queue
        assert any("secondary confidence" in r for r in reasons)

    def test_full_agreement_not_queued(self):
        primary = self._make_label()
        secondary = self._make_label()
        should_queue, reasons = should_queue_for_review(primary, secondary)
        assert not should_queue
        assert reasons == []

    def test_confidence_exactly_07_not_queued(self):
        """Confidence == 0.7 is NOT < 0.7, so it should not queue on its own."""
        primary = self._make_label(confidence=0.7)
        secondary = self._make_label(confidence=0.7)
        should_queue, reasons = should_queue_for_review(primary, secondary)
        assert not should_queue

    def test_multiple_disagreements_all_reported(self):
        primary = self._make_label(
            target_relevant=True, target_leakage=True,
            positive_entailment=True, task_useful=True,
        )
        secondary = self._make_label(
            target_relevant=False, target_leakage=False,
            positive_entailment=False, task_useful=False,
        )
        should_queue, reasons = should_queue_for_review(primary, secondary)
        assert should_queue
        assert len(reasons) >= 4  # At least 4 binary disagreements

    def test_sequence_reconstruction_disagreement_queues(self):
        primary = self._make_label(sequence_reconstructs_target=True)
        secondary = self._make_label(sequence_reconstructs_target=False)
        should_queue, reasons = should_queue_for_review(primary, secondary)
        assert should_queue
        assert any("sequence_reconstructs_target" in r for r in reasons)


class TestBuildReviewQueue:
    """Sec 59: Build the full review queue from label pairs."""

    def test_build_row_review_queue(self):
        primary_labels = [
            {"candidate_id": "c1", "target_relevant": True, "target_leakage": True,
             "positive_entailment": False, "task_useful": True,
             "leakage_strength": "partial", "confidence": 0.9, "uncertain": False},
            {"candidate_id": "c2", "target_relevant": True, "target_leakage": True,
             "positive_entailment": False, "task_useful": True,
             "leakage_strength": "none", "confidence": 0.9, "uncertain": False},
        ]
        secondary_labels = [
            {"candidate_id": "c1", "target_relevant": False, "target_leakage": True,
             "positive_entailment": False, "task_useful": True,
             "leakage_strength": "partial", "confidence": 0.9, "uncertain": False},
            {"candidate_id": "c2", "target_relevant": True, "target_leakage": True,
             "positive_entailment": False, "task_useful": True,
             "leakage_strength": "none", "confidence": 0.9, "uncertain": False},
        ]
        queue = build_review_queue(primary_labels, secondary_labels)
        # c1 disagrees on target_relevant, c2 agrees on everything
        assert len(queue) == 1
        assert queue[0]["candidate_id"] == "c1"
        assert queue[0]["item_type"] == "row"

    def test_build_includes_sequence_review(self):
        primary_row = [{"candidate_id": "c1", "target_relevant": True,
                        "target_leakage": True, "positive_entailment": True,
                        "task_useful": True, "leakage_strength": "none",
                        "confidence": 0.9, "uncertain": False}]
        secondary_row = [{"candidate_id": "c1", "target_relevant": True,
                          "target_leakage": True, "positive_entailment": True,
                          "task_useful": True, "leakage_strength": "none",
                          "confidence": 0.9, "uncertain": False}]
        primary_seq = [{"sequence_family_id": "sf1", "sequence_reconstructs_target": True,
                        "confidence": 0.9, "uncertain": False}]
        secondary_seq = [{"sequence_family_id": "sf1", "sequence_reconstructs_target": False,
                          "confidence": 0.9, "uncertain": False}]
        queue = build_review_queue(primary_row, secondary_row, primary_seq, secondary_seq)
        seq_items = [q for q in queue if q["item_type"] == "sequence"]
        assert len(seq_items) == 1

    def test_empty_labels_empty_queue(self):
        queue = build_review_queue([], [])
        assert queue == []


# ---------------------------------------------------------------------------
# Sec 60: Agreement statistics
# ---------------------------------------------------------------------------


class TestBinaryAgreement:
    """Sec 60: Binary agreement metrics with deterministic fixtures."""

    def test_perfect_agreement(self):
        labels = [True, False, True, False, True]
        result = compute_binary_agreement(labels, labels)
        assert result["raw_agreement"] == 1.0
        assert result["cohens_kappa"] == 1.0
        assert result["positive_agreement"] == 1.0
        assert result["negative_agreement"] == 1.0
        assert result["tp"] == 3
        assert result["tn"] == 2
        assert result["fp"] == 0
        assert result["fn"] == 0

    def test_balanced_disagreement(self):
        labels_a = [True, True, False, False]
        labels_b = [True, False, False, True]
        result = compute_binary_agreement(labels_a, labels_b)
        assert result["raw_agreement"] == 0.5
        assert result["tp"] == 1
        assert result["tn"] == 1
        assert result["fp"] == 1
        assert result["fn"] == 1

    def test_all_positive_prevalence(self):
        labels_a = [True, True, True, True]
        labels_b = [True, True, True, True]
        result = compute_binary_agreement(labels_a, labels_b)
        assert result["raw_agreement"] == 1.0
        # Kappa not estimable when only one class present
        assert result["cohens_kappa"] == "not_estimable"
        assert result["positive_agreement"] == 1.0
        # negative_agreement: tn=0, fp=0, fn=0 → 0/0 → 0.0
        assert result["negative_agreement"] == 0.0

    def test_all_negative_prevalence(self):
        labels_a = [False, False, False, False]
        labels_b = [False, False, False, False]
        result = compute_binary_agreement(labels_a, labels_b)
        assert result["raw_agreement"] == 1.0
        assert result["cohens_kappa"] == "not_estimable"
        assert result["negative_agreement"] == 1.0
        assert result["positive_agreement"] == 0.0

    def test_rare_positives(self):
        labels_a = [True, False, False, False, False, False, False, False, False, False]
        labels_b = [True, False, False, False, False, False, False, False, False, False]
        result = compute_binary_agreement(labels_a, labels_b)
        assert result["raw_agreement"] == 1.0
        # Kappa may not be estimable with very rare positives
        # (depends on marginals)
        assert result["tp"] == 1
        assert result["tn"] == 9

    def test_empty_labels(self):
        result = compute_binary_agreement([], [])
        assert result["raw_agreement"] == 0.0
        assert result["cohens_kappa"] == "not_estimable"
        assert result["n"] == 0

    def test_complete_disagreement(self):
        labels_a = [True, False, True, False]
        labels_b = [False, True, False, True]
        result = compute_binary_agreement(labels_a, labels_b)
        assert result["raw_agreement"] == 0.0
        assert result["tp"] == 0
        assert result["tn"] == 0
        assert result["fp"] == 2
        assert result["fn"] == 2

    def test_confusion_counts_sum_to_n(self):
        labels_a = [True, True, False, False, True]
        labels_b = [True, False, False, True, True]
        result = compute_binary_agreement(labels_a, labels_b)
        total = result["tp"] + result["fp"] + result["fn"] + result["tn"]
        assert total == result["n"]


# ---------------------------------------------------------------------------
# Sec 60 (extension): Categorical agreement (leakage strength)
# ---------------------------------------------------------------------------


class TestCategoricalAgreement:
    """Sec 37/60: Categorical agreement for leakage_strength."""

    def test_perfect_categorical(self):
        labels = ["none", "partial", "full", "none", "partial"]
        result = compute_categorical_agreement(labels, labels)
        assert result["exact_agreement"] == 1.0
        assert result["cohens_kappa"] == 1.0

    def test_single_category_not_estimable(self):
        labels = ["none", "none", "none"]
        result = compute_categorical_agreement(labels, labels)
        assert result["exact_agreement"] == 1.0
        assert result["cohens_kappa"] == "not_estimable"

    def test_balanced_categorical_disagreement(self):
        labels_a = ["none", "partial", "full", "none"]
        labels_b = ["partial", "none", "none", "full"]
        result = compute_categorical_agreement(labels_a, labels_b)
        assert result["exact_agreement"] == 0.0
        assert isinstance(result["cohens_kappa"], float)
        assert result["cohens_kappa"] < 0  # Worse than chance

    def test_confusion_matrix_present(self):
        labels_a = ["none", "partial", "full"]
        labels_b = ["none", "partial", "none"]
        result = compute_categorical_agreement(labels_a, labels_b)
        cm = result["confusion_matrix"]
        assert "none" in cm
        assert "partial" in cm
        assert "full" in cm
        # Diagonal should show agreements
        assert cm["none"]["none"] == 1
        assert cm["partial"]["partial"] == 1
        assert cm["full"]["none"] == 1  # full→none disagreement

    def test_empty_categorical(self):
        result = compute_categorical_agreement([], [])
        assert result["exact_agreement"] == 0.0
        assert result["cohens_kappa"] == "not_estimable"
        assert result["n"] == 0


# ---------------------------------------------------------------------------
# Sec 61: Sequence agreement
# ---------------------------------------------------------------------------


class TestSequenceAgreement:
    """Sec 61: Sequence-level agreement."""

    def _seq_label(self, family_id, reconstructs, step=None, strength="none"):
        return {
            "sequence_family_id": family_id,
            "sequence_reconstructs_target": reconstructs,
            "earliest_reconstruction_step": step,
            "reconstruction_strength": strength,
        }

    def test_both_reconstruct_same_step(self):
        primary = [self._seq_label("sf1", True, step=1, strength="full")]
        secondary = [self._seq_label("sf1", True, step=1, strength="full")]
        result = compute_sequence_agreement(primary, secondary)
        assert result["n"] == 1
        assert result["reconstruction_binary_agreement"]["raw_agreement"] == 1.0
        assert result["earliest_step_exact_agreement"] == 1.0
        assert result["earliest_step_n"] == 1

    def test_both_reconstruct_different_steps(self):
        primary = [self._seq_label("sf1", True, step=1, strength="partial")]
        secondary = [self._seq_label("sf1", True, step=2, strength="partial")]
        result = compute_sequence_agreement(primary, secondary)
        assert result["n"] == 1
        assert result["reconstruction_binary_agreement"]["raw_agreement"] == 1.0
        # Both reconstruct but at different steps
        assert result["earliest_step_exact_agreement"] == 0.0
        assert result["earliest_step_n"] == 1

    def test_one_reconstructs_one_does_not(self):
        primary = [self._seq_label("sf1", True, step=1, strength="partial")]
        secondary = [self._seq_label("sf1", False, step=None, strength="none")]
        result = compute_sequence_agreement(primary, secondary)
        assert result["n"] == 1
        assert result["reconstruction_binary_agreement"]["raw_agreement"] == 0.0
        # No sequences where both reconstruct → not_estimable
        assert result["earliest_step_exact_agreement"] == "not_estimable"
        assert result["earliest_step_n"] == 0

    def test_neither_reconstructs(self):
        primary = [self._seq_label("sf1", False, step=None, strength="none")]
        secondary = [self._seq_label("sf1", False, step=None, strength="none")]
        result = compute_sequence_agreement(primary, secondary)
        assert result["n"] == 1
        assert result["reconstruction_binary_agreement"]["raw_agreement"] == 1.0
        assert result["earliest_step_exact_agreement"] == "not_estimable"
        assert result["earliest_step_n"] == 0

    def test_multiple_sequences_mixed(self):
        primary = [
            self._seq_label("sf1", True, step=0, strength="full"),
            self._seq_label("sf2", True, step=2, strength="partial"),
            self._seq_label("sf3", False, step=None, strength="none"),
        ]
        secondary = [
            self._seq_label("sf1", True, step=0, strength="full"),
            self._seq_label("sf2", False, step=None, strength="none"),
            self._seq_label("sf3", False, step=None, strength="none"),
        ]
        result = compute_sequence_agreement(primary, secondary)
        assert result["n"] == 3
        # sf1: agree (both True), sf2: disagree, sf3: agree (both False)
        assert result["reconstruction_binary_agreement"]["raw_agreement"] == round(2 / 3, 4)
        # Only sf1 has both reconstructing → step agree = 1.0
        assert result["earliest_step_exact_agreement"] == 1.0
        assert result["earliest_step_n"] == 1

    def test_empty_sequence_agreement(self):
        result = compute_sequence_agreement([], [])
        assert result["n"] == 0

    def test_no_common_families(self):
        primary = [self._seq_label("sf1", True, step=0)]
        secondary = [self._seq_label("sf2", True, step=0)]
        result = compute_sequence_agreement(primary, secondary)
        assert result["n"] == 0


# ---------------------------------------------------------------------------
# Sec 40: Adjudication policy
# ---------------------------------------------------------------------------


class TestAdjudicationPolicy:
    """Sec 40: Development adjudication policy."""

    def test_consensus_when_agree(self):
        from experiments.trustparadox_u.empirical_annotation import adjudicate_row
        label = {
            "target_relevant": True, "target_leakage": True,
            "positive_entailment": False, "task_useful": True,
            "leakage_strength": "partial",
        }
        result = adjudicate_row(label, dict(label))
        assert result["status"] == "consensus"

    def test_unresolved_when_disagree(self):
        from experiments.trustparadox_u.empirical_annotation import adjudicate_row
        primary = {
            "target_relevant": True, "target_leakage": True,
            "positive_entailment": False, "task_useful": True,
            "leakage_strength": "partial",
        }
        secondary = {
            "target_relevant": False, "target_leakage": True,
            "positive_entailment": False, "task_useful": True,
            "leakage_strength": "partial",
        }
        result = adjudicate_row(primary, secondary)
        assert result["status"] == "unresolved"

    def test_human_adjudication_overrides(self):
        from experiments.trustparadox_u.empirical_annotation import adjudicate_row
        primary = {"target_relevant": True, "target_leakage": True,
                   "positive_entailment": False, "task_useful": True,
                   "leakage_strength": "partial"}
        secondary = {"target_relevant": False, "target_leakage": False,
                     "positive_entailment": False, "task_useful": True,
                     "leakage_strength": "none"}
        human = {"target_relevant": True, "target_leakage": False,
                 "positive_entailment": False, "task_useful": True,
                 "leakage_strength": "none"}
        result = adjudicate_row(primary, secondary, human_adjudication=human)
        assert result["status"] == "human_adjudicated"
        assert result["labels"] == human
