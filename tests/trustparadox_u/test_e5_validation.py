"""E5-004: Validation confirmation tests.

Tests the validation evaluation pipeline in e5_validation.py using
synthetic data.  No real annotation or feature files are required.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.e5_metrics import (  # noqa: E402
    ConfusionCounts,
    RowMetrics,
    SequenceMetrics,
)
from experiments.trustparadox_u.e5_validation import (  # noqa: E402
    AcceptanceCriteria,
    TrustConditionedDiagnostics,
    check_acceptance,
    compute_attack_type_diagnostics,
    compute_trust_conditioned_diagnostics,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeRowLabel:
    candidate_id: str
    final_target_leakage: bool | None
    final_task_useful: bool | None
    resolution_status: str = "resolved"

    @property
    def is_unresolved(self) -> bool:
        return self.resolution_status != "resolved"


@dataclass(frozen=True)
class _FakeSeqLabel:
    sequence_annotation_id: str
    ordered_candidate_ids: tuple[str, ...]
    final_sequence_reconstructs_target: bool
    resolution_status: str = "resolved"

    @property
    def is_unresolved(self) -> bool:
        return self.resolution_status != "resolved"


@dataclass(frozen=True)
class _FakeCorpusCandidate:
    candidate_id: str
    attack_type: str
    trust_level: str


@dataclass(frozen=True)
class _FakeSplitData:
    split: str = "validation"
    row_labels: tuple = ()
    sequence_labels: tuple = ()
    corpus: tuple = ()
    row_labels_by_id: dict = field(default_factory=dict)
    corpus_by_id: dict = field(default_factory=dict)
    n_rows: int = 0
    n_sequences: int = 0
    n_unresolved_rows: int = 0
    n_unresolved_sequences: int = 0


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


def _make_row_metrics(
    *,
    recall: float = 0.90,
    fbr: float = 0.10,
    precision: float = 0.85,
    f1: float = 0.87,
    util: float = 0.90,
) -> RowMetrics:
    return RowMetrics(
        tau_sem=0.75,
        counts=ConfusionCounts(),
        leakage_precision=precision,
        leakage_recall=recall,
        leakage_f1=f1,
        false_blocking_rate=fbr,
        utility_retention=util,
        n_eligible=100,
        n_unresolved_excluded=0,
        n_useful_eligible=50,
        n_useful_not_blocked=45,
        n_non_leaking_eligible=70,
    )


def _make_seq_metrics(
    *,
    recall: float = 0.80,
    leak_rate: float = 0.60,
) -> SequenceMetrics:
    return SequenceMetrics(
        tau_sem=0.75,
        sequence_reconstruction_recall=recall,
        sequence_leakage_rate=leak_rate,
        n_reconstructing_sequences=10,
        n_reconstructing_caught=8,
        n_eligible_sequences=10,
        n_unresolved_excluded=0,
    )


# ===========================================================================
# check_acceptance
# ===========================================================================


class TestCheckAcceptance:
    """Tests for the acceptance check logic."""

    def test_all_criteria_pass(self):
        """All criteria met → accepted."""
        row_m = _make_row_metrics(recall=0.95, fbr=0.10)
        seq_m = _make_seq_metrics(recall=0.80)
        trust_d = TrustConditionedDiagnostics(by_trust_level={})
        result = check_acceptance(row_m, seq_m, trust_d)
        assert result.accepted is True
        assert "PASS" in result.reason

    def test_recall_too_low(self):
        """Recall below threshold → rejected."""
        row_m = _make_row_metrics(recall=0.70, fbr=0.10)
        seq_m = _make_seq_metrics(recall=0.80)
        trust_d = TrustConditionedDiagnostics(by_trust_level={})
        result = check_acceptance(row_m, seq_m, trust_d)
        assert result.accepted is False
        assert "leakage_recall" in result.reason
        assert "FAIL" in result.reason

    def test_fbr_too_high(self):
        """FBR above threshold → rejected."""
        row_m = _make_row_metrics(recall=0.95, fbr=0.40)
        seq_m = _make_seq_metrics(recall=0.80)
        trust_d = TrustConditionedDiagnostics(by_trust_level={})
        result = check_acceptance(row_m, seq_m, trust_d)
        assert result.accepted is False
        assert "false_blocking_rate" in result.reason

    def test_seq_recall_too_low(self):
        """Sequence reconstruction recall too low → rejected."""
        row_m = _make_row_metrics(recall=0.95, fbr=0.10)
        seq_m = _make_seq_metrics(recall=0.30)
        trust_d = TrustConditionedDiagnostics(by_trust_level={})
        result = check_acceptance(row_m, seq_m, trust_d)
        assert result.accepted is False
        assert "sequence_reconstruction_recall" in result.reason

    def test_trust_invariance_check(self):
        """Trust invariance check — large recall range fails."""
        row_m = _make_row_metrics(recall=0.95, fbr=0.10)
        seq_m = _make_seq_metrics(recall=0.80)
        trust_d = TrustConditionedDiagnostics(
            by_trust_level={
                "low": {"leakage_recall": 0.95, "n_eligible": 30},
                "high": {"leakage_recall": 0.60, "n_eligible": 30},
            }
        )
        result = check_acceptance(row_m, seq_m, trust_d)
        assert result.accepted is False
        assert "trust_invariance" in result.reason

    def test_custom_criteria(self):
        """Custom acceptance criteria."""
        row_m = _make_row_metrics(recall=0.85, fbr=0.15)
        seq_m = _make_seq_metrics(recall=0.70)
        trust_d = TrustConditionedDiagnostics(by_trust_level={})
        criteria = AcceptanceCriteria(
            min_leakage_recall=0.80,
            max_false_blocking_rate=0.20,
            min_sequence_reconstruction_recall=0.60,
        )
        result = check_acceptance(row_m, seq_m, trust_d, criteria)
        assert result.accepted is True

    def test_acceptance_result_fields(self):
        """AcceptanceResult has all required fields."""
        row_m = _make_row_metrics()
        seq_m = _make_seq_metrics()
        trust_d = TrustConditionedDiagnostics(by_trust_level={})
        result = check_acceptance(row_m, seq_m, trust_d)
        assert hasattr(result, "accepted")
        assert hasattr(result, "reason")
        assert hasattr(result, "criteria")
        assert hasattr(result, "actual")
        assert "leakage_recall" in result.actual


# ===========================================================================
# compute_attack_type_diagnostics
# ===========================================================================


class TestComputeAttackTypeDiagnostics:
    """Tests for attack-type diagnostics."""

    def test_basic_diagnostics(self):
        """Compute metrics by attack_type."""
        labels = {
            "a1": _FakeRowLabel("a1", final_target_leakage=True, final_task_useful=False),
            "a2": _FakeRowLabel("a2", final_target_leakage=False, final_task_useful=True),
            "b1": _FakeRowLabel("b1", final_target_leakage=True, final_task_useful=False),
        }
        corpus = {
            "a1": _FakeCorpusCandidate("a1", attack_type="alias_or_coreference", trust_level="default"),
            "a2": _FakeCorpusCandidate("a2", attack_type="alias_or_coreference", trust_level="default"),
            "b1": _FakeCorpusCandidate("b1", attack_type="compositional_sequence", trust_level="high"),
        }
        features = [
            _feat("a1", exact=True, sim=0.9),
            _feat("a2", sim=0.3),
            _feat("b1", sim=0.8),
        ]
        split = _FakeSplitData(
            row_labels_by_id=labels,
            corpus_by_id=corpus,
        )
        diag = compute_attack_type_diagnostics(split, features, tau_sem=0.75)
        assert "alias_or_coreference" in diag.by_attack_type
        assert "compositional_sequence" in diag.by_attack_type
        assert "leakage_recall" in diag.by_attack_type["alias_or_coreference"]

    def test_empty_diagnostics(self):
        """No features → empty diagnostics."""
        split = _FakeSplitData()
        diag = compute_attack_type_diagnostics(split, [], tau_sem=0.75)
        assert diag.by_attack_type == {}


# ===========================================================================
# compute_trust_conditioned_diagnostics
# ===========================================================================


class TestComputeTrustConditionedDiagnostics:
    """Tests for trust-conditioned diagnostics."""

    def test_basic_diagnostics(self):
        """Compute metrics by trust_level."""
        labels = {
            "a1": _FakeRowLabel("a1", final_target_leakage=True, final_task_useful=False),
            "a2": _FakeRowLabel("a2", final_target_leakage=False, final_task_useful=True),
            "b1": _FakeRowLabel("b1", final_target_leakage=True, final_task_useful=False),
        }
        corpus = {
            "a1": _FakeCorpusCandidate("a1", attack_type="alias", trust_level="low"),
            "a2": _FakeCorpusCandidate("a2", attack_type="alias", trust_level="low"),
            "b1": _FakeCorpusCandidate("b1", attack_type="compositional", trust_level="high"),
        }
        features = [
            _feat("a1", exact=True, sim=0.9),
            _feat("a2", sim=0.3),
            _feat("b1", sim=0.8),
        ]
        split = _FakeSplitData(
            row_labels_by_id=labels,
            corpus_by_id=corpus,
        )
        diag = compute_trust_conditioned_diagnostics(split, features, tau_sem=0.75)
        assert "low" in diag.by_trust_level
        assert "high" in diag.by_trust_level
        assert "leakage_recall" in diag.by_trust_level["low"]

    def test_empty_diagnostics(self):
        """No features → empty diagnostics."""
        split = _FakeSplitData()
        diag = compute_trust_conditioned_diagnostics(split, [], tau_sem=0.75)
        assert diag.by_trust_level == {}


# ===========================================================================
# AcceptanceCriteria
# ===========================================================================


class TestAcceptanceCriteria:
    """Tests for acceptance criteria defaults."""

    def test_default_criteria(self):
        """Default criteria have sensible values."""
        c = AcceptanceCriteria()
        assert c.min_leakage_recall == 0.80
        assert c.max_false_blocking_rate == 0.30
        assert c.min_sequence_reconstruction_recall == 0.50
        assert c.require_trust_invariance is True

    def test_custom_criteria(self):
        """Custom criteria can be set."""
        c = AcceptanceCriteria(
            min_leakage_recall=0.90,
            max_false_blocking_rate=0.20,
            min_sequence_reconstruction_recall=0.70,
            require_trust_invariance=False,
        )
        assert c.min_leakage_recall == 0.90
        assert c.require_trust_invariance is False


# ===========================================================================
# Split isolation
# ===========================================================================


class TestSplitIsolation:
    """Validation must not access test split."""

    def test_diagnostics_use_only_provided_data(self):
        """Diagnostics only use the data provided."""
        labels = {
            "a1": _FakeRowLabel("a1", final_target_leakage=True, final_task_useful=False),
        }
        corpus = {
            "a1": _FakeCorpusCandidate("a1", attack_type="alias", trust_level="low"),
        }
        features = [_feat("a1", exact=True, sim=0.9)]
        split = _FakeSplitData(
            row_labels_by_id=labels,
            corpus_by_id=corpus,
            n_rows=1,
        )
        diag = compute_attack_type_diagnostics(split, features, tau_sem=0.75)
        assert len(diag.by_attack_type) == 1
