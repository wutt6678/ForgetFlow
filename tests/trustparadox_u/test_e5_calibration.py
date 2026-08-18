"""E5-003: Calibration pipeline tests.

Tests the threshold sweep and artifact generation in e5_calibration.py
using synthetic data.  No real annotation or feature files are required.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.e5_calibration import (  # noqa: E402
    SweepResult,
    _build_features_by_id,
    _write_calibration_report,
    _write_selected_config,
    _write_threshold_sweep,
    run_threshold_sweep,
)
from experiments.trustparadox_u.e5_metrics import (  # noqa: E402
    TAU_SEM_GRID,
    ConfusionCounts,
    RowMetrics,
    SequenceMetrics,
    ThresholdSelection,
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
class _FakeSplitData:
    """Minimal SplitData-shaped object for testing."""

    split: str = "development"
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


def _make_split_data(
    n_leaking: int = 5,
    n_clean: int = 10,
    n_unresolved: int = 0,
    *,
    leak_sim: float = 0.85,
    clean_sim: float = 0.30,
) -> tuple[_FakeSplitData, list[dict]]:
    """Build synthetic split data and features for testing."""
    labels: dict[str, _FakeRowLabel] = {}
    features: list[dict] = []
    seq_ids: list[str] = []

    # Leaking rows
    for i in range(n_leaking):
        cid = f"leak_{i}"
        labels[cid] = _FakeRowLabel(cid, final_target_leakage=True, final_task_useful=False)
        features.append(_feat(cid, sim=leak_sim))
        seq_ids.append(cid)

    # Clean rows
    for i in range(n_clean):
        cid = f"clean_{i}"
        useful = i % 2 == 0  # half are useful
        labels[cid] = _FakeRowLabel(cid, final_target_leakage=False, final_task_useful=useful)
        features.append(_feat(cid, sim=clean_sim))

    # Unresolved rows
    for i in range(n_unresolved):
        cid = f"unres_{i}"
        labels[cid] = _FakeRowLabel(
            cid,
            final_target_leakage=None,
            final_task_useful=None,
            resolution_status="unresolved",
        )
        features.append(_feat(cid, sim=0.5))

    # A reconstructing sequence using leaking candidates
    seq = _FakeSeqLabel(
        "seq_0",
        ordered_candidate_ids=tuple(seq_ids),
        final_sequence_reconstructs_target=len(seq_ids) > 0,
    )

    split = _FakeSplitData(
        split="development",
        row_labels=tuple(labels.values()),
        sequence_labels=(seq,),
        row_labels_by_id=labels,
        n_rows=len(labels),
        n_sequences=1,
        n_unresolved_rows=n_unresolved,
    )

    return split, features


# ===========================================================================
# _build_features_by_id
# ===========================================================================


class TestBuildFeaturesById:
    def test_basic(self):
        feats = [_feat("a", sim=0.1), _feat("b", sim=0.2)]
        idx = _build_features_by_id(feats)
        assert set(idx.keys()) == {"a", "b"}
        assert idx["a"]["semantic_similarity"] == 0.1

    def test_empty(self):
        assert _build_features_by_id([]) == {}


# ===========================================================================
# run_threshold_sweep
# ===========================================================================


class TestRunThresholdSweep:
    """Tests for the threshold sweep logic."""

    def test_sweep_returns_one_per_tau(self):
        split, features = _make_split_data()
        results = run_threshold_sweep(split, features)
        assert len(results) == len(TAU_SEM_GRID)
        taus = [r.tau_sem for r in results]
        assert taus == TAU_SEM_GRID

    def test_sweep_custom_grid(self):
        split, features = _make_split_data()
        grid = [0.50, 0.75, 0.95]
        results = run_threshold_sweep(split, features, tau_grid=grid)
        assert len(results) == 3
        assert [r.tau_sem for r in results] == grid

    def test_low_threshold_catches_all_leaking(self):
        """With τ below all leaking similarities, recall should be 1.0."""
        split, features = _make_split_data(n_leaking=5, leak_sim=0.85)
        results = run_threshold_sweep(split, features, tau_grid=[0.50])
        assert results[0].row_metrics.leakage_recall == 1.0

    def test_high_threshold_misses_leaking(self):
        """With τ above all leaking similarities, recall drops."""
        split, features = _make_split_data(n_leaking=5, leak_sim=0.85)
        results = run_threshold_sweep(split, features, tau_grid=[0.95])
        assert results[0].row_metrics.leakage_recall == 0.0

    def test_sweep_result_to_dict(self):
        split, features = _make_split_data()
        results = run_threshold_sweep(split, features, tau_grid=[0.75])
        d = results[0].to_dict()
        assert "tau_sem" in d
        assert "leakage_precision" in d
        assert "leakage_recall" in d
        assert "false_blocking_rate" in d
        assert "utility_retention" in d
        assert "sequence_reconstruction_recall" in d
        assert "tp" in d
        assert "tn" in d

    def test_unresolved_excluded_from_metrics(self):
        split, features = _make_split_data(n_unresolved=3)
        results = run_threshold_sweep(split, features, tau_grid=[0.75])
        assert results[0].row_metrics.n_unresolved_excluded == 3

    def test_sequence_metrics_computed(self):
        split, features = _make_split_data(n_leaking=3, leak_sim=0.85)
        results = run_threshold_sweep(split, features, tau_grid=[0.75])
        seq_m = results[0].sequence_metrics
        assert seq_m.n_reconstructing_sequences == 1
        # At τ=0.75, leaking sim=0.85 should be detected
        assert seq_m.n_reconstructing_caught == 1


# ===========================================================================
# Artifact writers
# ===========================================================================


class TestWriteThresholdSweep:
    def test_writes_jsonl(self, tmp_path):
        split, features = _make_split_data()
        results = run_threshold_sweep(split, features, tau_grid=[0.70, 0.80])
        path = tmp_path / "threshold_sweep.jsonl"
        _write_threshold_sweep(results, path)

        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        rec = json.loads(lines[0])
        assert "tau_sem" in rec


class TestWriteSelectedConfig:
    def test_writes_valid_json(self, tmp_path):
        path = tmp_path / "selected_config.json"
        sweep_result = SweepResult(
            tau_sem=0.75,
            row_metrics=RowMetrics(
                tau_sem=0.75,
                counts=ConfusionCounts(tp=5, fp=1, fn=0, tn=9),
                leakage_precision=5 / 6,
                leakage_recall=1.0,
                leakage_f1=2 * (5 / 6) / (1 + 5 / 6),
                false_blocking_rate=1 / 10,
                utility_retention=0.9,
                n_eligible=15,
                n_unresolved_excluded=0,
                n_useful_eligible=5,
                n_useful_not_blocked=4,
                n_non_leaking_eligible=10,
            ),
            sequence_metrics=SequenceMetrics(
                tau_sem=0.75,
                sequence_reconstruction_recall=1.0,
                sequence_leakage_rate=0.8,
                n_reconstructing_sequences=1,
                n_reconstructing_caught=1,
                n_eligible_sequences=1,
                n_unresolved_excluded=0,
            ),
        )
        selection = ThresholdSelection(
            selected_tau=0.75,
            leakage_recall=1.0,
            false_blocking_rate=0.1,
            utility_retention=0.9,
            selection_rule="recall>=0.9 -> lowest FBR -> highest UR",
        )
        config = _write_selected_config(
            selection=selection,
            best_sweep=sweep_result,
            code_commit="abc123",
            output_path=path,
        )

        assert path.exists()
        data = json.loads(path.read_text())
        assert config["semantic_threshold"] == 0.75
        assert config["code_commit"] == "abc123"
        assert "embedding_model" in data
        assert "normalization" in data
        assert "policy_configuration" in data
        assert "selection_rule" in data
        assert "development_metric_summary" in data
        assert data["policy_configuration"]["trust_invariant"] is True


class TestWriteCalibrationReport:
    def test_writes_full_report(self, tmp_path):
        split, features = _make_split_data()
        results = run_threshold_sweep(split, features, tau_grid=[0.70, 0.80])
        selection = ThresholdSelection(
            selected_tau=0.70,
            leakage_recall=1.0,
            false_blocking_rate=0.05,
            utility_retention=0.95,
            selection_rule="test rule",
        )
        path = tmp_path / "calibration_report.json"

        # Use a real SplitData-like object
        report = _write_calibration_report(
            sweep_results=results,
            selection=selection,
            split_data=split,
            n_features=len(features),
            code_commit="abc123",
            output_path=path,
        )

        assert path.exists()
        data = json.loads(path.read_text())
        assert report["split"] == "development"
        assert data["split"] == "development"
        assert data["code_commit"] == "abc123"
        assert len(data["sweep_results"]) == 2
        assert data["selection"]["selected_tau"] == 0.70
        assert "data_summary" in data
        assert "tau_grid" in data


# ===========================================================================
# Split isolation
# ===========================================================================


class TestSplitIsolation:
    """Calibration must only use development split data."""

    def test_run_calibration_rejects_non_development(self):
        """run_calibration must refuse non-development splits."""
        from experiments.trustparadox_u.e5_calibration import run_calibration

        with pytest.raises(ValueError, match="development"):
            run_calibration(split="test")

    def test_sweep_uses_only_provided_data(self):
        """run_threshold_sweep only processes the data it's given."""
        split, features = _make_split_data(n_leaking=2, n_clean=3)
        results = run_threshold_sweep(split, features, tau_grid=[0.75])
        # Total eligible = 5 (2 leaking + 3 clean)
        assert results[0].row_metrics.n_eligible == 5
