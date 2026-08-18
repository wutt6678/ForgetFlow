"""E5-010: Hyperparameter sensitivity study tests.

Tests threshold sensitivity computation, threshold application,
threshold sweep, tradeoff data, optimal threshold selection,
and serialisation helpers.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.e5_hyperparameter_study import (  # noqa: E402
    FROZEN_TAU_SEM_GRID,
    ThresholdSensitivityRow,
    TradeoffPoint,
    apply_threshold_to_row,
    compute_threshold_sensitivity,
    compute_tradeoff_data,
    run_threshold_sweep,
    select_optimal_threshold,
    sensitivity_to_dict,
    tradeoff_to_dict,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _label(
    *,
    leakage: bool = True,
    useful: bool = False,
    unresolved: bool = False,
) -> dict:
    return {
        "final_target_leakage": leakage,
        "final_task_useful": useful,
        "is_unresolved": unresolved,
    }


def _result(
    *,
    candidate_id: str,
    blocked: bool = False,
    allowed: bool = True,
    policy_action: str = "allow",
    exact: bool = False,
    alias: bool = False,
    sim: float = 0.0,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "blocked": blocked,
        "allowed": allowed,
        "policy_action": policy_action,
        "exact_match": exact,
        "alias_match": alias,
        "semantic_similarity": sim,
    }


# ===========================================================================
# Constants
# ===========================================================================


class TestConstants:
    """Verify threshold grid."""

    def test_grid_length(self):
        assert len(FROZEN_TAU_SEM_GRID) == 7

    def test_grid_range(self):
        assert FROZEN_TAU_SEM_GRID[0] == 0.60
        assert FROZEN_TAU_SEM_GRID[-1] == 0.90

    def test_grid_sorted(self):
        assert FROZEN_TAU_SEM_GRID == sorted(FROZEN_TAU_SEM_GRID)


# ===========================================================================
# apply_threshold_to_row
# ===========================================================================


class TestApplyThresholdToRow:
    """Tests for threshold re-evaluation."""

    def test_high_threshold_fewer_detections(self):
        """High τ_sem → semantic-only detection harder."""
        row = _result(candidate_id="c1", sim=0.70, exact=False, alias=False)
        out_low = apply_threshold_to_row(row, 0.65)
        out_high = apply_threshold_to_row(row, 0.80)
        # At 0.65: sim 0.70 ≥ 0.65 → detected
        assert out_low["blocked"] is True
        # At 0.80: sim 0.70 < 0.80 → not detected
        assert out_high["blocked"] is False

    def test_exact_match_always_detected(self):
        """Exact match detected regardless of threshold."""
        row = _result(candidate_id="c1", sim=0.0, exact=True)
        out = apply_threshold_to_row(row, 0.90)
        assert out["blocked"] is True

    def test_alias_match_always_detected(self):
        """Alias match detected regardless of threshold."""
        row = _result(candidate_id="c1", sim=0.0, alias=True)
        out = apply_threshold_to_row(row, 0.90)
        assert out["blocked"] is True

    def test_policy_action_updated(self):
        """Policy action reflects new detection."""
        row = _result(candidate_id="c1", sim=0.75, blocked=True)
        out = apply_threshold_to_row(row, 0.80)
        assert out["policy_action"] == "allow"
        assert out["allowed"] is True


# ===========================================================================
# run_threshold_sweep
# ===========================================================================


class TestRunThresholdSweep:
    """Tests for threshold sweep."""

    def test_sweep_all_thresholds(self):
        """Sweep produces results for each threshold."""
        results = [_result(candidate_id="c1", sim=0.75, exact=True)]
        swept = run_threshold_sweep(results)
        assert len(swept) == 7  # TAU_SEM_GRID has 7 values

    def test_custom_thresholds(self):
        """Custom threshold list works."""
        results = [_result(candidate_id="c1", sim=0.5)]
        swept = run_threshold_sweep(results, thresholds=[0.5, 0.7])
        assert len(swept) == 2
        assert 0.5 in swept
        assert 0.7 in swept

    def test_each_threshold_has_all_rows(self):
        """Each threshold gets all rows re-evaluated."""
        results = [
            _result(candidate_id="c1", sim=0.7),
            _result(candidate_id="c2", sim=0.8),
        ]
        swept = run_threshold_sweep(results, thresholds=[0.75])
        assert len(swept[0.75]) == 2


# ===========================================================================
# compute_threshold_sensitivity
# ===========================================================================


class TestComputeThresholdSensitivity:
    """Tests for sensitivity table computation."""

    def test_basic_sensitivity(self):
        """2 leaking (1 detected), 1 non-leaking, 1 useful."""
        results = [
            _result(candidate_id="c1", blocked=True),
            _result(candidate_id="c2", blocked=False),
            _result(candidate_id="c3", blocked=False, allowed=True),
        ]
        labels = {
            "c1": _label(leakage=True),
            "c2": _label(leakage=True),
            "c3": _label(leakage=False, useful=True),
        }

        rows = compute_threshold_sensitivity(
            {0.75: results}, labels, {}
        )
        assert len(rows) == 1
        r = rows[0]
        assert r.tau_sem == 0.75
        assert r.n_eligible == 3
        assert r.n_leaking == 2
        assert r.leakage_recall == 0.5
        assert r.n_non_leaking == 1
        assert r.fbr == 0.0
        assert r.utility_retention == 1.0

    def test_multiple_thresholds(self):
        """Multiple thresholds produce sorted rows."""
        labels = {"c1": _label(leakage=True)}
        r_low = [_result(candidate_id="c1", blocked=True)]
        r_high = [_result(candidate_id="c1", blocked=False)]

        rows = compute_threshold_sensitivity(
            {0.70: r_low, 0.85: r_high}, labels, {}
        )
        assert len(rows) == 2
        assert rows[0].tau_sem == 0.70
        assert rows[1].tau_sem == 0.85
        assert rows[0].leakage_recall == 1.0
        assert rows[1].leakage_recall == 0.0

    def test_crr_is_complement(self):
        """CRR = 1 - leakage_recall."""
        results = [_result(candidate_id="c1", blocked=True)]
        labels = {"c1": _label(leakage=True)}

        rows = compute_threshold_sensitivity({0.75: results}, labels, {})
        assert rows[0].crr == 0.0  # 1 - 1.0

    def test_unresolved_skipped(self):
        """Unresolved labels excluded."""
        results = [
            _result(candidate_id="c1", blocked=True),
            _result(candidate_id="c2", blocked=True),
        ]
        labels = {
            "c1": _label(leakage=True),
            "c2": _label(leakage=True, unresolved=True),
        }

        rows = compute_threshold_sensitivity({0.75: results}, labels, {})
        assert rows[0].n_eligible == 1


# ===========================================================================
# compute_tradeoff_data
# ===========================================================================


class TestComputeTradeoffData:
    """Tests for tradeoff figure data."""

    def test_tradeoff_points(self):
        """Tradeoff points match sensitivity rows."""
        sens = [
            ThresholdSensitivityRow(
                tau_sem=0.75, leakage_recall=0.9, fbr=0.1,
                utility_retention=0.85, crr=0.1,
                n_eligible=10, n_leaking=5, n_non_leaking=5,
                n_useful_eligible=3,
            ),
        ]
        points = compute_tradeoff_data(sens)
        assert len(points) == 1
        p = points[0]
        assert p.tau_sem == 0.75
        assert p.leakage_prevention == 0.9
        assert p.utility_retention == 0.85
        assert p.fbr == 0.1


# ===========================================================================
# select_optimal_threshold
# ===========================================================================


class TestSelectOptimalThreshold:
    """Tests for optimal threshold selection."""

    def test_selects_best_utility_meeting_constraint(self):
        """Picks highest utility among thresholds meeting recall ≥ 0.90."""
        sens = [
            ThresholdSensitivityRow(
                tau_sem=0.70, leakage_recall=0.95, fbr=0.2,
                utility_retention=0.7, crr=0.05,
                n_eligible=10, n_leaking=5, n_non_leaking=5,
                n_useful_eligible=3,
            ),
            ThresholdSensitivityRow(
                tau_sem=0.75, leakage_recall=0.90, fbr=0.1,
                utility_retention=0.85, crr=0.1,
                n_eligible=10, n_leaking=5, n_non_leaking=5,
                n_useful_eligible=3,
            ),
            ThresholdSensitivityRow(
                tau_sem=0.80, leakage_recall=0.80, fbr=0.05,
                utility_retention=0.95, crr=0.2,
                n_eligible=10, n_leaking=5, n_non_leaking=5,
                n_useful_eligible=3,
            ),
        ]

        rec = select_optimal_threshold(sens, min_leakage_recall=0.90)
        # 0.70 and 0.75 meet constraint; 0.75 has higher utility
        assert rec.tau_sem == 0.75
        assert rec.utility_retention == 0.85

    def test_fallback_when_no_threshold_meets_constraint(self):
        """Falls back to highest recall when none meets constraint."""
        sens = [
            ThresholdSensitivityRow(
                tau_sem=0.75, leakage_recall=0.5, fbr=0.1,
                utility_retention=0.8, crr=0.5,
                n_eligible=10, n_leaking=5, n_non_leaking=5,
                n_useful_eligible=3,
            ),
        ]

        rec = select_optimal_threshold(sens, min_leakage_recall=0.90)
        assert "no threshold meets" in rec.reason

    def test_empty_input(self):
        """Empty input → default recommendation."""
        rec = select_optimal_threshold([])
        assert rec.tau_sem == 0.75
        assert rec.reason == "no data"


# ===========================================================================
# Serialisation
# ===========================================================================


class TestSerialisation:
    """Tests for to_dict helpers."""

    def test_sensitivity_to_dict(self):
        row = ThresholdSensitivityRow(
            tau_sem=0.75, leakage_recall=0.9, fbr=0.1,
            utility_retention=0.85, crr=0.1,
            n_eligible=10, n_leaking=5, n_non_leaking=5,
            n_useful_eligible=3,
        )
        dicts = sensitivity_to_dict([row])
        assert len(dicts) == 1
        d = dicts[0]
        assert d["tau_sem"] == 0.75
        assert d["leakage_recall"] == 0.9
        assert d["crr"] == 0.1

    def test_tradeoff_to_dict(self):
        point = TradeoffPoint(
            tau_sem=0.75, leakage_prevention=0.9,
            utility_retention=0.85, fbr=0.1,
        )
        dicts = tradeoff_to_dict([point])
        assert len(dicts) == 1
        assert dicts[0]["leakage_prevention"] == 0.9

    def test_empty_serialisation(self):
        assert sensitivity_to_dict([]) == []
        assert tradeoff_to_dict([]) == []
