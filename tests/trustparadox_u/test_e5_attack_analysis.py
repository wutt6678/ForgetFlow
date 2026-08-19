"""E5-008: Attack robustness and trust-conditioned analysis tests.

Tests attack-type breakdown, trust-conditioned metrics, trust drift,
hard-negative overblocking, legitimate-task utility, and serialisation
helpers using synthetic data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.e5_attack_analysis import (  # noqa: E402
    ALL_ATTACK_TYPES,
    CONTROL_ATTACK_TYPES,
    LEAKAGE_ATTACK_TYPES,
    TRUST_LEVELS,
    AttackTypeRow,
    TrustConditionedRow,
    attack_robustness_to_dict,
    compute_attack_robustness,
    compute_hard_negative_analysis,
    compute_legitimate_task_analysis,
    compute_trust_conditioned,
    compute_trust_drift,
    trust_conditioned_to_dict,
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
    """Create a synthetic row label dict."""
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
) -> dict:
    """Create a synthetic row result dict."""
    return {
        "candidate_id": candidate_id,
        "blocked": blocked,
        "allowed": allowed,
        "policy_action": policy_action,
    }


def _corpus_entry(
    *,
    attack_type: str = "direct_disclosure",
    trust_level: str = "default",
) -> dict:
    return {"attack_type": attack_type, "trust_level": trust_level}


# ===========================================================================
# Constants
# ===========================================================================


class TestConstants:
    """Verify attack-type and trust-level constants."""

    def test_leakage_attack_types_count(self):
        assert len(LEAKAGE_ATTACK_TYPES) == 6

    def test_control_attack_types_count(self):
        assert len(CONTROL_ATTACK_TYPES) == 2

    def test_all_attack_types_is_concatenation(self):
        assert ALL_ATTACK_TYPES == LEAKAGE_ATTACK_TYPES + CONTROL_ATTACK_TYPES

    def test_trust_levels(self):
        assert TRUST_LEVELS == ("low", "default", "high")


# ===========================================================================
# compute_attack_robustness
# ===========================================================================


class TestComputeAttackRobustness:
    """Tests for attack robustness table (plan §52)."""

    def test_basic_robustness(self):
        """Two candidates: one leaking blocked by C4, one non-leaking."""
        row_results_by_condition = {
            "C0": [
                _result(candidate_id="c1", blocked=False, allowed=True),
                _result(candidate_id="c2", blocked=False, allowed=True),
            ],
            "C4": [
                _result(candidate_id="c1", blocked=True, allowed=False,
                        policy_action="block"),
                _result(candidate_id="c2", blocked=False, allowed=True),
            ],
        }
        row_labels_by_id = {
            "c1": _label(leakage=True, useful=False),
            "c2": _label(leakage=False, useful=False),
        }
        corpus_by_id = {
            "c1": _corpus_entry(attack_type="direct_disclosure"),
            "c2": _corpus_entry(attack_type="direct_disclosure"),
        }

        rows = compute_attack_robustness(
            row_results_by_condition, row_labels_by_id, corpus_by_id
        )

        # Find the direct_disclosure row
        dd_row = next(r for r in rows if r.attack_type == "direct_disclosure")
        assert dd_row.n == 2
        # C0: c1 leaking AND allowed → through rate = 1/1 = 1.0
        assert dd_row.baseline_leakage_through == 1.0
        # C4: c1 leaking but blocked (allowed=False) → through rate = 0/1
        assert dd_row.forgetflow_leakage_through == 0.0
        # relative reduction = (1.0 - 0.0) / 1.0 = 1.0
        assert dd_row.relative_leakage_reduction == 1.0

    def test_relative_reduction_computed(self):
        """When baseline > 0, relative reduction is computed."""
        # Under C0, we simulate blocking by marking blocked=True for leaking rows
        row_results_by_condition = {
            "C0": [
                _result(candidate_id="c1", blocked=True, allowed=False),
                _result(candidate_id="c2", blocked=False, allowed=True),
            ],
            "C4": [
                _result(candidate_id="c1", blocked=True, allowed=False),
                _result(candidate_id="c2", blocked=False, allowed=True),
            ],
        }
        row_labels_by_id = {
            "c1": _label(leakage=True),
            "c2": _label(leakage=True),
        }
        corpus_by_id = {
            "c1": _corpus_entry(attack_type="semantic_paraphrase"),
            "c2": _corpus_entry(attack_type="semantic_paraphrase"),
        }

        rows = compute_attack_robustness(
            row_results_by_condition, row_labels_by_id, corpus_by_id
        )
        sp_row = next(r for r in rows if r.attack_type == "semantic_paraphrase")
        # Baseline: 1/2 leaking allowed = 0.5
        assert sp_row.baseline_leakage_through == 0.5
        # ForgetFlow: 1/2 leaking allowed = 0.5
        assert sp_row.forgetflow_leakage_through == 0.5
        # Reduction = (0.5 - 0.5) / 0.5 = 0.0
        assert sp_row.relative_leakage_reduction == 0.0

    def test_full_reduction(self):
        """Baseline leaks 1.0, ForgetFlow leaks 0.0 → reduction = 1.0."""
        row_results_by_condition = {
            "C0": [
                _result(candidate_id="c1", blocked=False, allowed=True),
            ],
            "C4": [
                _result(candidate_id="c1", blocked=True, allowed=False),
            ],
        }
        row_labels_by_id = {
            "c1": _label(leakage=True),
        }
        corpus_by_id = {
            "c1": _corpus_entry(attack_type="alias_or_coreference"),
        }

        rows = compute_attack_robustness(
            row_results_by_condition, row_labels_by_id, corpus_by_id
        )
        ac_row = next(r for r in rows if r.attack_type == "alias_or_coreference")
        assert ac_row.baseline_leakage_through == 1.0
        assert ac_row.forgetflow_leakage_through == 0.0
        assert ac_row.relative_leakage_reduction == 1.0

    def test_missing_attack_type_gets_zero_row(self):
        """Attack types with no candidates get zero-filled rows."""
        rows = compute_attack_robustness(
            row_results_by_condition={"C0": [], "C4": []},
            row_labels_by_id={},
            corpus_by_id={},
        )
        assert len(rows) == len(ALL_ATTACK_TYPES)
        for r in rows:
            assert r.n == 0
            assert r.baseline_leakage_through == 0.0

    def test_unresolved_labels_skipped(self):
        """Unresolved labels are excluded from metrics."""
        row_results_by_condition = {
            "C0": [_result(candidate_id="c1", blocked=False)],
            "C4": [_result(candidate_id="c1", blocked=True)],
        }
        row_labels_by_id = {
            "c1": _label(leakage=True, unresolved=True),
        }
        corpus_by_id = {
            "c1": _corpus_entry(attack_type="direct_disclosure"),
        }

        rows = compute_attack_robustness(
            row_results_by_condition, row_labels_by_id, corpus_by_id
        )
        dd_row = rows[0]
        assert dd_row.n == 1  # still counted in corpus
        assert dd_row.baseline_leakage_through == 0.0  # no eligible after skip

    def test_utility_retention(self):
        """Useful candidates that are allowed → utility retention."""
        row_results_by_condition = {
            "C0": [
                _result(candidate_id="c1", blocked=False, allowed=True),
                _result(candidate_id="c2", blocked=False, allowed=True),
            ],
            "C4": [
                _result(candidate_id="c1", blocked=False, allowed=True),
                _result(candidate_id="c2", blocked=True, allowed=False),
            ],
        }
        row_labels_by_id = {
            "c1": _label(leakage=False, useful=True),
            "c2": _label(leakage=False, useful=True),
        }
        corpus_by_id = {
            "c1": _corpus_entry(attack_type="legitimate_task"),
            "c2": _corpus_entry(attack_type="legitimate_task"),
        }

        rows = compute_attack_robustness(
            row_results_by_condition, row_labels_by_id, corpus_by_id
        )
        lt_row = next(r for r in rows if r.attack_type == "legitimate_task")
        # 1 of 2 useful candidates preserved → 0.5
        assert lt_row.utility_retention == 0.5

    def test_fbr_computation(self):
        """FBR: non-leaking candidates blocked / total non-leaking."""
        row_results_by_condition = {
            "C0": [
                _result(candidate_id="c1", blocked=False),
                _result(candidate_id="c2", blocked=False),
            ],
            "C4": [
                _result(candidate_id="c1", blocked=True),
                _result(candidate_id="c2", blocked=False),
            ],
        }
        row_labels_by_id = {
            "c1": _label(leakage=False),
            "c2": _label(leakage=False),
        }
        corpus_by_id = {
            "c1": _corpus_entry(attack_type="hard_negative_control"),
            "c2": _corpus_entry(attack_type="hard_negative_control"),
        }

        rows = compute_attack_robustness(
            row_results_by_condition, row_labels_by_id, corpus_by_id
        )
        hn_row = next(r for r in rows if r.attack_type == "hard_negative_control")
        assert hn_row.fbr == 0.5  # 1 of 2 non-leaking blocked


# ===========================================================================
# compute_trust_conditioned
# ===========================================================================


class TestComputeTrustConditioned:
    """Tests for trust-conditioned metrics (plan §43)."""

    def test_basic_trust_conditioned(self):
        """One leaking candidate blocked, one non-leaking allowed."""
        row_results = [
            _result(candidate_id="c1", blocked=True, allowed=False,
                    policy_action="block"),
            _result(candidate_id="c2", blocked=False, allowed=True),
        ]
        row_labels_by_id = {
            "c1": _label(leakage=True, useful=False),
            "c2": _label(leakage=False, useful=True),
        }
        corpus_by_id = {
            "c1": _corpus_entry(trust_level="low"),
            "c2": _corpus_entry(trust_level="low"),
        }

        rows = compute_trust_conditioned(
            row_results, row_labels_by_id, corpus_by_id
        )
        assert len(rows) == 3  # one per trust level

        low_row = next(r for r in rows if r.trust_level == "low")
        assert low_row.n_eligible == 2
        assert low_row.n_leaking == 1
        assert low_row.n_non_leaking == 1
        assert low_row.leakage_prevention == 1.0  # 1/1 leaking blocked
        assert low_row.fbr == 0.0  # 0/1 non-leaking blocked
        assert low_row.utility_retention == 1.0  # 1/1 useful preserved

    def test_empty_trust_level(self):
        """Trust level with no candidates → zero metrics."""
        row_results = [
            _result(candidate_id="c1", blocked=False),
        ]
        row_labels_by_id = {
            "c1": _label(leakage=True),
        }
        corpus_by_id = {
            "c1": _corpus_entry(trust_level="high"),
        }

        rows = compute_trust_conditioned(
            row_results, row_labels_by_id, corpus_by_id
        )
        low_row = next(r for r in rows if r.trust_level == "low")
        assert low_row.n_eligible == 0
        assert low_row.leakage_prevention == 0.0

    def test_policy_action_distribution(self):
        """Policy action distribution is tracked."""
        row_results = [
            _result(candidate_id="c1", blocked=True, policy_action="block"),
            _result(candidate_id="c2", blocked=False, policy_action="allow"),
            _result(candidate_id="c3", blocked=False, policy_action="allow"),
        ]
        row_labels_by_id = {
            "c1": _label(leakage=True),
            "c2": _label(leakage=False),
            "c3": _label(leakage=True),
        }
        corpus_by_id = {
            "c1": _corpus_entry(trust_level="default"),
            "c2": _corpus_entry(trust_level="default"),
            "c3": _corpus_entry(trust_level="default"),
        }

        rows = compute_trust_conditioned(
            row_results, row_labels_by_id, corpus_by_id
        )
        def_row = next(r for r in rows if r.trust_level == "default")
        assert def_row.policy_action_distribution == {"block": 1, "allow": 2}


# ===========================================================================
# compute_trust_drift
# ===========================================================================


class TestComputeTrustDrift:
    """Tests for trust drift metrics (plan §45)."""

    def test_empty_input(self):
        """Empty input → zero drift."""
        drift = compute_trust_drift([])
        assert drift.leakage_rate_drift == 0.0
        assert drift.fbr_drift == 0.0
        assert drift.utility_drift == 0.0

    def test_drift_computation(self):
        """Drift = max - min across trust levels."""
        trust_conditioned = [
            TrustConditionedRow(
                trust_level="low",
                leakage_prevention=0.8,
                fbr=0.1,
                utility_retention=0.9,
                n_eligible=10,
                n_leaking=5,
                n_non_leaking=5,
                policy_action_distribution={"block": 5, "allow": 5},
            ),
            TrustConditionedRow(
                trust_level="default",
                leakage_prevention=0.9,
                fbr=0.05,
                utility_retention=0.95,
                n_eligible=10,
                n_leaking=5,
                n_non_leaking=5,
                policy_action_distribution={"block": 5, "allow": 5},
            ),
            TrustConditionedRow(
                trust_level="high",
                leakage_prevention=1.0,
                fbr=0.0,
                utility_retention=1.0,
                n_eligible=10,
                n_leaking=5,
                n_non_leaking=5,
                policy_action_distribution={"block": 5, "allow": 5},
            ),
        ]

        drift = compute_trust_drift(trust_conditioned)
        # leakage_rate = 1 - leakage_prevention: 0.2, 0.1, 0.0
        assert drift.leakage_rate_drift == pytest.approx(0.2)
        # fbr: 0.1, 0.05, 0.0
        assert drift.fbr_drift == pytest.approx(0.1)
        # utility: 0.9, 0.95, 1.0
        assert drift.utility_drift == pytest.approx(0.1)

    def test_single_level_zero_drift(self):
        """Single trust level → zero drift."""
        trust_conditioned = [
            TrustConditionedRow(
                trust_level="default",
                leakage_prevention=0.9,
                fbr=0.05,
                utility_retention=0.95,
                n_eligible=10,
                n_leaking=5,
                n_non_leaking=5,
                policy_action_distribution={},
            ),
        ]
        drift = compute_trust_drift(trust_conditioned)
        assert drift.leakage_rate_drift == 0.0
        assert drift.fbr_drift == 0.0
        assert drift.utility_drift == 0.0


# ===========================================================================
# compute_hard_negative_analysis
# ===========================================================================


class TestComputeHardNegativeAnalysis:
    """Tests for hard-negative overblocking (plan §53)."""

    def test_basic_overblocking(self):
        """3 hard negatives, 1 blocked → overblocking_rate = 1/3."""
        row_results = [
            _result(candidate_id="c1", blocked=True),
            _result(candidate_id="c2", blocked=False),
            _result(candidate_id="c3", blocked=False),
        ]
        row_labels_by_id = {
            "c1": _label(leakage=False),
            "c2": _label(leakage=False),
            "c3": _label(leakage=False),
        }
        corpus_by_id = {
            "c1": _corpus_entry(attack_type="hard_negative_control"),
            "c2": _corpus_entry(attack_type="hard_negative_control"),
            "c3": _corpus_entry(attack_type="hard_negative_control"),
        }

        result = compute_hard_negative_analysis(
            row_results, row_labels_by_id, corpus_by_id
        )
        assert result.n_hard_negatives == 3
        assert result.n_blocked == 1
        assert abs(result.overblocking_rate - 1 / 3) < 1e-9
        assert result.n_allowed == 2

    def test_no_hard_negatives(self):
        """No hard negatives → zero analysis."""
        result = compute_hard_negative_analysis(
            row_results=[],
            row_labels_by_id={},
            corpus_by_id={
                "c1": _corpus_entry(attack_type="direct_disclosure"),
            },
        )
        assert result.n_hard_negatives == 0
        assert result.overblocking_rate == 0.0

    def test_all_blocked(self):
        """All hard negatives blocked → overblocking_rate = 1.0."""
        row_results = [
            _result(candidate_id="c1", blocked=True),
            _result(candidate_id="c2", blocked=True),
        ]
        row_labels_by_id = {
            "c1": _label(leakage=False),
            "c2": _label(leakage=False),
        }
        corpus_by_id = {
            "c1": _corpus_entry(attack_type="hard_negative_control"),
            "c2": _corpus_entry(attack_type="hard_negative_control"),
        }

        result = compute_hard_negative_analysis(
            row_results, row_labels_by_id, corpus_by_id
        )
        assert result.overblocking_rate == 1.0
        assert result.n_allowed == 0

    def test_unresolved_skipped(self):
        """Unresolved labels are excluded."""
        row_results = [
            _result(candidate_id="c1", blocked=True),
            _result(candidate_id="c2", blocked=True),
        ]
        row_labels_by_id = {
            "c1": _label(leakage=False),
            "c2": _label(leakage=False, unresolved=True),
        }
        corpus_by_id = {
            "c1": _corpus_entry(attack_type="hard_negative_control"),
            "c2": _corpus_entry(attack_type="hard_negative_control"),
        }

        result = compute_hard_negative_analysis(
            row_results, row_labels_by_id, corpus_by_id
        )
        assert result.n_hard_negatives == 1  # c2 skipped


# ===========================================================================
# compute_legitimate_task_analysis
# ===========================================================================


class TestComputeLegitimateTaskAnalysis:
    """Tests for legitimate-task utility (plan §54)."""

    def test_basic_utility(self):
        """2 legitimate tasks, 1 allowed → utility_rate = 0.5."""
        row_results = [
            _result(candidate_id="c1", blocked=False, allowed=True,
                    policy_action="allow"),
            _result(candidate_id="c2", blocked=True, allowed=False,
                    policy_action="block"),
        ]
        row_labels_by_id = {
            "c1": _label(leakage=False),
            "c2": _label(leakage=False),
        }
        corpus_by_id = {
            "c1": _corpus_entry(attack_type="legitimate_task"),
            "c2": _corpus_entry(attack_type="legitimate_task"),
        }

        result = compute_legitimate_task_analysis(
            row_results, row_labels_by_id, corpus_by_id
        )
        assert result.n_legitimate == 2
        assert result.n_preserved == 1
        assert result.n_blocked == 1
        assert result.utility_rate == 0.5
        assert result.policy_action_distribution == {"allow": 1, "block": 1}

    def test_no_legitimate_tasks(self):
        """No legitimate tasks → zero analysis."""
        result = compute_legitimate_task_analysis(
            row_results=[],
            row_labels_by_id={},
            corpus_by_id={
                "c1": _corpus_entry(attack_type="direct_disclosure"),
            },
        )
        assert result.n_legitimate == 0
        assert result.utility_rate == 0.0

    def test_all_preserved(self):
        """All legitimate tasks preserved → utility_rate = 1.0."""
        row_results = [
            _result(candidate_id="c1", blocked=False, allowed=True),
            _result(candidate_id="c2", blocked=False, allowed=True),
        ]
        row_labels_by_id = {
            "c1": _label(leakage=False),
            "c2": _label(leakage=False),
        }
        corpus_by_id = {
            "c1": _corpus_entry(attack_type="legitimate_task"),
            "c2": _corpus_entry(attack_type="legitimate_task"),
        }

        result = compute_legitimate_task_analysis(
            row_results, row_labels_by_id, corpus_by_id
        )
        assert result.utility_rate == 1.0
        assert result.n_blocked == 0


# ===========================================================================
# Serialisation helpers
# ===========================================================================


class TestSerialisation:
    """Tests for to_dict helpers."""

    def test_attack_robustness_to_dict(self):
        """Serialises all fields correctly."""
        rows = [
            AttackTypeRow(
                attack_type="direct_disclosure",
                baseline_leakage_through=0.8,
                forgetflow_leakage_through=0.1,
                relative_leakage_reduction=0.875,
                utility_retention=0.95,
                fbr=0.05,
                n=100,
                n_leaking_eligible=50,
            ),
        ]
        dicts = attack_robustness_to_dict(rows)
        assert len(dicts) == 1
        d = dicts[0]
        assert d["attack_type"] == "direct_disclosure"
        assert d["baseline_leakage_through"] == 0.8
        assert d["forgetflow_leakage_through"] == 0.1
        assert d["relative_leakage_reduction"] == 0.875
        assert d["utility_retention"] == 0.95
        assert d["fbr"] == 0.05
        assert d["n"] == 100
        assert d["n_leaking_eligible"] == 50

    def test_trust_conditioned_to_dict(self):
        """Serialises all fields correctly."""
        rows = [
            TrustConditionedRow(
                trust_level="low",
                leakage_prevention=0.9,
                fbr=0.1,
                utility_retention=0.85,
                n_eligible=50,
                n_leaking=25,
                n_non_leaking=25,
                policy_action_distribution={"block": 20, "allow": 30},
            ),
        ]
        dicts = trust_conditioned_to_dict(rows)
        assert len(dicts) == 1
        d = dicts[0]
        assert d["trust_level"] == "low"
        assert d["leakage_prevention"] == 0.9
        assert d["fbr"] == 0.1
        assert d["n_eligible"] == 50
        assert d["policy_action_distribution"] == {"block": 20, "allow": 30}

    def test_empty_serialisation(self):
        """Empty lists serialise to empty lists."""
        assert attack_robustness_to_dict([]) == []
        assert trust_conditioned_to_dict([]) == []
