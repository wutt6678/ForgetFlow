"""E5-009: Component ablation study tests (repair §20-§26).

Tests ablation definitions, ablation overrides, metrics computation,
impact analysis, and serialisation helpers using synthetic data.

The old apply_ablation_to_row tests are removed because simulated
post-hoc ablations were replaced with real re-execution (§20).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.e5_ablation_study import (  # noqa: E402
    ABLATION_DESCRIPTIONS,
    ABLATION_DISABLED_COMPONENT,
    ABLATION_IDS,
    AblationImpact,
    AblationMetrics,
    AblationStudyResult,
    ablation_impacts_to_dict,
    ablation_metrics_to_dict,
    compute_ablation_impacts,
    compute_ablation_metrics,
    get_ablation_override,
    get_ablation_specs,
    run_ablation_study,
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


def _corpus(
    *,
    attack_type: str = "direct_disclosure",
    trust_level: str = "default",
) -> dict:
    return {"attack_type": attack_type, "trust_level": trust_level}


# ===========================================================================
# Constants and specs
# ===========================================================================


class TestAblationSpecs:
    """Tests for ablation definitions."""

    def test_five_ablations(self):
        assert len(ABLATION_IDS) == 5
        assert ABLATION_IDS == ("A0", "A1", "A2", "A3", "A4")

    def test_descriptions_match(self):
        for aid in ABLATION_IDS:
            assert aid in ABLATION_DESCRIPTIONS

    def test_disabled_components_match(self):
        for aid in ABLATION_IDS:
            assert aid in ABLATION_DISABLED_COMPONENT

    def test_get_specs_returns_five(self):
        specs = get_ablation_specs()
        assert len(specs) == 5

    def test_a0_full_system(self):
        specs = get_ablation_specs()
        a0 = specs[0]
        assert a0.ablation_id == "A0"
        assert a0.disabled_component is None
        assert a0.semantic_enabled is True
        assert a0.history_enabled is True
        assert a0.reconstruction_guard is True
        assert a0.purge_enabled is True

    def test_a1_no_semantic(self):
        specs = get_ablation_specs()
        a1 = specs[1]
        assert a1.ablation_id == "A1"
        assert a1.disabled_component == "semantic"
        assert a1.semantic_enabled is False
        assert a1.history_enabled is True

    def test_a2_no_history(self):
        specs = get_ablation_specs()
        a2 = specs[2]
        assert a2.ablation_id == "A2"
        assert a2.disabled_component == "history"
        assert a2.history_enabled is False

    def test_a3_no_reconstruction_guard(self):
        specs = get_ablation_specs()
        a3 = specs[3]
        assert a3.ablation_id == "A3"
        assert a3.disabled_component == "reconstruction_guard"
        assert a3.reconstruction_guard is False

    def test_a4_no_purge(self):
        specs = get_ablation_specs()
        a4 = specs[4]
        assert a4.ablation_id == "A4"
        assert a4.disabled_component == "purge"
        assert a4.purge_enabled is False


# ===========================================================================
# get_ablation_override (§20-§26)
# ===========================================================================


class TestGetAblationOverride:
    """Tests for the ablation → FirewallRunner override mapping."""

    def test_a0_empty_override(self):
        """A0 (full system) has no overrides."""
        spec = get_ablation_specs()[0]
        override = get_ablation_override(spec)
        assert override == {}

    def test_a1_semantic_disabled(self):
        spec = get_ablation_specs()[1]
        override = get_ablation_override(spec)
        assert override == {"semantic_enabled": False}

    def test_a2_history_disabled(self):
        spec = get_ablation_specs()[2]
        override = get_ablation_override(spec)
        assert override == {"history_enabled": False}

    def test_a3_reconstruction_disabled(self):
        spec = get_ablation_specs()[3]
        override = get_ablation_override(spec)
        assert override == {"reconstruction_guard": False}

    def test_a4_purge_disabled(self):
        spec = get_ablation_specs()[4]
        override = get_ablation_override(spec)
        assert override == {"purge_enabled": False}


# ===========================================================================
# compute_ablation_metrics
# ===========================================================================


class TestComputeAblationMetrics:
    """Tests for per-ablation metrics computation."""

    def test_basic_metrics(self):
        """2 leaking (1 blocked), 1 non-leaking (not blocked), 1 useful."""
        results = [
            _result(candidate_id="c1", blocked=True),
            _result(candidate_id="c2", blocked=False),
            _result(candidate_id="c3", blocked=False, allowed=True),
        ]
        labels = {
            "c1": _label(leakage=True, useful=False),
            "c2": _label(leakage=True, useful=True),
            "c3": _label(leakage=False, useful=True),
        }
        corpus = {
            "c1": _corpus(attack_type="direct_disclosure"),
            "c2": _corpus(attack_type="semantic_paraphrase"),
            "c3": _corpus(attack_type="legitimate_task"),
        }

        m = compute_ablation_metrics(results, labels, corpus, "A0", "full")
        assert m.n_eligible == 3
        assert m.n_leaking == 2
        assert m.n_leaking_blocked == 1
        assert m.leakage_prevention == 0.5
        assert m.n_non_leaking == 1
        assert m.n_fp == 0
        assert m.fbr == 0.0
        assert m.n_useful_eligible == 2
        assert m.n_useful_preserved == 2
        assert m.utility_retention == 1.0

    def test_empty_results(self):
        """No results → zero metrics."""
        m = compute_ablation_metrics([], {}, {}, "A0", "full")
        assert m.n_eligible == 0
        assert m.leakage_prevention == 0.0
        assert m.fbr == 0.0
        assert m.utility_retention == 0.0

    def test_attack_type_breakdown(self):
        """Attack-type breakdown is populated."""
        results = [
            _result(candidate_id="c1", blocked=True),
            _result(candidate_id="c2", blocked=False),
        ]
        labels = {
            "c1": _label(leakage=True),
            "c2": _label(leakage=True),
        }
        corpus = {
            "c1": _corpus(attack_type="direct_disclosure"),
            "c2": _corpus(attack_type="semantic_paraphrase"),
        }

        m = compute_ablation_metrics(results, labels, corpus, "A0", "full")
        assert "direct_disclosure" in m.attack_type_breakdown
        assert "semantic_paraphrase" in m.attack_type_breakdown
        dd = m.attack_type_breakdown["direct_disclosure"]
        assert dd["n"] == 1
        assert dd["leakage_prevention"] == 1.0

    def test_unresolved_skipped(self):
        """Unresolved labels are excluded."""
        results = [
            _result(candidate_id="c1", blocked=True),
            _result(candidate_id="c2", blocked=True),
        ]
        labels = {
            "c1": _label(leakage=True),
            "c2": _label(leakage=True, unresolved=True),
        }
        corpus = {
            "c1": _corpus(),
            "c2": _corpus(),
        }

        m = compute_ablation_metrics(results, labels, corpus, "A0", "full")
        assert m.n_eligible == 1


# ===========================================================================
# run_ablation_study (real re-execution §20)
# ===========================================================================


class TestRunAblationStudy:
    """Tests for the full ablation study runner (real re-execution)."""

    def _make_features_and_labels(self):
        """Create minimal features/labels for ablation study."""
        features_by_id = {
            "c1": {
                "exact_match": False,
                "alias_match": False,
                "semantic_similarity": 0.9,
            },
            "c2": {
                "exact_match": True,
                "alias_match": False,
                "semantic_similarity": 0.3,
            },
        }
        row_labels_by_id = {
            "c1": _label(leakage=True),
            "c2": _label(leakage=True),
        }
        return features_by_id, row_labels_by_id

    def test_returns_five_ablations(self):
        """Study returns metrics for all 5 ablations."""
        features_by_id, row_labels_by_id = self._make_features_and_labels()
        study = run_ablation_study(
            features_by_id, row_labels_by_id, tau_sem=0.75,
        )
        assert len(study.ablations) == 5
        assert study.baseline_id == "A0"

    def test_baseline_property(self):
        """Baseline property returns A0 metrics."""
        features_by_id, row_labels_by_id = self._make_features_and_labels()
        study = run_ablation_study(
            features_by_id, row_labels_by_id, tau_sem=0.75,
        )
        assert study.baseline.ablation_id == "A0"

    def test_all_ablations_have_metrics(self):
        """Each ablation produces valid AblationMetrics."""
        features_by_id, row_labels_by_id = self._make_features_and_labels()
        study = run_ablation_study(
            features_by_id, row_labels_by_id, tau_sem=0.75,
        )
        for a in study.ablations:
            assert a.ablation_id in ABLATION_IDS
            assert 0.0 <= a.leakage_prevention <= 1.0
            assert 0.0 <= a.fbr <= 1.0
            assert 0.0 <= a.utility_retention <= 1.0


# ===========================================================================
# compute_ablation_impacts
# ===========================================================================


class TestComputeAblationImpacts:
    """Tests for relative impact computation."""

    def test_impact_deltas(self):
        """Impact = baseline - ablation for leakage, ablation - baseline for FBR."""
        # Use synthetic AblationMetrics to test impact computation
        baseline = AblationMetrics(
            ablation_id="A0", description="full",
            n_eligible=2, n_leaking=2, n_leaking_blocked=2,
            n_non_leaking=0, n_fp=0,
            n_useful_eligible=0, n_useful_preserved=0,
            leakage_prevention=1.0, fbr=0.0, utility_retention=0.0,
            attack_type_breakdown={},
        )
        a1 = AblationMetrics(
            ablation_id="A1", description="-semantic",
            n_eligible=2, n_leaking=2, n_leaking_blocked=0,
            n_non_leaking=0, n_fp=0,
            n_useful_eligible=0, n_useful_preserved=0,
            leakage_prevention=0.0, fbr=0.0, utility_retention=0.0,
            attack_type_breakdown={},
        )
        study = AblationStudyResult(
            ablations=(baseline, a1),
            baseline_id="A0",
        )
        impacts = compute_ablation_impacts(study)
        assert len(impacts) == 1
        assert impacts[0].ablation_id == "A1"
        assert impacts[0].disabled_component == "semantic"
        assert impacts[0].leakage_prevention_delta == 1.0  # 1.0 - 0.0

    def test_baseline_excluded(self):
        """Baseline (A0) is not in impacts list."""
        features_by_id = {
            "c1": {"exact_match": True, "alias_match": False,
                    "semantic_similarity": 0.9},
        }
        row_labels_by_id = {"c1": _label(leakage=True)}

        study = run_ablation_study(
            features_by_id, row_labels_by_id, tau_sem=0.75,
        )
        impacts = compute_ablation_impacts(study)
        ids = [i.ablation_id for i in impacts]
        assert "A0" not in ids
        assert len(impacts) == 4  # A1-A4


# ===========================================================================
# Serialisation
# ===========================================================================


class TestSerialisation:
    """Tests for to_dict helpers."""

    def test_ablation_metrics_to_dict(self):
        m = AblationMetrics(
            ablation_id="A0",
            description="full",
            n_eligible=10,
            n_leaking=5,
            n_leaking_blocked=4,
            n_non_leaking=5,
            n_fp=1,
            n_useful_eligible=3,
            n_useful_preserved=2,
            leakage_prevention=0.8,
            fbr=0.2,
            utility_retention=0.667,
            attack_type_breakdown={"direct_disclosure": {"n": 5}},
        )
        dicts = ablation_metrics_to_dict([m])
        assert len(dicts) == 1
        d = dicts[0]
        assert d["ablation_id"] == "A0"
        assert d["leakage_prevention"] == 0.8
        assert d["attack_type_breakdown"] == {"direct_disclosure": {"n": 5}}

    def test_ablation_impacts_to_dict(self):
        i = AblationImpact(
            ablation_id="A1",
            disabled_component="semantic",
            leakage_prevention_delta=0.3,
            fbr_delta=-0.1,
            utility_delta=0.05,
        )
        dicts = ablation_impacts_to_dict([i])
        assert len(dicts) == 1
        assert dicts[0]["disabled_component"] == "semantic"
        assert dicts[0]["leakage_prevention_delta"] == 0.3

    def test_empty_serialisation(self):
        assert ablation_metrics_to_dict([]) == []
        assert ablation_impacts_to_dict([]) == []
