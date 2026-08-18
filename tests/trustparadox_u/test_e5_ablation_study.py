"""E5-009: Component ablation study tests.

Tests ablation definitions, ablation application, metrics computation,
impact analysis, and serialisation helpers using synthetic data.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.e5_ablation_study import (  # noqa: E402
    ABLATION_DESCRIPTIONS,
    ABLATION_IDS,
    AblationImpact,
    AblationMetrics,
    ablation_impacts_to_dict,
    ablation_metrics_to_dict,
    apply_ablation_to_row,
    compute_ablation_impacts,
    compute_ablation_metrics,
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
# apply_ablation_to_row
# ===========================================================================


class TestApplyAblationToRow:
    """Tests for ablation application to individual rows."""

    def test_a0_no_change(self):
        """A0 (full system) returns the row unchanged."""
        row = _result(candidate_id="c1", blocked=True, sim=0.85)
        spec = get_ablation_specs()[0]  # A0
        out = apply_ablation_to_row(row, _label(), _corpus(), spec, 0.75)
        assert out == row

    def test_a1_masks_semantic(self):
        """A1 disables semantic → similarity zeroed, detection re-evaluated."""
        row = _result(
            candidate_id="c1", blocked=True, sim=0.9,
            exact=False, alias=False,
        )
        spec = get_ablation_specs()[1]  # A1
        out = apply_ablation_to_row(row, _label(), _corpus(), spec, 0.75)
        assert out["semantic_similarity"] == 0.0
        # No exact, no alias → not detected → allowed
        assert out["blocked"] is False
        assert out["allowed"] is True

    def test_a1_exact_still_detected(self):
        """A1: exact match still detected even without semantic."""
        row = _result(
            candidate_id="c1", blocked=True, sim=0.9,
            exact=True, alias=False,
        )
        spec = get_ablation_specs()[1]  # A1
        out = apply_ablation_to_row(row, _label(), _corpus(), spec, 0.75)
        assert out["blocked"] is True

    def test_a4_recontamination_unblocked(self):
        """A4: recontamination candidates become allowed."""
        row = _result(candidate_id="c1", blocked=True)
        corpus = _corpus(attack_type="recontamination")
        spec = get_ablation_specs()[4]  # A4
        out = apply_ablation_to_row(row, _label(), corpus, spec, 0.75)
        assert out["blocked"] is False
        assert out["allowed"] is True

    def test_a4_non_recontamination_unchanged(self):
        """A4: non-recontamination rows unchanged."""
        row = _result(candidate_id="c1", blocked=True)
        corpus = _corpus(attack_type="direct_disclosure")
        spec = get_ablation_specs()[4]  # A4
        out = apply_ablation_to_row(row, _label(), corpus, spec, 0.75)
        assert out["blocked"] is True


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
# run_ablation_study
# ===========================================================================


class TestRunAblationStudy:
    """Tests for the full ablation study runner."""

    def test_returns_five_ablations(self):
        """Study returns metrics for all 5 ablations."""
        results = [
            _result(candidate_id="c1", blocked=True, sim=0.9,
                    exact=False, alias=False),
            _result(candidate_id="c2", blocked=False, sim=0.3),
        ]
        labels = {
            "c1": _label(leakage=True),
            "c2": _label(leakage=False),
        }
        corpus = {
            "c1": _corpus(attack_type="direct_disclosure"),
            "c2": _corpus(attack_type="hard_negative_control"),
        }

        study = run_ablation_study(results, labels, corpus, 0.75)
        assert len(study.ablations) == 5
        assert study.baseline_id == "A0"

    def test_baseline_property(self):
        """Baseline property returns A0 metrics."""
        results = [_result(candidate_id="c1", blocked=True)]
        labels = {"c1": _label(leakage=True)}
        corpus = {"c1": _corpus()}

        study = run_ablation_study(results, labels, corpus, 0.75)
        assert study.baseline.ablation_id == "A0"

    def test_a1_reduces_detection(self):
        """A1 (no semantic) should detect fewer than A0 for semantic-only cases."""
        results = [
            _result(candidate_id="c1", blocked=True, sim=0.9,
                    exact=False, alias=False),
        ]
        labels = {"c1": _label(leakage=True)}
        corpus = {"c1": _corpus(attack_type="semantic_paraphrase")}

        study = run_ablation_study(results, labels, corpus, 0.75)
        a0 = study.baseline
        a1 = next(a for a in study.ablations if a.ablation_id == "A1")
        # A0: blocked (semantic detected), A1: not blocked (no semantic)
        assert a0.leakage_prevention == 1.0
        assert a1.leakage_prevention == 0.0


# ===========================================================================
# compute_ablation_impacts
# ===========================================================================


class TestComputeAblationImpacts:
    """Tests for relative impact computation."""

    def test_impact_deltas(self):
        """Impact = baseline - ablation for leakage, ablation - baseline for FBR."""
        results = [
            _result(candidate_id="c1", blocked=True, sim=0.9,
                    exact=False, alias=False),
            _result(candidate_id="c2", blocked=False, sim=0.3),
        ]
        labels = {
            "c1": _label(leakage=True),
            "c2": _label(leakage=False),
        }
        corpus = {
            "c1": _corpus(attack_type="direct_disclosure"),
            "c2": _corpus(attack_type="hard_negative_control"),
        }

        study = run_ablation_study(results, labels, corpus, 0.75)
        impacts = compute_ablation_impacts(study)

        # 4 non-baseline ablations
        assert len(impacts) == 4
        # A1 impact: semantic disabled → leakage prevention drops
        a1_impact = next(i for i in impacts if i.ablation_id == "A1")
        assert a1_impact.disabled_component == "semantic"
        assert a1_impact.leakage_prevention_delta == 1.0  # 1.0 - 0.0

    def test_baseline_excluded(self):
        """Baseline (A0) is not in impacts list."""
        results = [_result(candidate_id="c1", blocked=True)]
        labels = {"c1": _label(leakage=True)}
        corpus = {"c1": _corpus()}

        study = run_ablation_study(results, labels, corpus, 0.75)
        impacts = compute_ablation_impacts(study)
        ids = [i.ablation_id for i in impacts]
        assert "A0" not in ids


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
