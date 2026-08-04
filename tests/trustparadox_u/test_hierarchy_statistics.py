"""Remediation §25/§26 behavioral tests: hierarchy-aware statistics.

§25: statistical units must match the data hierarchy — pairing units are
metric-specific, confidence intervals preserve the dependence structure
(scenario-cluster bootstrap), and candidate/sequence outcomes are never
pooled as independent rows.

§26: every primary comparison reports numerator, denominator, point
estimate, 95% CIs, paired effect size, cluster/scenario/secret-variant
counts and the number of independent pairing units.  Perfect observed
rates keep non-degenerate interval estimates.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.paired_statistics import (  # noqa: E402
    SMALL_CELL_MINIMUM,
    cluster_bootstrap_ci,
    compare_paired_outcomes,
    run_paired_statistics,
    wilson_score_ci,
)
from experiments.trustparadox_u.trial_artifacts import (  # noqa: E402
    STATUS_SUCCESS,
    CandidateTrial,
)

CONDITIONS = ("no_firewall", "full_mvp")


def _trial(
    candidate_id: str,
    condition: str,
    *,
    scenario: str,
    variant: str,
    attack_type: str = "direct",
    exposed: bool = False,
) -> CandidateTrial:
    return CandidateTrial(
        candidate_id=candidate_id,
        candidate_ids=(candidate_id,),
        condition_id=condition,
        scenario_id=scenario,
        trust_level="high",
        secret_variant_id=variant,
        attack_type=attack_type,
        target_forget_ids=("F001",),
        sequence_id="",
        episode_id=f"ep_{candidate_id}_{condition}",
        run_id="r1",
        seed=42,
        released_exposure_labels=("exact_value_disclosure",) if exposed else ("none",),
        eligible_opportunities=1,
        task_label="answer",
        task_success=True,
        blocked_legitimate=False,
        result_status=STATUS_SUCCESS,
        failure_reason=None,
    )


class TestWilsonScoreIntervalRemediation26:
    """§26: perfect observed rates must not be reported as certainty."""

    def test_perfect_rate_keeps_non_degenerate_interval(self) -> None:
        ci = wilson_score_ci(10, 10)
        assert ci is not None
        assert ci["rate"] == 1.0
        assert ci["upper"] == 1.0
        assert ci["lower"] < 1.0
        assert ci["lower"] > 0.0

    def test_zero_rate_keeps_non_degenerate_interval(self) -> None:
        ci = wilson_score_ci(0, 10)
        assert ci is not None
        assert ci["rate"] == 0.0
        assert ci["lower"] == 0.0
        assert ci["upper"] > 0.0

    def test_interior_rate_brackets_point_estimate(self) -> None:
        ci = wilson_score_ci(5, 10)
        assert ci is not None
        assert ci["lower"] < 0.5 < ci["upper"]
        assert ci["method"] == "wilson"
        assert ci["successes"] == 5
        assert ci["n"] == 10

    def test_empty_denominator_is_not_evaluable(self) -> None:
        assert wilson_score_ci(0, 0) is None


class TestClusterBootstrapRemediation25:
    """§25: CIs must preserve within-cluster dependence."""

    def test_empty_input_reports_no_clusters(self) -> None:
        ci = cluster_bootstrap_ci([], [])
        assert ci["lower"] is None
        assert ci["upper"] is None
        assert ci["n_clusters"] == 0
        assert ci["method"] == "cluster_bootstrap"

    def test_length_mismatch_fails_loudly(self) -> None:
        try:
            cluster_bootstrap_ci([1, 0], ["s1"])
        except ValueError:
            return
        raise AssertionError("expected ValueError for mismatched lengths")

    def test_deterministic_given_seed(self) -> None:
        diffs = [1, 1, 1, -1, -1, 0, 1, -1]
        clusters = ["s1", "s1", "s1", "s1", "s2", "s2", "s2", "s2"]
        first = cluster_bootstrap_ci(diffs, clusters, n_iterations=2000)
        second = cluster_bootstrap_ci(diffs, clusters, n_iterations=2000)
        assert first["lower"] == second["lower"]
        assert first["upper"] == second["upper"]

    def test_clustered_extremes_widen_the_interval(self) -> None:
        # Two clusters with opposite, internally identical outcomes: a
        # unit-level bootstrap would see many near-zero resamples, but the
        # cluster bootstrap can only draw whole clusters, so the interval
        # must span the cluster-level extremes.
        diffs = [1] * 20 + [-1] * 20
        clusters = ["s_pos"] * 20 + ["s_neg"] * 20
        ci = cluster_bootstrap_ci(diffs, clusters, n_iterations=4000)
        assert ci["n_clusters"] == 2
        assert ci["lower"] <= -1.0 + 1e-12
        assert ci["upper"] >= 1.0 - 1e-12


class TestHierarchyComparisonReportingRemediation2526:
    """§25/§26: comparisons carry design counts and scenario sensitivity."""

    def _comparison(self, **kwargs):
        outcomes_a = {"c1": True, "c2": True, "c3": False, "c4": True}
        outcomes_b = {"c1": False, "c2": False, "c3": False, "c4": False}
        scenarios = {"c1": "s1", "c2": "s1", "c3": "s2", "c4": "s2"}
        variants = {"c1": "v1", "c2": "v1", "c3": "v2", "c4": "v2"}
        return compare_paired_outcomes(
            outcomes_a,
            outcomes_b,
            condition_a="no_firewall",
            condition_b="full_mvp",
            metric="exposure",
            pairing_unit="candidate_id",
            n_permutations=200,
            n_bootstrap=200,
            unit_scenarios=kwargs.get("unit_scenarios", scenarios),
            unit_secret_variants_a=kwargs.get("variants", variants),
            unit_secret_variants_b=kwargs.get("variants", variants),
        )

    def test_numerators_denominators_and_rate_cis(self) -> None:
        comp = self._comparison()
        assert comp["numerator_a"] == 3
        assert comp["denominator_a"] == 4
        assert comp["numerator_b"] == 0
        assert comp["denominator_b"] == 4
        assert comp["rate_ci_95_a"] is not None
        assert comp["rate_ci_95_b"] is not None
        assert comp["rate_ci_95_a"]["lower"] <= comp["rate_a"] <= comp["rate_ci_95_a"]["upper"]

    def test_perfect_rate_flagged_with_note(self) -> None:
        comp = self._comparison()
        # rate_b = 0.0 is a boundary rate: must never be read as certainty.
        assert comp["perfect_rate_observed"] is True
        assert comp["interpretation_note"]
        assert comp["rate_ci_95_b"]["upper"] > 0.0

    def test_cluster_ci_and_design_summary(self) -> None:
        comp = self._comparison()
        cluster_ci = comp["cluster_bootstrap_ci_95"]
        assert cluster_ci is not None
        assert cluster_ci["method"] == "cluster_bootstrap"
        assert cluster_ci["n_clusters"] == 2
        design = comp["design_summary"]
        assert design["n_clusters"] == 2
        assert design["n_scenarios"] == 2
        assert design["n_secret_variants_a"] == 2
        assert design["n_secret_variants_b"] == 2
        assert design["cluster_unit"] == "scenario_id"

    def test_small_cells_identified(self) -> None:
        comp = self._comparison()
        # Both scenarios carry only 2 pairs < SMALL_CELL_MINIMUM.
        assert set(comp["design_summary"]["small_cell_scenarios"]) == {"s1", "s2"}
        assert comp["design_summary"]["min_scenario_pairs"] == 2
        assert SMALL_CELL_MINIMUM > 2

    def test_scenario_sensitivity_rates(self) -> None:
        comp = self._comparison()
        sensitivity = comp["scenario_sensitivity"]
        assert set(sensitivity) == {"s1", "s2"}
        assert sensitivity["s1"]["rate_a"] == 1.0
        assert sensitivity["s1"]["rate_b"] == 0.0
        assert sensitivity["s2"]["rate_a"] == 0.5
        assert sensitivity["s2"]["n"] == 2

    def test_backward_compatible_without_clusters(self) -> None:
        comp = self._comparison(unit_scenarios=None, variants=None)
        assert comp["cluster_bootstrap_ci_95"] is None
        assert comp["design_summary"]["n_clusters"] is None
        assert comp["scenario_sensitivity"] == {}
        # Numerators, denominators and Wilson CIs are still reported.
        assert comp["numerator_a"] == 3
        assert comp["rate_ci_95_a"] is not None


class TestRunPairedStatisticsHierarchyRemediation25:
    """§25: the full pipeline attaches hierarchy metadata per metric."""

    def _inputs(self) -> dict:
        trials: list[CandidateTrial] = []
        for index, (scenario, variant) in enumerate(
            [("pilot_credential", "v1"), ("pilot_authorization", "v2")]
        ):
            for condition in CONDITIONS:
                exposed = condition == "no_firewall"
                trials.append(
                    _trial(
                        f"c_{scenario}_{index}",
                        condition,
                        scenario=scenario,
                        variant=variant,
                        exposed=exposed,
                    )
                )
        reconstruction = [
            {
                "condition": condition,
                "episode_id": "ep_seq",
                "sequence_id": "seq1",
                "forget_id": "F001",
                "scenario_id": "pilot_credential",
                "eligible": True,
                "recovered": condition == "no_firewall",
            }
            for condition in CONDITIONS
        ]
        recontamination = [
            {
                "condition": condition,
                "episode_id": "ep_recont",
                "candidate_id": "c_recont",
                "agent_id": "agent_1",
                "forget_id": "F003",
                "scenario_id": "pilot_authorization",
                "probe_executed": True,
                "probe_recovered_target": condition == "no_firewall",
            }
            for condition in CONDITIONS
        ]
        return {
            "candidate_trials": trials,
            "reconstruction_records": reconstruction,
            "recontamination_records": recontamination,
            "utility_trials": [],
        }

    def test_every_comparison_is_hierarchy_aware(self) -> None:
        comparisons = run_paired_statistics(
            self._inputs(),
            condition_pairs=[CONDITIONS],
            n_permutations=100,
            n_bootstrap=100,
        )
        metrics = {comp["metric"] for comp in comparisons}
        # No legitimate candidates in the fixture: false-block is skipped.
        assert metrics == {"exposure", "reconstruction", "recontamination"}
        for comp in comparisons:
            assert comp["cluster_bootstrap_ci_95"] is not None
            assert comp["cluster_bootstrap_ci_95"]["n_clusters"] >= 1
            assert comp["design_summary"]["n_scenarios"] >= 1
            assert comp["rate_ci_95_a"] is not None
            assert comp["numerator_a"] is not None
            assert comp["scenario_sensitivity"]

    def test_metric_specific_pairing_units_preserved(self) -> None:
        comparisons = run_paired_statistics(
            self._inputs(),
            condition_pairs=[CONDITIONS],
            n_permutations=100,
            n_bootstrap=100,
        )
        units = {comp["metric"]: comp["pairing_unit"] for comp in comparisons}
        assert units["reconstruction"] == "sequence_id"
        assert units["exposure"] == "candidate_id"
        assert units["recontamination"] == "candidate_id"
