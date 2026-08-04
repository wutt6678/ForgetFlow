"""FF92-017 behavioral tests: paired statistics from trial artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u import paired_statistics  # noqa: E402
from experiments.trustparadox_u.paired_statistics import (  # noqa: E402
    BASELINE_CONDITION,
    compare_paired_outcomes,
    exact_mcnemar_test,
    exposure_outcomes_by_condition,
    false_block_outcomes_by_condition,
    load_paired_inputs,
    reconstruction_outcomes_by_condition,
    recontamination_outcomes_by_condition,
    run_paired_statistics,
    write_paired_statistics,
)
from experiments.trustparadox_u.trial_artifacts import (  # noqa: E402
    STATUS_SUCCESS,
    CandidateTrial,
    UtilityTrial,
)

CONDITIONS = ("no_firewall", "full_mvp")


def _trial(
    candidate_id: str,
    condition: str,
    attack_type: str,
    *,
    exposed: bool = False,
    blocked: bool = False,
    status: str = STATUS_SUCCESS,
) -> CandidateTrial:
    return CandidateTrial(
        candidate_id=candidate_id,
        candidate_ids=(candidate_id,),
        condition_id=condition,
        scenario_id="pilot_credential",
        trust_level="high",
        secret_variant_id="v1",
        attack_type=attack_type,
        target_forget_ids=("F001",),
        sequence_id="",
        episode_id=f"ep_{candidate_id}_{condition}",
        run_id="r1",
        seed=42,
        released_exposure_labels=("exact_value_disclosure",) if exposed else ("none",),
        eligible_opportunities=1,
        task_label="answer",
        task_success=not blocked,
        blocked_legitimate=blocked,
        result_status=status,
        failure_reason=None,
    )


def _build_candidate_trials() -> list[CandidateTrial]:
    """Attack candidates: exposed under no_firewall only; one unmatched per side."""
    trials: list[CandidateTrial] = []
    for condition in CONDITIONS:
        exposed = condition == "no_firewall"
        trials.append(_trial("c_direct", condition, "direct", exposed=exposed))
        trials.append(_trial("c_alias", condition, "alias", exposed=False))
        trials.append(_trial("c_paraphrase", condition, "paraphrase", exposed=False))
        trials.append(_trial("c_legit", condition, "legitimate_task", blocked=exposed))
    # Unmatched pairing units: one only under no_firewall, one only under full_mvp.
    trials.append(_trial("c_only_baseline", "no_firewall", "direct", exposed=True))
    trials.append(_trial("c_only_firewall", "full_mvp", "direct"))
    # Failed trials must be excluded from pairing.
    trials.append(_trial("c_failed", "no_firewall", "direct", status="error"))
    return trials


def _build_reconstruction_records() -> list[dict]:
    return [
        {
            "condition": condition,
            "episode_id": "ep_seq",
            "sequence_id": "seq1",
            "forget_id": "F001",
            "eligible": True,
            "recovered": condition == "no_firewall",
        }
        for condition in CONDITIONS
    ] + [
        {
            "condition": "no_firewall",
            "episode_id": "ep_seq_ineligible",
            "sequence_id": "seq2",
            "forget_id": "F001",
            "eligible": False,
            "recovered": True,
        }
    ]


def _build_recontamination_records() -> list[dict]:
    return [
        {
            "condition": condition,
            "episode_id": "ep_recont",
            "candidate_id": "c_recont",
            "agent_id": "agent_1",
            "forget_id": "F003",
            "probe_executed": True,
            "probe_recovered_target": condition == "no_firewall",
        }
        for condition in CONDITIONS
    ] + [
        {
            "condition": "no_firewall",
            "episode_id": "ep_recont_skip",
            "candidate_id": "c_recont_skip",
            "agent_id": "agent_2",
            "forget_id": "F003",
            "probe_executed": False,
            "probe_recovered_target": False,
        }
    ]


def _build_utility_trials() -> list[UtilityTrial]:
    return [
        UtilityTrial(
            candidate_id="c_legit",
            scenario_id="pilot_credential",
            trust_level="high",
            secret_variant_id="v1",
            baseline_condition="no_firewall",
            firewall_condition="full_mvp",
            baseline_task_success=True,
            firewall_task_success=False,
            firewall_blocked=True,
        ),
        UtilityTrial(
            candidate_id="c_benign",
            scenario_id="pilot_credential",
            trust_level="high",
            secret_variant_id="v1",
            baseline_condition="no_firewall",
            firewall_condition="full_mvp",
            baseline_task_success=False,  # ineligible: excluded from success pairs
            firewall_task_success=False,
            firewall_blocked=False,
        ),
    ]


def _build_inputs() -> dict:
    return {
        "candidate_trials": _build_candidate_trials(),
        "reconstruction_records": _build_reconstruction_records(),
        "recontamination_records": _build_recontamination_records(),
        "utility_trials": _build_utility_trials(),
    }


class TestExactMcNemar:
    """Known contingency tables must produce the expected exact p-values."""

    def test_known_table_2_8(self) -> None:
        # n=10, k=2: 2*sum(C(10,i) for i<=2)/1024 = 2*(1+10+45)/1024
        result = exact_mcnemar_test(2, 8)
        assert result["p_value"] == pytest.approx(112 / 1024)

    def test_known_table_0_10(self) -> None:
        result = exact_mcnemar_test(0, 10)
        assert result["p_value"] == pytest.approx(2 / 1024)

    def test_balanced_discordant(self) -> None:
        assert exact_mcnemar_test(5, 5)["p_value"] == 1.0

    def test_no_discordant(self) -> None:
        result = exact_mcnemar_test(0, 0)
        assert result["p_value"] == 1.0
        assert result["discordant"] == 0

    def test_exact_not_asymptotic(self) -> None:
        # Small sample: an asymptotic chi-square approximation would differ.
        result = exact_mcnemar_test(1, 3)
        assert result["p_value"] == pytest.approx(2 * (1 + 4) / 16)
        assert result["test"] == "exact_mcnemar"


class TestOutcomeIndexing:
    def test_exposure_indexes_attacks_only(self) -> None:
        outcomes = exposure_outcomes_by_condition(_build_candidate_trials())
        baseline = outcomes["no_firewall"]
        assert baseline["c_direct"] is True
        assert baseline["c_alias"] is False
        assert "c_legit" not in baseline  # legitimate tasks are not exposures
        assert "c_failed" not in baseline  # failed trials excluded
        assert outcomes["full_mvp"]["c_direct"] is False

    def test_exposure_duplicate_candidate_raises(self) -> None:
        trials = [
            _trial("c_dup", "full_mvp", "direct"),
            _trial("c_dup", "full_mvp", "direct"),
        ]
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            exposure_outcomes_by_condition(trials)

    def test_reconstruction_keyed_by_sequence_identity(self) -> None:
        outcomes = reconstruction_outcomes_by_condition(_build_reconstruction_records())
        assert outcomes["no_firewall"] == {"ep_seq|seq1|F001": True}
        assert outcomes["full_mvp"] == {"ep_seq|seq1|F001": False}
        assert "ep_seq_ineligible|seq2|F001" not in outcomes["no_firewall"]  # ineligible excluded

    def test_reconstruction_duplicate_sequence_raises(self) -> None:
        records = [
            {"condition": "c", "sequence_id": "s1", "forget_id": "F001", "eligible": True},
            {"condition": "c", "sequence_id": "s1", "forget_id": "F001", "eligible": True},
        ]
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            reconstruction_outcomes_by_condition(records)

    def test_recontamination_skips_unexecuted_probes(self) -> None:
        outcomes = recontamination_outcomes_by_condition(_build_recontamination_records())
        assert outcomes["no_firewall"] == {"ep_recont|agent_1|F003": True}
        assert outcomes["full_mvp"] == {"ep_recont|agent_1|F003": False}

    def test_false_block_indexes_legitimate_only(self) -> None:
        outcomes = false_block_outcomes_by_condition(_build_candidate_trials())
        assert outcomes["no_firewall"] == {"c_legit": True}
        assert outcomes["full_mvp"] == {"c_legit": False}
        assert "c_direct" not in outcomes["no_firewall"]


class TestComparePairedOutcomes:
    def test_unmatched_units_reported(self) -> None:
        comp = compare_paired_outcomes(
            {"a": True, "b": False},
            {"a": True, "c": True},
            condition_a="A",
            condition_b="B",
            metric="exposure",
            pairing_unit="candidate_id",
        )
        assert comp["unmatched"]["only_in_a"] == ["b"]
        assert comp["unmatched"]["only_in_b"] == ["c"]
        assert comp["n_pairs"] == 1

    def test_full_report_fields(self) -> None:
        outcomes_a = {f"c{i}": i < 6 for i in range(10)}
        outcomes_b = {f"c{i}": i < 2 for i in range(10)}
        comp = compare_paired_outcomes(
            outcomes_a,
            outcomes_b,
            condition_a="A",
            condition_b="B",
            metric="exposure",
            pairing_unit="candidate_id",
            n_permutations=1000,
            n_bootstrap=1000,
        )
        assert comp["n_pairs"] == 10
        assert comp["rate_a"] == pytest.approx(0.6)
        assert comp["rate_b"] == pytest.approx(0.2)
        assert comp["risk_difference"] == pytest.approx(0.4)
        assert comp["relative_risk"] == pytest.approx(3.0)
        assert comp["contingency"]["both_positive"] == 2
        assert comp["contingency"]["a_only"] == 4
        assert comp["contingency"]["b_only"] == 0
        assert comp["contingency"]["both_negative"] == 4
        assert comp["mcnemar"]["p_value"] == pytest.approx(2 * 1 / 16)
        assert 0.0 <= comp["permutation"]["p_value"] <= 1.0
        ci = comp["bootstrap_ci_95"]
        assert ci["lower"] <= comp["risk_difference"] <= ci["upper"]
        assert comp["cohens_h"] == pytest.approx(comp["effect_size"])
        assert comp["cohens_h"] > 0

    def test_relative_risk_none_when_rate_b_zero(self) -> None:
        comp = compare_paired_outcomes(
            {"a": True},
            {"a": False},
            condition_a="A",
            condition_b="B",
            metric="exposure",
            pairing_unit="candidate_id",
        )
        assert comp["rate_b"] == 0.0
        assert comp["relative_risk"] is None

    def test_no_common_pairs_reports_zero_n(self) -> None:
        comp = compare_paired_outcomes(
            {"a": True},
            {"b": False},
            condition_a="A",
            condition_b="B",
            metric="exposure",
            pairing_unit="candidate_id",
        )
        assert comp["n_pairs"] == 0
        assert comp["risk_difference"] is None


class TestRunPairedStatistics:
    def test_end_to_end_comparison_set(self) -> None:
        comparisons = run_paired_statistics(
            _build_inputs(),
            condition_pairs=[("no_firewall", "full_mvp")],
            n_permutations=1000,
            n_bootstrap=1000,
        )
        metrics = {c["metric"] for c in comparisons}
        assert metrics == {
            "exposure",
            "reconstruction",
            "recontamination",
            "false_block",
            "utility",
            "utility_false_block",
        }

        by_metric = {c["metric"]: c for c in comparisons}
        exposure = by_metric["exposure"]
        assert exposure["rate_a"] > exposure["rate_b"]
        # Only c_direct is exposed, and only under the baseline condition.
        assert exposure["n_pairs"] == 3
        assert exposure["contingency"]["a_only"] == 1
        assert exposure["contingency"]["both_negative"] == 2
        assert exposure["unmatched"]["only_in_a"] == ["c_only_baseline"]
        assert exposure["unmatched"]["only_in_b"] == ["c_only_firewall"]

        reconstruction = by_metric["reconstruction"]
        assert reconstruction["pairing_unit"] == "sequence_id"
        assert reconstruction["rate_a"] == 1.0
        assert reconstruction["rate_b"] == 0.0

        recontamination = by_metric["recontamination"]
        assert recontamination["rate_a"] == 1.0
        assert recontamination["rate_b"] == 0.0

        false_block = by_metric["false_block"]
        assert false_block["rate_a"] == 1.0  # baseline blocked the legitimate task
        assert false_block["rate_b"] == 0.0

        utility = by_metric["utility"]
        assert utility["n_pairs"] == 1  # ineligible trial excluded
        assert utility["rate_a"] == 1.0  # baseline succeeds by definition
        assert utility["rate_b"] == 0.0

        utility_false_block = by_metric["utility_false_block"]
        assert utility_false_block["rate_b"] == 0.5  # one of two candidates blocked

    def test_utility_only_for_baseline_pairs(self) -> None:
        comparisons = run_paired_statistics(
            _build_inputs(),
            condition_pairs=[("full_mvp", "no_monitoring")],
            n_permutations=100,
            n_bootstrap=100,
        )
        metrics = {c["metric"] for c in comparisons}
        assert "utility" not in metrics
        assert "utility_false_block" not in metrics

    def test_conditions_never_pooled(self) -> None:
        comparisons = run_paired_statistics(
            _build_inputs(),
            condition_pairs=[("no_firewall", "full_mvp")],
            n_permutations=100,
            n_bootstrap=100,
        )
        for comp in comparisons:
            assert comp["condition_a"] == "no_firewall"
            assert comp["condition_b"] == "full_mvp"

    def test_deterministic_given_seed(self) -> None:
        inputs = _build_inputs()
        kwargs = dict(
            condition_pairs=[("no_firewall", "full_mvp")],
            n_permutations=500,
            n_bootstrap=500,
        )
        first = run_paired_statistics(inputs, **kwargs)
        second = run_paired_statistics(inputs, **kwargs)
        assert first == second


class TestInputsAndOutput:
    def test_missing_inputs_raise(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="frozen_replay"):
            load_paired_inputs(tmp_path)

    def test_write_creates_report(self, tmp_path: Path) -> None:
        comparisons = run_paired_statistics(
            _build_inputs(),
            condition_pairs=[("no_firewall", "full_mvp")],
            n_permutations=100,
            n_bootstrap=100,
        )
        write_paired_statistics(comparisons, tmp_path)
        path = tmp_path / "paired_statistics.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["num_comparisons"] == len(comparisons)
        assert data["baseline_condition"] == BASELINE_CONDITION
        # table4 consumer keys are preserved.
        comp = data["comparisons"][0]
        assert "rate_a" in comp and "cohens_h" in comp
        assert "p_value" in comp["mcnemar"]

    def test_no_shallow_counter_in_source(self) -> None:
        source = Path(paired_statistics.__file__).read_text()
        assert "cleaned_agents_exposed" not in source
