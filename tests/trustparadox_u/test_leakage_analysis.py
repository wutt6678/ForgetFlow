"""FF92-016 tests: leakage analysis computed from trial artifacts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u import leakage_analysis  # noqa: E402
from experiments.trustparadox_u.leakage_analysis import (  # noqa: E402
    load_leakage_inputs,
    run_leakage_analysis,
    target_type_for,
    write_leakage_analysis,
)
from experiments.trustparadox_u.trial_artifacts import (  # noqa: E402
    STATUS_SUCCESS,
    CandidateTrial,
)

CONDITIONS = ("no_firewall", "full_mvp")


def _trial(
    candidate_id: str,
    condition: str,
    attack_type: str,
    *,
    labels: tuple[str, ...] = (),
    blocked: bool = False,
    forget_ids: tuple[str, ...] = ("F001",),
    scenario: str = "pilot_credential",
    trust: str = "high",
    sequence_id: str = "",
) -> CandidateTrial:
    return CandidateTrial(
        candidate_id=candidate_id,
        candidate_ids=(candidate_id,),
        condition_id=condition,
        scenario_id=scenario,
        trust_level=trust,
        secret_variant_id="v1",
        attack_type=attack_type,
        target_forget_ids=forget_ids,
        sequence_id=sequence_id,
        episode_id=f"ep_{candidate_id}_{condition}",
        run_id="r1",
        seed=42,
        released_exposure_labels=labels,
        eligible_opportunities=len(labels),
        task_label=None,
        task_success=True,
        blocked_legitimate=blocked,
        result_status=STATUS_SUCCESS,
        failure_reason=None,
    )


def _build_trials() -> list[CandidateTrial]:
    trials: list[CandidateTrial] = []
    for condition in CONDITIONS:
        exposed = condition == "no_firewall"
        trials.append(
            _trial(
                "c_direct",
                condition,
                "direct",
                labels=("exact_value_disclosure",) if exposed else ("none",),
            )
        )
        trials.append(_trial("c_benign", condition, "benign_control", labels=("none",)))
        trials.append(
            _trial(
                "c_legit",
                condition,
                "legitimate_task",
                labels=("none",),
                blocked=condition == "full_mvp",
                forget_ids=("F002",),
                scenario="pilot_attribute",
            )
        )
        trials.append(
            _trial(
                "c_seq",
                condition,
                "temporal_fragmentation",
                sequence_id="seq1",
            )
        )
        trials.append(
            _trial(
                "c_recont",
                condition,
                "recontamination",
                forget_ids=("F003",),
                scenario="pilot_auth",
            )
        )
        trials.append(
            _trial(
                "c_multi",
                condition,
                "direct",
                labels=("alias_disclosure",) if exposed else ("none",),
                forget_ids=("F001", "F003"),
                scenario="pilot_multi",
            )
        )
    return trials


def _build_reconstruction() -> list[dict[str, Any]]:
    return [
        {
            "condition": condition,
            "episode_id": f"ep_c_seq_{condition}",
            "sequence_id": "seq1",
            "forget_id": "F001",
            "eligible": True,
            "recovered": condition == "no_firewall",
        }
        for condition in CONDITIONS
    ]


def _build_recontamination() -> list[dict[str, Any]]:
    return [
        {
            "condition": condition,
            "episode_id": f"ep_c_recont_{condition}",
            "candidate_id": "c_recont",
            "agent_id": "agent_1",
            "forget_id": "F003",
            "eligible": True,
            "probe_executed": True,
            "probe_recovered_target": condition == "no_firewall",
        }
        for condition in CONDITIONS
    ]


def _build_annotations() -> list[dict[str, Any]]:
    attack_by_candidate = {
        "c_direct": "direct",
        "c_benign": "benign_control",
        "c_legit": "legitimate_task",
        "c_seq": "temporal_fragmentation",
        "c_recont": "recontamination",
        "c_multi": "direct",
    }
    return [
        {"candidate_id": cid, "attack_type": attack} for cid, attack in attack_by_candidate.items()
    ]


def _inputs() -> dict[str, Any]:
    return {
        "candidate_trials": _build_trials(),
        "reconstruction_records": _build_reconstruction(),
        "recontamination_records": _build_recontamination(),
        "annotations": _build_annotations(),
    }


class TestGroupings:
    """FF92-016: condition, attack type, scenario, trust, target type."""

    def test_all_required_tables_present(self) -> None:
        analysis = run_leakage_analysis(_inputs())
        for key in (
            "by_condition",
            "by_condition_and_attack",
            "by_condition_and_scenario",
            "by_condition_and_trust",
            "by_condition_and_target_type",
            "global",
        ):
            assert key in analysis

    def test_conditions_never_pooled(self) -> None:
        analysis = run_leakage_analysis(_inputs())
        assert set(analysis["by_condition"]) == set(CONDITIONS)
        for table in (
            "by_condition_and_attack",
            "by_condition_and_scenario",
            "by_condition_and_trust",
            "by_condition_and_target_type",
        ):
            assert set(analysis[table]) == set(CONDITIONS)

    def test_target_type_grouping(self) -> None:
        analysis = run_leakage_analysis(_inputs())
        types = set()
        for cells in analysis["by_condition_and_target_type"].values():
            types.update(cells)
        assert types == {"credential", "private_attribute", "authorization", "multi_target"}

    def test_target_type_for_multi(self) -> None:
        assert target_type_for(("F001",)) == "credential"
        assert target_type_for(("F001", "F003")) == "multi_target"
        with pytest.raises(ValueError):
            target_type_for(())


class TestMetricCells:
    """Breakdowns represent the defined metrics."""

    def test_no_firewall_direct_high_exposure(self) -> None:
        analysis = run_leakage_analysis(_inputs())
        cell = analysis["by_condition_and_attack"]["no_firewall"]["direct"]
        assert cell["pu_rer"]["evaluable"]
        assert cell["pu_rer"]["value"] == 1.0

    def test_full_mvp_blocks_direct(self) -> None:
        analysis = run_leakage_analysis(_inputs())
        cell = analysis["by_condition_and_attack"]["full_mvp"]["direct"]
        assert cell["pu_rer"]["numerator"] == 0
        assert cell["pu_rer"]["denominator"] > 0

    def test_crr_rr_fbr_per_condition(self) -> None:
        analysis = run_leakage_analysis(_inputs())
        nf = analysis["by_condition"]["no_firewall"]
        fm = analysis["by_condition"]["full_mvp"]
        assert nf["crr"]["numerator"] == 1 and nf["crr"]["denominator"] == 1
        assert fm["crr"]["numerator"] == 0 and fm["crr"]["denominator"] == 1
        assert nf["rr"]["numerator"] == 1 and fm["rr"]["numerator"] == 0
        assert fm["fbr"]["numerator"] == 1 and nf["fbr"]["numerator"] == 0


class TestValidationChecks:
    """FF92-016 required validation: all failures raise."""

    def test_validation_passes_on_sound_inputs(self) -> None:
        analysis = run_leakage_analysis(_inputs())
        for check in analysis["validation"].values():
            assert check["passed"] is True

    def test_benign_exposure_raises(self) -> None:
        inputs = _inputs()
        inputs["candidate_trials"] = [
            CandidateTrial(
                candidate_id=trial.candidate_id,
                candidate_ids=trial.candidate_ids,
                condition_id=trial.condition_id,
                scenario_id=trial.scenario_id,
                trust_level=trial.trust_level,
                secret_variant_id=trial.secret_variant_id,
                attack_type=trial.attack_type,
                target_forget_ids=trial.target_forget_ids,
                sequence_id=trial.sequence_id,
                episode_id=trial.episode_id,
                run_id=trial.run_id,
                seed=trial.seed,
                released_exposure_labels=("exact_value_disclosure",)
                if trial.attack_type == "benign_control"
                else trial.released_exposure_labels,
                eligible_opportunities=1
                if trial.attack_type == "benign_control"
                else trial.eligible_opportunities,
                task_label=trial.task_label,
                task_success=trial.task_success,
                blocked_legitimate=trial.blocked_legitimate,
                result_status=trial.result_status,
                failure_reason=trial.failure_reason,
            )
            for trial in inputs["candidate_trials"]
        ]
        with pytest.raises(ValueError, match="[Bb]enign"):
            run_leakage_analysis(inputs)

    def test_condition_counts_sum_to_global(self) -> None:
        analysis = run_leakage_analysis(_inputs())
        for metric in ("pu_rer", "crr", "rr", "fbr"):
            cond_num = sum(cell[metric]["numerator"] for cell in analysis["by_condition"].values())
            assert cond_num == analysis["global"][metric]["numerator"]

    def test_annotation_mismatch_raises(self) -> None:
        inputs = _inputs()
        inputs["annotations"] = [a for a in inputs["annotations"] if a["candidate_id"] != "c_multi"]
        with pytest.raises(ValueError, match="annotation"):
            run_leakage_analysis(inputs)


class TestInputsAndOutput:
    def test_missing_inputs_fail_fast(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_leakage_inputs(tmp_path, tmp_path / "corpus_annotations.jsonl")

    def test_write_creates_file(self, tmp_path: Path) -> None:
        analysis = run_leakage_analysis(_inputs())
        write_leakage_analysis(analysis, tmp_path)
        assert (tmp_path / "leakage_analysis.json").exists()

    def test_no_shallow_exposure_counter_in_source(self) -> None:
        source = Path(leakage_analysis.__file__).read_text()
        assert "cleaned_agents_exposed" not in source
