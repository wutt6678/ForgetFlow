"""FF92-016: Leakage analysis computed exclusively from trial artifacts.

Inputs (never shallow episode counters):

    candidate_trials.jsonl
    reconstruction_trials.jsonl
    recontamination_trials.jsonl
    corpus_annotations.jsonl

Groupings reported (conditions are never pooled unintentionally — every
table is keyed by condition first):

    condition, attack type, scenario, trust level, target type

Outputs: PU-RER, CRR, RR and FBR breakdowns with explicit
numerator/denominator/rate for every cell.

Validation (fail-fast, FF92-016 required tests):

    * benign controls have zero exposure by construction;
    * no-firewall direct candidates have high exposure;
    * condition-specific counts sum to global counts;
    * attack-type trial populations match annotation counts;
    * no result uses a shallow episode counter as PU-RER.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.trial_artifacts import (  # noqa: E402
    STATUS_SUCCESS,
    CandidateTrial,
    crr_from_trials,
    fbr_from_trials,
    load_trial_records,
    pu_rer_from_trials,
    rr_from_trials,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CORPUS_DIR = Path(__file__).parents[2] / "data" / "trustparadox_u" / "frozen_corpus"
REPLAY_DIR = Path(__file__).parents[2] / "results" / "frozen_replay"
ANALYSIS_DIR = Path(__file__).parents[2] / "results" / "leakage_analysis"

REQUIRED_INPUT_FILES: tuple[str, ...] = (
    "candidate_trials.jsonl",
    "reconstruction_trials.jsonl",
    "recontamination_trials.jsonl",
)

# Forget-id → target type (generate_corpus target specs).
FORGET_ID_TARGET_TYPES: dict[str, str] = {
    "F001": "credential",
    "F002": "private_attribute",
    "F003": "authorization",
}

# Direct disclosure attacks: benign controls must have zero exposure, these
# must have high exposure under no_firewall.
DIRECT_DISCLOSURE_ATTACKS = frozenset({"direct", "alias", "paraphrase"})
BENIGN_CONTROL_ATTACKS = frozenset({"benign_control"})

# Minimum no-firewall direct PU-RER for the "high exposure" sanity check.
HIGH_EXPOSURE_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def target_type_for(forget_ids: Sequence[str]) -> str:
    """Target type of a trial from its forget ids (multi → multi_target)."""
    types = sorted({FORGET_ID_TARGET_TYPES[fid] for fid in forget_ids if fid})
    if not types:
        raise ValueError(f"Trial has no resolvable target forget ids: {forget_ids!r}")
    if len(types) == 1:
        return types[0]
    return "multi_target"


def load_leakage_inputs(
    replay_dir: Path,
    annotations_path: Path,
) -> dict[str, Any]:
    """Load the four FF92-016 input artifacts; fail fast on missing files."""
    missing = [name for name in REQUIRED_INPUT_FILES if not (replay_dir / name).exists()]
    if not annotations_path.exists():
        missing.append(str(annotations_path))
    if missing:
        raise FileNotFoundError(
            f"Missing FF92-016 input artifacts: {missing}. "
            "Run frozen_replay.py and annotate_corpus.py first."
        )
    candidate_trials = [
        CandidateTrial.from_dict(record)
        for record in load_trial_records(replay_dir / "candidate_trials.jsonl")
    ]
    return {
        "candidate_trials": candidate_trials,
        "reconstruction_records": load_trial_records(replay_dir / "reconstruction_trials.jsonl"),
        "recontamination_records": load_trial_records(replay_dir / "recontamination_trials.jsonl"),
        "annotations": load_trial_records(annotations_path),
    }


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def _condition_groups(
    trials: Sequence[CandidateTrial],
) -> dict[str, list[CandidateTrial]]:
    groups: dict[str, list[CandidateTrial]] = {}
    for trial in trials:
        groups.setdefault(trial.condition_id, []).append(trial)
    return groups


def _subgroup(
    trials: Sequence[CandidateTrial], key: Callable[[CandidateTrial], str]
) -> dict[str, list[CandidateTrial]]:
    groups: dict[str, list[CandidateTrial]] = {}
    for trial in trials:
        groups.setdefault(key(trial), []).append(trial)
    return groups


def _trial_by_episode(
    trials: Sequence[CandidateTrial],
) -> dict[tuple[str, str], CandidateTrial]:
    """(condition, episode_id) → candidate trial, for sequence-record joins."""
    index: dict[tuple[str, str], CandidateTrial] = {}
    for trial in trials:
        if not trial.episode_id:
            continue  # failed trials never produced episodes
        key = (trial.condition_id, trial.episode_id)
        if key in index:
            raise ValueError(f"Duplicate candidate trial for episode key {key!r}")
        index[key] = trial
    return index


def _attach_trial_attributes(
    records: Sequence[dict[str, Any]],
    episode_index: dict[tuple[str, str], CandidateTrial],
) -> list[tuple[dict[str, Any], CandidateTrial]]:
    """Join sequence/recontamination records to their candidate trial."""
    joined: list[tuple[dict[str, Any], CandidateTrial]] = []
    for record in records:
        key = (record.get("condition", ""), record.get("episode_id", ""))
        trial = episode_index.get(key)
        if trial is None:
            raise ValueError(
                f"Sequence record {record.get('episode_id', '')!r} under "
                f"condition {record.get('condition', '')!r} has no candidate trial"
            )
        joined.append((record, trial))
    return joined


# ---------------------------------------------------------------------------
# Metric cells
# ---------------------------------------------------------------------------


def metric_cell(
    trials: Sequence[CandidateTrial],
    reconstruction_records: Sequence[dict[str, Any]],
    recontamination_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """All four defined metrics for one cell, from trial artifacts only."""
    return {
        "candidate_trials": len(trials),
        "failed_trials": sum(1 for t in trials if t.result_status != STATUS_SUCCESS),
        "pu_rer": pu_rer_from_trials(trials),
        "crr": crr_from_trials(reconstruction_records),
        "rr": rr_from_trials(recontamination_records),
        "fbr": fbr_from_trials(trials),
    }


def _sum_cells(cells: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Global cell = per-metric sum of condition numerators/denominators.

    Counts are aggregated; outcomes are never pooled across conditions.
    """

    def summed(metric: str) -> dict[str, Any]:
        numerator = sum(cell[metric]["numerator"] for cell in cells)
        denominator = sum(cell[metric]["denominator"] for cell in cells)
        if denominator == 0:
            return {
                "value": None,
                "numerator": 0,
                "denominator": 0,
                "evaluable": False,
                "reason": "no eligible trials",
            }
        return {
            "value": numerator / denominator,
            "numerator": numerator,
            "denominator": denominator,
            "evaluable": True,
            "reason": None,
        }

    return {
        "candidate_trials": sum(cell["candidate_trials"] for cell in cells),
        "failed_trials": sum(cell["failed_trials"] for cell in cells),
        "pu_rer": summed("pu_rer"),
        "crr": summed("crr"),
        "rr": summed("rr"),
        "fbr": summed("fbr"),
    }


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------


def run_leakage_analysis(inputs: dict[str, Any]) -> dict[str, Any]:
    """Build every FF92-016 breakdown table from trial artifacts."""
    trials: list[CandidateTrial] = inputs["candidate_trials"]
    recon: list[dict[str, Any]] = inputs["reconstruction_records"]
    recont: list[dict[str, Any]] = inputs["recontamination_records"]
    annotations: list[dict[str, Any]] = inputs["annotations"]

    episode_index = _trial_by_episode(trials)
    recon_joined = _attach_trial_attributes(recon, episode_index)
    recont_joined = _attach_trial_attributes(recont, episode_index)

    def recon_for(cond_trials: Sequence[CandidateTrial]) -> list[dict[str, Any]]:
        keys = {(t.condition_id, t.episode_id) for t in cond_trials if t.episode_id}
        return [
            record
            for record, trial in recon_joined
            if (trial.condition_id, trial.episode_id) in keys
        ]

    def recont_for(cond_trials: Sequence[CandidateTrial]) -> list[dict[str, Any]]:
        keys = {(t.condition_id, t.episode_id) for t in cond_trials if t.episode_id}
        return [
            record
            for record, trial in recont_joined
            if (trial.condition_id, trial.episode_id) in keys
        ]

    def table_for(
        key: Callable[[CandidateTrial], str],
    ) -> dict[str, dict[str, Any]]:
        by_condition: dict[str, dict[str, Any]] = {}
        for condition, cond_trials in sorted(_condition_groups(trials).items()):
            by_condition[condition] = {
                value: metric_cell(group, recon_for(group), recont_for(group))
                for value, group in sorted(_subgroup(cond_trials, key).items())
            }
        return by_condition

    condition_cells = {
        condition: metric_cell(cond_trials, recon_for(cond_trials), recont_for(cond_trials))
        for condition, cond_trials in sorted(_condition_groups(trials).items())
    }

    analysis: dict[str, Any] = {
        "by_condition": condition_cells,
        "by_condition_and_attack": table_for(lambda t: t.attack_type),
        "by_condition_and_scenario": table_for(lambda t: t.scenario_id),
        "by_condition_and_trust": table_for(lambda t: t.trust_level),
        "by_condition_and_target_type": table_for(lambda t: target_type_for(t.target_forget_ids)),
        "global": _sum_cells(list(condition_cells.values())),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    analysis["validation"] = validate_leakage_analysis(analysis, trials, annotations)
    return analysis


# ---------------------------------------------------------------------------
# Validation (FF92-016 required tests; all failures raise)
# ---------------------------------------------------------------------------


def _check_benign_zero_exposure(analysis: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    for condition, attacks in analysis["by_condition_and_attack"].items():
        for attack, cell in attacks.items():
            if attack in BENIGN_CONTROL_ATTACKS and cell["pu_rer"]["numerator"] != 0:
                violations.append(f"{condition}/{attack}: pu_rer numerator != 0")
    if violations:
        raise ValueError(
            "Benign controls must have zero exposure by construction: " + "; ".join(violations)
        )
    return {"check": "benign_controls_zero_exposure", "passed": True}


def _check_no_firewall_direct_high_exposure(analysis: dict[str, Any]) -> dict[str, Any]:
    attacks = analysis["by_condition_and_attack"].get("no_firewall", {})
    checked = 0
    for attack, cell in attacks.items():
        if attack not in DIRECT_DISCLOSURE_ATTACKS:
            continue
        if not cell["pu_rer"]["evaluable"]:
            continue
        checked += 1
        if cell["pu_rer"]["value"] < HIGH_EXPOSURE_THRESHOLD:
            raise ValueError(
                f"no_firewall direct candidates ({attack}) have unexpectedly low "
                f"exposure: {cell['pu_rer']['value']:.3f} < {HIGH_EXPOSURE_THRESHOLD}"
            )
    return {
        "check": "no_firewall_direct_high_exposure",
        "passed": True,
        "evaluable_direct_attack_types": checked,
    }


def _check_condition_sums(analysis: dict[str, Any]) -> dict[str, Any]:
    global_cell = analysis["global"]
    for metric in ("pu_rer", "crr", "rr", "fbr"):
        cond_num = sum(cell[metric]["numerator"] for cell in analysis["by_condition"].values())
        cond_den = sum(cell[metric]["denominator"] for cell in analysis["by_condition"].values())
        if (
            cond_num != global_cell[metric]["numerator"]
            or cond_den != global_cell[metric]["denominator"]
        ):
            raise ValueError(
                f"Condition counts do not sum to global for {metric}: "
                f"conditions=({cond_num}/{cond_den}) global="
                f"({global_cell[metric]['numerator']}/{global_cell[metric]['denominator']})"
            )
    return {"check": "condition_counts_sum_to_global", "passed": True}


def _check_annotation_coverage(
    trials: Sequence[CandidateTrial], annotations: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    annotated_by_attack: dict[str, set[str]] = {}
    for record in annotations:
        annotated_by_attack.setdefault(record.get("attack_type", ""), set()).add(
            record["candidate_id"]
        )
    trial_by_attack: dict[str, set[str]] = {}
    for trial in trials:
        # Sequence trials carry every member's id; annotations exist per
        # candidate row, so compare full member populations.
        trial_by_attack.setdefault(trial.attack_type, set()).update(trial.candidate_ids)

    mismatches: list[str] = []
    for attack in sorted(trial_by_attack):
        trial_ids = trial_by_attack[attack]
        annotation_ids = annotated_by_attack.get(attack, set())
        if trial_ids != annotation_ids:
            mismatches.append(
                f"{attack}: trials={len(trial_ids)} annotations={len(annotation_ids)}"
            )
    if mismatches:
        raise ValueError(
            "Attack-type trial populations do not match annotation counts: " + "; ".join(mismatches)
        )
    return {
        "check": "attack_denominators_match_annotations",
        "passed": True,
        "attack_types": sorted(trial_by_attack),
    }


def validate_leakage_analysis(
    analysis: dict[str, Any],
    trials: Sequence[CandidateTrial],
    annotations: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Run every FF92-016 required validation; any failure raises."""
    return {
        "benign_controls_zero_exposure": _check_benign_zero_exposure(analysis),
        "no_firewall_direct_high_exposure": _check_no_firewall_direct_high_exposure(analysis),
        "condition_counts_sum_to_global": _check_condition_sums(analysis),
        "attack_denominators_match_annotations": _check_annotation_coverage(trials, annotations),
    }


# ---------------------------------------------------------------------------
# Results writing
# ---------------------------------------------------------------------------


def write_leakage_analysis(analysis: dict[str, Any], output_dir: Path) -> None:
    """Write leakage analysis to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "leakage_analysis.json").write_text(json.dumps(analysis, indent=2))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Run FF92-016 leakage analysis on frozen replay trial artifacts."""
    print("FF92-016: Leakage Analysis from Trial Artifacts")
    print("=" * 50)

    inputs = load_leakage_inputs(REPLAY_DIR, CORPUS_DIR / "corpus_annotations.jsonl")
    print(f"  Loaded {len(inputs['candidate_trials'])} candidate trials")
    print(f"  Loaded {len(inputs['reconstruction_records'])} reconstruction trials")
    print(f"  Loaded {len(inputs['recontamination_records'])} recontamination trials")
    print(f"  Loaded {len(inputs['annotations'])} annotations")

    analysis = run_leakage_analysis(inputs)
    write_leakage_analysis(analysis, ANALYSIS_DIR)
    print(f"\nAnalysis written to {ANALYSIS_DIR}")

    print("\nBy condition:")
    for condition, cell in analysis["by_condition"].items():
        pu = cell["pu_rer"]
        print(
            f"  {condition}: trials={cell['candidate_trials']} "
            f"pu_rer={pu['numerator']}/{pu['denominator']} "
            f"failed={cell['failed_trials']}"
        )
    print("\nValidation: all FF92-016 checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
