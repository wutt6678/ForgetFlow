"""Remediation §37: curated failure examples with decision traces.

Aggregate rates explain *how much* leakage or false blocking occurs, but
not *why*. This module extracts a curated, privacy-safe sample of failure
examples from the committed frozen-replay trial artifacts and the message
audit, one sample per declared failure category:

- ``false_negative_leakage``: an attack candidate was released and exposed
  a forget target (pu_rer numerator contribution);
- ``false_positive_block``: a legitimate message was wrongly blocked
  (fbr numerator contribution) — zero observed in this study, reported
  explicitly;
- ``partial_reconstruction``: a multi-step attack recovered some but not
  all fragments of the forgotten target;
- ``incorrect_abstraction``: the detector fired and the policy acted, but
  the released text still exposed the target;
- ``monitoring_failure``: a recontamination attempt succeeded under a
  condition whose monitoring was enabled (rr numerator contribution);
- ``trust_conditioned_difference``: the same candidate produced divergent
  exposure across trust levels — zero observed, reported explicitly.

Acceptance (§37):
- selection follows the declared, deterministic procedure below;
- the corpus is fully synthetic, so examples cannot expose real secrets
  or sensitive personal data (stated in every artifact);
- each example carries an ``error_attribution`` distinguishing detector,
  history, policy, monitoring, and annotation errors.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

RESULTS_DIR = _PROJECT_ROOT / "results" / "frozen_replay"
OUTPUT_DIR = _PROJECT_ROOT / "results" / "failure_examples"
OUTPUT_PATH = OUTPUT_DIR / "failure_examples.json"

SCHEMA_VERSION = "1.0"
MAX_EXAMPLES_PER_CATEGORY = 3

CATEGORIES: tuple[str, ...] = (
    "false_negative_leakage",
    "false_positive_block",
    "partial_reconstruction",
    "incorrect_abstraction",
    "monitoring_failure",
    "trust_conditioned_difference",
)

# §37 acceptance: the failure analysis must distinguish these sources.
ATTRIBUTION_SOURCES: tuple[str, ...] = (
    "detector",
    "history",
    "policy",
    "monitoring",
    "annotation",
    "none_observed",
)

SELECTION_PROCEDURE = (
    "Deterministic, declared selection: within each category the matching "
    "trials are filtered by the exact predicate declared for that category, "
    "sorted lexicographically by (candidate_id or trial_key, episode_id, "
    f"condition_id), and the first {MAX_EXAMPLES_PER_CATEGORY} are taken. "
    "No manual curation, no post-hoc reordering. Categories with zero "
    "matching trials are reported with count 0 rather than omitted."
)

PRIVACY_STATEMENT = (
    "All example texts come from the fully synthetic TrustParadox-U frozen "
    "corpus. No real secrets, credentials, or sensitive personal data appear "
    "in any example; identifiers and values are fabricated for the study."
)

# Conditions whose design includes recontamination monitoring; a
# recontamination there is a monitoring failure, not just an absence of
# protection.
MONITORED_CONDITIONS = frozenset({"full_mvp", "exact_only"})


# ---------------------------------------------------------------------------
# Loading and joining
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_inputs() -> dict[str, Any]:
    """Load trial artifacts and index the message audit for joins."""
    candidate_trials = _load_jsonl(RESULTS_DIR / "candidate_trials.jsonl")
    reconstruction_trials = _load_jsonl(RESULTS_DIR / "reconstruction_trials.jsonl")
    recontamination_trials = _load_jsonl(RESULTS_DIR / "recontamination_trials.jsonl")

    audit_index: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in _load_jsonl(RESULTS_DIR / "message_audit.jsonl"):
        key = (
            str(record.get("episode_id", "")),
            str(record.get("candidate_id", "")),
            str(record.get("condition_id", "")),
        )
        audit_index.setdefault(key, []).append(record)

    return {
        "candidate_trials": candidate_trials,
        "reconstruction_trials": reconstruction_trials,
        "recontamination_trials": recontamination_trials,
        "audit_index": audit_index,
    }


def audit_trace(
    audit_index: dict[tuple[str, str, str], list[dict[str, Any]]],
    episode_id: str,
    candidate_id: str,
    condition_id: str,
) -> dict[str, Any]:
    """Decision trace for one trial from the message audit.

    Returns the candidate/released text and the policy decision of the
    first POST_FORGET_ATTACK message, plus every decision seen in the
    episode for that candidate and condition.
    """
    rows = audit_index.get((episode_id, candidate_id, condition_id), [])
    attack_rows = [r for r in rows if r.get("phase") == "POST_FORGET_ATTACK"]
    first = attack_rows[0] if attack_rows else (rows[0] if rows else {})
    return {
        "candidate_text": first.get("candidate_text", ""),
        "released_text": first.get("released_text", ""),
        "decision_action": first.get("decision_action"),
        "detector_evidence": list(first.get("decision_reason_codes") or []),
        "policy_version": first.get("policy_version"),
        "decision_actions_seen": sorted(
            {str(r.get("decision_action")) for r in rows if r.get("decision_action")}
        ),
        "audit_message_count": len(rows),
    }


def _target_type_for(forget_ids: list[str]) -> str:
    from experiments.trustparadox_u.leakage_analysis import target_type_for

    return target_type_for(forget_ids)


# ---------------------------------------------------------------------------
# Category predicates and example builders
# ---------------------------------------------------------------------------


def _sort_key(trial: dict[str, Any], id_field: str) -> tuple[str, str, str]:
    return (
        str(trial.get(id_field, "")),
        str(trial.get("episode_id", "")),
        str(trial.get("condition_id", "")),
    )


def _select(trials: list[dict[str, Any]], id_field: str) -> list[dict[str, Any]]:
    return sorted(trials, key=lambda t: _sort_key(t, id_field))[:MAX_EXAMPLES_PER_CATEGORY]


def _false_negative_leakage(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    """Attacks released with target exposure: a miss somewhere downstream."""
    examples = []
    for trial in _select(
        [t for t in inputs["candidate_trials"] if t.get("released_exposure_positive")],
        "candidate_id",
    ):
        trace = audit_trace(
            inputs["audit_index"],
            trial["episode_id"],
            trial["candidate_id"],
            trial["condition_id"],
        )
        condition = trial["condition_id"]
        if condition == "no_firewall":
            attribution = "policy"
            attribution_note = "baseline condition declares no protection; expected leakage"
        elif not trace["detector_evidence"]:
            attribution = "detector"
            attribution_note = "detector produced no evidence, so the message passed"
        else:
            attribution = "policy"
            attribution_note = (
                "detector evidence existed but the policy action did not remove exposure"
            )
        examples.append(
            {
                "category": "false_negative_leakage",
                "candidate_id": trial["candidate_id"],
                "episode_id": trial["episode_id"],
                "condition": condition,
                "target_type": _target_type_for(trial["target_forget_ids"]),
                "target_forget_ids": trial["target_forget_ids"],
                "trust_level": trial.get("trust_level"),
                "attack_type": trial.get("attack_type"),
                "candidate_text": trace["candidate_text"],
                "released_text": trace["released_text"],
                "detector_evidence": trace["detector_evidence"],
                "policy_action": trace["decision_action"],
                "oracle_label": {
                    "disclosure_class": trial.get("disclosure_class"),
                    "released_exposure_labels": trial.get("released_exposure_labels"),
                },
                "metric_contribution": "pu_rer numerator (released exposure under protection)",
                "error_attribution": attribution,
                "attribution_note": attribution_note,
            }
        )
    return examples


def _false_positive_block(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    """Legitimate messages wrongly blocked (fbr numerator). Zero observed."""
    trials = [t for t in inputs["candidate_trials"] if t.get("blocked_legitimate")]
    examples = []
    for trial in _select(trials, "candidate_id"):
        trace = audit_trace(
            inputs["audit_index"],
            trial["episode_id"],
            trial["candidate_id"],
            trial["condition_id"],
        )
        examples.append(
            {
                "category": "false_positive_block",
                "candidate_id": trial["candidate_id"],
                "episode_id": trial["episode_id"],
                "condition": trial["condition_id"],
                "target_type": _target_type_for(trial["target_forget_ids"]),
                "target_forget_ids": trial["target_forget_ids"],
                "candidate_text": trace["candidate_text"],
                "released_text": trace["released_text"],
                "detector_evidence": trace["detector_evidence"],
                "policy_action": trace["decision_action"],
                "oracle_label": {
                    "blocked_legitimate": trial.get("blocked_legitimate"),
                    "task_label": trial.get("task_label"),
                },
                "metric_contribution": "fbr numerator (blocked legitimate message)",
                "error_attribution": "detector",
                "attribution_note": "a legitimate message was blocked",
            }
        )
    return examples


def _partial_reconstruction(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    """Multi-step attacks that recovered some but not all fragments."""
    trials = [
        t
        for t in inputs["reconstruction_trials"]
        if t.get("eligible")
        and t.get("complete")
        and not t.get("recovered")
        and int(t.get("fragment_count", 0)) > 0
    ]
    examples = []
    for trial in _select(trials, "trial_key"):
        trace = audit_trace(
            inputs["audit_index"],
            trial["episode_id"],
            "",
            trial["condition"],
        )
        examples.append(
            {
                "category": "partial_reconstruction",
                "trial_key": trial["trial_key"],
                "sequence_id": trial.get("sequence_id"),
                "episode_id": trial["episode_id"],
                "condition": trial["condition"],
                "target_type": _target_type_for([trial["forget_id"]]),
                "target_forget_ids": [trial["forget_id"]],
                "candidate_text": trace["candidate_text"],
                "released_text": trace["released_text"],
                "detector_evidence": trace["detector_evidence"],
                "policy_action": trace["decision_action"],
                "oracle_label": {
                    "recovered": trial.get("recovered"),
                    "fragment_count": trial.get("fragment_count"),
                    "executed_step_count": trial.get("executed_step_count"),
                    "expected_step_count": trial.get("expected_step_count"),
                },
                "metric_contribution": "crr denominator without numerator (incomplete reconstruction)",
                "error_attribution": "history",
                "attribution_note": (
                    "forgetting removed the direct record but residual fragments "
                    "remained reachable; the failure is in what the history "
                    "retained, not in a single release decision"
                ),
            }
        )
    return examples


def _incorrect_abstraction_trials(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    """Exposed trials where a detector/policy intervention actually ran."""
    trials = []
    for trial in [t for t in inputs["candidate_trials"] if t.get("released_exposure_positive")]:
        trace = audit_trace(
            inputs["audit_index"],
            trial["episode_id"],
            trial["candidate_id"],
            trial["condition_id"],
        )
        if (
            trace["decision_action"] in {"abstract", "redact", "block"}
            or trace["detector_evidence"]
        ):
            trial = dict(trial)
            trial["_trace"] = trace
            trials.append(trial)
    return trials


def _incorrect_abstraction(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    """The detector fired and the policy acts, yet exposure survived."""
    examples = []
    for trial in _select(_incorrect_abstraction_trials(inputs), "candidate_id"):
        trace = trial.pop("_trace")
        examples.append(
            {
                "category": "incorrect_abstraction",
                "candidate_id": trial["candidate_id"],
                "episode_id": trial["episode_id"],
                "condition": trial["condition_id"],
                "target_type": _target_type_for(trial["target_forget_ids"]),
                "target_forget_ids": trial["target_forget_ids"],
                "attack_type": trial.get("attack_type"),
                "candidate_text": trace["candidate_text"],
                "released_text": trace["released_text"],
                "detector_evidence": trace["detector_evidence"],
                "policy_action": trace["decision_action"],
                "oracle_label": {
                    "disclosure_class": trial.get("disclosure_class"),
                    "released_exposure_labels": trial.get("released_exposure_labels"),
                },
                "metric_contribution": "pu_rer numerator despite a detector/policy intervention",
                "error_attribution": "detector",
                "attribution_note": (
                    "an intervention ran but the rewrite/redaction did not remove "
                    "the entailed exposure; the detector matched a surface form "
                    "without covering the paraphrase or abstraction path"
                ),
            }
        )
    return examples


def _monitoring_failure(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    """Recontamination succeeded under a condition with monitoring enabled."""
    trials = [
        t
        for t in inputs["recontamination_trials"]
        if t.get("evaluable")
        and t.get("final_status") == "recontaminated"
        and t.get("condition_id") in MONITORED_CONDITIONS
    ]
    examples = []
    for trial in _select(trials, "trial_key"):
        trace = audit_trace(inputs["audit_index"], trial["episode_id"], "", trial["condition_id"])
        examples.append(
            {
                "category": "monitoring_failure",
                "trial_key": trial["trial_key"],
                "episode_id": trial["episode_id"],
                "condition": trial["condition_id"],
                "agent_id": trial.get("agent_id"),
                "target_type": _target_type_for([trial["forget_id"]]),
                "target_forget_ids": [trial["forget_id"]],
                "candidate_text": trial.get("probe_text", "") or trace["candidate_text"],
                "released_text": trace["released_text"],
                "detector_evidence": trace["detector_evidence"],
                "policy_action": trace["decision_action"],
                "oracle_label": {
                    "pre_attempt_status": trial.get("pre_attempt_status"),
                    "final_status": trial.get("final_status"),
                    "target_reached_recipient": trial.get("target_reached_recipient"),
                    "target_recoverable_after_monitor": trial.get(
                        "target_recoverable_after_monitor"
                    ),
                },
                "metric_contribution": "rr numerator under a monitored condition",
                "error_attribution": "monitoring",
                "attribution_note": (
                    "monitoring was enabled for this condition yet the "
                    "recontamination probe re-established the forgotten target"
                ),
            }
        )
    return examples


def _trust_conditioned_difference(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    """Same candidate, divergent exposure across trust levels. Zero observed."""
    by_key: dict[tuple[str, str], dict[str, bool]] = {}
    for trial in inputs["candidate_trials"]:
        key = (trial["candidate_id"], trial["condition_id"])
        by_key.setdefault(key, {})[str(trial.get("trust_level"))] = bool(
            trial.get("released_exposure_positive")
        )
    divergent: list[dict[str, Any]] = [
        {
            "category": "trust_conditioned_difference",
            "candidate_id": cid,
            "condition": cond,
            "target_type": "all pooled",
            "candidate_text": "",
            "released_text": "",
            "detector_evidence": [],
            "policy_action": None,
            "oracle_label": {"exposure_by_trust_level": levels},
            "metric_contribution": "trust-level interaction term on pu_rer",
            "error_attribution": "policy",
            "attribution_note": (
                "the same candidate produced divergent exposure across trust "
                "levels under one condition"
            ),
        }
        for (cid, cond), levels in sorted(by_key.items())
        if len(levels) > 1 and len(set(levels.values())) > 1
    ]
    return divergent[:MAX_EXAMPLES_PER_CATEGORY]


def _category_observed_counts(inputs: dict[str, Any]) -> dict[str, int]:
    """Total matching trials per category (before the example cap)."""
    cand = inputs["candidate_trials"]
    rec = inputs["reconstruction_trials"]
    rc = inputs["recontamination_trials"]
    by_key: dict[tuple[str, str], dict[str, bool]] = {}
    for trial in cand:
        by_key.setdefault((trial["candidate_id"], trial["condition_id"]), {})[
            str(trial.get("trust_level"))
        ] = bool(trial.get("released_exposure_positive"))
    divergent = sum(
        1 for levels in by_key.values() if len(levels) > 1 and len(set(levels.values())) > 1
    )
    return {
        "false_negative_leakage": sum(1 for t in cand if t.get("released_exposure_positive")),
        "false_positive_block": sum(1 for t in cand if t.get("blocked_legitimate")),
        "partial_reconstruction": sum(
            1
            for t in rec
            if t.get("eligible")
            and t.get("complete")
            and not t.get("recovered")
            and int(t.get("fragment_count", 0)) > 0
        ),
        "incorrect_abstraction": len(_incorrect_abstraction_trials(inputs)),
        "monitoring_failure": sum(
            1
            for t in rc
            if t.get("evaluable")
            and t.get("final_status") == "recontaminated"
            and t.get("condition_id") in MONITORED_CONDITIONS
        ),
        "trust_conditioned_difference": divergent,
    }


# ---------------------------------------------------------------------------
# Artifact builder and CLI
# ---------------------------------------------------------------------------


def build_failure_examples() -> dict[str, Any]:
    """Build the §37 failure-example artifact from committed trial artifacts."""
    from experiments.trustparadox_u.research_protocol import PROTOCOL_VERSION
    from experiments.trustparadox_u.status import STUDY_CLASS_DIAGNOSTIC

    inputs = load_inputs()
    builders = {
        "false_negative_leakage": _false_negative_leakage,
        "false_positive_block": _false_positive_block,
        "partial_reconstruction": _partial_reconstruction,
        "incorrect_abstraction": _incorrect_abstraction,
        "monitoring_failure": _monitoring_failure,
        "trust_conditioned_difference": _trust_conditioned_difference,
    }
    categories: dict[str, dict[str, Any]] = {}
    counts = _category_observed_counts(inputs)
    for category in CATEGORIES:
        examples = builders[category](inputs)
        observed = counts[category]
        categories[category] = {
            "observed_count": observed,
            "example_count": len(examples),
            "examples": examples,
            "note": (
                "no matching trials observed in this study; category reported "
                "with count 0 per the declared selection procedure"
                if observed == 0
                else ""
            ),
        }

    study_class = STUDY_CLASS_DIAGNOSTIC
    run_manifest_path = RESULTS_DIR / "run_manifest.json"
    if run_manifest_path.exists():
        study_class = str(json.loads(run_manifest_path.read_text()).get("study_class", study_class))

    return {
        "schema_version": SCHEMA_VERSION,
        "remediation_items": ["37"],
        "study_class": study_class,
        "protocol_version": PROTOCOL_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection_procedure": SELECTION_PROCEDURE,
        "max_examples_per_category": MAX_EXAMPLES_PER_CATEGORY,
        "privacy_statement": PRIVACY_STATEMENT,
        "attribution_sources": list(ATTRIBUTION_SOURCES),
        "categories": categories,
    }


def main() -> int:
    print("Remediation §37: Curated Failure Examples and Decision Traces")
    print("=" * 60)
    artifact = build_failure_examples()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    total = 0
    for category in CATEGORIES:
        info = artifact["categories"][category]
        total += info["example_count"]
        print(
            f"  {category}: observed={info['observed_count']} " f"examples={info['example_count']}"
        )
    print(f"\nWrote {total} failure examples to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
