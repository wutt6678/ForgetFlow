"""FF92-020: Deterministic reproducibility validation.

Formerly misnamed "closed-loop validation". Re-running a deterministic
frozen replay twice and comparing outputs is reproducibility validation,
not closed-loop validation. A genuine closed-loop experiment would
generate a message, pass it through the firewall, release the
transformed output, update each agent's visible state, generate the
next message from that state, and repeat for multiple rounds; that is
reserved for future work (FF92-020 acceptance: closed-loop claims are
reserved for adaptive multi-round generation).

Reproducibility criteria (FF92-020): two identical replay runs must
match on
- candidate-level outputs (same candidates executed, same outcomes),
- trial-level outputs (per-trial content hash of released text and
  firewall actions),
- metric counts (numerators/denominators exactly, not just aggregates),
- hashes (config/condition hashes and condition content hashes),
not only aggregate metric values.

Exit criterion:
  Every comparison layer passes; any mismatch fails the run.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.frozen_replay import (  # noqa: E402
    ConditionResult,
    build_config_for_condition,
    run_frozen_replay,
)
from experiments.trustparadox_u.runner import EpisodeResult  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

VALIDATION_DIR = Path(__file__).parents[2] / "results" / "deterministic_reproducibility_validation"


# ---------------------------------------------------------------------------
# Trial-level fingerprinting
# ---------------------------------------------------------------------------


def _trial_record(result: EpisodeResult) -> dict[str, Any]:
    """Deterministic per-trial record of everything the run produced."""
    return {
        "episode_id": result.episode_id,
        "candidate_sample_id": result.candidate_sample_id,
        "scenario_id": result.scenario_id,
        "trust_level": result.trust_level,
        "seed": result.seed,
        "task_success": result.task_success,
        "task_label": result.task_label,
        "cleaned_agents_exposed": result.cleaned_agents_exposed,
        "recontaminated_agents": result.recontaminated_agents,
        "turns": [
            {
                "turn_id": turn.turn_id,
                "phase": turn.phase,
                "sender_id": turn.sender_id,
                "recipient_id": turn.recipient_id,
                "released_text": turn.released_text,
                "firewall_action": (turn.decision.action if turn.decision is not None else None),
                "reason_codes": (
                    list(turn.decision.reason_codes) if turn.decision is not None else []
                ),
            }
            for turn in result.turns
        ],
    }


def trial_hash(result: EpisodeResult) -> str:
    """Content hash of one trial's outputs."""
    record = _trial_record(result)
    payload = json.dumps(record, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def condition_content_hash(result: ConditionResult) -> str:
    """Hash over the full candidate -> trial-hash mapping of a condition."""
    mapping = {r.candidate_sample_id: trial_hash(r) for r in result.episode_results}
    payload = json.dumps(mapping, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Layered comparison
# ---------------------------------------------------------------------------


def compare_candidate_level(
    run1: dict[str, ConditionResult], run2: dict[str, ConditionResult]
) -> list[dict[str, Any]]:
    """Same candidates executed per condition, with the same failures."""
    mismatches: list[dict[str, Any]] = []
    for condition in sorted(run1):
        first = {r.candidate_sample_id for r in run1[condition].episode_results}
        second = {r.candidate_sample_id for r in run2[condition].episode_results}
        if first != second:
            mismatches.append(
                {
                    "layer": "candidate_level",
                    "condition": condition,
                    "detail": "candidate sets differ",
                    "only_in_run1": sorted(first - second),
                    "only_in_run2": sorted(second - first),
                }
            )
        fail1 = {f["candidate_id"] for f in run1[condition].failed_candidates}
        fail2 = {f["candidate_id"] for f in run2[condition].failed_candidates}
        if fail1 != fail2:
            mismatches.append(
                {
                    "layer": "candidate_level",
                    "condition": condition,
                    "detail": "failure sets differ",
                    "only_in_run1": sorted(fail1 - fail2),
                    "only_in_run2": sorted(fail2 - fail1),
                }
            )
    return mismatches


def compare_trial_level(
    run1: dict[str, ConditionResult], run2: dict[str, ConditionResult]
) -> list[dict[str, Any]]:
    """Per-candidate trial content hashes must be identical."""
    mismatches: list[dict[str, Any]] = []
    for condition in sorted(run1):
        trials1 = {r.candidate_sample_id: trial_hash(r) for r in run1[condition].episode_results}
        trials2 = {r.candidate_sample_id: trial_hash(r) for r in run2[condition].episode_results}
        for candidate_id in sorted(set(trials1) | set(trials2)):
            if trials1.get(candidate_id) != trials2.get(candidate_id):
                mismatches.append(
                    {
                        "layer": "trial_level",
                        "condition": condition,
                        "candidate_id": candidate_id,
                        "detail": "trial content hash differs",
                    }
                )
    return mismatches


def compare_metric_counts(
    run1: dict[str, ConditionResult], run2: dict[str, ConditionResult]
) -> list[dict[str, Any]]:
    """Exact numerator/denominator/evaluable equality for every metric.

    Aggregate values alone can hide pipeline changes; the counts are the
    pipeline's actual evidence, so they must match exactly.
    """
    mismatches: list[dict[str, Any]] = []
    for condition in sorted(run1):
        metrics1 = run1[condition].metrics
        metrics2 = run2[condition].metrics
        for metric_name in sorted(set(metrics1) | set(metrics2)):
            m1 = metrics1.get(metric_name)
            m2 = metrics2.get(metric_name)
            if m1 is None or m2 is None:
                mismatches.append(
                    {
                        "layer": "metric_counts",
                        "condition": condition,
                        "metric": metric_name,
                        "detail": "metric missing in one run",
                    }
                )
                continue
            for field in ("numerator", "denominator", "evaluable", "reason", "value"):
                if m1.get(field) != m2.get(field):
                    mismatches.append(
                        {
                            "layer": "metric_counts",
                            "condition": condition,
                            "metric": metric_name,
                            "field": field,
                            "run1": m1.get(field),
                            "run2": m2.get(field),
                            "detail": "metric field differs",
                        }
                    )
    return mismatches


def compare_hashes(
    run1: dict[str, ConditionResult],
    run2: dict[str, ConditionResult],
    seed: int,
) -> list[dict[str, Any]]:
    """Config hashes, condition hashes, and condition content hashes."""
    mismatches: list[dict[str, Any]] = []
    for condition in sorted(run1):
        config1 = build_config_for_condition(condition, seed=seed)
        config2 = build_config_for_condition(condition, seed=seed)
        if config1.config_hash() != config2.config_hash():
            mismatches.append(
                {
                    "layer": "hashes",
                    "condition": condition,
                    "detail": "config hash not deterministic",
                }
            )
        if config1.condition_hash() != config2.condition_hash():
            mismatches.append(
                {
                    "layer": "hashes",
                    "condition": condition,
                    "detail": "condition hash not deterministic",
                }
            )
        content1 = condition_content_hash(run1[condition])
        content2 = condition_content_hash(run2[condition])
        if content1 != content2:
            mismatches.append(
                {
                    "layer": "hashes",
                    "condition": condition,
                    "detail": "condition content hash differs",
                    "run1": content1,
                    "run2": content2,
                }
            )
    return mismatches


# ---------------------------------------------------------------------------
# Validation run
# ---------------------------------------------------------------------------


def run_deterministic_reproducibility_validation(
    max_candidates: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the same frozen replay subset twice and compare every layer.

    FF92-020: this is a deterministic reproducibility check, not a
    closed-loop experiment; the comparison covers candidate outputs,
    trial outputs, metric counts, and hashes.
    """
    print(f"Run 1: {max_candidates} candidates...")
    run1 = run_frozen_replay(
        max_candidates_per_condition=max_candidates,
        seed=seed,
        run_id="repro_run1",
    )
    print(f"Run 2: {max_candidates} candidates...")
    run2 = run_frozen_replay(
        max_candidates_per_condition=max_candidates,
        seed=seed,
        run_id="repro_run2",
    )

    layers = {
        "candidate_level": compare_candidate_level(run1, run2),
        "trial_level": compare_trial_level(run1, run2),
        "metric_counts": compare_metric_counts(run1, run2),
        "hashes": compare_hashes(run1, run2, seed),
    }
    checks = {
        name: {"passed": len(mismatches) == 0, "num_mismatches": len(mismatches)}
        for name, mismatches in layers.items()
    }
    all_mismatches = [m for mismatches in layers.values() for m in mismatches]
    passed = all(check["passed"] for check in checks.values())

    return {
        "validation": "deterministic_reproducibility",
        "repair_item": "FF92-020",
        "scope_note": (
            "Deterministic rerun comparison. Closed-loop validation "
            "(adaptive multi-round generation) is future work."
        ),
        "passed": passed,
        "max_candidates": max_candidates,
        "seed": seed,
        "num_conditions": len(run1),
        "checks": checks,
        "num_mismatches": len(all_mismatches),
        "mismatches": all_mismatches[:20],
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Run deterministic reproducibility validation."""
    print("FF92-020: Deterministic Reproducibility Validation")
    print("=" * 50)

    result = run_deterministic_reproducibility_validation(max_candidates=50)

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    (VALIDATION_DIR / "validation_result.json").write_text(json.dumps(result, indent=2))

    for name, check in result["checks"].items():
        status = "PASSED" if check["passed"] else "FAILED"
        print(f"  {name}: {status} ({check['num_mismatches']} mismatches)")

    if result["passed"]:
        print("\nExit criterion: PASSED (all comparison layers match)")
        return 0
    for m in result["mismatches"][:5]:
        print(f"  mismatch: {m.get('layer')}/{m.get('condition')}: {m.get('detail')}")
    print("\nExit criterion: FAILED (runs are not reproducible)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
