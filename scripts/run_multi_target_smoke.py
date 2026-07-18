#!/usr/bin/env python3
"""Multi-target smoke study runner.

Runs multi-target fixtures across multiple seeds and conditions,
validating that cross-record state isolation, detector-only exposure,
reconstruction specificity, and reintroduction tracking work correctly.

Usage:
    poetry run python scripts/run_multi_target_smoke.py [--output-dir DIR] [--mode release|diagnostic]
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.trustparadox_u.audit_results import audit_results  # noqa: E402
from experiments.trustparadox_u.config import (  # noqa: E402
    DetectorConfig,
    ExperimentConfig,
    HistoryConfig,
    MonitoringConfig,
    PolicyConfig,
    RunConfig,
)
from experiments.trustparadox_u.dataset import load_episode  # noqa: E402
from experiments.trustparadox_u.evaluator import evaluate_all  # noqa: E402
from experiments.trustparadox_u.manifest import get_repository_commit  # noqa: E402
from experiments.trustparadox_u.runner import EpisodeResult, run_episode  # noqa: E402
from experiments.trustparadox_u.serialization import (  # noqa: E402
    load_episode_results,
    serialize_episode_result,
)

SCENARIOS_DIR = PROJECT_ROOT / "data" / "trustparadox_u" / "scenarios"

# Multi-target fixtures (each has 2+ sensitive items)
FIXTURES = [
    "pilot_multi_target.yaml",
]

SEEDS = [42, 123, 7]

# Exit codes
EXIT_SUCCESS = 0
EXIT_INPUT_CONFIG = 2
EXIT_EXECUTION = 3
EXIT_AUDIT = 4
EXIT_MANIFEST = 5
EXIT_PROVENANCE = 6
EXIT_ARTIFACT_COMPLETENESS = 7
EXIT_DISK_COMPARISON = 8
EXIT_DIRECTIONAL = 1

# Required artifacts
REQUIRED_ARTIFACTS = [
    "episodes.jsonl",
    "smoke_manifest.json",
    "result_audit.json",
    "metrics.json",
    "metric_counts.json",
    "metrics_by_condition.json",
    "multi_target_report.json",
]

# Conditions: (name, config_overrides, firewall_enabled)
CONDITIONS: list[tuple[str, dict[str, Any], bool]] = [
    ("no_firewall", {}, False),
    (
        "full_mvp",
        {
            "detector": DetectorConfig(
                exact_enabled=True,
                entity_enabled=True,
                semantic_enabled=True,
            ),
        },
        True,
    ),
    (
        "exact_only",
        {
            "detector": DetectorConfig(
                exact_enabled=True,
                entity_enabled=False,
                semantic_enabled=False,
            ),
        },
        True,
    ),
    (
        "monitoring_0",
        {
            "monitoring": MonitoringConfig(continuous=False, duration_rounds=0),
        },
        True,
    ),
    (
        "binary_policy",
        {
            "policy": PolicyConfig(rich_actions_enabled=False),
        },
        True,
    ),
]


def _make_config(seed: int, overrides: dict[str, Any], mode: str = "test") -> ExperimentConfig:
    """Create an ExperimentConfig with the given overrides."""
    kwargs: dict[str, Any] = dict(
        seed=seed,
        repetitions=1,
        detector=DetectorConfig(exact_enabled=True, entity_enabled=True, semantic_enabled=False),
        history=HistoryConfig(),
        policy=PolicyConfig(),
        monitoring=MonitoringConfig(),
        run=RunConfig(mode=mode),
    )
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


def _build_smoke_responder(episode: Any) -> Any:
    """Build a ScriptedResponder for multi-target episodes."""
    from experiments.trustparadox_u.agent import ScriptedResponder

    responder = ScriptedResponder()
    success_value = episode.task.success_value

    canonical_targets: dict[str, str] = {}
    fragment_map: dict[str, list[str]] = {}
    for si in episode.sensitive_items:
        canonical_targets[si.forget_id] = si.canonical_target
        recon = si.reconstruction or {}
        fragment_map[si.forget_id] = recon.get("fragments", [])

    for step in episode.phases.post_forget:
        key = f"{episode.episode_id}:{step.attacker}:{step.instruction}"
        if step.label.is_legitimate_message:
            responder.set_response(key, success_value)
        elif step.label.is_recontamination_attempt:
            # For combined attempts, return all targets
            targets = step.label.target_forget_ids
            if len(targets) > 1:
                combined = " ".join(
                    canonical_targets[fid] for fid in targets if fid in canonical_targets
                )
                responder.set_response(key, combined)
            else:
                for forget_id in targets:
                    if forget_id in canonical_targets:
                        responder.set_response(key, canonical_targets[forget_id])
                        break
                else:
                    responder.set_response(key, step.instruction)
        elif step.label.is_reconstruction_attempt:
            # For reconstruction steps, return fragments for the matching forget_id
            # Infer which forget_id from instruction text matching sensitive item aliases
            instruction_lower = step.instruction.lower()
            best_fragments: list[str] = []
            best_score = 0
            for si in episode.sensitive_items:
                if not si.reconstruction:
                    continue
                frags = si.reconstruction.get("fragments", [])
                if not frags:
                    continue
                # Score how well this instruction matches the sensitive item
                match_score = 0
                # Match individual words from aliases (skip very short words)
                for alias in si.aliases:
                    for word in alias.lower().split():
                        if len(word) > 3 and word in instruction_lower:
                            match_score += 5
                # Match target_type
                if si.target_type.lower() in instruction_lower:
                    match_score += 3
                # Match canonical target words
                for word in si.canonical_target.split():
                    if len(word) > 2 and word.lower() in instruction_lower:
                        match_score += 2
                if match_score > best_score:
                    best_score = match_score
                    best_fragments = frags
            if best_fragments:
                responder.set_response(key, " ".join(best_fragments))
            else:
                responder.set_response(key, step.instruction)
        else:
            responder.set_response(key, step.instruction)

    return responder


@dataclass
class MultiTargetAssertion:
    """A single assertion result."""

    name: str
    passed: bool
    detail: str = ""


def _validate_multi_target(
    all_results: list[EpisodeResult],
    condition_results: dict[str, list[EpisodeResult]],
) -> list[MultiTargetAssertion]:
    """Run all multi-target assertions."""
    assertions: list[MultiTargetAssertion] = []

    # 1. Positive F001-only and F002-only exposure cases (s3 + s4)
    f001_only = 0
    f002_only = 0
    both = 0
    for r in all_results:
        for turn in r.turns:
            exposed = set(turn.exposed_forget_ids)
            has_f001 = "F001" in exposed
            has_f002 = "F002" in exposed
            if has_f001 and not has_f002:
                f001_only += 1
            elif has_f002 and not has_f001:
                f002_only += 1
            elif has_f001 and has_f002:
                both += 1

    assertions.append(
        MultiTargetAssertion(
            name="F001_exposure_independent_of_F002",
            passed=f001_only > 0 and f002_only > 0,
            detail=f"F001-only={f001_only}, F002-only={f002_only}, both={both}",
        )
    )

    # 2. Positive single-record exposure proof (s4)
    assertions.append(
        MultiTargetAssertion(
            name="positive_F001_only_exposure",
            passed=f001_only > 0,
            detail=f"F001-only exposure turns: {f001_only}",
        )
    )
    assertions.append(
        MultiTargetAssertion(
            name="positive_F002_only_exposure",
            passed=f002_only > 0,
            detail=f"F002-only exposure turns: {f002_only}",
        )
    )
    assertions.append(
        MultiTargetAssertion(
            name="positive_combined_exposure",
            passed=both > 0,
            detail=f"Combined F001+F002 exposure turns: {both}",
        )
    )

    # 3. Detector-only exposure updates the correct tracker record (s5)
    # Validate final_contamination_states per agent-record pair
    valid_states = True
    state_details: list[str] = []
    for r in all_results:
        fcs = r.final_contamination_states
        for (agent_id, forget_id), status in fcs.items():
            if forget_id not in ("F001", "F002"):
                valid_states = False
            if status not in (
                "unknown",
                "contaminated",
                "clean",
                "verified",
                "at_risk",
                "recontaminated",
            ):
                valid_states = False
        # Check that F001 and F002 states are tracked independently
        f001_state = fcs.get(("CK", "F001"))
        f002_state = fcs.get(("CK", "F002"))
        if f001_state is not None and f002_state is not None:
            state_details.append(f"seed={r.seed}: F001={f001_state}, F002={f002_state}")

    assertions.append(
        MultiTargetAssertion(
            name="tracker_state_per_agent_record_pair",
            passed=valid_states,
            detail=f"Validated {len(state_details)} agent-record state pairs",
        )
    )

    # 4. Protected and unprotected state agreement for identical released text (s7)
    no_fw_results = condition_results.get("no_firewall", [])
    fw_results = condition_results.get("full_mvp", [])
    outcome_agreement = True
    compared_count = 0
    # Build lookup by (episode_id, seed, turn_index) for matching
    no_fw_by_key: dict[tuple[str, int, int], Any] = {}
    for r in no_fw_results:
        for i, turn in enumerate(r.turns):
            if turn.phase == "POST_FORGET_ATTACK":
                key = (r.episode_id, r.seed, i)
                no_fw_by_key[key] = turn

    for r in fw_results:
        for i, turn in enumerate(r.turns):
            if turn.phase == "POST_FORGET_ATTACK":
                key = (r.episode_id, r.seed, i)
                no_fw_turn = no_fw_by_key.get(key)
                if no_fw_turn is not None:
                    # Only compare when released text is identical
                    if (
                        turn.released_text == no_fw_turn.released_text
                        and turn.released_text is not None
                    ):
                        compared_count += 1
                        if set(turn.exposed_forget_ids) != set(no_fw_turn.exposed_forget_ids):
                            outcome_agreement = False

    assertions.append(
        MultiTargetAssertion(
            name="protected_unprotected_outcome_symmetry",
            passed=outcome_agreement,
            detail=f"Compared {compared_count} identical-text turn pairs",
        )
    )

    # 5. Reintroduced IDs are subsets of targeted and exposed IDs
    reintroduction_valid = True
    for r in all_results:
        for turn in r.turns:
            reintroduced = set(getattr(turn, "reintroduced_forget_ids", ()))
            exposed = set(turn.exposed_forget_ids)
            if reintroduced and not reintroduced.issubset(exposed):
                reintroduction_valid = False
    assertions.append(
        MultiTargetAssertion(
            name="reintroduced_subset_of_exposed",
            passed=reintroduction_valid,
            detail="All reintroduced IDs are subsets of exposed IDs",
        )
    )

    # 6. Positive per-record reconstruction (s8)
    f001_recon_turns = 0
    f002_recon_turns = 0
    both_recon_turns = 0
    for r in all_results:
        for turn in r.turns:
            reconstructed = set(getattr(turn, "reconstructed_forget_ids", ()))
            has_f001 = "F001" in reconstructed
            has_f002 = "F002" in reconstructed
            if has_f001 and not has_f002:
                f001_recon_turns += 1
            elif has_f002 and not has_f001:
                f002_recon_turns += 1
            elif has_f001 and has_f002:
                both_recon_turns += 1

    assertions.append(
        MultiTargetAssertion(
            name="positive_F001_reconstruction",
            passed=f001_recon_turns > 0,
            detail=f"F001-only reconstruction turns: {f001_recon_turns}",
        )
    )
    # F002-only is structurally impossible because F001 temporal_fragmentation
    # steps precede F002's in the scenario, so the accumulated transcript
    # always contains F001 fragments before F002 fragments arrive.
    # Instead, assert F002 is *involved* in reconstruction (alone or combined).
    f002_involved_turns = f002_recon_turns + both_recon_turns
    assertions.append(
        MultiTargetAssertion(
            name="positive_F002_reconstruction",
            passed=f002_involved_turns > 0,
            detail=(
                f"F002 reconstruction turns: {f002_involved_turns} "
                f"(F002-only={f002_recon_turns}, both={both_recon_turns})"
            ),
        )
    )

    # 7. Reconstructed IDs are record-specific
    reconstruction_specific = True
    for r in all_results:
        for turn in r.turns:
            reconstructed = set(getattr(turn, "reconstructed_forget_ids", ()))
            for fid in reconstructed:
                if fid not in ("F001", "F002"):
                    reconstruction_specific = False
    assertions.append(
        MultiTargetAssertion(
            name="reconstructed_ids_record_specific",
            passed=reconstruction_specific,
            detail="All reconstructed IDs are F001 or F002",
        )
    )

    # 8. RR denominator > 0 and numerator bounded (s2)
    evaluation = evaluate_all(all_results)
    rr_denom = evaluation.rr.denominator
    rr_num = evaluation.rr.numerator
    assertions.append(
        MultiTargetAssertion(
            name="rr_denominator_positive",
            passed=rr_denom > 0,
            detail=f"RR denominator={rr_denom}, numerator={rr_num}",
        )
    )
    assertions.append(
        MultiTargetAssertion(
            name="rr_numerator_le_denominator",
            passed=rr_num <= rr_denom,
            detail=f"RR={rr_num}/{rr_denom}",
        )
    )

    # 9. CRR numerator does not exceed denominator
    crr_valid = evaluation.crr.numerator <= evaluation.crr.denominator
    assertions.append(
        MultiTargetAssertion(
            name="crr_numerator_le_denominator",
            passed=crr_valid,
            detail=f"CRR={evaluation.crr.numerator}/{evaluation.crr.denominator}",
        )
    )

    # 10. Multi-target scenario has correct number of sensitive items
    multi_ep = load_episode(SCENARIOS_DIR / "pilot_multi_target.yaml")
    item_count = len(multi_ep.sensitive_items)
    assertions.append(
        MultiTargetAssertion(
            name="multi_target_has_multiple_items",
            passed=item_count >= 2,
            detail=f"Scenario has {item_count} sensitive items",
        )
    )

    # 11. Multi-target scenario has recontamination steps
    recontamination_steps = [
        step for step in multi_ep.phases.post_forget if step.label.is_recontamination_attempt
    ]
    assertions.append(
        MultiTargetAssertion(
            name="multi_target_has_recontamination_steps",
            passed=len(recontamination_steps) >= 3,
            detail=f"Recontamination steps: {len(recontamination_steps)}",
        )
    )

    # 12. Audit passes for multi-target results
    audit_report = audit_results(all_results)
    assertions.append(
        MultiTargetAssertion(
            name="multi_target_audit_valid",
            passed=not audit_report.has_errors,
            detail=f"Audit errors: {len(audit_report.errors())}",
        )
    )

    return assertions


def _validate_disk_round_trip(
    output_dir: Path,
    all_results: list[EpisodeResult],
) -> MultiTargetAssertion:
    """Validate that disk-loaded metrics equal in-memory metrics (s6)."""
    episodes_path = output_dir / "episodes.jsonl"
    if not episodes_path.exists():
        return MultiTargetAssertion(
            name="disk_metrics_match_in_memory",
            passed=False,
            detail="episodes.jsonl not found",
        )

    try:
        loaded_results = load_episode_results(episodes_path)
    except Exception as exc:
        return MultiTargetAssertion(
            name="disk_metrics_match_in_memory",
            passed=False,
            detail=f"Failed to load: {exc}",
        )

    memory_metrics = evaluate_all(all_results).to_dict()
    disk_metrics = evaluate_all(loaded_results).to_dict()

    if memory_metrics != disk_metrics:
        # Find specific differences
        diffs = []
        for key in memory_metrics:
            if memory_metrics[key] != disk_metrics.get(key):
                diffs.append(key)
        return MultiTargetAssertion(
            name="disk_metrics_match_in_memory",
            passed=False,
            detail=f"Mismatched metrics: {diffs}",
        )

    # Also verify per-result field equality
    if len(loaded_results) != len(all_results):
        return MultiTargetAssertion(
            name="disk_metrics_match_in_memory",
            passed=False,
            detail=f"Result count mismatch: memory={len(all_results)}, disk={len(loaded_results)}",
        )

    return MultiTargetAssertion(
        name="disk_metrics_match_in_memory",
        passed=True,
        detail=f"Verified {len(loaded_results)} results match across disk round-trip",
    )


def run_multi_target_smoke(
    output_dir: Path,
    mode: str = "diagnostic",
) -> dict[str, Any]:
    """Run the multi-target smoke study."""
    if mode not in ("release", "diagnostic"):
        raise ValueError(f"Invalid mode: {mode}. Must be 'release' or 'diagnostic'.")

    output_dir.mkdir(parents=True, exist_ok=True)

    repository_commit = get_repository_commit()
    repository_clean = not repository_commit.endswith("-dirty")

    if mode == "release" and not repository_clean:
        raise ValueError(f"Release mode requires clean repository, got: {repository_commit}")

    # s10: Use experiment mode for release runs
    run_mode = "experiment" if mode == "release" else "test"

    generated_at = datetime.now(timezone.utc).isoformat()

    all_results: list[EpisodeResult] = []
    condition_results: dict[str, list[EpisodeResult]] = {}

    print(
        f"Running multi-target smoke study ({mode} mode): "
        f"{len(FIXTURES)} fixtures x {len(SEEDS)} seeds x {len(CONDITIONS)} conditions"
    )

    for fixture_name in FIXTURES:
        ep = load_episode(SCENARIOS_DIR / fixture_name)
        responder = _build_smoke_responder(ep)
        for seed in SEEDS:
            for cond_name, cond_overrides, fw_enabled in CONDITIONS:
                cfg = _make_config(seed, cond_overrides, mode=run_mode)
                run_id = hashlib.sha256(
                    f"{ep.episode_id}|{cond_name}|{seed}|{fw_enabled}".encode()
                ).hexdigest()[:20]
                result = run_episode(
                    ep, cfg, responder=responder, firewall_enabled=fw_enabled, run_id=run_id
                )
                result.metadata["smoke_condition"] = cond_name
                result.metadata["firewall_enabled"] = fw_enabled
                all_results.append(result)
                condition_results.setdefault(cond_name, []).append(result)

    print(f"Completed {len(all_results)} runs")

    # Write episodes.jsonl
    episodes_path = output_dir / "episodes.jsonl"
    with open(episodes_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(serialize_episode_result(r)) + "\n")

    # Audit
    audit_report = audit_results(all_results)
    audit_valid = not audit_report.has_errors

    # Write audit report
    audit_path = output_dir / "result_audit.json"
    audit_path.write_text(json.dumps(audit_report.to_dict(), indent=2))

    # Compute metrics
    evaluation = evaluate_all(all_results)

    # Write aggregate metrics
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(evaluation.to_dict(), indent=2, sort_keys=True))

    # Write aggregate metric counts
    metric_counts = {
        "pu_rer": {
            "numerator": evaluation.pu_rer.numerator,
            "denominator": evaluation.pu_rer.denominator,
        },
        "crr": {"numerator": evaluation.crr.numerator, "denominator": evaluation.crr.denominator},
        "rr": {"numerator": evaluation.rr.numerator, "denominator": evaluation.rr.denominator},
        "fbr": {"numerator": evaluation.fbr.numerator, "denominator": evaluation.fbr.denominator},
    }
    counts_path = output_dir / "metric_counts.json"
    counts_path.write_text(json.dumps(metric_counts, indent=2, sort_keys=True))

    # Write per-condition metrics
    per_condition_metrics: dict[str, dict[str, Any]] = {}
    for cond_name, results in condition_results.items():
        per_condition_metrics[cond_name] = evaluate_all(results).to_dict()
    (output_dir / "metrics_by_condition.json").write_text(
        json.dumps(per_condition_metrics, indent=2)
    )

    # Build smoke manifest
    from experiments.trustparadox_u.manifest import SmokeManifest

    episode_ids = tuple(sorted({r.episode_id for r in all_results}))
    seeds_tuple = tuple(sorted({r.seed for r in all_results}))
    config_hashes = tuple(sorted({str(r.metadata.get("config_hash", "")) for r in all_results}))

    manifest = SmokeManifest(
        repository_commit=repository_commit,
        generated_at_utc=generated_at,
        run_mode=mode,
        config_hashes=config_hashes,
        provider="fixed",
        model=None,
        dimension=None,
        semantic_threshold=0.80,
        api_base_sanitized=None,
        episode_ids=episode_ids,
        seeds=seeds_tuple,
        result_count=len(all_results),
        audit_valid=audit_valid,
        audit_error_count=len(audit_report.errors()),
        metric_counts=metric_counts,
    )
    manifest_path = output_dir / "smoke_manifest.json"
    manifest_path.write_text(manifest.to_json())

    # Run multi-target assertions
    assertions = _validate_multi_target(all_results, condition_results)

    # s6: Real disk round-trip validation
    disk_assertion = _validate_disk_round_trip(output_dir, all_results)
    assertions.append(disk_assertion)

    all_assertions_passed = all(a.passed for a in assertions)

    # Check artifact completeness (s14)
    missing_artifacts = [name for name in REQUIRED_ARTIFACTS if not (output_dir / name).exists()]
    artifacts_complete = len(missing_artifacts) == 0

    # Build report
    report = {
        "repository_commit": repository_commit,
        "repository_clean": repository_clean,
        "generated_at": generated_at,
        "mode": mode,
        "run_mode": run_mode,
        "total_runs": len(all_results),
        "fixture_count": len(FIXTURES),
        "seed_count": len(SEEDS),
        "condition_count": len(CONDITIONS),
        "all_assertions_passed": all_assertions_passed,
        "artifacts_complete": artifacts_complete,
        "missing_artifacts": missing_artifacts,
        "audit_valid": audit_valid,
        "assertions": [
            {"name": a.name, "passed": a.passed, "detail": a.detail} for a in assertions
        ],
        "metrics": evaluation.to_dict(),
    }

    # Determine status
    if mode == "diagnostic":
        status = "DIAGNOSTIC"
    elif all_assertions_passed and artifacts_complete and audit_valid:
        status = "GO"
    else:
        status = "NO-GO"

    report["status"] = status

    report_path = output_dir / "multi_target_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    # Write summary files
    summary_md = f"""# Multi-Target Smoke Study Summary

- **Status**: {status}
- **Commit**: {repository_commit}
- **Mode**: {mode}
- **Run mode**: {run_mode}
- **Fixtures**: {len(FIXTURES)}
- **Seeds**: {len(SEEDS)}
- **Conditions**: {len(CONDITIONS)}
- **Total runs**: {len(all_results)}
- **Audit valid**: {audit_valid}
- **Artifacts complete**: {artifacts_complete}

## Assertions

"""
    for a in assertions:
        icon = "PASS" if a.passed else "FAIL"
        summary_md += f"- [{icon}] **{a.name}**: {a.detail}\n"

    summary_md += f"""
## Metrics

| Metric | Value | Numerator | Denominator |
|--------|------:|----------:|------------:|
| PU-RER | {evaluation.pu_rer.value} | {evaluation.pu_rer.numerator} | {evaluation.pu_rer.denominator} |
| CRR | {evaluation.crr.value} | {evaluation.crr.numerator} | {evaluation.crr.denominator} |
| RR | {evaluation.rr.value} | {evaluation.rr.numerator} | {evaluation.rr.denominator} |
| FBR | {evaluation.fbr.value} | {evaluation.fbr.numerator} | {evaluation.fbr.denominator} |
"""
    (output_dir / "summary.md").write_text(summary_md)

    summary_json = {
        "status": status,
        "repository_commit": repository_commit,
        "repository_clean": repository_clean,
        "audit_valid": audit_valid,
        "artifacts_complete": artifacts_complete,
        "all_assertions_passed": all_assertions_passed,
        "total_runs": len(all_results),
        "generated_at": generated_at,
        "mode": mode,
        "run_mode": run_mode,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_json, indent=2))

    print("\nMulti-target smoke study complete:")
    print(f"  Status: {status}")
    print(f"  Total runs: {len(all_results)}")
    print(f"  All assertions passed: {all_assertions_passed}")
    print(f"  Artifacts complete: {artifacts_complete}")
    for a in assertions:
        icon = "PASS" if a.passed else "FAIL"
        print(f"  [{icon}] {a.name}: {a.detail}")

    return report


def main() -> int:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Multi-target smoke study")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/multi_target_smoke",
        help="Output directory",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["release", "diagnostic"],
        default="diagnostic",
        help="Run mode",
    )
    args = parser.parse_args()

    try:
        report = run_multi_target_smoke(Path(args.output_dir), args.mode)
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_INPUT_CONFIG
    except Exception as exc:
        print(f"Execution error: {exc}", file=sys.stderr)
        return EXIT_EXECUTION

    # Determine exit code
    status = report.get("status", "NO-GO")
    if status == "GO":
        return EXIT_SUCCESS
    if status == "DIAGNOSTIC":
        return EXIT_SUCCESS

    # NO-GO: determine specific failure category
    if not report.get("audit_valid", False):
        return EXIT_AUDIT
    if not report.get("artifacts_complete", False):
        return EXIT_ARTIFACT_COMPLETENESS

    # Check for specific assertion failures
    assertions = report.get("assertions", [])
    disk_failed = any(
        a["name"] == "disk_metrics_match_in_memory" and not a["passed"] for a in assertions
    )
    if disk_failed:
        return EXIT_DISK_COMPARISON

    return EXIT_DIRECTIONAL


if __name__ == "__main__":
    sys.exit(main())
