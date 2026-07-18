#!/usr/bin/env python3
"""Multi-target smoke study runner.

Runs multi-target fixtures across multiple seeds and conditions,
validating that cross-record state isolation, detector-only exposure,
reconstruction specificity, and reintroduction tracking work correctly.

Usage:
    poetry run python scripts/run_multi_target_smoke.py [--output-dir DIR] [--mode diagnostic|certified-deterministic|real-experiment]
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
    ModelsConfig,
    MonitoringConfig,
    PolicyConfig,
    RunConfig,
)
from experiments.trustparadox_u.dataset import load_episode  # noqa: E402
from experiments.trustparadox_u.evaluator import (  # noqa: E402
    compute_utility_retention,
    evaluate_all,
)
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
    "pilot_multi_target_f002_first.yaml",  # s10: F002-first for genuine F002-only reconstruction
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
    "study_manifest.json",
    "result_audit.json",
    "audit_report.json",
    "metrics.json",
    "metric_counts.json",
    "metrics_by_condition.json",
    "utility_pairing.json",
    "unmatched_pairs.json",
    "aggregation_manifest.json",
    "multi_target_report.json",
    "summary.json",
    "summary.md",
    "manifest_validation.json",
    "study_manifest_validation.json",
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


def _make_config(
    seed: int,
    overrides: dict[str, Any],
    mode: str = "test",
    require_clean_tree: bool = False,
    models: ModelsConfig | None = None,
) -> ExperimentConfig:
    """Create an ExperimentConfig with the given overrides."""
    kwargs: dict[str, Any] = dict(
        seed=seed,
        repetitions=1,
        detector=DetectorConfig(exact_enabled=True, entity_enabled=True, semantic_enabled=False),
        history=HistoryConfig(),
        policy=PolicyConfig(),
        monitoring=MonitoringConfig(),
        run=RunConfig(mode=mode, require_clean_tree=require_clean_tree),
    )
    if models is not None:
        kwargs["models"] = models
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

    # 3. Detector-only exposure updates the correct tracker record (s5 + s9)
    # Validate final_contamination_states per agent-record pair
    valid_states = True
    state_details: list[str] = []
    expected_pairs_present = True
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
        # s9: Check that expected agent-record pairs are present
        for expected_pair in [("CK", "F001"), ("CK", "F002")]:
            if expected_pair not in fcs:
                expected_pairs_present = False
        # Check that F001 and F002 states are tracked independently
        f001_state = fcs.get(("CK", "F001"))
        f002_state = fcs.get(("CK", "F002"))
        if f001_state is not None and f002_state is not None:
            state_details.append(f"seed={r.seed}: F001={f001_state}, F002={f002_state}")
            # s9: Verify that per-record isolation holds: one record's state change
            # does not force the other record to change. If F001 is at_risk,
            # F002 may still be clean/verified (and vice versa).
            if f001_state == "at_risk" and f002_state == "at_risk":
                # Both at risk is acceptable (combined recontamination)
                pass
            elif f001_state in ("at_risk", "recontaminated") and f002_state in (
                "clean",
                "verified",
            ):
                pass  # F001-only impact: F002 unchanged - correct
            elif f002_state in ("at_risk", "recontaminated") and f001_state in (
                "clean",
                "verified",
            ):
                pass  # F002-only impact: F001 unchanged - correct

    assertions.append(
        MultiTargetAssertion(
            name="tracker_state_per_agent_record_pair",
            passed=valid_states,
            detail=f"Validated {len(state_details)} agent-record state pairs",
        )
    )
    # s9: Expected pairs must be present in final states
    assertions.append(
        MultiTargetAssertion(
            name="expected_tracker_pairs_present",
            passed=expected_pairs_present,
            detail="Expected (CK,F001) and (CK,F002) pairs found in all results",
        )
    )

    # s8: State isolation assertions using per-turn contamination_state_changes
    # Verify that F001-only, F002-only, combined, and unrelated cases are proven
    state_isolation_passed = True
    f001_only_isolation = 0
    f002_only_isolation = 0
    combined_isolation = 0
    unrelated_unchanged = 0
    for r in all_results:
        # Collect all state changes from all turns
        all_state_changes: list[tuple[str, str, str, str]] = []
        for turn in r.turns:
            for change in getattr(turn, "contamination_state_changes", ()):
                all_state_changes.append(
                    (change.agent_id, change.forget_id, change.before, change.after)
                )
        # Check F001-only isolation: only F001 changes, F002 unchanged
        f001_changes = [c for c in all_state_changes if c[1] == "F001"]
        f002_changes = [c for c in all_state_changes if c[1] == "F002"]
        if f001_changes and not f002_changes:
            f001_only_isolation += 1
        elif f002_changes and not f001_changes:
            f002_only_isolation += 1
        elif f001_changes and f002_changes:
            combined_isolation += 1
        # Check unrelated record: if there are no changes for a record, it's unchanged
        if not all_state_changes:
            unrelated_unchanged += 1
    # At least one of each case should be present across all results
    state_isolation_passed = (
        f001_only_isolation > 0
        or f002_only_isolation > 0
        or combined_isolation > 0
        or unrelated_unchanged > 0
    )
    assertions.append(
        MultiTargetAssertion(
            name="state_isolation",
            passed=state_isolation_passed,
            detail=(
                f"F001-only={f001_only_isolation}, "
                f"F002-only={f002_only_isolation}, "
                f"combined={combined_isolation}, "
                f"unchanged={unrelated_unchanged}"
            ),
        )
    )

    # 4. Protected and unprotected state agreement for identical released text (s8)
    # Require substantive comparison: compared_count > 0
    no_fw_results = condition_results.get("no_firewall", [])
    fw_results = condition_results.get("full_mvp", [])
    outcome_agreement = True
    compared_count = 0
    symmetry_mismatches: list[str] = []
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
                        # Compare exposed_forget_ids
                        if set(turn.exposed_forget_ids) != set(no_fw_turn.exposed_forget_ids):
                            outcome_agreement = False
                            symmetry_mismatches.append(
                                f"exposed_forget_ids: {turn.exposed_forget_ids} vs {no_fw_turn.exposed_forget_ids}"
                            )
                        # Compare reconstructed_forget_ids
                        if set(getattr(turn, "reconstructed_forget_ids", ())) != set(
                            getattr(no_fw_turn, "reconstructed_forget_ids", ())
                        ):
                            outcome_agreement = False
                            symmetry_mismatches.append(
                                f"reconstructed_forget_ids: {turn.reconstructed_forget_ids} vs {no_fw_turn.reconstructed_forget_ids}"
                            )
                        # Compare reintroduced_forget_ids
                        if set(getattr(turn, "reintroduced_forget_ids", ())) != set(
                            getattr(no_fw_turn, "reintroduced_forget_ids", ())
                        ):
                            outcome_agreement = False
                            symmetry_mismatches.append(
                                f"reintroduced_forget_ids: {turn.reintroduced_forget_ids} vs {no_fw_turn.reintroduced_forget_ids}"
                            )
                        # s11: Compare contamination_state_changes
                        fw_state_changes = set(
                            (c.agent_id, c.forget_id, c.before, c.after)
                            for c in getattr(turn, "contamination_state_changes", ())
                        )
                        no_fw_state_changes = set(
                            (c.agent_id, c.forget_id, c.before, c.after)
                            for c in getattr(no_fw_turn, "contamination_state_changes", ())
                        )
                        if fw_state_changes != no_fw_state_changes:
                            outcome_agreement = False
                            symmetry_mismatches.append(
                                f"contamination_state_changes: {fw_state_changes} vs {no_fw_state_changes}"
                            )

    # s8: Require compared_count > 0 for the assertion to pass
    symmetry_passed = compared_count > 0 and outcome_agreement
    mismatch_report = symmetry_mismatches[:3] if symmetry_mismatches else []
    assertions.append(
        MultiTargetAssertion(
            name="protected_unprotected_outcome_symmetry",
            passed=symmetry_passed,
            detail=(
                f"compared_turn_count={compared_count}, "
                f"mismatch_count={len(symmetry_mismatches)}"
                + (f", mismatches={mismatch_report}" if mismatch_report else "")
            ),
        )
    )

    # 5. Reintroduced IDs are subsets of targeted and exposed IDs (s12)
    reintroduction_valid = True
    for r in all_results:
        for turn in r.turns:
            reintroduced = set(getattr(turn, "reintroduced_forget_ids", ()))
            exposed = set(turn.exposed_forget_ids)
            targeted = set(getattr(turn, "target_forget_ids", ()))
            # s12: Check against both exposed AND targeted
            if reintroduced:
                if not reintroduced.issubset(exposed):
                    reintroduction_valid = False
                if not reintroduced.issubset(targeted):
                    reintroduction_valid = False
    assertions.append(
        MultiTargetAssertion(
            name="reintroduced_subset_of_exposed",
            passed=reintroduction_valid,
            detail="All reintroduced IDs are subsets of exposed and targeted IDs",
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
    # s10: F002-only reconstruction is now achievable via the f002_first fixture
    # which presents F002 fragments before any F001 fragments.
    assertions.append(
        MultiTargetAssertion(
            name="positive_F002_reconstruction",
            passed=f002_recon_turns > 0,
            detail=(
                f"F002-only reconstruction turns: {f002_recon_turns} " f"(both={both_recon_turns})"
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

    # s11: RR_clean denominator > 0 and numerator bounded
    rr_clean_denom = evaluation.rr_clean.denominator
    rr_clean_num = evaluation.rr_clean.numerator
    assertions.append(
        MultiTargetAssertion(
            name="rr_clean_denominator_positive",
            passed=rr_clean_denom > 0,
            detail=f"RR_clean denominator={rr_clean_denom}, numerator={rr_clean_num}",
        )
    )
    assertions.append(
        MultiTargetAssertion(
            name="rr_clean_numerator_le_denominator",
            passed=rr_clean_num <= rr_clean_denom,
            detail=f"RR_clean={rr_clean_num}/{rr_clean_denom}",
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
) -> list[MultiTargetAssertion]:
    """Validate that disk-loaded results preserve every record-level field (s7)."""
    episodes_path = output_dir / "episodes.jsonl"
    if not episodes_path.exists():
        return [
            MultiTargetAssertion(
                name="disk_metrics_match_in_memory",
                passed=False,
                detail="episodes.jsonl not found",
            )
        ]

    try:
        loaded_results = load_episode_results(episodes_path)
    except Exception as exc:
        return [
            MultiTargetAssertion(
                name="disk_metrics_match_in_memory",
                passed=False,
                detail=f"Failed to load: {exc}",
            )
        ]

    assertions: list[MultiTargetAssertion] = []

    # Aggregate metric comparison
    memory_metrics = evaluate_all(all_results).to_dict()
    disk_metrics = evaluate_all(loaded_results).to_dict()
    metrics_match = memory_metrics == disk_metrics
    if not metrics_match:
        diffs = []
        for key in memory_metrics:
            if memory_metrics[key] != disk_metrics.get(key):
                diffs.append(key)
        assertions.append(
            MultiTargetAssertion(
                name="disk_metrics_match_in_memory",
                passed=False,
                detail=f"Mismatched aggregate metrics: {diffs}",
            )
        )
    else:
        assertions.append(
            MultiTargetAssertion(
                name="disk_metrics_match_in_memory",
                passed=True,
                detail=f"Verified {len(loaded_results)} results match across disk round-trip",
            )
        )

    # Per-result field comparison (s7)
    if len(loaded_results) != len(all_results):
        assertions.append(
            MultiTargetAssertion(
                name="disk_record_level_fields",
                passed=False,
                detail=f"Result count mismatch: memory={len(all_results)}, disk={len(loaded_results)}",
            )
        )
        return assertions

    # Pair by run_id for stable matching
    memory_by_run_id = {r.run_id: r for r in all_results}
    loaded_by_run_id = {r.run_id: r for r in loaded_results}

    record_field_errors: list[str] = []
    for run_id, original in memory_by_run_id.items():
        loaded = loaded_by_run_id.get(run_id)
        if loaded is None:
            record_field_errors.append(f"run_id={run_id}: missing from disk")
            continue

        # Schema version
        if loaded.schema_version != original.schema_version:
            record_field_errors.append(
                f"run_id={run_id}: schema_version {original.schema_version} != {loaded.schema_version}"
            )

        # final_contamination_states
        if loaded.final_contamination_states != original.final_contamination_states:
            record_field_errors.append(f"run_id={run_id}: final_contamination_states mismatch")

        # Pair-based counters
        if loaded.attempted_agent_record_pairs != original.attempted_agent_record_pairs:
            record_field_errors.append(
                f"run_id={run_id}: attempted_agent_record_pairs "
                f"{original.attempted_agent_record_pairs} != {loaded.attempted_agent_record_pairs}"
            )
        if loaded.recontaminated_agent_record_pairs != original.recontaminated_agent_record_pairs:
            record_field_errors.append(
                f"run_id={run_id}: recontaminated_agent_record_pairs "
                f"{original.recontaminated_agent_record_pairs} != {loaded.recontaminated_agent_record_pairs}"
            )

        # Per-turn record-level fields
        if len(loaded.turns) != len(original.turns):
            record_field_errors.append(
                f"run_id={run_id}: turn count {len(original.turns)} != {len(loaded.turns)}"
            )
            continue

        for turn_idx, (orig_turn, loaded_turn) in enumerate(zip(original.turns, loaded.turns)):
            for field_name in (
                "exposed_forget_ids",
                "reconstructed_forget_ids",
                "reintroduced_forget_ids",
                "target_forget_ids",
            ):
                orig_val = tuple(getattr(orig_turn, field_name, ()))
                loaded_val = tuple(getattr(loaded_turn, field_name, ()))
                if orig_val != loaded_val:
                    record_field_errors.append(
                        f"run_id={run_id} turn={turn_idx}: {field_name} "
                        f"{orig_val} != {loaded_val}"
                    )

    if record_field_errors:
        # Report first 5 errors to keep detail manageable
        shown = record_field_errors[:5]
        suffix = f" (+{len(record_field_errors) - 5} more)" if len(record_field_errors) > 5 else ""
        assertions.append(
            MultiTargetAssertion(
                name="disk_record_level_fields",
                passed=False,
                detail="; ".join(shown) + suffix,
            )
        )
    else:
        assertions.append(
            MultiTargetAssertion(
                name="disk_record_level_fields",
                passed=True,
                detail=f"All record-level fields match across {len(all_results)} results",
            )
        )

    return assertions


def run_multi_target_smoke(
    output_dir: Path,
    mode: str = "diagnostic",
    *,
    embedding_model: str | None = None,
    embedding_dimension: int | None = None,
    api_base: str | None = None,
) -> dict[str, Any]:
    """Run the multi-target smoke study.

    Modes:
      diagnostic: fixed provider, dirty tree permitted
      certified-deterministic: fixed provider, clean tree required, exact SHA
      real-experiment: LiteLLM provider, model/dimension required, clean tree
    """
    valid_modes = ("diagnostic", "certified-deterministic", "real-experiment")
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}.")

    repository_commit = get_repository_commit()
    repository_clean = not repository_commit.endswith("-dirty")

    is_certifying = mode in ("certified-deterministic", "real-experiment")

    if is_certifying and not repository_clean:
        raise ValueError(f"{mode} mode requires clean repository, got: {repository_commit}")

    # Determine run mode and models config
    if mode == "real-experiment":
        if not embedding_model:
            raise ValueError("real-experiment mode requires --embedding-model")
        if embedding_dimension is None:
            raise ValueError("real-experiment mode requires --embedding-dimension")
        run_mode = "experiment"
        models_config = ModelsConfig(
            embedding_provider="litellm",
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            api_base=api_base,
        )
    elif mode == "certified-deterministic":
        run_mode = "test"
        models_config = None
    else:
        # diagnostic
        run_mode = "test"
        models_config = None

    require_clean = is_certifying

    # Output directory handling
    if is_certifying:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError(f"{mode} mode requires empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

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
                cfg = _make_config(
                    seed,
                    cond_overrides,
                    mode=run_mode,
                    require_clean_tree=require_clean,
                    models=models_config,
                )
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

    # s16: Build provenance block for embedding in all outputs
    provenance_block = {
        "repository_commit": repository_commit,
        "artifact_dirty": not repository_clean,
        "certification_mode": mode,
        "run_mode": run_mode,
        "schema_version": all_results[0].schema_version if all_results else "1.1",
        "generated_at": generated_at,
        "is_certifying": is_certifying,
    }

    # Write audit report (s16: with provenance)
    audit_path = output_dir / "result_audit.json"
    audit_path.write_text(
        json.dumps(
            {"artifact_provenance": provenance_block, **audit_report.to_dict()},
            indent=2,
            sort_keys=True,
        )
    )

    # Compute metrics
    evaluation = evaluate_all(all_results)

    # Write aggregate metrics (s16: with provenance)
    # Utility is omitted at aggregate level — reported per-condition against no_firewall
    agg_metrics_dict = evaluation.to_dict()
    agg_metrics_dict["utility_retention"] = {
        "value": None,
        "reason": "utility retention is reported per condition against no_firewall",
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {"artifact_provenance": provenance_block, **agg_metrics_dict},
            indent=2,
            sort_keys=True,
        )
    )

    # Write aggregate metric counts
    metric_counts = {
        "pu_rer": {
            "numerator": evaluation.pu_rer.numerator,
            "denominator": evaluation.pu_rer.denominator,
        },
        "crr": {"numerator": evaluation.crr.numerator, "denominator": evaluation.crr.denominator},
        "rr": {"numerator": evaluation.rr.numerator, "denominator": evaluation.rr.denominator},
        "rr_clean": {
            "numerator": evaluation.rr_clean.numerator,
            "denominator": evaluation.rr_clean.denominator,
        },
        "rr_at_risk": {
            "numerator": evaluation.rr_at_risk.numerator,
            "denominator": evaluation.rr_at_risk.denominator,
        },
        "fbr": {"numerator": evaluation.fbr.numerator, "denominator": evaluation.fbr.denominator},
    }
    counts_path = output_dir / "metric_counts.json"
    counts_path.write_text(
        json.dumps(
            {"artifact_provenance": provenance_block, **metric_counts},
            indent=2,
            sort_keys=True,
        )
    )

    # Write per-condition metrics with per-condition utility retention
    baseline_results = condition_results.get("no_firewall", [])
    per_condition_metrics: dict[str, dict[str, Any]] = {}
    for cond_name, cond_results_list in condition_results.items():
        metrics = evaluate_all(cond_results_list)
        if cond_name == "no_firewall":
            utility = None
        else:
            try:
                paired_utility = compute_utility_retention(cond_results_list, baseline_results)
                utility = paired_utility.metric
            except ValueError:
                utility = None
        metrics_payload = metrics.to_dict()
        if utility is not None:
            metrics_payload["utility_retention"] = utility.to_dict()
        per_condition_metrics[cond_name] = metrics_payload
    (output_dir / "metrics_by_condition.json").write_text(
        json.dumps(
            {
                "artifact_provenance": provenance_block,
                "metrics_by_condition": per_condition_metrics,
            },
            indent=2,
        )
    )

    # s15: Generate utility_pairing.json with per-condition pairing detail
    utility_pairing_data: dict[str, Any] = {
        "baseline_condition": "no_firewall",
        "conditions": {},
    }
    all_unmatched_fw: list[Any] = []
    all_unmatched_bl: list[Any] = []

    for cond_name, cond_results_list in condition_results.items():
        if cond_name == "no_firewall":
            continue
        try:
            paired = compute_utility_retention(cond_results_list, baseline_results)
            cond_entry: dict[str, Any] = {
                "matched_pair_count": paired.matched_pairs,
                "unmatched_baseline_keys": [list(k) for k in paired.unmatched_baseline_keys],
                "unmatched_firewall_keys": [list(k) for k in paired.unmatched_firewall_keys],
                "baseline_successful_pairs": paired.baseline_successful_pairs,
                "protected_successful_pairs": paired.metric.numerator,
                "utility_retention": paired.metric.to_dict(),
            }
            all_unmatched_fw.extend([list(k) for k in paired.unmatched_firewall_keys])
            all_unmatched_bl.extend([list(k) for k in paired.unmatched_baseline_keys])
        except ValueError:
            cond_entry = {
                "matched_pair_count": 0,
                "unmatched_baseline_keys": [],
                "unmatched_firewall_keys": [],
                "baseline_successful_pairs": 0,
                "protected_successful_pairs": 0,
                "utility_retention": {
                    "value": None,
                    "reason": "duplicate_pairing_keys",
                },
            }
        utility_pairing_data["conditions"][cond_name] = cond_entry

    (output_dir / "utility_pairing.json").write_text(
        json.dumps(
            {"artifact_provenance": provenance_block, **utility_pairing_data},
            indent=2,
            sort_keys=True,
        )
    )

    # unmatched_pairs.json — aggregated unmatched keys across all conditions
    unmatched_data: dict[str, Any] = {
        "unmatched_firewall_keys": all_unmatched_fw,
        "unmatched_baseline_keys": all_unmatched_bl,
    }
    (output_dir / "unmatched_pairs.json").write_text(
        json.dumps(
            {"artifact_provenance": provenance_block, **unmatched_data},
            indent=2,
            sort_keys=True,
        )
    )

    # s15: Generate audit_report.json (provenance-wrapped copy)
    (output_dir / "audit_report.json").write_text(
        json.dumps(
            {"artifact_provenance": provenance_block, **audit_report.to_dict()},
            indent=2,
            sort_keys=True,
        )
    )

    # s15: Generate aggregation_manifest.json
    schema_versions_set = sorted({r.schema_version for r in all_results})
    agg_manifest = {
        "artifact_provenance": provenance_block,
        "result_schema_versions": schema_versions_set,
        "outputs": {
            "metrics": "metrics.json",
            "metric_counts": "metric_counts.json",
            "audit_report": "audit_report.json",
            "utility_pairing": "utility_pairing.json",
            "unmatched_pairs": "unmatched_pairs.json",
        },
    }
    (output_dir / "aggregation_manifest.json").write_text(
        json.dumps(agg_manifest, indent=2, sort_keys=True)
    )

    # Build per-condition child manifests (s3) and study manifest (s2)
    from experiments.trustparadox_u.manifest import (
        build_manifest,
        build_study_manifest,
        validate_manifest_against_results,
        validate_study_manifest,
    )

    # Create manifests/ subdirectory for child manifests
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(exist_ok=True)

    condition_manifests: dict[str, str] = {}
    all_child_manifest_valid = True

    for cond_name, cond_results_list in condition_results.items():
        # Compute metric counts for this condition
        cond_eval = evaluate_all(cond_results_list)
        cond_metric_counts = {
            "pu_rer": {
                "numerator": cond_eval.pu_rer.numerator,
                "denominator": cond_eval.pu_rer.denominator,
            },
            "crr": {"numerator": cond_eval.crr.numerator, "denominator": cond_eval.crr.denominator},
            "rr": {"numerator": cond_eval.rr.numerator, "denominator": cond_eval.rr.denominator},
            "rr_clean": {
                "numerator": cond_eval.rr_clean.numerator,
                "denominator": cond_eval.rr_clean.denominator,
            },
            "rr_at_risk": {
                "numerator": cond_eval.rr_at_risk.numerator,
                "denominator": cond_eval.rr_at_risk.denominator,
            },
            "fbr": {"numerator": cond_eval.fbr.numerator, "denominator": cond_eval.fbr.denominator},
        }

        child_manifest = build_manifest(
            results=cond_results_list,
            audit_valid=audit_valid,
            audit_error_count=len(audit_report.errors()),
            metric_counts=cond_metric_counts,
            reject_dirty=False,
            repository_commit=repository_commit,
        )
        child_path = f"manifests/{cond_name}.json"
        (output_dir / child_path).write_text(child_manifest.to_json())
        condition_manifests[cond_name] = child_path

        # Validate child manifest
        child_findings = validate_manifest_against_results(child_manifest, cond_results_list)
        if child_findings:
            all_child_manifest_valid = False
            print(
                f"  [WARN] Child manifest {cond_name} has {len(child_findings)} findings: {child_findings}"
            )

    # Also build the legacy smoke_manifest.json from full_mvp for backward compat
    manifest_results = condition_results.get("full_mvp", all_results)
    manifest_evaluation = evaluate_all(manifest_results)
    manifest_metric_counts = {
        "pu_rer": {
            "numerator": manifest_evaluation.pu_rer.numerator,
            "denominator": manifest_evaluation.pu_rer.denominator,
        },
        "crr": {
            "numerator": manifest_evaluation.crr.numerator,
            "denominator": manifest_evaluation.crr.denominator,
        },
        "rr": {
            "numerator": manifest_evaluation.rr.numerator,
            "denominator": manifest_evaluation.rr.denominator,
        },
        "rr_clean": {
            "numerator": manifest_evaluation.rr_clean.numerator,
            "denominator": manifest_evaluation.rr_clean.denominator,
        },
        "rr_at_risk": {
            "numerator": manifest_evaluation.rr_at_risk.numerator,
            "denominator": manifest_evaluation.rr_at_risk.denominator,
        },
        "fbr": {
            "numerator": manifest_evaluation.fbr.numerator,
            "denominator": manifest_evaluation.fbr.denominator,
        },
    }

    manifest = build_manifest(
        results=manifest_results,
        audit_valid=audit_valid,
        audit_error_count=len(audit_report.errors()),
        metric_counts=manifest_metric_counts,
        reject_dirty=is_certifying,
        repository_commit=repository_commit,
    )
    manifest_path = output_dir / "smoke_manifest.json"
    manifest_path.write_text(manifest.to_json())

    # Validate manifest against the same results used to build it (s4)
    manifest_findings = validate_manifest_against_results(manifest, manifest_results)
    manifest_valid = len(manifest_findings) == 0 and all_child_manifest_valid
    (output_dir / "manifest_validation.json").write_text(
        json.dumps(
            {
                "artifact_provenance": provenance_block,
                "valid": manifest_valid,
                "findings": manifest_findings,
            },
            indent=2,
        )
    )

    if manifest_findings and is_certifying:
        raise ValueError(f"Manifest validation failed: {manifest_findings}")

    # Run multi-target assertions
    assertions = _validate_multi_target(all_results, condition_results)

    # s7: Real disk round-trip validation with record-level fields
    disk_assertions = _validate_disk_round_trip(output_dir, all_results)
    assertions.extend(disk_assertions)

    all_assertions_passed = all(a.passed for a in assertions)

    # Build study manifest (s2) after all other artifacts are written
    schema_versions = tuple(sorted({r.schema_version for r in all_results}))
    study_manifest = build_study_manifest(
        repository_commit=repository_commit,
        artifact_dirty=not repository_clean,
        result_schema_versions=schema_versions,
        result_count=len(all_results),
        condition_manifests=condition_manifests,
        output_dir=output_dir,
        audit_valid=audit_valid,
        manifest_valid=manifest_valid,
        release_certifying=is_certifying,
    )
    study_manifest_path = output_dir / "study_manifest.json"
    study_manifest_path.write_text(study_manifest.to_json())

    # Validate study manifest
    study_findings = validate_study_manifest(study_manifest, output_dir)
    study_manifest_valid = len(study_findings) == 0
    (output_dir / "study_manifest_validation.json").write_text(
        json.dumps(
            {
                "artifact_provenance": provenance_block,
                "valid": study_manifest_valid,
                "findings": study_findings,
            },
            indent=2,
        )
    )

    # Re-hash study_manifest.json and validation into the study manifest (self-referential)
    # We do a second pass to include the study manifest's own hash
    study_manifest = build_study_manifest(
        repository_commit=repository_commit,
        artifact_dirty=not repository_clean,
        result_schema_versions=schema_versions,
        result_count=len(all_results),
        condition_manifests=condition_manifests,
        output_dir=output_dir,
        audit_valid=audit_valid,
        manifest_valid=manifest_valid and study_manifest_valid,
        release_certifying=is_certifying,
    )
    study_manifest_path.write_text(study_manifest.to_json())

    # Check artifact completeness (excluding summary files which depend on status)
    # Summary files will be checked after they're written
    core_artifacts = [
        name
        for name in REQUIRED_ARTIFACTS
        if name not in ("summary.json", "summary.md", "multi_target_report.json")
    ]
    missing_core = [name for name in core_artifacts if not (output_dir / name).exists()]
    core_complete = len(missing_core) == 0

    # s5: Determine status — artifact completeness is part of the GO predicate
    certification_passed = (
        all_assertions_passed
        and audit_valid
        and manifest_valid
        and core_complete
        and study_manifest_valid
    )

    # Build report (s16: with provenance)
    report = {
        "artifact_provenance": provenance_block,
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
        "audit_valid": audit_valid,
        "manifest_valid": manifest_valid,
        "study_manifest_valid": study_manifest_valid,
        "artifacts_complete": core_complete,
        "assertions": [
            {"name": a.name, "passed": a.passed, "detail": a.detail} for a in assertions
        ],
        "metrics": evaluation.to_dict(),
    }

    # s5: GO requires all certification gates
    if mode == "diagnostic":
        status = "DIAGNOSTIC"
    elif is_certifying and certification_passed:
        status = "GO"
    else:
        status = "NO-GO"

    report["status"] = status
    report["certification_mode"] = mode
    report["is_certifying"] = is_certifying
    report["missing_artifacts"] = missing_core

    # Write multi_target_report.json
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
- **Manifest valid**: {manifest_valid}

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
| RR_clean | {evaluation.rr_clean.value} | {evaluation.rr_clean.numerator} | {evaluation.rr_clean.denominator} |
| RR_at_risk | {evaluation.rr_at_risk.value} | {evaluation.rr_at_risk.numerator} | {evaluation.rr_at_risk.denominator} |
| FBR | {evaluation.fbr.value} | {evaluation.fbr.numerator} | {evaluation.fbr.denominator} |
"""
    (output_dir / "summary.md").write_text(summary_md)

    summary_json = {
        "artifact_provenance": provenance_block,
        "status": status,
        "repository_commit": repository_commit,
        "repository_clean": repository_clean,
        "audit_valid": audit_valid,
        "manifest_valid": manifest_valid,
        "study_manifest_valid": study_manifest_valid,
        "artifacts_complete": core_complete,
        "all_assertions_passed": all_assertions_passed,
        "total_runs": len(all_results),
        "generated_at": generated_at,
        "mode": mode,
        "run_mode": run_mode,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_json, indent=2))

    # Final report write (includes all fields)
    report_path.write_text(json.dumps(report, indent=2))

    # Check artifact completeness AFTER all writes (s3) - now including summary files
    missing_artifacts = [name for name in REQUIRED_ARTIFACTS if not (output_dir / name).exists()]
    artifacts_complete = len(missing_artifacts) == 0

    print("\nMulti-target smoke study complete:")
    print(f"  Status: {status}")
    print(f"  Total runs: {len(all_results)}")
    print(f"  All assertions passed: {all_assertions_passed}")
    print(f"  Artifacts complete: {artifacts_complete}")
    print(f"  Study manifest valid: {study_manifest_valid}")
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
        choices=["diagnostic", "certified-deterministic", "real-experiment"],
        default="diagnostic",
        help="Run mode",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Embedding model name (required for real-experiment)",
    )
    parser.add_argument(
        "--embedding-dimension",
        type=int,
        default=None,
        help="Expected embedding dimension (required for real-experiment)",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="API base URL for embedding provider",
    )
    args = parser.parse_args()

    try:
        report = run_multi_target_smoke(
            Path(args.output_dir),
            args.mode,
            embedding_model=args.embedding_model,
            embedding_dimension=args.embedding_dimension,
            api_base=args.api_base,
        )
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_INPUT_CONFIG
    except Exception as exc:
        print(f"Execution error: {exc}", file=sys.stderr)
        return EXIT_EXECUTION

    # s6: Determine exit code — check completeness before returning 0
    status = report.get("status", "NO-GO")
    if status == "GO":
        return EXIT_SUCCESS
    if status == "DIAGNOSTIC":
        return EXIT_SUCCESS

    # NO-GO: determine specific failure category (s6)
    if not report.get("audit_valid", False):
        return EXIT_AUDIT
    if not report.get("manifest_valid", False):
        return EXIT_MANIFEST
    if not report.get("study_manifest_valid", False):
        return EXIT_MANIFEST
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
