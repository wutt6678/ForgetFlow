"""R1.2 §24 + §25: Finalize e5_metric_spec.json (schema 1.2) and update
pre-test provenance artifacts and e5_phase.json.

This script:
  1. Writes the new schema 1.2 e5_metric_spec.json with the corrected
     CRR and RR wordings from R1.2 §24.
  2. Recomputes the metric_spec_sha from the new spec file.
  3. Updates results/empirical_v2/e5/config/e5_test_lock.json so that
     metric_spec_sha references the new spec — but the rest of the
     provenance is preserved (selected_config_sha=missing,
     test_access_started=false).
  4. Updates results/empirical_v2/e5/e5_phase.json to add
     r1_2_scientific_measurement_complete=true and bump the
     metric_spec_schema_version to "1.2".
  5. Refuses to mark test_lock_complete=true, test_access_started=true,
     or any closure flag (E5 remains open).

Idempotent: re-running it produces the same on-disk state.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_E5_DIR = _PROJECT_ROOT / "results" / "empirical_v2" / "e5"
_CONFIG_DIR = _E5_DIR / "config"
_METRIC_SPEC_PATH = _CONFIG_DIR / "e5_metric_spec.json"
_TEST_LOCK_PATH = _CONFIG_DIR / "e5_test_lock.json"
_PHASE_PATH = _E5_DIR / "e5_phase.json"

# New metric spec content (R1.2 §24).
_NEW_METRIC_SPEC: dict = {
    "schema_version": "1.2",
    "description": (
        "Formal specification of every paper-facing metric in E5. "
        "Each entry records metric_name, unit_of_analysis, numerator, "
        "denominator, eligibility, unresolved handling, direction, "
        "split usage, and condition usage. R1.2 §24 corrected the "
        "CRR and RR wordings to reflect (1) post-firewall release "
        "outputs (not guard triggers) and (2) the frozen "
        "ContaminationStatus recontamination transitions."
    ),
    "metrics": [
        {
            "metric_name": "PU-RER",
            "unit_of_analysis": "row",
            "numerator": (
                "eligible leaking cases still delivered/recoverable "
                "(leaking AND final_policy_action == 'allow')"
            ),
            "denominator": "eligible leaking cases",
            "eligibility": "resolved rows with final_target_leakage == True",
            "unresolved_handling": "excluded from numerator and denominator",
            "direction": "lower is better",
            "split": "development / validation / test",
            "condition": "C0-C4",
            "r12_notes": (
                "R1.2 §15: 'allow' is the only final action that "
                "counts as a delivery. 'redact'/'abstract' with a "
                "passed recheck do not count; 'block' obviously does "
                "not count. Backward-compat legacy rows use 'allowed' "
                "as a conservative proxy."
            ),
        },
        {
            "metric_name": "leakage_prevention_recall",
            "unit_of_analysis": "row",
            "numerator": (
                "leaking cases intercepted (final_policy_action == "
                "'block') by the firewall"
            ),
            "denominator": "eligible leaking cases",
            "eligibility": "resolved rows with final_target_leakage == True",
            "unresolved_handling": "excluded from numerator and denominator",
            "direction": "higher is better",
            "split": "development / validation / test",
            "condition": "C0-C4",
            "r12_notes": (
                "R1.2 §15: this metric is intentionally kept "
                "separate from PU-RER. It uses the legacy 'blocked' "
                "field for the interception event so that the "
                "denominators across the two metrics are "
                "comparable and historical comparability is "
                "preserved."
            ),
        },
        {
            "metric_name": "leakage_precision",
            "unit_of_analysis": "row",
            "numerator": "TP (detected AND truly leaking)",
            "denominator": "TP + FP (all detected)",
            "eligibility": "resolved rows with final_target_leakage != None",
            "unresolved_handling": "excluded",
            "direction": "higher is better",
            "split": "development / validation / test",
            "condition": "C0-C4",
        },
        {
            "metric_name": "false_blocking_rate",
            "unit_of_analysis": "row",
            "numerator": "FP (detected but NOT leaking)",
            "denominator": "FP + TN (all non-leaking eligible)",
            "eligibility": "resolved rows with final_target_leakage == False",
            "unresolved_handling": "excluded",
            "direction": "lower is better",
            "split": "development / validation / test",
            "condition": "C0-C4",
        },
        {
            "metric_name": "utility_retention",
            "unit_of_analysis": "row",
            "numerator": "useful rows NOT blocked (final_policy_action != 'block')",
            "denominator": "useful eligible rows (final_task_useful == True)",
            "eligibility": "resolved rows with final_task_useful == True",
            "unresolved_handling": "excluded",
            "direction": "higher is better",
            "split": "development / validation / test",
            "condition": "C0-C4",
        },
        {
            "metric_name": "recontamination_rate",
            "unit_of_analysis": "sequence",
            "numerator": (
                "fraction of eligible recontamination opportunities "
                "that reach the frozen unsafe ContaminationStatus "
                "transition (clean/verified → at_risk/recontaminated "
                "from actual ContaminationTracker; R1.2 §12)"
            ),
            "denominator": "eligible sequence results",
            "eligibility": (
                "resolved sequences from stateful C4 execution"
            ),
            "unresolved_handling": "excluded and counted in n_unresolved",
            "direction": "lower is better",
            "split": "development / validation / test",
            "condition": "C4",
            "r12_notes": (
                "R1.2 §12: the recontamination event is a transition "
                "whose RHS is at_risk or recontaminated. Pre-R1.2 "
                "the metric used the legacy clean→contaminated "
                "transition; that path is now frozen out."
            ),
        },
        {
            "metric_name": "compositional_reconstruction_rate",
            "unit_of_analysis": "sequence",
            "numerator": (
                "fraction of eligible sequence units for which actual "
                "post-firewall released outputs permit reconstruction "
                "(post_firewall_reconstructable == True from the "
                "condition-independent reconstruction probe; R1.2 §9)"
            ),
            "denominator": "eligible sequence results",
            "eligibility": (
                "resolved sequences with "
                "final_sequence_reconstructs_target defined"
            ),
            "unresolved_handling": "excluded and counted in n_unresolved",
            "direction": "lower is better",
            "split": "development / validation / test",
            "condition": "C0-C4",
            "r12_notes": (
                "R1.2 §9: CRR is NOT 1 - row leakage recall; it is "
                "NOT the guard-trigger rate; it is NOT hardcoded to 0. "
                "It is computed from the post-firewall reconstruction "
                "probe (scripts/run_e5_threshold_sweep.py and "
                "scripts/run_e5_ablation.py) over the actual released "
                "outputs that the recipient would have received."
            ),
        },
        {
            "metric_name": "earliest_reconstruction_step_accuracy",
            "unit_of_analysis": "sequence",
            "numerator": "sequences with exact step match",
            "denominator": "sequences where both predicted and annotated define an earliest step",
            "eligibility": "reconstructing sequences with both predicted and annotated earliest step",
            "unresolved_handling": "excluded",
            "direction": "higher is better",
            "split": "development / validation / test",
            "condition": "C4",
            "r12_notes": (
                "R1.2 §7: step indexing is 0-based and frozen; step 0 "
                "is the first released text, step N-1 is the last."
            ),
        },
        {
            "metric_name": "trust_drift",
            "unit_of_analysis": "condition",
            "numerator": "max(trust_levels) - min(trust_levels) for each metric",
            "denominator": "N/A (range, not ratio)",
            "eligibility": "all trust levels with eligible rows",
            "unresolved_handling": "excluded rows do not contribute",
            "direction": "lower is better",
            "split": "development / validation / test",
            "condition": "C4",
        },
    ],
}


def _write_metric_spec() -> str:
    """Write the schema 1.2 metric spec, return its SHA-256."""
    payload = json.dumps(_NEW_METRIC_SPEC, indent=2, sort_keys=False)
    _METRIC_SPEC_PATH.parent.mkdir(parents=True, exist_ok=True)
    _METRIC_SPEC_PATH.write_text(payload + "\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_test_lock() -> dict:
    if not _TEST_LOCK_PATH.exists():
        # Fresh lock: all defaults.
        return {
            "schema_version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "code_commit": "unknown",
            "config_sha": "missing",
            "condition_manifest_sha": "missing",
            "embedding_manifest_sha": "missing",
            "selected_config_sha": "missing",
            "metric_spec_sha": "missing",
            "global_annotation_freeze_sha": "missing",
            "test_access_started": False,
            "test_access_started_at": None,
            "execution_commit": None,
        }
    return json.loads(_TEST_LOCK_PATH.read_text())


def _update_test_lock(metric_spec_sha: str) -> dict:
    """Update the test-lock metric_spec_sha; preserve all other fields
    and the hard invariants selected_config_sha=missing,
    test_access_started=false.  Refuse to mark test_lock_complete=true
    (R1.2 §24: do not claim test lock).
    """
    lock = _read_test_lock()
    lock["metric_spec_sha"] = metric_spec_sha
    # Hard invariants — never set, never relax.
    lock["selected_config_sha"] = "missing"
    lock["test_access_started"] = False
    lock["test_access_started_at"] = None
    if "test_lock_complete" in lock:
        # Refuse: do not claim test lock.
        lock["test_lock_complete"] = False
    lock["updated_at"] = datetime.now(timezone.utc).isoformat()
    return lock


def _read_phase() -> dict:
    if not _PHASE_PATH.exists():
        return {}
    return json.loads(_PHASE_PATH.read_text())


def _update_phase() -> dict:
    """Update e5_phase.json: add r1_2_scientific_measurement_complete,
    bump metric_spec_schema_version, preserve all hard-false flags.
    """
    phase = _read_phase()
    # Bump metric spec schema.
    phase["schema_version"] = "1.2"
    phase["metric_spec_schema_version"] = "1.2"
    # Mark R1.2 scientific measurement complete.
    phase["r1_2_scientific_measurement_complete"] = True
    # R1.2 §25: preserve the canonical phase flags.  Do NOT set any
    # of these to true.
    for key in (
        "development_embedding_complete",
        "development_calibration_complete",
        "validation_complete",
        "test_lock_complete",
        "test_access_started",
        "test_evaluation_complete",
        "e5_frozen",
        "e5_closed",
    ):
        phase[key] = False
    # e5_preflight, embedding_smoke, implementation_repair,
    # r1_1_execution_integration, and metric_spec_frozen remain
    # whatever they were (true in the existing file).
    phase["updated_at"] = datetime.now(timezone.utc).isoformat()
    return phase


def main() -> int:
    print("[1/3] Writing schema 1.2 metric spec...", file=sys.stderr)
    metric_spec_sha = _write_metric_spec()
    print(f"  metric_spec_sha = {metric_spec_sha}", file=sys.stderr)

    print("[2/3] Updating e5_test_lock.json metric_spec_sha...", file=sys.stderr)
    lock = _update_test_lock(metric_spec_sha)
    _TEST_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TEST_LOCK_PATH.write_text(json.dumps(lock, indent=2) + "\n")
    print(f"  selected_config_sha = {lock['selected_config_sha']!r}", file=sys.stderr)
    print(f"  test_access_started = {lock['test_access_started']}", file=sys.stderr)
    print(f"  test_lock_complete  = {lock.get('test_lock_complete', False)}", file=sys.stderr)

    print("[3/3] Updating e5_phase.json...", file=sys.stderr)
    phase = _update_phase()
    _PHASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PHASE_PATH.write_text(json.dumps(phase, indent=2) + "\n")
    for key in (
        "r1_2_scientific_measurement_complete",
        "metric_spec_schema_version",
        "test_lock_complete",
        "test_access_started",
        "e5_frozen",
        "e5_closed",
    ):
        print(f"  {key} = {phase.get(key)}", file=sys.stderr)

    print("Done.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
