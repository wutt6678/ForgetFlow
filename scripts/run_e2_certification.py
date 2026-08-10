#!/usr/bin/env python3
"""E2R-FIX-035: re-run the repaired E2 certification.

Execution sequence:
1. Lock E3 (verify phase is E2_PROMPTS_FROZEN).
2. Verify immutable 90 raw G attempts.
3. Load all artifacts from disk.
4. Compute SHA-256 hashes for all artifacts.
5. Run full E2 completion checker.
6. If all pass, transition to E2_COMPLETE.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.empirical_corpus import EMPIRICAL_PHASE_FILE  # noqa: E402
from experiments.trustparadox_u.run_e2_completion_check import (  # noqa: E402
    E2_ARTIFACT_PATHS,
    check_label_completeness_from_files,
    check_raw_pilot_completeness,
    run_completion_check,
    save_completion_report,
    sha256_file,
    transition_to_e2_complete,
)

# Artifact file paths for loading.
_ARTIFACT_LOAD_PATHS = {
    "manifest": _PROJECT_ROOT
    / "results"
    / "empirical_v2"
    / "e2_primary_trust_pilot"
    / "pilot_manifest.json",
    "request_schedule": _PROJECT_ROOT
    / "results"
    / "empirical_v2"
    / "e2_primary_trust_pilot"
    / "request_schedule.json",
    "labeling_report": _PROJECT_ROOT
    / "results"
    / "empirical_v2"
    / "e2_primary_pilot_labels"
    / "labeling_report.json",
    "agreement_report": _PROJECT_ROOT
    / "results"
    / "empirical_v2"
    / "e2_primary_pilot_labels"
    / "label_agreement_report.json",
    "pilot_analysis": _PROJECT_ROOT
    / "results"
    / "empirical_v2"
    / "e2_reanalysis"
    / "e2_reanalysis_report.json",
    "floor_effect_diagnostic": _PROJECT_ROOT
    / "results"
    / "empirical_v2"
    / "e2_reanalysis"
    / "floor_effect_diagnostic.json",
    "bounded_revision_report": _PROJECT_ROOT
    / "results"
    / "empirical_v2"
    / "e2_reanalysis"
    / "bounded_revision_report.json",
    "frozen_prompt_manifest": _PROJECT_ROOT
    / "results"
    / "empirical_v2"
    / "e2_prompt_freeze"
    / "frozen_prompt_manifest.json",
    "synthetic_regression_report": _PROJECT_ROOT
    / "results"
    / "empirical_v2"
    / "e2_synthetic_regression"
    / "synthetic_regression_report.json",
}

_EVALUATOR_RAW_PATH = (
    _PROJECT_ROOT
    / "results"
    / "empirical_v2"
    / "e2_primary_pilot_labels"
    / "evaluator_raw_responses.jsonl"
)


def _load_json(path: Path) -> dict | None:
    """Load a JSON file, returning None if missing."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, returning empty list if missing."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    return [json.loads(line) for line in lines if line.strip()]


def main() -> int:
    """Run the full E2 certification pipeline."""
    print("E2R-FIX-035: Re-running repaired E2 certification")
    print("=" * 60)

    # Step 1: Verify phase file.
    print("\n[1/6] Verifying phase state...")
    if not EMPIRICAL_PHASE_FILE.exists():
        print(f"  ERROR: Phase file not found: {EMPIRICAL_PHASE_FILE}")
        return 1
    phase_data = json.loads(EMPIRICAL_PHASE_FILE.read_text(encoding="utf-8"))
    phase = phase_data.get("phase")
    print(f"  Current phase: {phase}")
    if phase != "E2_PROMPTS_FROZEN":
        print(f"  ERROR: Phase must be E2_PROMPTS_FROZEN, got {phase}")
        return 1
    print("  Phase OK")

    # Step 2: Verify raw generation attempts.
    print("\n[2/6] Verifying raw generation attempts...")
    raw_result = check_raw_pilot_completeness()
    if not raw_result.passed:
        print(f"  ERROR: Raw pilot completeness check failed: {raw_result.failure_code}")
        return 1
    print(f"  Raw attempts: {raw_result.details.get('raw_count', '?')} records")

    # Step 3: Verify label completeness.
    print("\n[3/6] Verifying label completeness...")
    label_result = check_label_completeness_from_files()
    if not label_result.passed:
        print(f"  ERROR: Label completeness check failed: {label_result.failure_code}")
        return 1
    print(f"  Labels: {label_result.details.get('label_count', '?')} records")

    # Step 4: Load all artifacts.
    print("\n[4/6] Loading artifacts...")
    artifacts = {}
    for name, path in _ARTIFACT_LOAD_PATHS.items():
        data = _load_json(path)
        if data is not None:
            artifacts[name] = data
            print(f"  {name}: loaded ({path.name})")
        else:
            print(f"  {name}: MISSING ({path.name})")

    # Load evaluator raw responses.
    evaluator_raw = _load_jsonl(_EVALUATOR_RAW_PATH)
    print(f"  evaluator_raw_responses: {len(evaluator_raw)} records")

    # Extract key artifacts for run_completion_check.
    manifest = artifacts.get("manifest", {})
    raw_schedule = artifacts.get("request_schedule")
    # Wrap list-based schedule in dict expected by check_schedule.
    schedule = {"requests": raw_schedule} if isinstance(raw_schedule, list) else raw_schedule
    labels_report = artifacts.get("labeling_report", {})
    analysis = artifacts.get("pilot_analysis", {})
    freeze_manifest = artifacts.get("frozen_prompt_manifest", {})
    bounded_revision_report = artifacts.get("bounded_revision_report")
    synthetic_regression_report = artifacts.get("synthetic_regression_report")
    agreement_report = artifacts.get("agreement_report")

    # Step 5: Compute SHA-256 hashes.
    print("\n[5/6] Computing artifact hashes...")
    artifact_hashes: dict[str, str | None] = {}
    for name, path in E2_ARTIFACT_PATHS.items():
        h = sha256_file(path)
        artifact_hashes[name] = h
        if h:
            print(f"  {name}: {h[:16]}...")
        else:
            print(f"  {name}: MISSING")

    # Connectivity and pilot config (from manifest).
    connectivity_config = {
        "provider": manifest.get("generator_provider", "openai"),
        "model": manifest.get("generator_model", "qwen3.7-plus"),
    }
    pilot_config = {
        "provider": manifest.get("generator_provider", "openai"),
        "model": manifest.get("generator_model", "qwen3.7-plus"),
    }

    # Step 6: Run full completion check.
    print("\n[6/6] Running full E2 completion checker...")
    report = run_completion_check(
        artifacts={"manifest": manifest},
        phase_file=phase_data,
        connectivity_config=connectivity_config,
        pilot_config=pilot_config,
        pilot_manifest=manifest,
        schedule=schedule,
        labels_report=labels_report,
        analysis=analysis,
        freeze_manifest=freeze_manifest,
        bounded_revision_report=bounded_revision_report,
        synthetic_regression_report=synthetic_regression_report,
        artifact_hashes=artifact_hashes,
        agreement_report=agreement_report,
        evaluator_raw_responses=evaluator_raw,
    )

    # Print results.
    print(f"\nCompletion check results: {len(report.checks)} checks")
    passed = sum(1 for c in report.checks.values() if c.passed)
    failed = sum(1 for c in report.checks.values() if not c.passed)
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")

    if failed > 0:
        print("\nFailed checks:")
        for c in report.checks.values():
            if not c.passed:
                print(f"  - {c.check_name}: {c.failure_code}")
                if c.details:
                    for k, v in c.details.items():
                        print(f"      {k}: {v}")

    # Save completion report.
    save_completion_report(report)
    print("\nCompletion report saved.")

    if not report.all_passed:
        print("\nCERTIFICATION FAILED: Not all checks passed.")
        print("Cannot transition to E2_COMPLETE.")
        return 1

    # Transition to E2_COMPLETE.
    print("\nAll checks passed! Transitioning to E2_COMPLETE...")
    try:
        new_phase = transition_to_e2_complete(report)
        print(f"  New phase: {new_phase.get('phase')}")
        print(
            f"  Full corpus generation authorized: {new_phase.get('full_corpus_generation_authorized')}"
        )
        print("\nE2 CERTIFICATION COMPLETE")
        return 0
    except RuntimeError as e:
        print(f"\n  ERROR during transition: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
