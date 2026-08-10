#!/usr/bin/env python3
"""E2R-FIX-010/011/012/013/014: Re-run pilot analysis from J primary labels.

Loads raw generation attempts and independent primary labels, then runs the
full reanalysis pipeline to produce:

- e2_pairing_audit.json
- e2_reanalysis_report.json
- floor_effect_diagnostic.json
- bounded_revision_report.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.trustparadox_u.empirical_corpus import (  # noqa: E402
    EmpiricalGenerationAttempt,
)
from experiments.trustparadox_u.empirical_reanalysis import (  # noqa: E402
    REANALYSIS_OUTPUT_DIR,
    run_reanalysis,
)
from experiments.trustparadox_u.empirical_relabeling import (  # noqa: E402
    IndependentPrimaryLabel,
)

# Paths
RAW_G_PATH = (
    _PROJECT_ROOT
    / "results"
    / "empirical_v2"
    / "e2_primary_trust_pilot"
    / "raw_generation_attempts.jsonl"
)
PRIMARY_LABELS_PATH = (
    _PROJECT_ROOT / "results" / "empirical_v2" / "e2_primary_pilot_labels" / "primary_labels.jsonl"
)


def load_raw_attempts() -> list[EmpiricalGenerationAttempt]:
    """Load all 90 raw generation attempts from JSONL."""
    attempts: list[EmpiricalGenerationAttempt] = []
    with RAW_G_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                data = json.loads(line)
                attempts.append(EmpiricalGenerationAttempt(**data))
    return attempts


def load_primary_labels() -> list[IndependentPrimaryLabel]:
    """Load all 90 primary J labels from JSONL."""
    labels: list[IndependentPrimaryLabel] = []
    with PRIMARY_LABELS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                data = json.loads(line)
                labels.append(IndependentPrimaryLabel(**data))
    return labels


def main() -> None:
    """Execute the full reanalysis pipeline."""
    print("E2R-FIX-010: Re-running pilot analysis from J primary labels")
    print("=" * 60)

    # Load data
    print("\nLoading raw generation attempts...")
    attempts = load_raw_attempts()
    print(f"  Loaded {len(attempts)} attempts from {RAW_G_PATH.name}")

    print("Loading primary J labels...")
    labels = load_primary_labels()
    print(f"  Loaded {len(labels)} labels from {PRIMARY_LABELS_PATH.name}")

    # Validate counts
    assert len(attempts) == 90, f"Expected 90 attempts, got {len(attempts)}"
    assert len(labels) == 90, f"Expected 90 labels, got {len(labels)}"
    print("  Validated: 90 attempts, 90 labels")

    # Validate ID set match (E2R-FIX-004)
    attempt_ids = {a.generation_attempt_id for a in attempts}
    label_ids = {lb.generation_attempt_id for lb in labels}
    assert attempt_ids == label_ids, (
        f"ID mismatch: attempts-labels={attempt_ids - label_ids}, "
        f"labels-attempts={label_ids - attempt_ids}"
    )
    print(f"  E2R-FIX-004: G↔J ID sets match exactly ({len(attempt_ids)} IDs)")

    # Run reanalysis pipeline
    print("\nRunning reanalysis pipeline...")
    report = run_reanalysis(attempts, labels, output_dir=REANALYSIS_OUTPUT_DIR)

    # Print summary
    print("\nReanalysis report summary:")
    print(f"  input_file: {report.get('input_file')}")
    print(f"  label_source: {report.get('label_source')}")
    overall = report.get("overall_metrics", {})
    print(f"  n_total_attempts: {overall.get('n_total_attempts')}")
    print(f"  n_positive_disclosures: {overall.get('n_positive_disclosures')}")
    print(f"  n_behavioral_refusals: {overall.get('n_behavioral_refusals')}")
    print(f"  n_task_compliant: {overall.get('n_task_compliant')}")
    print(f"  n_task_relevant: {overall.get('n_task_relevant')}")

    pairing = report.get("pairing_audit", {})
    print("\n  Pairing audit:")
    print(f"    total_families: {pairing.get('total_families')}")
    print(f"    complete_families: {pairing.get('complete_families')}")
    print(f"    incomplete_families: {pairing.get('incomplete_families')}")

    paired = report.get("paired_effects", {})
    print("\n  Paired effects:")
    print(f"    n_families: {paired.get('n_families')}")
    print(f"    bootstrap_method: {paired.get('bootstrap_method')}")
    print(f"    bootstrap_iterations: {paired.get('bootstrap_iterations')}")
    print(f"    bootstrap_seed: {paired.get('bootstrap_seed')}")
    hml = paired.get("high_minus_low", {})
    print(f"    disclosure_risk_difference: {hml.get('disclosure_risk_difference')}")
    print(f"    disclosure_ci95: {hml.get('disclosure_ci95')}")

    floor = report.get("floor_effect_diagnostic", {})
    print("\n  Floor effect:")
    print(f"    decision: {floor.get('decision')}")
    print(f"    overall_disclosure_rate: {floor.get('overall_disclosure_rate')}")

    print(f"\n  Bounded revision decision: {report.get('bounded_revision_decision')}")

    consistency = report.get("consistency_checks", {})
    print(f"\n  Consistency checks: all_passed={consistency.get('all_passed')}")
    for chk in consistency.get("checks", []):
        status = "OK" if chk["passed"] else "FAIL"
        print(
            f"    {chk['metric']}: overall={chk['overall']}, "
            f"trust_sum={chk['trust_level_sum']}, "
            f"scenario_sum={chk['scenario_trust_sum']} [{status}]"
        )

    # Verify artifact files exist
    expected_files = [
        "e2_pairing_audit.json",
        "e2_reanalysis_report.json",
        "floor_effect_diagnostic.json",
        "bounded_revision_report.json",
    ]
    print("\nArtifact verification:")
    for fname in expected_files:
        fpath = REANALYSIS_OUTPUT_DIR / fname
        exists = fpath.exists()
        size = fpath.stat().st_size if exists else 0
        status = "OK" if exists else "MISSING"
        print(f"  {fname}: {status} ({size} bytes)")

    print("\nIteration D reconstruction complete.")


if __name__ == "__main__":
    main()
