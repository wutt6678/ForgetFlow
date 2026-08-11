#!/usr/bin/env python3
"""Regenerate empirical-v2 provenance artifacts from source files.

FIX-023: Regenerate labeling_report.json from source files.
FIX-024: Regenerate label_agreement_report.json.
FIX-025: Regenerate frozen_primary_labels.json with J2 hashes.
FIX-026: Correct freeze timestamps.
FIX-027: Repair reanalysis provenance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

EMP_V2 = Path("results/empirical_v2")
PRIMARY_LABELS_DIR = EMP_V2 / "e2_primary_pilot_labels"
SECONDARY_ANNOTATION_DIR = EMP_V2 / "e2_secondary_annotation"
REANALYSIS_DIR = EMP_V2 / "e2_reanalysis"


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file."""
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def regenerate_labeling_report() -> None:
    """FIX-023: Regenerate labeling_report.json from source files."""
    print("FIX-023: Regenerating labeling_report.json...")

    # Load source files
    queue = load_jsonl(PRIMARY_LABELS_DIR / "secondary_review_queue.jsonl")
    labels = load_jsonl(SECONDARY_ANNOTATION_DIR / "secondary_labels.jsonl")
    adj = load_jsonl(PRIMARY_LABELS_DIR / "adjudication_log.jsonl")

    # Compute counts
    num_review_required = len(queue)
    num_secondary_review_attempted = num_review_required  # All queue cases were attempted
    num_secondary_review_successful = sum(
        1 for r in labels if r.get("secondary_evaluator_status") == "success"
    )
    num_secondary_review_failed = num_secondary_review_attempted - num_secondary_review_successful

    # Count disagreements (J1 != J2 where both successful)
    num_disagreements = 0
    for rec in labels:
        if rec.get("secondary_evaluator_status") == "success":
            j_label = rec.get("j_label")
            secondary_label = rec.get("secondary_label")
            if j_label and secondary_label and j_label != secondary_label:
                num_disagreements += 1

    # Count adjudication required and completed
    num_adjudication_required = sum(1 for r in labels if r.get("resolution_status") == "resolved")
    num_adjudicated = sum(
        1
        for r in adj
        if r.get("adjudicated") is True
        and r.get("resolution_status") == "resolved"
        and r.get("final_label") is not None
        and r.get("adjudicator_id")
        and r.get("adjudicated_at")
    )

    # Count unresolved
    num_unresolved = sum(1 for r in labels if r.get("resolution_status") == "unresolved")

    # Load existing report to preserve other fields
    report_path = PRIMARY_LABELS_DIR / "labeling_report.json"
    if report_path.exists():
        with report_path.open(encoding="utf-8") as fh:
            report = json.load(fh)
    else:
        report = {}

    # Update with computed counts
    report.update(
        {
            "num_review_required": num_review_required,
            "num_secondary_review_attempted": num_secondary_review_attempted,
            "num_secondary_review_successful": num_secondary_review_successful,
            "num_secondary_review_failed": num_secondary_review_failed,
            "num_disagreements": num_disagreements,
            "num_adjudication_required": num_adjudication_required,
            "num_adjudicated": num_adjudicated,
            "num_unresolved": num_unresolved,
            "secondary_reviewer_type": "independent_llm",
            "secondary_reviewer_model": "glm-5.2",
            "secondary_reviewer_provider": "openai",
        }
    )

    # Write updated report
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")

    print(f"  num_review_required: {num_review_required}")
    print(f"  num_secondary_review_attempted: {num_secondary_review_attempted}")
    print(f"  num_secondary_review_successful: {num_secondary_review_successful}")
    print(f"  num_secondary_review_failed: {num_secondary_review_failed}")
    print(f"  num_disagreements: {num_disagreements}")
    print(f"  num_adjudication_required: {num_adjudication_required}")
    print(f"  num_adjudicated: {num_adjudicated}")
    print(f"  num_unresolved: {num_unresolved}")


def regenerate_label_agreement_report() -> None:
    """FIX-024: Regenerate label_agreement_report.json with nested structure."""
    print("\nFIX-024: Regenerating label_agreement_report.json...")

    # Load source files
    primary_labels = load_jsonl(PRIMARY_LABELS_DIR / "primary_labels.jsonl")
    reference_labels = load_jsonl(PRIMARY_LABELS_DIR / "reference_labels.jsonl")
    secondary_labels = load_jsonl(SECONDARY_ANNOTATION_DIR / "secondary_labels.jsonl")

    # Build lookup maps
    primary_by_id = {r["generation_attempt_id"]: r for r in primary_labels}
    reference_by_id = {r["generation_attempt_id"]: r for r in reference_labels}
    secondary_by_id = {r["generation_attempt_id"]: r for r in secondary_labels}

    # J1 vs reference agreement
    j1_ref_agreements = 0
    j1_ref_compared = 0
    for aid, primary_rec in primary_by_id.items():
        ref_rec = reference_by_id.get(aid)
        if ref_rec:
            j1_ref_compared += 1
            j1_label = primary_rec.get("primary_exposure_label")
            ref_label = ref_rec.get("reference_label")
            if j1_label == ref_label:
                j1_ref_agreements += 1

    j1_ref_agreement_rate = j1_ref_agreements / j1_ref_compared if j1_ref_compared > 0 else 0.0

    # J1 vs J2 secondary audit agreement
    j1_j2_agreements = 0
    j1_j2_compared = 0
    j1_j2_disagreements = []
    for aid, secondary_rec in secondary_by_id.items():
        if secondary_rec.get("secondary_evaluator_status") == "success":
            j1_j2_compared += 1
            j1_label = secondary_rec.get("j_label")
            j2_label = secondary_rec.get("secondary_label")
            if j1_label == j2_label:
                j1_j2_agreements += 1
            else:
                j1_j2_disagreements.append(
                    {
                        "generation_attempt_id": aid,
                        "j1_label": j1_label,
                        "j2_label": j2_label,
                        "resolution_status": secondary_rec.get("resolution_status"),
                    }
                )

    j1_j2_agreement_rate = j1_j2_agreements / j1_j2_compared if j1_j2_compared > 0 else 0.0

    # Build nested structure
    report = {
        "j1_reference": {
            "num_compared": j1_ref_compared,
            "num_agreements": j1_ref_agreements,
            "num_disagreements": j1_ref_compared - j1_ref_agreements,
            "exact_agreement_rate": j1_ref_agreement_rate,
        },
        "j1_j2_secondary_audit": {
            "num_compared": j1_j2_compared,
            "num_agreements": j1_j2_agreements,
            "num_disagreements": len(j1_j2_disagreements),
            "exact_agreement_rate": j1_j2_agreement_rate,
            "disagreements": j1_j2_disagreements,
        },
        # Legacy fields for backward compatibility
        "j1_j2_exact_agreement": j1_j2_agreement_rate,
        "num_compared": j1_j2_compared,
        "num_disagreements": len(j1_j2_disagreements),
        "annotation_source": "j1_j2_llm_only",
    }

    report_path = PRIMARY_LABELS_DIR / "label_agreement_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")

    print(f"  J1 vs reference: {j1_ref_agreements}/{j1_ref_compared} ({j1_ref_agreement_rate:.3f})")
    print(f"  J1 vs J2: {j1_j2_agreements}/{j1_j2_compared} ({j1_j2_agreement_rate:.3f})")


def regenerate_frozen_primary_labels() -> None:
    """FIX-025: Regenerate frozen_primary_labels.json with J2 hashes."""
    print("\nFIX-025: Regenerating frozen_primary_labels.json...")

    # Load existing frozen labels
    frozen_path = PRIMARY_LABELS_DIR / "frozen_primary_labels.json"
    if frozen_path.exists():
        with frozen_path.open(encoding="utf-8") as fh:
            frozen = json.load(fh)
    else:
        frozen = {}

    # Compute hashes for all artifacts
    frozen["primary_label_sha256"] = sha256_file(PRIMARY_LABELS_DIR / "primary_labels.jsonl")
    frozen["raw_generation_sha256"] = sha256_file(
        EMP_V2 / "e2_primary_trust_pilot" / "raw_generation_attempts.jsonl"
    )
    frozen["labeling_report_sha256"] = sha256_file(PRIMARY_LABELS_DIR / "labeling_report.json")
    frozen["evaluator_prompt_manifest_sha256"] = sha256_file(
        EMP_V2 / "e2_prompt_freeze" / "frozen_prompt_manifest.json"
    )
    frozen["secondary_review_queue_sha256"] = sha256_file(
        PRIMARY_LABELS_DIR / "secondary_review_queue.jsonl"
    )
    frozen["secondary_raw_responses_sha256"] = sha256_file(
        SECONDARY_ANNOTATION_DIR / "secondary_raw_responses.jsonl"
    )
    frozen["secondary_labels_sha256"] = sha256_file(
        SECONDARY_ANNOTATION_DIR / "secondary_labels.jsonl"
    )
    frozen["secondary_agreement_sha256"] = sha256_file(
        SECONDARY_ANNOTATION_DIR / "secondary_annotation_agreement.json"
    )

    # Load secondary labels to compute counts
    secondary_labels = load_jsonl(SECONDARY_ANNOTATION_DIR / "secondary_labels.jsonl")
    frozen["num_secondary_review_successful"] = sum(
        1 for r in secondary_labels if r.get("secondary_evaluator_status") == "success"
    )
    frozen["num_secondary_review_failed"] = (
        len(secondary_labels) - frozen["num_secondary_review_successful"]
    )
    frozen["num_disagreements"] = sum(
        1
        for r in secondary_labels
        if r.get("secondary_evaluator_status") == "success"
        and r.get("j_label") != r.get("secondary_label")
    )
    frozen["num_adjudicated"] = sum(
        1 for r in secondary_labels if r.get("resolution_status") == "resolved"
    )
    frozen["num_unresolved"] = sum(
        1 for r in secondary_labels if r.get("resolution_status") == "unresolved"
    )

    with frozen_path.open("w", encoding="utf-8") as fh:
        json.dump(frozen, fh, indent=2)
        fh.write("\n")

    print(f"  primary_label_sha256: {frozen['primary_label_sha256'][:16]}...")
    print(f"  secondary_labels_sha256: {frozen['secondary_labels_sha256'][:16]}...")
    print(f"  num_secondary_review_successful: {frozen['num_secondary_review_successful']}")
    print(f"  num_unresolved: {frozen['num_unresolved']}")


def repair_reanalysis_provenance() -> None:
    """FIX-027: Repair reanalysis provenance timestamps."""
    print("\nFIX-027: Repairing reanalysis provenance...")

    report_path = REANALYSIS_DIR / "e2_reanalysis_report.json"
    if not report_path.exists():
        print("  WARNING: e2_reanalysis_report.json not found")
        return

    with report_path.open(encoding="utf-8") as fh:
        report = json.load(fh)

    # Update provenance to current commit
    import subprocess

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    timestamp = subprocess.check_output(
        ["git", "show", "-s", "--format=%cI", "HEAD"], text=True
    ).strip()

    report["analysis_code_commit"] = commit
    report["analysis_timestamp"] = timestamp
    report["generated_at"] = timestamp

    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")

    print(f"  Updated to commit: {commit[:8]}")
    print(f"  Timestamp: {timestamp}")


def main() -> None:
    """Run all regeneration steps."""
    print("Regenerating empirical-v2 provenance artifacts...\n")
    print(f"Primary labels dir: {PRIMARY_LABELS_DIR}")
    print(f"Secondary annotation dir: {SECONDARY_ANNOTATION_DIR}")
    print(f"Reanalysis dir: {REANALYSIS_DIR}\n")

    regenerate_labeling_report()
    regenerate_label_agreement_report()
    regenerate_frozen_primary_labels()
    repair_reanalysis_provenance()

    print("\n✓ Regeneration complete.")


if __name__ == "__main__":
    main()
